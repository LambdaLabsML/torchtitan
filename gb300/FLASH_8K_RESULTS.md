# `deepseek_v4_flash` on 64x GB300 — 8k optimization round

Branch `dsv4_flash_64xgb300`, worktree `/mnt/dgxc/worktrees/dsv4-flash`.
All numbers are 30-step runs at `seq_len=8192`, 64 ranks, mean TFLOP/s over
steps 15+ (skipping warmup/compile), peak memory from the last step line.

## Headline

| | config | TFLOP/s | peak |
|---|---|---:|---:|
| baseline (stock recipe, 4096) | `deepseek_v4_flash_64xgb300` | 24.14 | 56.14 GiB (20.3 %) |
| **8k reference** (same recipe at 8192) | `deepseek_v4_flash_8k` | **18.29** | 95.14 GiB (34.4 %) |
| best, config only | `deepseek_v4_flash_8k_ep4_blk32` | 25.45 | 76.58 GiB (27.7 %) |
| PR #18 | same config + PR #18 (sink out of the kernel), branch `dsv4_flash_pr18_attn` | 92.15 | 76.18 GiB (27.6 %) |
| **best** | PR #18 + `minimal_async_ep` + `mixed_precision_reduce=bfloat16`, `deepseek_v4_flash_pr18_asyncep_bf16reduce` on `dsv4_flash_pr18_sweep` | **100.09** | 74.78 GiB (27.0 %) |

Config tuning alone is **+39.1 %** over the 8k reference. With PR #18 on top
it is **5.0x the 8k reference (3.62x the tuned config)**, at the same memory,
with a normal loss trajectory (12.18 -> 2.82 at step 30 vs the reference's
2.84). See "The 72.5 % kernel" below for what PR #18 actually removes. Everything below is measured against the 8k reference, not the
4096 baseline: doubling context changes attention work per token, so a
cross-length comparison would not be clean.

**The replicate spread is 1.3 %**, measured the hard way -- three runs that were
meant to be different configs turned out identical (see the AC bug below) and
returned 24.74 / 24.62 / 24.42. Treat anything under ~1.5 % as noise. This is
tighter than the ~2 % assumed from earlier rounds.

## What worked

Two levers, and they compose almost exactly multiplicatively (1.170 x 1.158 =
1.355 predicted, 1.353 observed) because they attack disjoint costs: MoE
dispatch scope and attention score area.

### 1. Expert parallel degree 64 -> 4 (+20.5 % alone)

Stock is EP=64, which spreads every layer's dispatch across all 16 nodes. The
sweep, at `block_size` 128:

| EP | TFLOP/s | peak |
|---:|---:|---:|
| 64 (stock) | 18.29 | 95.14 GiB |
| 16 | 21.39 | 65.70 GiB |
| 8 | 21.84 | 67.69 GiB |
| **4** | **22.04** | 76.02 GiB |
| 1 | 20.80 | 128.98 GiB |

Composed with `block_size` 32:

| EP | TFLOP/s | peak |
|---:|---:|---:|
| 16 | 24.74 | 72.40 GiB |
| **4** | **25.45** | **76.58 GiB** |
| 2 | 25.22 | 94.66 GiB |

EP=4 is the optimum, and the reason is the node shape: 4 GPUs per node means an
EP=4 group fits entirely inside one node's NVLink mesh, so expert dispatch
never crosses InfiniBand. EP=2 ties it on throughput (0.9 %, inside noise) but
costs 18 GiB more, and EP=1 collapses -- with no expert parallelism every rank
holds all 256 routed experts for FSDP to all-gather, which is both slower and
129 GiB.

### 2. FlexAttention `block_size` 128 -> 32 (+15.8 % alone)

`deepseek_v4_flash_8k_blocksize32`: 21.18 TFLOP/s, and it *frees* 19.3 GiB.
This is the DSA block-mask granularity: at 128 the mask keeps whole 128-token
blocks that top-k only partly selected, so the kernel computes score area it
then discards. Transfers almost exactly from `pro` (+19 % there), which
confirms the mechanism is mask granularity and not flavor-specific.

`block_size` 16 was never measured -- the one untested tuning question left.

## What did not work

| lever | result | why it is a dead end, not a tuning problem |
|---|---|---|
| no activation checkpointing | **OOM** | Wants ~265 GiB at the 8k reference; still OOMs on the composed base. FullAC is load-bearing at 8192 (it was optional at 4096). |
| `SelectiveAC` | **hard error** | `Tensor cached during selective activation checkpoint has been mutated` -- a correctness guard in the AC machinery, not a memory limit. Some op in this model mutates a tensor SAC cached. |
| microbatch 2x and 4x | **hang** | Three attempts, none reached step 1. See below -- the cause is NOT settled. |
| `mixed_precision_reduce=bfloat16` | **noise** | 18.58 vs 18.29 = +1.6 %, under the 1.3-1.5 % floor. Frees 20.7 GiB though, so it is a memory lever to spend elsewhere, not a speed lever. |

### The batch hang (unresolved)

The most valuable open problem here. Peak memory at the best config is 76.58 GiB
of ~276.5 GiB -- **~200 GiB unused** -- and FullAC keeps the *stored* activation
term small by construction, so a doubled microbatch should cost tens of GiB.
Memory is not the constraint. Yet:

| job | config | batch | outcome |
|---|---|---|---|
| 330 | `ep16_blk32_batch4` | 4x | 3 h wall clock, step 0 |
| 341 | `ep16_blk32_batch2` | 2x | cancelled, 30 min static, step 0 |
| 365 | `ep4_blk32_batch2` | 2x | 6 h clock, cancelled after 43 min static, step 0 |

**The cause is not established, and an earlier diagnosis in this file was
wrong.** This was first recorded as a FlexAttention autotune hang, on the
reasoning that 341 stopped on an autotune line and its log held 64 AUTOTUNE
events "one per rank". Both halves were mistaken: the launcher passes
`--local-ranks-filter 0`, so those 64 events are 64 separate *kernel*
autotunings on rank 0 alone, not one per rank. And job 365 -- the same lever on
the best config, with double the wall clock -- froze with **zero** autotune
blocks, wedged in the DTensor redistribute / weight-materialization phase of
model init, long before any compilation.

So the failure is in init at 2x, not in autotune. Note the tiles genuinely are
pinned: all 7 reported choices carry identical `BLOCK_M=32, BLOCK_M1=16, ...`
and times within 0.2 ms of one another (311.77-311.86 ms), so autotuning is
re-benchmarking each flex kernel rather than searching tile space.

One caveat against calling it a hard deadlock: job 346 (EP=2, 1x) sat at
*exactly* the same 27389-byte log offset for over 10 minutes and then recovered
and finished at 25.22. That phase is legitimately slow. 365 sat there 43 minutes.

Next step is a stack dump from a wedged rank (`py-spy dump`, or
`PYTORCH_DISTRIBUTED_DEBUG=DETAIL` plus the NCCL flight recorder) to see which
collective is blocked -- not another long run at a different batch size.

## Attention backward: the 72.5 % kernel, and why tiles cannot fix it

The profile of the best config puts ONE kernel,
`triton_tem_fused_flex_attention_backward_2`, at **72.5 % of GPU time**:
517 ms/call mean, against 7.3 ms for its own forward. It alternates by layer --
890 ms on the 21 `compress_ratio=4` (CSA) layers, 166 ms on the 20
`compress_ratio=128`, 129 ms on the 2 `compress_ratio=1` -- so 21 layers carry
84 % of it. The trace shows the kernel at 255 registers/thread (the maximum),
99,328 B shared memory (`limitingFactors=SMEM`), `num_stages=1`. Full
analysis: `/mnt/dgxc/profiles/dsv4_flash_8k_ep4_blk32/ATTENTION_BACKWARD_FINDING.md`.

Three tile experiments on the best config, all 30 steps:

| change | TFLOP/s | vs 25.45 | verdict |
|---|---:|---:|---|
| `num_stages` 1 -> 2 (jobs 374, 382) | 25.43 / 25.41 | flat | pipelining is not the bottleneck; also widened autotune 7 -> 13 candidates (~40 min startup) |
| M1/N1/M2/N2 = 32/32/32/32 (383) | **24.06** | **-5.5 %** | larger tiles are *worse* |
| M1=64, N1=32 / M2=32, N2=64 (381) | -- | -- | **illegal**: tiles must divide `block_size=32` |

Two facts fell out of this that were not known before:

1. **`block_size` caps the backward tiles.** FlexAttention requires
   `SPARSE_Q/KV_BLOCK_SIZE` (= `block_size`) to be divisible by every tile,
   so at `block_size=32` no tile can exceed 32. The +15.8 % `block_size`
   128 -> 32 win therefore also locked the backward into its smallest tiles;
   a coarser mask paired with the larger tiles it permits was never measured.
2. Observed shared memory is exactly `2*(M1+N1)*head_dim*2` B, so GB300's
   232,448 B caps `M1+N1 <= 113`.

**What it actually was (PR #18, C. Lowman, `dsv4_flash_pr18_attn`).** The
sink was a zero-valued KV token whose score a `score_mod` replaced with the
learned `attn_sink[h]` -- a tensor that *requires grad*. FlexAttention then
has to produce `grad_score_mod_captured` through the joint graph inside the
backward template on every block, a known slow path. PR #18 removes the sink
from the kernel and applies it afterwards as `out * sigmoid(lse - sink[h])`,
which is the identical softmax (the sink's V row was zero, so it only ever
scaled the denominator; verified to 1e-15 in float64, gradients included).
Result on the tuned config: **92.15 TFLOP/s, 3.62x**, same memory. The
register/SMEM pressure in the trace was real but was a *consequence* of that
fused captured-buffer gradient, not the fundamental cost of `head_dim=512`.

**Verdict:** tile geometry cannot fix this kernel, and it turned out not to
need fixing -- it needed the captured buffer taken out of it. The fix is the seam `DSV4FlexAttention`'s own
docstring names: replace `forward` for CSA with a kernel that consumes the raw
tensors. That is branch `dsv4_flash_csa_gather_attention`
(`/mnt/dgxc/worktrees/dsv4-csa-gather`): K is V, the KV stream is one head
shared by all 64 query heads, and each query attends <=641 explicit positions,
so `dsa_gather_attention` gathers those rows once and runs the attention as
batched cuBLAS GEMMs. Config `deepseek_v4_flash_8k_ep4_blk32_gather`; result
pending as of this writing.

### The gather kernel, measured (branch `dsv4_flash_csa_gather_attention`)

Single GPU, the flash CSA shape (T=8192, H=64, D=512, n_cmp=2048, topk=512,
window=128), fwd+bwd per layer call, 5 iters after 3 warmups:

| kernel | ms/call | peak |
|---|---:|---:|
| sink-token flex (the shipped path) | 932-935 | 5.7 GiB |
| PR #18 flex (sink out of the kernel) | 520 | 5.7 GiB |
| gather, fp32 post-gather math | **92.0** | 19.6 GiB |

Correctness (job 393, 3/3 passed; `tests/unit_tests/gpu/test_dsv4_csa_gather.py`):
against an fp32 dense masked softmax over the exact positions both paths
attend, the gather path is *closer to truth than flex on every tensor* --
dswa_k 3.3e-3 vs 4.5e-3, dcmp_k 3.3e-3 vs 4.5e-3, dattn_sink 2.6e-3 vs
6.8e-3 rel-norm. A first version gathered bf16 rows and accumulated the KV
gradient in bf16 through `index_select`'s backward (1.5-2.7 % error on the KV
grads); gathering from an fp32 copy fixed it and was also faster (fp32
`index_add` beats bf16 atomics) and smaller (30 -> 19.6 GiB).

**Do not read the 520 vs 92 as the in-training comparison.** The bench feeds
random indexer tensors, so each query's 512 top-k picks scatter uniformly and
the 32 queries in a block touch nearly every KV block: a near-dense block mask,
i.e. a worst case for flex. With real data the selections cluster and flex
skips blocks; the PR #18 training run (92.15 TFLOP/s, ~8.5 s/step total) is
only possible if its attention costs far less than 21 x 520 ms. The gather
kernel's cost is *fixed* at 641 keys/query (~92 ms/layer, ~1.9 s/step)
whatever the pattern. Which is faster on real data is decided by the PR #18
profile (`/mnt/dgxc/profiles/dsv4_flash_8k_ep4_blk32_pr18/`), not by this
bench. The old kernel matched between bench (932) and profile (890 + fwd)
because its captured-buffer-gradient path was insensitive to block sparsity --
part of why it was so slow on real data.

## After PR #18: where the step goes now

Profile of the tuned config on the PR #18 attention
(`/mnt/dgxc/profiles/dsv4_flash_8k_ep4_blk32_pr18/`, job 391; 79.86 TFLOP/s
with the profiler on, 92.15 without). Same 2-step window as the sink-token
profile; `before_after.md` there has the full side-by-side.

| bucket | sink-token ms (share) | PR #18 ms (share) | change |
|---|---:|---:|---:|
| attention backward (flex) | 44,512 (72.5 %) | 4,165 (20.6 %) | **0.09x** |
| NCCL collectives | 6,977 (11.4 %) | 6,069 (30.1 %) | 0.87x |
| elementwise / reductions | 4,999 (8.1 %) | 5,203 (25.8 %) | 1.04x |
| other | 2,035 (3.3 %) | 2,023 (10.0 %) | 0.99x |
| GEMMs (dense + expert) | 1,622 (2.6 %) | 1,571 (7.8 %) | 0.97x |
| attention forward (flex) | 1,253 (2.0 %) | 1,148 (5.7 %) | 0.92x |
| **total GPU kernel time** | **61,399** | **20,178** | **0.33x** |
| NCCL exposed (serialised) | 2,709 (4.7 %) | 3,169 (17.5 %) | -- |

The attention backward kernel runs at **48 ms/call** on real data (vs 517
before). Every other bucket is unchanged in absolute terms; they only look
bigger because the denominator shrank 3x. The next levers, in order of size:

1. **NCCL, 30 % of kernel time and 17.5 % exposed.** Absolute comm time did
   not move; the compute that used to hide it is gone. MoE SendRecv is the top
   NCCL kernel (2,664 ms). Re-tune EP under PR #18 (the EP=4 choice was made
   when attention dominated), and the comm-overlap levers (async EP, DeepEP)
   are worth more now than when they were first tried.
2. **Elementwise, 26 %, ~153k launches at ~34 us.** The HC-branch fp32
   upcasts and their backward, RMS norms, casts -- all unfused because the
   flash recipe runs `compile.enable=False`. torch.compile of the blocks is
   the natural lever; the mhc.py cast cleanup that looked like <1 % against
   the old step is now a several-percent item.
3. **Attention backward, 21 %.** Still the single largest kernel. The gather
   branch does NOT help here: on real data PR #18's flex costs ~55 ms/layer
   fwd+bwd against the gather kernel's fixed ~92 ms (its cost does not fall
   with clustered selections; flex's does). Superseded -- see the gather
   branch's `gb300/CSA_GATHER_VERDICT.md`.

## Sweep on top of PR #18 (branch `dsv4_flash_pr18_sweep`, in progress)

Each config is the 92.15 TFLOP/s tuned config with one lever moved.

| lever | config | TFLOP/s | vs 92.15 | peak | note |
|---|---|---:|---:|---:|---|
| `mixed_precision_reduce=bfloat16` | `pr18_bf16reduce` | **95.54** | **+3.7 %** | 72.03 GiB | real (spread 1.3 %); was noise under the old kernel, now NCCL is 30 % of kernel time |
| `compile.enable=True` (model+loss) | `pr18_compile` | 92.18 / 92.02 (attempts 3 / 4) | +0.0 % | 77 GiB | runs after fixes, **fuses nothing** (profiled twice: eager launches unchanged). Block body silently runs eager under Dynamo. Closed. See below. |
| `moe_comm_backend=minimal_async_ep` | `pr18_asyncep` | **96.66** | **+4.9 %** | 78.12 GiB | real; +1.9 GiB only (no Kimi-style buffer blow-up), 0 allocator retries. Needed no code change: the shared dispatcher-capacity code fills `num_max_tokens_per_rank` at config time. |
| 2x microbatch | `pr18_bs2` | -- | -- | -- | crash at step 0: `CheckpointError`, MoE routed-token count differs between forward and FullAC recompute. See below. |
| 1x, `debug.deterministic` | `pr18_det` | 92.66 | +0.6 % | 76.86 GiB | deterministic mode is **free** at 1x -- so the batch numbers below are clean |
| 2x microbatch, `debug.deterministic` | `pr18_bs2_det` | 80.90 | **-12.7 %** vs 1x-det | 101.04 GiB | batch hurts |
| 3x microbatch, `debug.deterministic` | `pr18_bs3_det` | 70.55 | **-23.9 %** vs 1x-det | 134.92 GiB | batch hurts more; memory was never the limit |

**Round 2:** `pr18_asyncep_bf16reduce` -- the two comm levers stacked --
**100.09 TFLOP/s (+8.6 %)** at 74.78 GiB. Multiplicative prediction from the
solo runs was ~100.4, so they compose as expected (different collectives: MoE
all-to-all overlap vs FSDP reduce-scatter bytes). This is the current best.

**2x microbatch vs FullAC: non-deterministic MoE routing.** At 16384
tokens/rank the router dispatched two fewer tokens to the local experts during
the activation-checkpoint recompute than in the original forward (`[67898]` vs
`[67896]` int64 indices; `[N, 4096]` / `[N, 2048]` expert inputs). Top-k
routing flipped on a near-tie, i.e. floating-point nondeterminism in the gate
path at this shape -- consistent with cuBLAS choosing a split-K / atomic GEMM
at 16384 tokens that it does not choose at 8192, where 30-step runs have been
reproducible dozens of times. Because routing feeds an all-to-all, one rank's
flip changes every rank's received count. Note this is a different failure
from the old-kernel 2x/4x attempts (330/341/365), which stalled without
reaching this point; whether they would have hit it is unknown.
`determinism_check="none"` is NOT the fix -- the shapes really differ.
Retry with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (job 400) failed identically
(`[105470]` vs `[105469]`), so deterministic cuBLAS is not (the whole) answer;
remaining suspects are cuBLASLt algorithm choice, Triton kernels, the bf16x9
fp32-emulation path, or an unlisted op. **Under `debug.deterministic`
(torch.use_deterministic_algorithms, warn-only; job 402) 2x reaches 30 steps:
80.90 TFLOP/s at 101.04 GiB, zero fallback warnings.** So the flip is inside
that flag's coverage but not cuBLAS -- a float scatter/index-class op in the
block forward. 80.90 mixes the 2x effect with the deterministic-mode tax;
`pr18_det` (1x, deterministic) isolates the tax and `pr18_bs3_det` extends the
curve. The surgical fix is to find and replace that one op (torchtitan already
did exactly this for the MoE combine: `deterministic_scatter_add`). Ruled
out: cuBLASLt -- this torch's default BLAS preference is already cuBLAS, so
a "prefer cuBLAS" bisect run would have replicated job 400 (cancelled). No
float scatter/index op exists in the block forward either. `torch.topk` was
the next suspect, but `use_deterministic_algorithms` does not touch it, and
every op the flag *does* list that appears in this forward path is
integer-typed (exact atomics). So the flag worked through a side effect --
and the one that fits everything is **Inductor**: `inductor.config.deterministic`
follows the flag and disables benchmark-driven autotuning. FlexAttention
compiles with `max_autotune` + `coordinate_descent_tuning`; at an uncached
shape the forward and the FullAC recompute can each autotune and pick
different tile variants, whose rounding differences flip router near-ties by
a token or two. At 1x the node-local Inductor cache is warm so both pick the
same kernel. This also explains the old-kernel 2x/4x runs sitting in autotune
for hours. Tested on branch `dsv4_flash_pr18_noautotune`
(`max_autotune=False`, `coordinate_descent_tuning=False`): **refuted** -- 2x
still throws the CheckpointError with autotune off (job 407, `[111283]` vs
`[111282]`). Three mechanisms proposed (cuBLAS split-K, cuBLASLt, Inductor
autotune), three refuted; **the source of the flip is unidentified.** What is
established: it needs >1x microbatch to appear, `debug.deterministic=True`
prevents it and is free (92.66 vs 92.15), and it is moot for this model
because >1x microbatch is counter-productive anyway. If someone needs it, the
next step is a per-rank dump of router top-k between forward and recompute,
not more hypotheses.

The autotune-off branch did establish something else: **autotune off is
throughput-neutral at 1x (92.01 vs 92.15, job 406) with zero autotune blocks
and a 6.9-minute 30-step job.** With `kernel_options` pinned on every flex
layer -- which GB300 forces at head_dim=512 -- timing-based autotune only
ever re-benchmarked the pinned choice against variants, at a cost of up to
~10 minutes per block on any new shape (the 2x runs sat in it for hours).
Recommend adopting `max_autotune=False, coordinate_descent_tuning=False` on
the PR #18 branch: same speed, deterministic kernel choice, fast startup.

**Batch size: dead as a throughput lever on DSv4 flash.** With deterministic
mode shown to be free (92.66 vs 92.15 at 1x), the curve is clean:
1x 92.66 -> 2x 80.90 -> 3x 70.55 TFLOP/s, monotonically worse, while memory
climbs 77 -> 101 -> 135 GiB of 276. Larger microbatches are affordable and
counter-productive. Likely mechanism: the DSA indexer scores every query
against every compressed key of the whole folded microbatch
(`Indexer.select`, `einsum("shd,td->sht")` with s = t = T) and the selection
mask is a dense `[T, T + n_cmp]`, so attention-side cost per token grows with
T instead of staying flat; the extra tokens do not amortise anything the
step was paying per microbatch. (Whether attention over the folded stream is
meant to span packed documents is a model question, not addressed here.)
No bs4, no batch stack. Round-3 `pr18_stack_bs2` is not run.

## The 100.09 config, profiled on 8 nodes (32 GPUs)

`/mnt/dgxc/profiles/dsv4_flash_8k_best100_32xgb300/` (job 417). Run on 8
nodes because 5 of 16 were down; **no config change was needed** (dp_shard=-1
resolves to the world size, EP=4 divides 32, per-rank quantities unchanged) --
only the launcher's hardcoded `--nnodes 16` became a parameter that follows
the allocation.

| | TFLOP/s per GPU | peak |
|---|---:|---:|
| 100.09 config, 16 nodes, unprofiled (399) | 100.09 | 74.78 GiB |
| 100.09 config, 8 nodes, profiler on (417) | **88.86** | **108.83 GiB** |
| step 3 of the same run | 100.84 | 106.21 GiB |

The profiler costs ~13 % on this model (92.15 -> 79.86 measured on the eager
config), so 88.86 profiled corresponds to ~101-102 unprofiled: **per-GPU
throughput at 8 nodes is the same as at 16, a hair better**, as expected from
FSDP groups of 32 instead of 64. Memory +34 GiB, exactly the doubled FSDP
shard of weights + optimizer state.

Buckets vs the 16-node profile of the 92.15 config (not like-for-like: node
count AND the two comm levers differ): NCCL 30.1 % -> 22.6 % with launches
1,240 -> 552 (MinimalAsyncEP replaces NCCL SendRecv with its own
symmetric-memory kernels, which land in "other": 2,023 -> 2,585 ms); attention
backward and elementwise unchanged in absolute ms (4,124 / 4,833); total GPU
kernel time 20,178 -> 18,485 ms.

## EP re-sweep on the 100.09 recipe (round 4, branch `dsv4_flash_pr18_sweep`)

EP=4 was chosen under a *blocking* all-to-all, where a group that fits one
node's NVLink mesh won. MinimalAsyncEP overlaps that all-to-all, so the
optimum could have moved. It did not:

| EP | config | TFLOP/s | vs EP=4 | peak |
|---:|---|---:|---:|---:|
| **4** | `pr18_asyncep_bf16reduce` | **100.09** | -- | 74.78 GiB |
| 8 | `pr18_best_ep8` | 94.87 | -5.2 % | 71.36 GiB |
| 16 | `pr18_best_ep16` | 82.90 | -17.2 % | 75.95 GiB |
| 32 | `pr18_best_ep32` | 67.44 | -32.6 % | 93.35 GiB |
| 64 | `pr18_best_ep64` | not run | | (cancelled: curve already unambiguous) |

Everything but `expert_parallel_degree` is identical to the 100.09 config
(verified in-process before submission). Steeply monotonic: keeping each
expert group inside a single 4-GPU node is worth ~5 % per doubling and more
beyond, overlap or not -- cross-node dispatch costs more than its collective's
latency. Peak memory *rises* with EP because MinimalAsyncEP's symmetric-memory
buffers scale with group size.

Below 4: EP=1 is illegal with MinimalAsyncEP (the dispatcher requires EP > 1)
and lost anyway with the standard dispatcher (20.80 vs 22.04 at 129 GiB, old
kernel) because every rank then FSDP-all-gathers all 256 experts per layer
(~12.9 GB/layer); EP=2 tied EP=4 at +18 GiB on the old kernel. **EP=4 is a
real optimum on 4-GPU nodes: exactly one node per expert group.**

**torch.compile vs the DSA block mask.** `dsa_mask_mod` indexes a dense
`selected_mask` tensor that `_build_block_mask` builds *inside* the
transformer block. Under torchtitan's per-block compile that tensor is an
Inductor intermediate whose layout is not yet fixed when the flex template
renders the `mask_mod`, and Inductor asserts. Standalone flex never sees this:
the tensor is a real input. Attempt 2 -- `@torch.compiler.disable` on `_build_block_mask` so the mask is
built eagerly -- failed with `torch._dynamo.exc.Unsupported: Skip inlining
torch.compiler.disable()'d function`: torchtitan compiles each block with
`fullgraph=True` (`distributed/compile.py:70`), so the graph break the
workaround needs is fatal.

Attempt 3 -- `fullgraph=False` on this branch (job 401) -- **runs, and is
flat: 92.18 vs 92.15**, +0.9 GiB. So compile is *possible* on DSv4 flash but
as configured buys nothing. The likely reason is fragmentation: every block
breaks at the eager mask build and again at the all-to-all dispatcher's
`.tolist()` host sync, so the HC-branch elementwise chains that motivated
compiling are split across small subgraphs. **Confirmed by profile (job 408,
`/mnt/dgxc/profiles/dsv4_flash_8k_ep4_blk32_pr18_compiled/`, `fusion.md`):
compile fused nothing in the blocks.** Eager elementwise launches 150,512 ->
150,428; Inductor fused kernels 258 -> 306 where the 258 are flex's own and
the extra ~24/step are the compiled loss. The blocks ran eagerly -- Dynamo
graph-broke pervasively (SPMD typecheck contexts, the dispatcher's `.tolist()`
sync, the eager mask build). Hoisting the mask removes one break source of
several; making the block compile-friendly is a real engineering task with a
bounded prize (elementwise is 25.8 % of kernel time; fusing half is ~10-13 %).
**`TORCH_LOGS=graph_breaks` (job 413) found exactly one break per block**, and
it explains the whole thing: the `torch.compiler.disable`'d `_build_block_mask`
call at `attention.py:270` is reached *inside* `with spmd.no_typecheck():`, and
Dynamo cannot split a graph inside a context manager it does not model
("Attempted to graph break in an active context manager that doesn't support
graph breaking"), so it abandoned the entire block frame. Nothing compiled;
the dispatcher's `.tolist()` never even appeared as a break because tracing
never reached it. Attempt 4 (job 414) built the mask in a disabled helper *before* entering
the context: **still flat, 92.02** (tps 1038 vs 1040 eager). So either
another break inside a context manager is abandoning the frame (the outer
`Attention.forward` uses `spmd.local()` blocks; the flex call itself is still
inside `no_typecheck()`), or the block compiled and fusion bought nothing.
The attempt-4 `TORCH_LOGS=graph_breaks` run (job 415) shows **one clean
break per block** at the disabled `_eager_block_mask` call -- no "active
context manager" clause this time, resume point in the sharding wrapper -- so
Dynamo now splits the block around the mask build rather than abandoning it,
and **no `.tolist()` break appears**, so the MoE half traced through as well.
That leaves one explanation for the flat 92.02: the block compiled and fusion
bought nothing. **Profiled (job 416, `/mnt/dgxc/profiles/dsv4_flash_8k_ep4_blk32_pr18_compiled4/`):
still nothing fused** -- eager elementwise launches 150,512 -> 150,428 ->
150,376 (eager / attempt 3 / attempt 4), Inductor fused kernels 258 -> 306 ->
306 (the loss). Even with a clean split, the block body runs eagerly: the
graphs Dynamo emits contain none of it. Remaining explanation: silent frame
skipping for a reason `graph_breaks` does not log (SPMD `local()` /
`no_typecheck()` or the redistribution wrapper are the candidates;
`TORCH_LOGS=graph_code` would show what each graph contains). That is a
Dynamo-level investigation, not a config change.

**Verdict: torch.compile is not a lever on DSv4 flash today.** Four attempts
(Inductor FlexibleLayout assert; fullgraph forbidding the fix; clean-breaking
reorder; profile confirming no fusion), two graph-break runs, two profiles.
Prize if it ever works: ~10-13 % of step time. Not stacked with the comm
levers.

## Configs added

In `torchtitan/models/deepseek_v4/config_registry.py`:

- `deepseek_v4_flash_8k` — the 8k reference
- `deepseek_v4_flash_8k_ep{1,2,4,8,16}` — EP sweep at `block_size` 128
- `deepseek_v4_flash_8k_blocksize32`, `_bf16reduce`, `_no_ac`, `_sac`
- `_flash_8k_ep_blk(ep, block_size)` — the composition helper
- **`deepseek_v4_flash_8k_ep4_blk32`** — the best config
- `deepseek_v4_flash_8k_ep{2,8,16}_blk32`, `_ep16_blk16`
- `_ep16_blk32_{sac,no_ac,batch2,batch4}`

Supporting changes outside the registry:

- `models/deepseek_v4/__init__.py`: raised the `deepseek_v4_flash` context cap
  from 4096 to 65536 so 8192 is expressible at all.
- `config/configs.py`: widened `mixed_precision_reduce` to accept `bfloat16`.

## A measurement bug worth not repeating

The first SAC and no-AC configs assigned `config.model_spec.ac`, which is not a
dataclass field on `model_spec`. Python accepted it as a stray attribute on a
non-slots dataclass and it controlled nothing, so both runs silently executed
plain FullAC and looked like flat results with slightly different memory. The
operative field is `config.activation_checkpoint`. Two jobs were wasted and
`no_ac` was briefly reported as "fits and is flat" when it had never run.

**A second one.** `git add -A` in this worktree swept a colleague's *uncommitted*
`mhc.py` edit (fp32 HC parameters, which FSDP2 rejects -- it killed jobs
377/378) into ledger commit `172428400`. Restored verbatim in `f4dedd8f7`; the
edit is recoverable with `git show 172428400 -- torchtitan/models/deepseek_v4/mhc.py`.
No measurement was launched from the contaminated state. In a shared worktree,
stage paths explicitly.

Assert on the field you think you set, in-process, before queuing 64 ranks --
every config in this round is now verified by constructing it and printing the
values that matter.

## Round 5: the elementwise bucket, attacked at the source (8 nodes)

Motivation: the 8-node profile of the 100.09 config attributes 36 % of kernel
time to ~75k elementwise/copy/reduce launches per step averaging 44 us --
memory-bound passes over hidden-state-sized fp32 tensors, not launch
overhead. By autograd node, MulBackward0 alone is 46 % of that bucket and,
with forward `aten::mul`, Pow/Mean/Div backward and the bf16<->fp32 copies,
two thirds of it come from the hyper-connection branches in `mhc.py`. Since
block-level `torch.compile` never fused anything (round 4), three branches
were cut from `dsv4_flash_pr18_sweep` (the 100.09 config), each moving one
thing. All runs 8 nodes x 4 GPUs, EP=4, 30 steps, mean TFLOP/s over steps 5-30
(5 nodes were down; every number in this table is same-node-count).

| job | branch | config | TFLOP/s | vs 8-node baseline | peak mem |
|---|---|---|---|---|---|
| 418 | `dsv4_flash_pr18_sweep` | `deepseek_v4_flash_pr18_asyncep_bf16reduce` | **99.50** | baseline (the 100.09 recipe at 8 nodes) | 108.71 GiB |
| 419 | `dsv4_flash_hc_einsum` | `deepseek_v4_flash_best_hc_einsum` | **109.00** | **+9.5 %** | 106.66 GiB |
| 420 | `dsv4_flash_hc_compile` | `deepseek_v4_flash_best_hc_compile` | **123.31** | **+23.9 %** | 107.54 GiB |
| 421 | `dsv4_flash_flex_sac` | `deepseek_v4_flash_best_flex_sac` | crashed at config time | -- | -- |

**419, `hc_einsum` (worktree `/mnt/dgxc/worktrees/hc_einsum`, commit 0edc9386a).**
`HcPre`/`HcHead`: the branch reduction `sum(pre[:, m, None] * x[:, m])`
becomes a batched `[1, M] @ [M, D]` GEMM and the RMS statistic uses
`vector_norm` (no full-size fp32 `x^2`, no PowBackward pass). `HcPost`: see
the finding below. Same function of the same inputs -- CPU check against the
original agrees to 9e-7 (fp32) / 3e-5 (bf16) rel-norm on outputs and every
gradient.

**420, `hc_compile` (worktree `/mnt/dgxc/worktrees/hc_compile`, commit ae46cc3d7).**
The HC pre/post/head math moved into three module-level functions, each
`torch.compile(fullgraph=True, dynamic=False)`; the modules call them. Nothing
else in the block is compiled. This is the compile that round 4 could not get:
the functions contain only tensor ops, so there is no SPMD context manager
for Dynamo to trip on. `HC_COMPILE=0` runs the same functions eagerly
(compiled vs eager: 3e-5 rel-norm). **+23.9 % from fusing one module's math**
-- the largest single lever since PR #18. The two branches are alternatives
for the same bytes, not a stack: the compiled version already avoids the
intermediates the einsum rewrite removes by hand, and does it for the
sinkhorn too.

**Finding: `HcPost`'s residual term is a per-branch scale, not a branch mix.**
`torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` broadcasts
`[T, M, M, 1]` against `[T, M, 1, D]` -- product index `[t, i, j, d] =
comb[t,i,j] * residual[t,i,d]` -- and reduces over j, so it equals
`residual * comb.sum(-1, keepdim=True)` exactly (verified 1e-6 fp32). A true
M x M mix would reduce over i (`dim=1`, i.e. `bmm(comb^T, residual)`).
Upstream torchtitan main has the identical line. Both branches preserve the
existing semantics; whether that is the intended model is a question for the
model owners -- if it is a bug, the 20 sinkhorn iterations feed a near-1
scalar.

Loss is not evidence either way here (init/data seed is not fixed; step-1
losses differ across all three runs).

**421, `flex_sac`: `ValueError: MinimalAsyncEP requires full recompute`** from
`distributed/minimal_async_ep/api.py` -- a plain `isinstance(..., FullAC.Config)`
gate on the AC policy. `FlexSaveAC` saves only the flex region and recomputes
the dispatcher's ops exactly as FullAC does, so the gate was widened to accept
it; requeued. Job 422 (same branch, 2x microbatch with saved top-k) was
cancelled before it hit the same error and requeued behind it.

### Round 5, continued: the AC policies (8 nodes)

| job | config | TFLOP/s | peak mem | note |
|---|---|---|---|---|
| 423 | `deepseek_v4_flash_best_flex_sac` (save flex outputs only, recompute the rest) | **99.69** | 152.86 GiB | flat vs 99.50; +44 GiB. Skipping the flex forward recompute buys nothing: either it was smaller than the profile suggested or SAC's per-op dispatch overhead ate it. |
| 424 | `deepseek_v4_flash_best_flex_sac_topk_bs2` (2x microbatch, flex + **aten.topk saved**, NO deterministic mode) | **90.24** | 239.50 GiB | **reached steps** -- the first >1x run to survive without `debug.deterministic`. Saving the router's top-k across the recompute is a structural fix for the CheckpointError. 2x is still -9.5 % vs the same policy at 1x (99.69), consistent with the 16-node curve. |

To get 421 (and 423/424) to run at all, `distributed/minimal_async_ep/api.py`'s
full-recompute gate had to accept SAC policies: it was a plain
`isinstance(..., FullAC.Config)`. The gate's real reason -- verified in
`_dispatch_to_experts`, which returns the process-global symmetric receive
buffer itself -- is that without recompute autograd would hold expert inputs
aliasing a buffer the next layer overwrites. Eager SAC recomputes the
dispatcher's ops (none is in any save set), so it is as safe as FullAC; no-AC
(`None`) stays rejected.

**Stock `SelectiveAC` mutation guard, root cause found.** Every stock-SAC run
of this model died with "Tensor cached during selective activation checkpoint
has been mutated", traceback ending in `Indexer.select`
(`compressor.py:193`). The einsum there lowers to `aten.bmm`, whose output the
stock policy caches, and the next line applied `relu_()` to it in place
(reproduced on CPU: version counter 0 -> 1 on the cached tensor). Fixed on
branch `dsv4_flash_flex_sac`: out-of-place relu, indexer and block-mask build
under `torch.no_grad()` (its aux loss is dropped, nothing differentiable
consumes it -- gradients unchanged). Stock SAC on the 100.09 recipe is job 425
(`deepseek_v4_flash_best_sac_mm`).

| 425 | `deepseek_v4_flash_best_sac_mm` (stock `SelectiveAC` + the `relu_` fix) | **96.96** | 227.51 GiB | ran 30 steps -- the mutation-guard root cause is confirmed -- but **-2.6 % vs FullAC and +118 GiB**. |

**Verdict on "less AC" for DSv4 flash: closed.** Both SAC flavours are
flat-to-worse against FullAC at 8 nodes (flex-only 99.69, stock 96.96 vs
99.50) while costing 44-118 GiB. The recompute this model pays is dominated
by cheap-to-recompute elementwise work whose *eager* cost is the problem, and
SAC's per-op Python dispatch mode adds overhead on ~75k ops/step. The
productive direction is the one that already paid: make the recomputed work
cheap (leaf compile, +24 %), not skip it. The stock policy is still the right
tool for saving router top-k across the recompute if >1x microbatch is ever
needed (job 424 pattern), and the `relu_` fix is required for it.

## Round 6: leaf compile beyond the HC branches (8 nodes)

`torchtitan/tools/leaf_compile.py` (branch `dsv4_flash_hc_compile`, commit
dd4d70759) generalises the mhc.py pattern -- compile a pure-tensor function on
its own, `fullgraph=True`, per-group env switches -- and adds three sites:
`attn` (deepseek_v4/attention.py: per-head q RMS-norm + rope tail rotation +
cat as one graph, and the inverse rope on o; complex rotation written in real
arithmetic, cache passed as `view_as_real`), `moe` (common/moe.py: the SwiGLU
between the expert grouped GEMMs, `dynamic=True`), `sink` (common/attention.py:
the sink rescale). Eager mode reproduces the original code bitwise in bf16;
compiled bf16 sites differ at bf16 rounding level and are closer to an fp64
reference than eager (q-norm 1.7e-3 vs 3.8e-3; SwiGLU 1.6e-3 vs 2.4e-3).

| job | groups | TFLOP/s | vs FullAC baseline 99.50 | peak mem |
|---|---|---|---|---|
| 420 | `hc` | 123.31 | +23.9 % | 107.54 GiB |
| 427 | `hc,attn,moe,sink` (all) | **133.80** | **+34.5 %** | 106.48 GiB |
| 428 | `hc,attn` | 132.17 | +32.8 % | 105.52 GiB |

Attribution: `attn` (q-norm + rope + cat, o inverse rope) is worth +8.9
points over `hc` alone; `moe` + `sink` add +1.6, at the edge of the 1.3-1.5 %
run-to-run floor but positive and free. Keep all four on.

Config `deepseek_v4_flash_best_leaf_compile` = the 100.09 recipe; the groups
come from `TORCHTITAN_LEAF_COMPILE` in the job environment. No recompile or
graph-break warnings in the 427 log.

## The >1x microbatch CheckpointError: where the divergence starts

torch's own checkpoint debug mode cannot run on this model (job 426: its
`LoggingTensorMode` has no rule for the `flex_attention` HOP). Job 429 ran a
targeted instrument instead (`common/moe.py`, `DSV4_ROUTER_DEBUG=1`, branch
`dsv4_flash_pr18_sweep`, commit 79e9d3025): the router records its input
`x_TD`, post-sigmoid `scores_TE` and top-k on the forward call and diffs them
on the FullAC recompute. 2x microbatch, determinism check off, 8 nodes. Every
one of the 8 reporting ranks, first compared layer:

| rank | router input equal? | elements differing (of 67.1 M) | max |x diff| | scores max diff | top-k rows changed (of 16384) |
|---|---|---|---|---|---|
| 0..7 | **no** | 0.59 M - 1.77 M (0.9-2.6 %) | 3.1e-2 / 4.7e-2 (1 bf16 ulp at |x|~4-8) | 3.5e-4 - 4.8e-4 | 11 - 34 |

`expert_bias_equal=True` everywhere. The flipped rows are 6th-slot near-ties
(e.g. row 6430: experts 28 vs 180 at 0.9114022 vs 0.9114074 in the forward,
0.9114000 vs 0.9113990 in the recompute).

So the router and top-k are not the culprit: **the block's hidden state is
already not bitwise reproducible between forward and recompute at T=16384,
upstream of the router** -- in attention, the HC branches or the norms --
while at T=8192 thirty runs never tripped the check. One-ulp bf16 flips in
~2 % of elements is the signature of an fp32 accumulation whose order varies
run to run (split-K / atomics) somewhere that feeds the whole hidden state.
Prime suspect: the HC mixing linear, `F.linear(x.float(), hc_fn.float())`
with K=16384 and N=24, an extremely skinny fp32 GEMM that runs through the
bf16x9 emulation path; cuBLAS heuristics pick split-K by M, which would
explain the shape dependence, and deterministic mode (which fixed 2x) is
the one knob that constrains cuBLAS/cuBLASLt algorithm choice. A single-GPU
run-to-run test of every GEMM and reduction on the block's path at M=8192
and M=16384, under bfx9/ieee and with/without deterministic mode, is
`gb300/gemm_determinism.py` (branch `dsv4_flash_pr18_sweep`).

**GEMM/reduction microbench (job 430, one GB300): suspect refuted.** Every
GEMM and reduction on the block's path to the router -- the skinny HC mixing
linear (K=16384, N=24) and HC head linear under bf16x9, the HC RMS statistic
and branch sum, the router gate, wq_a/wq_b/wkv/wo_a/wo_b, rms_norm and the
q-norm -- is bitwise identical across 6 runs at both M=8192 and M=16384,
under bfx9 and ieee, with and without deterministic mode. So the
nondeterminism is not in any individual dense op run in isolation. Remaining
candidates: FlexAttention's forward (Triton template) at the 2x mask shape,
the DSA index selection (`Indexer.select`: bf16 scores with many relu ties
feeding `topk`, n_cmp 2048 -> 4096 may cross a kernel-selection threshold),
or an interaction (allocator/stream state) that a single op in a loop does
not reproduce. Next: `gb300/block_determinism.py` runs one real CSA+MoE
block (layer 4) on one GPU, same weights and input, four times at T=8192 and
T=16384, with forward hooks on every submodule, and names the first one
whose output is not bitwise stable.

**FOUND (job 431, one GB300): `Indexer.select` is not run-to-run
deterministic at the 2x shape.** Same bf16 inputs, six calls:

| T | n_cmp (candidates per query) | identical indices | identical *set* |
|---|---|---|---|
| 8192 | 2048 | yes | yes |
| 16384 | 4096 | **no** | **no** |

`Indexer.select` (`deepseek_v4/compressor.py:183-208`) sums relu'd bf16 index
scores over 64 heads and takes `topk(512)` per query over the n_cmp
compressed keys. bf16 scores have many exact ties (relu zeros and 8-bit
mantissas), so the 512th-place boundary is usually a tie; `torch.topk`'s
choice among tied candidates is arbitrary, and at slice length 4096 it is
not even stable call to call (at 2048 it is). A different key set for some
queries changes those queries' attention output, which is the 1-ulp bf16
flips in ~2 % of the hidden state that job 429 saw at the router, which flip
6th-slot near-ties in the router's own top-k, which changes the routed-token
counts -- the CheckpointError. The chain is now traced end to end, and the
same thing will happen at 1x for any seq_len whose n_cmp crosses the
threshold (seq_len 16384 -> n_cmp 4096), i.e. this is a long-context
correctness issue too, not just a batch-size one. Every individual GEMM was
a red herring (job 430). The fix is to break ties deterministically in
`Indexer.select` (fp32 scores with an index-ordered tiebreak, or a stable
sort), which costs nothing at these sizes.

**Job 432 (one GB300, whole block):** the real CSA+MoE block (layer 4, 6.6 B
params, bf16 weights) run four times on identical input is bitwise identical
at T=8192 and NOT at T=16384; the first diverging submodule is
`attention.inner_attention` (the flex CSA output, 6.4 % of elements), and
everything downstream inherits it (router input 6.1 %, router top-k 7.1 % of
slots). Under `torch.use_deterministic_algorithms(True)` the same block is
bitwise identical at T=16384 -- reproducing the training-side observation on
one GPU. Two corrections to the story above: `torch.topk` itself IS
deterministic on tied bf16 rows at every slice length tested (2048/4096/8192),
and a tie-broken fp32 variant of `select` is still nondeterministic at
T=16384 -- so the nondeterminism is in the index-score computation
(`einsum("shd,td->sht")`, relu, `* idx_w`, head-sum) rather than in top-k.
`gb300/select_determinism.py` isolates the op.

**Settled (jobs 433, 434).** Every op producing the index scores is bitwise
deterministic at both shapes (433). `torch.topk` on bf16 rows is stable even
with boundary ties at every width (434) -- but the original `select` promotes
its scores to fp32 (the `where(mask, finfo.min, 0)` add), and the fp32
top-k over 4096-wide rows is where the tie order becomes unstable: the
original's *effective* selection differs in 5/5 calls at T=16384 and 0/5 at
8192; under `use_deterministic_algorithms` 0/5 (torch routes topk through a
deterministic path). The stable-sort formulation
(`masked_fill` + `torch.sort(descending, stable=True)[:, :k]`) is 0/5 at
both shapes and agrees with the original in 100 % of rows at 8192, where the
original is itself deterministic. **Landed in `compressor.py` on
`dsv4_flash_pr18_sweep` and `dsv4_flash_hc_compile`.** Proof runs (8 nodes):
plain FullAC 2x without deterministic mode or saved top-k
(`deepseek_v4_flash_pr18_bs2`, tag `_fix`), the all-leaves 1x recipe with
the fix (regression check), and all-leaves at 2x.

**Proof runs of the fix (8 nodes, 30 steps each, all completed, exit 0):**

| job | branch | config | AC / determinism | TFLOP/s | peak mem |
|---|---|---|---|---|---|
| 438 | `dsv4_flash_pr18_sweep` @4f005a26b | `deepseek_v4_flash_pr18_bs2` (92.15 recipe, **2x**) | plain FullAC, no deterministic mode, no saved top-k | 80.84 | 142.72 GiB |
| 439 | `dsv4_flash_hc_compile` @2bbaa16d9 | `deepseek_v4_flash_best_leaf_compile` (1x) | FullAC | 133.77 | 106.05 GiB |
| 440 | `dsv4_flash_hc_compile` @2bbaa16d9 | `deepseek_v4_flash_best_leaf_compile_bs2` (**2x**) | plain FullAC | 107.85 | 142.88 GiB |

- 438 is the first >1x run to complete under plain FullAC with nothing else
  changed: **the CheckpointError is fixed at the source.** 80.84 matches the
  16-node 2x-deterministic number (80.90), as expected.
- 439 vs 427 (133.77 vs 133.80): the stable sort costs nothing measurable.
- 440: 2x is still **-19 % vs 1x** on the best recipe (107.85 vs 133.77),
  same direction as every earlier batch measurement -- the DSA single-
  sequence cost (indexer and masks scaling with T^2, see the batch section).
  The crash is gone; the throughput case for batch still needs the batched
  DSA path.

**State of the best recipe:** `deepseek_v4_flash_best_leaf_compile` on
`dsv4_flash_hc_compile` (leaf compiles hc/attn/moe/sink + deterministic
Indexer.select), 133.77 TFLOP/s at 8 nodes vs 99.50 for the 100.09 recipe
at the same node count (+34.4 %). Not yet measured at 16 nodes.

## Profile of the all-leaves recipe (job 441, 8 nodes, 133.8 TFLOP/s class)

`/mnt/dgxc/profiles/dsv4_flash_leaf_all_32xgb300/rank0_trace.json.gz` (2
profiled steps). Kernel time 14,636 ms vs 18,383 for the 100.09 recipe; launches
96,480 vs ~150k.

| bucket | ms / 2 steps | share | note |
|---|---|---|---|
| flex attention backward | 4,147 | 28.3 % | unchanged in absolute terms; now the largest item |
| NCCL (FSDP all-gather / reduce-scatter) | 3,596 | 24.6 % | |
| GEMMs incl. grouped (cutlass) | 1,733 | 11.8 % | grouped expert GEMM alone 658 ms |
| symmetric-memory (async EP dispatch/combine, FSDP AG copies) | 1,555 | 10.6 % | |
| flex attention forward (x2: recompute) | 1,204 | 8.2 % | |
| eager elementwise | 853 | 5.8 % | was 4,613 (25.1 %) |
| Inductor leaf kernels | 840 | 5.7 % | the compiled hc/attn/moe/sink work |
| cat/split copies | 423 | 2.9 % | mostly FSDP2 all-gather copy-out (`split_with_sizes_copy`) and reduce-scatter staging (`_chunk_cat`) |
| sort/topk/scatter/index | 96 | 0.7 % | the stable sort is 17 ms of this |

**Wall-clock view (union of kernel intervals over the 11.78 s window):
compute busy 79.9 %, communication 38.4 %, EXPOSED communication 17.6 %,
fully idle 2.5 %.** With compute 20 % cheaper, the same communication volume
is exposed three times more than in the 100.09 profile (5.5 %). Communication
is now the largest non-kernel inefficiency, ahead of anything compile can
reach.

**What is left for compile.** The remaining eager elementwise/copy/reduce
work is ~1.86 s / 2 steps (12.7 % of kernel time) once the cutlass grouped
GEMM (which the name-based bucket mis-caught) is removed; of that ~0.55 s is
FSDP2's own copy-in/copy-out and reduce-scatter staging, ~0.1 s async-EP
metadata, and the model-side remainder is spread thin: `aten::mul` 280 ms,
`copy_` 174, `add_` 100, `sum` 75, `clamp_min` 70 (block-mask build and the
indexer relu), `cat` 63, `mul_` 57. Round 7 (`dsv4_flash_leaf2`) targets the
indexer (relu*weight*head-sum over the 2 GiB score tensor, q rope/cat, scale
folded into the hadamard), the compressor pooling, the router and the
shared-expert SwiGLU; after it the compile lever is within ~3-4 % of
exhausted on this model. The big remaining items are the flex kernels (36 %
of kernel time, needs kernel work), exposed communication (17.6 % of wall)
and the FSDP2 copies (~4 %).

## Round 7: leaves beyond the first four (8 nodes, 30 steps)

Branch `dsv4_flash_leaf2` (worktree `/mnt/dgxc/worktrees/leaf2`), off the
all-leaves branch. Config `deepseek_v4_flash_best_leaf2` (= the 100.09
recipe; all levers code-level).

| job | commit | what | TFLOP/s | vs 133.77 | peak mem |
|---|---|---|---|---|---|
| 442 | 498480c8c | indexer leaves (`_index_q_rope`, `_index_scores`), indexer + block-mask build under `no_grad`, compressor pooling leaf, router leaves, shared-expert SwiGLU | **137.02** | **+2.4 %** | 105.31 GiB |
| 444 | 0bc5311e2 | + rotation scale folded into the cached hadamard (one GEMM instead of GEMM + pass) | **137.28** | +2.6 % | 106.56 GiB |

The fold is +0.2 % over 442, inside the run-to-run floor; kept because it is
free. **`dsv4_flash_leaf2` @ 0bc5311e2, config
`deepseek_v4_flash_best_leaf2`, is the current best: 137.28 TFLOP/s at 8
nodes, +38 % over the 100.09 recipe at the same node count.**

**Grouped GEMM layout (job 445, one GB300):** `_grouped_mm` with the model's
`[E, F, D].transpose(-2, -1)` weights vs a pre-transposed contiguous copy:
2.0 vs 1.9 ms fwd+bwd at the flash shape, same cutlass kernels, no copy
kernels. The 658 ms/2 steps under `aten::_grouped_mm` in the profile is the
grouped GEMM itself (4.5 % of kernel time, real compute), not layout copies.
Closed.

### Verdict on compile for DSv4 flash

Rounds 5-7 moved the recipe 99.50 -> 123.31 (hc) -> 133.80 (+attn, moe,
sink) -> 137.28 (+indexer, compressor, router, shared FFN, hadamard fold), all
at 8 nodes, all with identical memory, every leaf verified against the
original math. After round 7 the eager elementwise/copy/reduce work is
~11 % of kernel time, of which FSDP2's own copy-in/out and reduce-scatter
staging is ~4 %, async-EP metadata ~1 %, and the model-side remainder is
scattered over sub-1 % sites (`mul` 1.9 %, `copy_` 1.2 %, `add_` 0.7 %,
block-mask build ~0.5 %). Compiling more of the model (the router's top-k
selection, the dispatcher glue, `_build_block_mask`) is possible but each is
worth well under 1 %; there is no remaining single site of the kind that
paid in rounds 5-7. **The compile lever is exhausted on this model** short
of a fused attention kernel. What remains, by size:

1. Exposed communication: 17.6 % of wall time (FSDP all-gather twice per
   step under FullAC + resharding; async-EP dispatch/combine). Levers:
   `parallelism.fsdp_reshard_after_forward="never"` (halves all-gather bytes;
   holds unsharded bf16 params -- fits at 16 nodes, likely not at 8), the EP
   backend sweep (deepep / hybridep / minimal_async_ep / standard), fewer
   all-gather bytes (fp8 params).
2. FlexAttention: 36 % of kernel time (backward 28 %, forward x2 8 %). Needs
   a purpose-built DSA kernel; every template-level lever is measured flat.
3. Batch size: crash fixed; throughput needs the batched DSA path (T^2
   indexer/mask cost).
4. FSDP2 copies ~4 %: framework; `enable_fsdp_symm_mem` is the knob to try.

No recompile / graph-break warnings. Eager mode of every new leaf reproduces
the original math bitwise (CPU); compiled differs at bf16 rounding only.

## Round 8: communication (branch `dsv4_comms_reduction_attempt`, 8 nodes)

Base: the 137.28 recipe (`deepseek_v4_flash_best_leaf2`), one lever each.

| job | config | lever | TFLOP/s | vs 137.28 | peak mem |
|---|---|---|---|---|---|
| 448 | `comms_ep_standard` | stock NCCL all-to-all dispatch instead of MinimalAsyncEP | **119.33** | -13.1 % | 103.4 GiB |
| 449 | `comms_ep_deepep` | DeepEP v2.1.0 dispatch (EP=4 intra-node, GIN disabled) | **128.21** | -6.6 % | 102.3 GiB |
| 450 | `comms_reshard_never` | `fsdp_reshard_after_forward="never"`: params stay unsharded after forward, no second all-gather for the FullAC recompute/backward | **141.52** | **+3.1 %** | **241.17 GiB (87 %)** |

| 451 | `comms_symm_mem` | `enable_fsdp_symm_mem=True` on the async-EP recipe | **crash at init** | -- | -- |

`enable_fsdp_symm_mem` and MinimalAsyncEP are mutually exclusive in one
process: FSDP's symm-mem collectives call `symm_mem.set_backend("NCCL")`
(`_fsdp_collectives.py`) while MinimalAsyncEP requires the "CUDA" backend
(`minimal_async_ep/api.py:221`) and has already initialised it -- "Backend
can not be changed after use". Re-tested as `comms_deepep_symm` (FSDP
symm-mem over DeepEP dispatch, which does not use torch symmetric memory).

| 464 | `comms_deepep_symm` | FSDP symm-mem collectives over DeepEP dispatch | **125.28** | -8.7 % (-2.3 % vs DeepEP alone) | 105.1 GiB |

FSDP symmetric-memory collectives are a loss on this fabric (NCCL's MNNVL
path is already at ~0.5 TB/s per rank), and they cannot be combined with
the best dispatcher anyway. Closed.

Reshard-never fits at 8 nodes after all (the 141 GiB of unsharded bf16
params land on top of 106 GiB static: 241 GiB peak, 35 GiB headroom) and is
the first communication lever to beat 137.28. At 16 nodes the same setting
would peak near 190 GiB.

So MinimalAsyncEP is worth +15 % on the current recipe (it was +4.9 % on the
92.15 one): the cheaper compute gets, the more the dispatch overlap matters.

**Topology correction (probe jobs 453/455).** The cluster is NOT a per-node
NVLink mesh with IB between nodes, as every earlier note assumed. Each GPU
has all 18 NVLinks into NVSwitches (`nvidia-smi topo -m`: NV18 for every
pair; remotes `FFFFFFFF:FF:FF.0`), `Fabric State: Completed` with one
ClusterUUID across nodes, and `nvidia-imex` is active with all 16 nodes in
its domain: a multi-node NVLink (NVL72-class) fabric. Whether NCCL already
carries inter-node collectives over it (MNNVL) or over the 4 IB HCAs is what
the 2-node all_gather probe (`nccl_probe.slurm`) measures; if it is IB today,
enabling MNNVL is potentially the largest communication lever available.

**NCCL already uses the NVLink fabric across nodes (probe job 463, 2 nodes x 4
GPUs, 128 MiB per rank):**

| collective | default env | `NCCL_MNNVL_ENABLE=1` |
|---|---|---|
| all_gather, per-rank receive bandwidth | **489 GB/s** (1.92 ms) | 371 GB/s |
| reduce_scatter, per-rank send bandwidth | 394 GB/s | 392 GB/s |

NCCL logs `MNNVL 1 ... nvlDomainSize 8`, `nNodes 1 localRanks 8` (the two
nodes are one NVLink domain to NCCL), every channel `via P2P/MNNVL`, NVLS
multicast available. The IB HCAs are not used at all: NCCL reports
`Using network Socket` with "GPU Direct RDMA Disabled" -- if any collective
ever had to leave the NVLink domain it would run over TCP. So inter-node
FSDP traffic is already at ~0.5 TB/s per GPU, and the exposed 17.6 % is
volume and latency at NVLink speed. Levers that cut bytes (reshard-never:
+3.1 %) or add overlap are the ones that can move it; there is no IB->NVLink
step left to take. MNNVL also means the EP degree is not bounded by the
node: EP=8..32 groups are NVLink-connected too (the earlier EP re-sweep
that found EP=4 optimal was therefore not comparing NVLink vs IB, but
dispatcher fan-out at equal link speed).

With `NCCL_MNNVL_ENABLE=0` the same 2-node all_gather runs at **0.5 GB/s**
(1.9 s per 128 MiB): NCCL's IB transport is not functional here (no IB
plugin picked up; `Using network Socket`, GPU Direct RDMA disabled), so
without the NVLink fabric every inter-node collective would be TCP. The
`NCCL_IB_HCA` setting in the launcher is inert. Every multi-node result in
this ledger therefore ran over MNNVL.

### Profile of the reshard-never recipe (job 466, 8 nodes)

`/mnt/dgxc/profiles/dsv4_flash_reshard_never_32xgb300/`. Kernel time 11,265 ms
/ 2 steps (was 14,636 at 137.28); NCCL 1,536 ms (was 3,596 -- the second
all-gather is gone).

| | 137.28 recipe | + reshard-never (141.52) |
|---|---|---|
| compute busy | 79.9 % | 75.9 % |
| communication total | 38.4 % | 21.2 % |
| **exposed communication** | **17.6 %** | **12.9 %** |
| idle (no kernel at all) | 2.5 % | **11.2 %** |

Halving the all-gather bytes did exactly what it should to the comm term,
but ~9 points of it reappeared as *idle*: the step is now partly
launch-bound rather than comm-bound. The idle is NOT a few big stalls -- only
114 ms sits in gaps >1 ms (the largest being `Optimizer.step#AdamW` at 56 ms
and the profiler's own step boundary) -- it is ~1.2 s of sub-millisecond gaps
spread across 92,746 launches per 2 steps. Levers for that are fewer, larger
kernels (CUDA graphs, more aggressive fusion) rather than communication.

Remaining kernel-time shares: **flex backward 36.8 %**, flex forward 9.7 %,
GEMMs 15.1 %, NCCL 13.6 %, all-gather/symm copies 8.3 %, Inductor leaves
8.0 %, elementwise 4.2 %, cat/split 2.5 %.

Collectives are `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` (776 ms) and
`ncclDevKernel_AllGather_RING_LL` (718 ms) -- NCCL picks the **LL protocol**,
which carries 8 bytes of payload per 16-byte line (half the link's usable
bandwidth), and the RING algorithm even though the probe showed NVLS
multicast available. `NCCL_PROTO=Simple` and `NCCL_ALGO=NVLS` are therefore
the next two cheap experiments (jobs 472/473).

### NCCL protocol / algorithm on the reshard-never winner (8 nodes)

| job | env | TFLOP/s | vs 141.52 |
|---|---|---|---|
| 472 | `NCCL_PROTO=Simple` | **142.72** | +0.85 % (at the 1.3 % noise floor, but positive and free) |
| 473 | `NCCL_ALGO=NVLS` (global) | **crash** | `No algorithm/protocol available for function Broadcast with datatype ncclInt8` -- NVLS has no Broadcast path, and a global NCCL_ALGO applies to every collective including the bootstrap broadcasts. Per-collective syntax is the correct form. |
| 475 | `NCCL_ALGO=allgather:nvls,reducescatter:nvls` | **crash** | `Unrecognized element token "reducescatter"` -- NCCL's parser wants its CamelCase function names. The accepted tokens in this build (2.30.7, from `strings libnccl.so.2`) are `AllGather ReduceScatter AllReduce Broadcast Reduce SendRecv` and algorithms `TREE RING NVLS NVLS_TREE COLLNET_DIRECT`; job 476 uses `AllGather:NVLS,ReduceScatter:NVLS`. |

(The 473 failure also confirms the exported env does reach the ranks, which
is how the `NCCL_PROTO=Simple` run is known to have taken effect.)

### EP dispatch backends, head to head on the 137.28 recipe (8 nodes, EP=4)

| backend | config | TFLOP/s | vs best |
|---|---|---|---|
| **MinimalAsyncEP** | `deepseek_v4_flash_best_leaf2` | **137.28** | -- |
| HybridEP | `comms_ep_hybridep` (job 474) | 133.25 | -2.9 % |
| DeepEP v2.1.0 | `comms_ep_deepep` (job 449) | 128.21 | -6.6 % |
| standard NCCL all-to-all | `comms_ep_standard` (job 448) | 119.33 | -13.1 % |

All four measured with identical model code, parallelism, AC and leaf
compiles; only the dispatcher differs. MinimalAsyncEP wins; the sweep is
complete for EP=4.

**Installing HybridEP on this cluster** (it is the `hybrid-ep` branch of
deepseek-ai/DeepEP, not a separate package; torchtitan imports
`deep_ep.HybridEPBuffer`). Checked out at `/mnt/dgxc/DeepEP-hybrid`, build
job `build_hybridep.slurm` there. Five environmental blockers, all fixed,
none architectural:

1. multinode path needs DOCA or NIXL, neither installed -> built with
   `HYBRID_EP_MULTINODE=0` (intranode only; fine at EP=4, which is one node);
2. `cuda/std/tuple` -- the branch's setup.py never adds the CCCL include dir;
   CCCL ships in the `nvidia-cu13` wheel, symlinked as
   `$CUDA_HOME/include/cccl` and put on `CPATH`/`NVCC_PREPEND_FLAGS`;
3. `cuda_profiler_api.h` absent from the wheel-assembled shadow toolkit ->
   two-declaration stub written into `/mnt/dgxc/cuda13/include`;
4. link `-lnvtx3interop`: the wheel ships only `libnvtx3interop.so.1` ->
   unversioned symlink in `/mnt/dgxc/deepep-deps/libs`, and that dir plus
   `nvidia/cu13/lib` on `LIBRARY_PATH` (and `LD_LIBRARY_PATH` at runtime);
5. HybridEP JIT-compiles its backend at *runtime* and passes
   `-L$CUDA_HOME/lib64`, while the shadow toolkit only had `lib` ->
   `lib64 -> lib` symlink. This one failed only inside a real run (job 471).

Runtime knobs are in the launcher behind `HYBRIDEP=1`
(`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN`, `USE_MNNVL`,
`HYBRIDEP_NUM_SMS_{DISPATCH,COMBINE}`).

Not tried: HybridEP with EP>4. Because the cluster is one NVLink domain (see
the MNNVL finding), an EP=8/16/32 group is still NVLink-connected, and
HybridEP's TMA intranode path plus a larger
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` is the one configuration where it
could plausibly beat MinimalAsyncEP. That needs the EP sweep re-run per
backend, which is a round of its own.

## The batched DSA path (branch `dsv4_batched_dsa`)

The DSA path treated the whole per-rank stream as one sequence:
`_forward_impl` used `seqlen = q.size(0)` with `bsz=1`, so the indexer scored
every query against every compressed key of the entire microbatch
(`einsum` T x T/4) and the selection mask was a dense `[T, T + n_cmp]`.
Attention-side cost was quadratic in the microbatch, which is why 2x measured
-19 % and 3x -24 % per token even after the CheckpointError was fixed.

Every DSA component now knows the packed sequence length (threaded from the
three model builders into `DSV4FlexAttention.Config.seq_len` and
`Compressor.Config.seq_len`), and a T-token stream is split into
`T // seq_len` independent sequences:

- `deepseek_v4/attention.py`: `_batch_shape()`, per-sequence KV streams
  `[B, L + n_cmp, H, D]`, block mask built with `bsz=B`, indexer tensors
  viewed as `[B, L, ...]`;
- `common/attention.py`: `FlexAttention.forward` accepts 4D `[B, L, H, K]`
  and folds the result back to `[T, H, V]`, so `out_transform` and all callers
  are unchanged;
- `compressor.py`: `Indexer.select` batched (`einsum("bshd,btd->bsht")`), so
  the score tensor is `B*L*(L/4)` rather than `(B*L)*(B*L/4)`;
- `leaf_ops.py`: a batched variant of the index-score reduction.

`seq_len=0`, or a stream exactly one sequence long, keeps the folded path
unchanged.

**A second correctness bug, found while doing this:** `Compressor.
_overlap_transform` took each compressed group's "previous group" from the
flat stream, so with more than one packed sequence the first group of sequence
n was fed from the tail of sequence n-1. Now the shift stays inside each
sequence. This was wrong on the folded path too, at every document boundary
inside a packed row.

**Equivalence test** `tests/unit_tests/gpu/test_dsv4_batched_dsa.py`
(job 483, 6 passed): for all three DSA variants and the compressor, a batch of
B reproduces, per sequence, what that sequence produces alone -- outputs and
gradients, rel-err 0.000e+00 on every tensor except one `dcmp_k` at 7.2e-6.
It also pins `_batch_shape` and rejects a token count that is not a multiple
of `seq_len`. Two bugs it caught before any cluster time was spent:
`Indexer.select`'s stable sort returned `order[:, :k]` (slices the sequence
dim once batched) and the batched fold-back dropped the head dim.

### The NVLink domain at working scale, and EP>4 on it (jobs 481, 478-480)

The 8-node probe settles the topology question at the scale we actually run:
NCCL reports `cliqueSize 32 nvlDomainSize 32` -- all 32 GPUs of an 8-node job
are one NVLink clique -- and bandwidth *improves* with scale:

| ranks | all_gather per rank | reduce_scatter per rank |
|---|---|---|
| 8 (2 nodes) | 489 GB/s | 394 GB/s |
| 32 (8 nodes) | **526 GB/s** | **527 GB/s** |

So the fabric is not the constraint and nothing about it is being
underutilised. What about using it for *expert* dispatch? With HybridEP's
NVLink-domain knob set to the real domain size and `USE_MNNVL=1`:

| config | TFLOP/s | peak mem |
|---|---|---|
| HybridEP EP=4 (`ranks_per_nvlink_domain=4`) | 133.25 | 104.85 GiB |
| HybridEP EP=8 (`=8`, `USE_MNNVL=1`) | 130.44 | 100.42 GiB |
| HybridEP EP=32 (`=32`, `USE_MNNVL=1`) | 89.73 | 124.47 GiB |

Monotonically worse, exactly as MinimalAsyncEP was (4 -> 100.09, 8 -> 94.87,
16 -> 82.90, 32 -> 67.44 on the older recipe). **EP=4 is optimal for
dispatcher-fan-out reasons, not link reasons**: a wider group means more
peers, smaller per-peer transfers and more metadata, and that cost does not
care whether the links are NVLink or IB. The earlier interpretation ("EP=4
wins because it stays inside a node's NVLink mesh") was wrong about the
mechanism while right about the conclusion.

`NCCL_ALGO=AllGather:NVLS;ReduceScatter:NVLS` (semicolon syntax, which does
parse -- the launcher echo confirms it reached the ranks) fails at runtime:
**NVLS has no bf16 ReduceScatter** ("No algorithm/protocol available for
function ReduceScatter with datatype ncclBfloat16"). All-gather-only NVLS is
job 487.

### Batched DSA throughput (8 nodes, reshard-never + NCCL_PROTO=Simple)

| job | config | microbatch | TFLOP/s | peak mem |
|---|---|---|---|---|
| 484 | `bdsa_bs1` | 1x | **143.29** | 241.17 GiB |

The 1x regression check: the batched code path plus the compressor overlap fix
cost nothing at one sequence per rank (143.29 vs 142.72 baseline, inside
noise), which is the prerequisite for trusting the >1x numbers.

**Job 485 (2x) did not reach step 1 in 13 minutes** -- not a bug in the
batched path but Inductor autotune on the new shape, ~115 s per block across
43 layers for forward and backward. The autotune banner is itself the
confirmation that the batch dim is live in the kernel:
`flex_attention(2x64x8192x512, 2x64x10240x512, ...)`, i.e. 2 sequences of
8192 queries against their own 10240-long KV streams, where the folded path
would have shown one 16384-query sequence against 20480 KV. Since every flex
layer pins `kernel_options` (required on GB300 at head_dim=512), autotune only
re-benchmarks the pinned choice; it is measured free at 1x
(`dsv4_flash_pr18_noautotune`, 92.01 vs 92.15). **Turned off on this branch**
(`max_autotune=False, coordinate_descent_tuning=False` in
`common/attention.py`) and the batch runs requeued as 489 (2x), 490 (4x),
491 (1x re-check). This is the same trap the original 2x/4x attempts
(jobs 330/341/365) fell into.

**Jobs 489-491 all failed without steps -- two separate causes, one of them a
config mistake of mine:**

1. **No memory headroom.** The batch configs were based on reshard-never,
   which already peaks at 241 GiB of 276 because it keeps every block's
   parameters unsharded. There is nothing left for a bigger microbatch: job
   490 (4x) died with a genuine `OutOfMemoryError` (3 GiB request, 777 MiB
   free). Fixed by basing >1x on the leaf-compile recipe (137.28, 106 GiB) and
   keeping reshard-never only for the 1x comparison. The two levers are
   memory-competing, not additive.
2. **Cold Inductor cache over NFS.** Jobs 489 and 491 died with
   `Operation timed out after 300 s` on their FIRST collective, on every rank.
   Editing the flex `inductor_configs` (autotune off) changed the Inductor
   cache key, so all 32 ranks recompiled every flex kernel into the shared
   `~/.triton` at once -- the same NFS contention that has bitten this cluster
   before -- and blew the 300 s `comm.init_timeout_seconds` default. Raised to
   1800 s on these configs. Note 491 also *overwrote* 484's log because both
   used the same TAG; use distinct tags for re-runs.

Also: `NCCL_ALGO=AllGather:NVLS` (job 488) = **140.74**, below the 142.72 of
`NCCL_PROTO=Simple` alone. NVLS multicast is not a win for FSDP's all-gather
here, and it has no bf16 reduce-scatter at all. **Communication round closed
at 142.72** (leaf2 + reshard-never + `NCCL_PROTO=Simple`).

### Batched DSA: microbatch is now a WIN (8 nodes, 30 steps, `NCCL_PROTO=Simple`)

| job | config | microbatch | base | TFLOP/s | tok/s/GPU | peak mem |
|---|---|---|---|---|---|---|
| 494 | `bdsa_bs1` | 1x | reshard-never | 141.23 | 1,593 | 241.17 GiB |
| 492 | `bdsa_bs2` | **2x** | leaf2 (137.28) | **150.57** | 1,699 | 127.70 GiB |
| 493 | `bdsa_bs4` | **4x** | leaf2 (137.28) | **158.74** | 1,791 | 173.96 GiB |

No recompiles, no graph breaks, no CheckpointError, loss descending normally.

**The sign flipped.** On the folded path a 2x microbatch cost -19 %
(107.85 vs 133.77); batched it gains **+9.7 %** (150.57 vs 137.28) and 4x
gains **+15.7 %** -- a ~35-point swing, and 4x is the best number this
campaign has produced, **158.74 TFLOP/s**, 8.7x the 18.29 flash baseline and
+11 % over the previous best (142.72). Memory also behaves as predicted now
that the transients are per-sequence: 2x costs only +22 GiB over 1x on the
same base (128 vs 106) where the folded 2x needed 241.

Note the 1x row uses the reshard-never base (241 GiB) and so is not directly
comparable to 2x/4x; the honest comparisons are 2x/4x against the leaf2 base
at 1x (137.28), and the two memory-heavy levers cannot be stacked naively
(see job 490's OOM). `bdsa_bs2_rn` tests whether reshard-never + 2x fits.

### The batch curve, and where it ends (8 nodes, 30 steps)

| job | microbatch | tokens/rank | TFLOP/s | tok/s/GPU | peak mem |
|---|---|---|---|---|---|
| -- | 1x | 8,192 | 137.28 | ~1,517 | 106 GiB |
| 492 | 2x | 16,384 | 150.57 | 1,699 | 128 GiB |
| 493 | 4x | 32,768 | **158.74** | 1,791 | 174 GiB |
| 495 | 6x | 49,152 | **160.04** | 1,806 | 221 GiB |
| 496 | 8x | 65,536 | **27.96** | 315 | 245 GiB |

**4x is the operating point.** 6x buys +0.8 % for +47 GiB (and 160.04 vs
158.74 is inside the run-to-run floor), and 8x falls off a cliff: the log
shows `expandable_segments: memory mapping failed with OOM` and allocator
retries, with throughput decaying step by step (35.4 at step 5, 33.7 at 10,
21.3 at 20) as the allocator thrashes. It never OOMs outright, it just grinds
-- worth knowing as a failure signature: a *slow* run at ~88 % memory is the
allocator, not the model.

**Stacking with reshard-never (job 497): fits, but is not worth it.** 2x +
reshard-never = 151.75 at **262.91 GiB (95.1 %)**, versus 150.57 at 128 GiB
for 2x alone. +0.8 % for +135 GiB and no headroom left. The two levers are
memory-competing and batch is the far better use of the memory.

### Profile of the 4x batched step (job 498)

`/mnt/dgxc/profiles/dsv4_flash_bdsa_bs4_32xgb300/`. Kernel time 39,615 ms per
2 steps over 94,804 launches -- i.e. **4x the tokens for 2.6x the launches**,
which is the whole point of the batched path.

| | reshard-never 1x | **batched 4x** |
|---|---|---|
| compute busy | 75.9 % | **87.0 %** |
| exposed communication | 12.9 % | 11.8 % |
| idle | 11.2 % | **1.2 %** |

The launch-bound idle that appeared after reshard-never is gone: bigger
per-kernel work refills the pipeline. Kernel shares: flex backward 40.7 %,
GEMMs 15.6 %, flex forward 11.2 %, all-gather/symm copies 10.4 %, NCCL 8.3 %,
Inductor leaves 8.1 %, elementwise 4.1 %, and the DSA index/sort work is
**0.9 %** (it was the quadratic term before). Attention is now 52 % of kernel
time and everything else is small: the next lever is a DSA-specific attention
kernel, nothing else.

## Why 6x barely helps and 8x collapses (jobs 499-501)

**The "spare" memory was not spare.** The trainer's `memory: X GiB(Y%)` line
is PyTorch's peak *reserved* over the card total, so it misses the CUDA
context, NCCL buffers and -- the big one -- MinimalAsyncEP's symmetric-memory
receive buffers, which are sized
`ep_size * num_max_tokens_per_rank * min(top_k, num_local_experts)` and grow
linearly with the microbatch: 3 GiB at 1x, 12 at 4x, 18 at 6x, **24 at 8x**.
Driver-level sampling (new `MEM_SAMPLE=1` knob in the launcher) gives the real
picture:

| microbatch | torch reserved | driver peak used | driver min free |
|---|---|---|---|
| 4x | 174 GiB | 193.3 GiB | 83.2 GiB |
| 6x | 221 GiB | 246.4 GiB | 30.1 GiB |
| 8x (async EP) | 245 GiB | -- (not sampled) | ~0: 245 + 24 symm + ~8.5 ctx = 277.5 = the whole card |
| 8x (standard dispatch) | 250 GiB | 258.4 GiB | 18.1 GiB |

**The 8x collapse is the symmetric buffers, proven by removing them.** Same 8x
microbatch with standard all-to-all dispatch (which allocates none):

| 8x run | TFLOP/s | allocator "mapping failed"/retries | trend |
|---|---|---|---|
| MinimalAsyncEP | 27.96 | **956** | decaying: 35.4 -> 33.7 -> 21.3 |
| standard all-to-all | **141.11** | **0** | rising: 138.5 -> 140.4 -> 141.9 |

So 8x is perfectly feasible; it just cannot be paid for twice. 141.11 is still
below 4x async EP (158.74) because standard dispatch costs ~13 % on its own,
so the trade is not worth taking -- but the failure was memory, not batch.

**6x flattens because there is nothing left to amortize.** Batch only ever
amortizes *per-step* costs, and by 4x those are nearly gone. Comparing the two
profiles (`dsv4_flash_bdsa_bs4_32xgb300` vs `dsv4_flash_bdsa_bs6_32xgb300`):

| | 4x | 6x |
|---|---|---|
| compute busy | 87.0 % | 86.7 % |
| exposed comm | 11.8 % | 12.1 % |
| idle | 1.2 % | 1.1 % |
| NCCL (parameter collectives, per-step fixed) | 8.3 % | **5.8 %** |
| flex bwd + fwd | 51.9 % | **53.8 %** |
| MoE dispatch/combine (`symm/AG`, per-token) | 10.4 % | 11.1 % |
| kernel ms per 1k tokens | 604.5 | **575.4** |

The only line that improves is the per-step NCCL parameter traffic (8.3 % ->
5.8 %, worth the ~2-3 % actually observed); everything else is per-token and
its share is flat or rising. Attention alone is 54 % of kernel time and
exactly linear in tokens, so it sets a per-token floor that no microbatch can
lower. Exposed communication does not amortize either, because it is now
dominated by the MoE dispatch/combine, which is per-token by nature.

The memory-sampled 4x re-run (job 501) also settles the 4x-vs-6x question by
replication: **159.97** TFLOP/s, against 160.04 at 6x. The two are
indistinguishable, so 6x buys nothing at all for its extra 47 GiB -- the
earlier +0.8 % was noise.

**Conclusion: 4x is the operating point** (158.74-159.97, 83 GiB of driver
headroom).
6x is +1-3 % for +47 GiB and only 30 GiB of headroom. The batch lever is spent;
the remaining ceiling is the flex attention kernel (54 % of kernel time) and
the ~12 % exposed MoE dispatch.
