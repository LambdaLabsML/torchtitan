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
| `compile.enable=True` (model+loss) | `pr18_compile` | 92.18 (attempt 3) | +0.0 % | 77.04 GiB | attempts 1-2 failed (Inductor `FlexibleLayout` assert; then `fullgraph=True` forbidding the graph break the fix needs); attempt 3 with `fullgraph=False` runs but is flat. See below. |
| `moe_comm_backend=minimal_async_ep` | `pr18_asyncep` | **96.66** | **+4.9 %** | 78.12 GiB | real; +1.9 GiB only (no Kimi-style buffer blow-up), 0 allocator retries. Needed no code change: the shared dispatcher-capacity code fills `num_max_tokens_per_rank` at config time. |
| 2x microbatch | `pr18_bs2` | -- | -- | -- | crash at step 0: `CheckpointError`, MoE routed-token count differs between forward and FullAC recompute. See below. |
| 2x microbatch, `debug.deterministic` | `pr18_bs2_det` | 80.90 | -12.2 % (confounded) | 101.04 GiB | runs; number includes the deterministic-mode tax -- 1x-det baseline queued |

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
for hours. Under test on branch `dsv4_flash_pr18_noautotune`
(`max_autotune=False`, `coordinate_descent_tuning=False` -- the state the
FlexAttention docstring itself prescribes once `kernel_options` are pinned):
1x for the baseline effect, 2x for the flip.

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
compiling are split across small subgraphs. Confirming that needs a profile
of the compiled run (does the ~153k-launch elementwise bucket shrink at all?).
Not stacked with the comm levers -- nothing to add.

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

Assert on the field you think you set, in-process, before queuing 64 ranks --
every config in this round is now verified by constructing it and printing the
values that matter.
