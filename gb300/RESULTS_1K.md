# GPT-OSS-120B: the push from 787 to 1000 TFLOP/s/GPU on 64x GB300

16 nodes x 4 GB300, `seq_len=8192`, SelectiveAC + model compile, C4 from locally
staged shards, `--training.disable-cuda-graphs`. 200 steps per screening arm;
**steady state is steps >= 100**, the same window every number in
`TUNING_RESULTS_120B.md` uses. Peak for MFU is GB300 dense bf16 ~2503
TFLOP/s/GPU.

Reference is **job 56**, `gpt_oss_120b_gb300_sac_compile` at bs=18: **786.9
TFLOP/s/GPU**, 232.96 GiB (84.25%). Target is 1000, i.e. **+27.1%**.

## Read this before quoting any number here

These are on torchtitan's **stock FLOP accounting**, identical to job 56, so
they are comparable to it and to the rest of the sweep. That accounting charges
sliding-window layers the full O(L^2) attention cost, which GPT-OSS-120B does
not pay on 18 of its 36 layers. Branch `perf/flops-sliding-window` corrects it
and every 120B figure drops ~18% (786.9 -> 665.0; a corrected 1000 would be
~846). It is deliberately not merged into this branch: changing the denominator
mid-comparison would make these runs incomparable to the reference they are
trying to beat. **Any number here quoted outside this sweep needs that -18%
applied, or needs saying which accounting it is on.**

## Why these levers

The 120B sweep spent the memory and batch knobs: bs=20 at 98.7% bought -0.1%,
never-reshard is mechanically unavailable (240 GB of unsharded bf16 parameters
against 276 GB of HBM), and HSDP was settled on 20B. +27% was never going to
come from there.

What it never touched is that 120B runs at **EP=1 with fp32 gradient
reduction**, which makes three costs large at once:

| lever | what it targets | prior evidence |
| --- | --- | --- |
| MXFP8 grouped experts + attention Linears | ~72% of per-token FLOPs | +5.7% on dsv4, where expert GEMMs were only 1.6% of the step |
| bf16 gradient reduce-scatter | the largest single collective | ~25% of the step on Qwen3.5-122B |
| expert parallelism | a ~229 GB/step weight all-gather | +13.7% on dsv4 (EP 64->16) |

MXFP8 was recorded as blocked in `TUNING_RESULTS.md` ("SM100 kernels absent").
That was a packaging gap, not a hardware limit; see `MXFP8_120B.md`.

## Gates

Both quantization gates ran on one node in under four minutes each, and both
failed. That is the gates working, not the plan failing -- each would otherwise
have been discovered by a 16-node job.

| check | result |
| --- | --- |
| torchao 0.19 sm_103 build against torch 2.14 | **PASS** -- `cuda_kernels=True triton_kernels=True mxfp8_quantize_op=True` |
| `gpt_oss_debugmodel_1k_mxfp8` (job 246) | **FAIL** -- `AssertionError: K must be divisible by 128` |
| `gpt_oss_debugmodel_1k_fp8` (job 252) | **FAIL** -- `CUDA error: unspecified launch failure` inside Inductor's autotuner |

### MXFP8: the packaging gap is fixed, the shape gap is not

The build works. The kernel does not fit this model. `K` is the GEMM contraction
dim, which for every expert GEMM in gpt-oss is `dim` = 2880, and
**2880 % 128 = 64**. `pad_multiple` cannot help -- it pads per-expert *token*
groups (M), not K -- and torchao's grouped-MXFP8 path calls
`mxfp8_quantize_2d_1x32_cutedsl` unconditionally, with no Triton cast to fall
back to (`flydsl` is the ROCm sibling). MXFP8 here needs a padded-K kernel or a
model dimension gpt-oss does not have.

This is a **different** blocker from the one `TUNING_RESULTS.md` recorded. That
one -- "no aarch64 build, SM100 kernels absent" -- is genuinely fixed, and the
recipe is kept in `MXFP8_120B.md` because it is correct and reusable on any
model whose dims are multiples of 128. Worth correcting the earlier note: the
reason MXFP8 is unavailable for gpt-oss was never the hardware or the packaging.

### FP8 rowwise: the gate crashed in the autotuner, not the kernel

```
triton_poi_fused_..._triton_fp8_rowwise_2d_scale_and_cast_25.run(...)
  -> autotune_to_one_config -> benchmark_all_configs -> synchronize
torch.AcceleratorError: CUDA error: unspecified launch failure
```

The failing launch is one of the autotuner's *trial* configs for a fused fp8-cast
kernel. Not assumed to transfer to 120B: the gate runs `gpt_oss_debugmodel`,
which has `dim=256` where 120B has 2880 while sharing `hidden_dim=2880`, and the
crashing kernel was a 189 M-element fusion over the MoE dispatch -- a shape the
120B model does not produce. Job 255 runs fp8 at the real shape, and
`GPTOSS_1K_NO_POINTWISE_AUTOTUNE=1` is ready if it reproduces.

FP8 rowwise is the lever either way, because its alignment requirement is one
this model meets: `PAD_MULTIPLE` 16, and 2880 % 16 == 0. It also needs nothing
built -- `torch._scaled_grouped_mm` is compiled into the torch 2.14 wheel.

## Profile of the reference step (job 253) -- this inverted the plan

15 steps, profiler active at step 12, all 64 ranks traced. Read on rank1 and
re-read on rank33 (a different node) because a single rank could be atypical;
the two agree to within 1%.

**Streams, rank1.** The numbers that matter are not the kernel totals but which
stream they are on and whether they overlap:

| | busy | share of span |
| --- | --- | --- |
| compute stream | 9490 ms | **98.1%** |
| NCCL streams | 5599 ms | 57.9% |
| union of both | 9652 ms | 99.8% |
| **exposed NCCL** | **158 ms** | **1.6%** |
| GPU idle | 22 ms | 0.2% |

**97% of all communication is hidden behind compute, and the GPU is idle 0.2% of
the step.** GPT-OSS-120B at bs=18 is compute-bound.

**This contradicts the reading in `TUNING_RESULTS_120B.md`**, which said 120B is
"more COMMUNICATION-bound: more parameters to all-gather per token of useful
work". That was inferred from 120B reaching 56% of 20B's per-GPU throughput at
the same batch, not measured, and it is wrong at this operating point. The
all-gather is real and large -- 3666 ms, the single biggest kernel total in the
trace -- it is simply overlapped.

**What is actually on the critical path**, compute stream only (9480 ms):

| bucket | ms | % | what it is |
| --- | --- | --- | --- |
| **expert grouped GEMM** | 4708 | **49.7%** | the MoE FFN; cutlass `enable_3x_kernel_for_sm10*` |
| MoE dispatch/combine movement | 1730 | 18.2% | `tma_scatter_add`, `chunk_cat`, `split_with_sizes_copy`, `indexing_backward` |
| triton pointwise (fused) | 1121 | 11.8% | routing, norms, activations |
| **dense GEMM** | 936 | 9.9% | attention projections + lm_head; `nvjet_sm103_*` |
| attention | 481 | 5.1% | flash-attn-4 fwd+bwd |
| triton reduction | 278 | 2.9% | |
| other / elementwise | 227 | 2.4% | |

### How the queue was re-aimed

**Half the critical path is one thing, and fp8 is pointed straight at it.**
Expert GEMMs plus dense GEMMs are **59.6%** of the compute stream. At 1.5-2x on
those kernels that is +25% to +42% on the step, which spans the target.

**Two arms were cancelled as aimed at nothing.** `bf16reduce` standalone (job
256) halves a 1942 ms reduce-scatter that is 97% hidden, and `ep8`/`ep16` (258,
260) trade a hidden all-gather for an all-to-all that would also be hidden. On
this profile they cannot pay.

**But bf16 reduce is kept in combination, and that is not a hedge.** Comm is
hidden only while compute is long enough to hide it: 5599 ms of NCCL sits under
9490 ms of compute today, and if fp8 takes ~30% off the compute stream that
becomes ~6600 ms against the same 5599 ms. Comm stops being free at roughly that
point, so `fp8_bf16reduce` (259) tests a knob that the profile predicts is
worthless *now* and valuable *after* fp8. Ordering matters more than the knob.

**EP is kept only in combination with fp8**, and for a different reason than it
was first queued: not the comm it removes, but GEMM shape. At EP=1 each rank
runs 128 expert groups of ~4.6 k rows; at EP=8 it runs 16 groups of ~37 k rows.
Same arithmetic, fewer and larger GEMMs. That is a compute-stream argument, and
it is the only version of the EP hypothesis this profile leaves standing.

### What the profile says about the two rejected memory levers

The trace also prices, quantitatively, the trade that
`gpt_oss_120b_1k_sac_gmm_bs6` and every other "recompute less" idea sits on.

Expert grouped GEMMs show ~324 launches per step over 36 layers, about 9 per
layer, consistent with 2 forward + 2 recomputed + 4 backward. SelectiveAC does
not save `aten._grouped_mm`, so roughly 2 of those 9 -- about **1030 ms, or 11%
of the compute stream** -- is recompute that saving the expert GEMM would remove.

That looks like an 11% lever until the NCCL number is put next to it. Saving the
expert GEMM costs ~17.7 GiB per batch unit on 36 layers, which forces bs=18 down
to about 6. Compute scales with tokens; the 5599 ms of NCCL does **not** -- it is
set by parameter count and is the same at any batch. At bs=6 the compute stream
falls to roughly 3160 ms against 5599 ms of communication, so **the comm stops
being hidden and starts setting the step time**, for a third of the tokens:

    bs=18   compute 9490 ms > NCCL 5599 ms   -> compute-bound, comm free
    bs=6    compute ~3160 ms < NCCL 5599 ms  -> comm-bound, ~0.6x throughput

So the reason not to trade batch for recompute is not that recompute is cheap.
It is that **batch is what hides the communication**, and there is no batch-neutral
way to buy the 11% back. This is the same mechanism that makes the reference's
bs=18 worth 5x the bs=1 baseline, now visible directly rather than inferred.

It also says the complement to fp8 is a *larger* batch, not a smaller one: fp8
shortens the compute stream, which erodes the margin hiding 5599 ms of NCCL, and
raising batch restores it while amortising the same fixed comm over more tokens.

### Caveat on the profile

Taken at step 12, where the run is at ~540 TFLOP/s against a steady 787 -- so
the compute stream at steady state is shorter than the 9490 ms measured here
while the NCCL bytes are unchanged. That makes comm *relatively* larger in
steady state than this trace shows, which is the direction that matters for the
argument above and is why the bf16-reduce combination is still on the list. The
kernel *mix* is what was read from it, and mix is not what changes between step
12 and step 120.

## Measured

Control is job 263, which reproduces job 56: **771.9 TFLOP/s at 232.96 GiB
(84.3%)**, the same memory footprint job 56 reports to the decimal, with 0
dataloader retries and 0 Triton cubin misses. The 1.9% below job 56's 786.9 is
inside this cluster's ~2% noise floor, so the launcher and worktree are sound.

| job | arm | bs | TFLOP/s/GPU | cluster | mem | vs control | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 56 | `sac_compile` (the reference) | 16 | 786.9 | 50.36 PF | 84.2% | +1.9% | 250 steps, prior sweep |
| **263** | **`_1k_ref` (control)** | 16 | **771.9** | **49.40 PF** | **84.3%** | — | clean |
| 254 | `_1k_ref_bs18` | 18 | 730.6 | 46.76 PF | 92.1% | -5.4% | **contaminated**, see below |
| 281 | `_1k_fp8_linear` | 16 | 618.8 | 39.60 PF | 70.8% | **-19.8%** | clean; frees 37 GiB |
| 272 | `_1k_ep8` | 16 | — | — | OOM | — | wanted 6.6 GiB, 5.0 free |
| 273 | `_1k_ep16` | 16 | — | — | OOM | — | wanted 8.7 GiB, 3.2 free |
| 286 | `_1k_mxfp8_experts` | 16 | — | — | — | — | **grad_norm NaN at step 1** |
| 287 | `_1k_mxfp8` | 16 | — | — | — | — | cancelled after 286 |

**Nothing beat the control.** The target was +29.6% over 771.9 (or +27.1% over
job 56's 786.9) and the best measured arm is the control itself.

### fp8 on the dense Linears: -19.8%, and the reason is not the GEMMs

Dense GEMMs are 9.9% of the compute stream, so they cannot lose 20% by
themselves even if fp8 made them infinitely slow. `Float8LinearConverter` sets
`torch._inductor.config.emulate_precision_casts = True` (float8.py:122), which
is global -- it changes codegen for the **whole model**, not the converted
Linears. That is the most likely cost, and it is a property of how torchtitan
enables the rowwise recipe rather than of fp8.

Worth noting separately: it freed **37 GiB** (232.96 -> 195.72), which is 3.4
batch units. On a model where batch is what hides communication that headroom
has value -- just not at -20%.

### Expert parallelism: OOM, for a reason the bs=1 evidence hid

Job 105 showed EP=8 costing 48.36 GiB against the baseline's 68.33, which reads
as EP being cheap. That was at **bs=1**. At bs=16 both EP=8 and EP=16 OOM, and
the amount they overshoot by grows with EP degree (6.6 GiB at EP=8, 8.7 at
EP=16). The all-to-all staging buffers scale with tokens *and* with the number
of peers, so EP's memory cost is invisible at bs=1 and binding at bs=16.

EP could still be run at a smaller batch, and was not, because the profile had
already removed its motive: the collective EP replaces is 97% hidden, so the
only surviving argument for it was expert-GEMM *shape*, and buying that with
batch is the trade the next section prices at roughly 0.6x.

### The bs=18 row is contaminated, by me

Job 254 ran while NFS- and `/tmp`-heavy commands were being run on the login
node -- which is one of the 16 allocated nodes. It logged 5 dataloader retries
against job 56's 0 and job 263's 0, and its throughput dips line up in time
with those commands. It is in the table because it is the measurement that
uncovered the bs=16/bs=18 mix-up, not because -5.4% is a result. Job 294
re-measures it on a quiet cluster.

## A confound in the fp8 arms, recorded up front

`Float8GroupedExpertsConverter.convert` calls `swap_token_dispatcher`, which
replaces `AllToAllTokenDispatcher` with `TorchAOTokenDispatcher` (it needs a
dispatcher that pads token groups to a multiple of 16). So **every fp8 arm
changes two things**: the expert GEMMs become fp8, and the token dispatcher
changes.

That matters because the profile puts MoE dispatch/combine data movement at
18.2% of the compute stream -- the second largest item. A gain measured here is
"fp8 plus whatever the TorchAO dispatcher does differently", not fp8 alone, and
the two cannot be separated by configuration: the converter will not accept the
old dispatcher. Separating them would need a run with the dispatcher swapped and
no quantization, which is not expressible in this config surface.

Not a reason to avoid the arm -- the combination is what a user would deploy --
but the attribution has to be stated that way rather than credited to fp8.

## Predictions on record

Written before the runs, so they can be scored rather than reconstructed:

1. **MXFP8 is the largest single lever, worth +15-25%.** The MoE FFN is ~57% of
   per-token FLOPs and attention projections ~15%. This is the opposite call to
   dsv4, where MXFP8 gave +0.5% -- there a FlexAttention backward was 58% of GPU
   time and the expert GEMMs it bypassed were 1.6%. GPT-OSS uses flash-attn
   varlen with a 128-token window on half its layers, so it should have no such
   dominant attention term. **If MXFP8 comes in flat, this reasoning is wrong
   and the profile will say what actually costs.**
2. **bf16 reduce is worth +8-15%**, more than the +0.6% it gave dsv4, because at
   EP=1 it touches all ~117 B parameters rather than only the ~21.6 B non-expert
   ones.
3. **EP at bs=18 is roughly neutral to negative.** What EP removes (the
   per-layer 128-expert weight all-gather) is fixed, what it adds (a token
   all-to-all) scales with batch, and these run at 18x the batch where EP=8 last
   looked good. EP's real value here is the ~20 GiB it frees, which buys batch.
4. **MXFP8 and bf16 reduce are close to additive**, since one is arithmetic and
   the other is bytes on the wire.


## Why 1000 is not reachable on this software stack

Half the critical path is one thing -- the bf16 expert grouped GEMM, 49.7% --
and it runs at roughly **898 TFLOP/s against a 2503 bf16 peak, about 36%**.
Closing a 29.6% gap means either making that GEMM narrower (quantize it) or
making it more efficient (better shapes or kernels). Both routes were tried and
both are closed by things outside this repository.

### Quantizing the expert GEMM: two independent blockers

| path | status on GB300 / sm_103 |
| --- | --- |
| `torch._scaled_mm` 2D rowwise | **works** -- but only reaches the 9.9% dense GEMMs |
| `torch._grouped_mm` bf16 | works (this is what the reference runs) |
| `torch._scaled_grouped_mm` fp8 | **aborts** -- CUTLASS arch-conditional MMA, identical on torch 2.14.0 and 2.15.0.dev20260907 |
| MXFP8 grouped, stock torchao | **asserts** -- CuTeDSL cast needs K % 128 == 0, K is 2880 |
| MXFP8 grouped, patched (this branch) | **runs, and trains to NaN** |

The fp8 abort is worth being precise about: CUTLASS calls `abort()` rather than
raising, so it kills the process with no Python traceback. That is why the first
120B fp8 job appeared to fail inside Inductor's autotuner -- the autotuner was
simply the first caller to touch the kernel. Disabling the autotuner moved the
message, not the failure.

### The MXFP8 patch works at the kernel level and is verified

`gb300/patches/torchao-mxfp8-nonmultiple-128-K.patch` routes the activation cast
through Triton when K % 128 != 0. Auditing the path showed the 128 constraint was
isolated to that single kernel -- weights already use the Triton cast, the 3D
weight quantize needs only %32, and the scale swizzle has no K constraint.

It is verified, not assumed:

| check | result |
| --- | --- |
| K=2880, 128 groups, 524288 rows, fwd+bwd | runs, all finite |
| rel err vs bf16 (output / dA / dB) | 0.03763 / 0.03764 / 0.03775 -- mxfp8's own error |
| vs CuTeDSL at K=2944, 2816 (128 groups) | **0.00000000 on both** -- bit-identical |
| 128 unequal zero-padded groups, N=2880 and N=5760 | finite in fwd, dA, dB |
| debugmodel end-to-end, 4 GPUs, compile + SelectiveAC | 5 steps, loss 8.10 -> 4.48 |

### Where MXFP8 actually fails, and what was ruled out

At 64 GPUs, **`grad_norm` is NaN at step 1** while the loss reads 12.730.

That loss is *not* evidence the forward is correct. At step 1 the model is
randomly initialised, so cross-entropy sits near ln(201088) = 12.21 almost
regardless of what the experts compute. The honest statement is only that the
gradient is non-finite; the forward is unverified at scale.

Ruled out by measurement rather than argument:

| hypothesis | test | verdict |
| --- | --- | --- |
| zero-padded token groups mint inf/NaN scales | cast an all-zero row | **refuted** -- e8m0 scale 0, not the 255 NaN encoding |
| padding corrupts wgrad, which contracts over tokens | 128 unequal padded groups, fwd+bwd | **refuted** -- finite |
| Inductor lowers the mxfp8 path wrongly | same config with `compile.enable=False` | **refuted** -- NaN either way |
| the patched swizzle disagrees with CuTeDSL | compare at K where both are legal | **refuted** -- bit-identical |

What remains untested, and is where the next session should start:

1. **`GptOssGroupedExperts` passes `offs` that do not cover all of `x_RD`.**
   `offsets_E` is the cumsum of real token counts, while `x_RD` carries extra
   `tail_slack` rows. The CuTeDSL cast is offset-aware; `triton_to_mxfp8_dim0`
   is not -- it casts every row, and the swizzle then sizes itself from the full
   row count rather than from `offs[-1]`. If the grouped GEMM expects a scale
   layout sized by groups, that is a layout mismatch the isolated probes would
   not reproduce, because they construct `offs` that cover exactly all rows.
   **This is the leading hypothesis and it was not tested.**
2. FSDP/DTensor interaction. `GptOssGroupedExperts.forward` calls `.to_local()`
   on all four expert parameters; the dsv4 sweep hit a closely related failure
   where an MXFP8 tensor subclass met a raw-weight read under FSDP.
3. Per-parameter gradient inspection to name which tensor goes non-finite
   first, which would settle 1 vs 2 immediately and needs one instrumented
   12-step run.

### What would actually unblock the target

- An **sm_103 fp8 grouped-GEMM kernel** in torch. This is the clean fix: it
  needs no patching here and reaches 49.7% of the step.
- Or resolving the MXFP8 NaN above, which is a bounded debugging task with a
  named leading hypothesis.
- Neither is a configuration change, which is why no config in this registry
  reaches 1000.

## Corrections this sweep makes to the existing write-ups

1. **`TUNING_RESULTS_120B.md` records job 56 as bs=18. It was bs=16.** Job 56's
   step-1 memory (219.44 GiB, 79.36%) matches the registry's own bs=16 probe
   note; bs=16 -> bs=20 spans 84.25% -> 98.73%, i.e. 3.62 points per batch unit,
   predicting 91.5% for bs=18; and bs=18 measures 92.07%. Job 56 ran the
   registry default with no override.
2. **"120B is more COMMUNICATION-bound" is wrong at this operating point.** It
   was inferred from a throughput ratio. Measured: 97% of NCCL is hidden, the
   compute stream is busy 98.1% of the step, and the GPU is idle 0.2%.
3. **"MXFP8 is blocked because torchao ships no aarch64 kernels" is no longer
   the blocker.** That gap is fixed and the build recipe is in
   `MXFP8_120B.md` -- the kernels load and register on sm_103 under torch 2.14.
   MXFP8 on GPT-OSS is blocked by K=2880 instead, which the patch addresses at
   the kernel level.
4. **Every TFLOP/s figure in `gb300/*.md`, including the 786.9 reference and
   anything here, is on an accounting that overcharges sliding-window layers by
   ~18%.** `perf/flops-sliding-window` corrects it: 786.9 -> 665.0. Not merged
   here deliberately, so these numbers stay comparable to job 56 -- but a
   corrected 1000 is ~846, and the two must never be mixed in one table.
