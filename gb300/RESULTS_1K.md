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

## Profile of the reference step

_pending -- runs first, and aims everything below._

## Measured

| job | arm | bs | EP | reduce | MXFP8 | TFLOP/s/GPU | cluster | mem | vs ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _pending_ | | | | | | | | | |

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
