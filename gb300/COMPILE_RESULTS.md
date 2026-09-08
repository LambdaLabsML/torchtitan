# compile.components=["loss"] on DeepSeek V4 Pro / 64x GB300

Job 80, `deepseek_v4_pro_64xgb300_compile`, 25 steps, **exit 0**.
Compared against the baseline over the same step range.

| metric (steps 11-25) | baseline | compile(loss) | delta |
|---|---|---|---|
| mean TFLOP/s per GPU | **31.39** | **29.50** | **-6.0 %** |
| mean MFU | 1.256 % | 1.180 % | -0.076 pp |
| peak memory | 243.21 GiB | 236.02 GiB | -7.2 GiB |
| final loss (step 25) | -- | 3.21748 | -- |

**Loss-only compile is a ~6 % throughput regression.** It is not free: it costs
more than it saves. It does return ~7 GiB of memory, but not enough to buy a
larger microbatch (see below).

`ChunkedLossWrapper` already chunks the logits, so there was little for compile
to fuse, and the compiled region adds overhead around a loss that was already
cheap. On a model where 1.55 T of 1.573 T parameters are experts and the step is
attention-kernel bound, the loss is not where the time goes.

Worth noting the run is noisier than the baseline: step 20 read 26.50 TFLOP/s
and step 25 read 30.21, so single-step readings are unreliable here. The -6 %
is the 15-step mean.

## Model compile is blocked, and that is where the value would have been

See `COMPILE_FINDINGS.md`. `components=["model","loss"]` fails in Inductor
lowering dsv4's data-dependent block-mask construction, and two fixes were
attempted and reverted. Model-level fusion is the part that might have mattered;
loss-only compile is the part that was reachable, and it does not help.

## The microbatch cannot grow

6144 tokens (1.5x) was tried and **OOMs at step 3** (job 79):

```
CUDA out of memory. Tried to allocate 15.75 GiB. GPU 3 has a total capacity of
276.50 GiB of which 10.96 GiB is free
```

It cleared step 2 at 245.94 GiB, which is why an early reading looked like a
fit. **Peak memory keeps climbing past step 2** -- the baseline ran 233.76 GiB
at step 2, 242.83 at step 4 and 246.72 by step 6 -- so a two-step run is not
evidence that a batch size fits. Judge memory at step 6 or later.

The 31.56 GiB that looks free in the baseline is mostly consumed by that drift,
not available for a larger batch.

## Where the throughput actually is

Neither lever in this round moved the number, and both failed for the same
underlying reason: the baseline is attention-kernel bound at 1.26 % MFU, and
memory is too tight at 88.6 % peak to trade anything for speed.

The remaining levers are upstream changes, not configuration:

1. **`head_dim=512` shared memory.** Only a 32x32 forward / 16x32-32x16 backward
   Triton tile launches on GB300's 232448 B budget. Larger tiles would be the
   single biggest win.
2. **Hoist block-mask construction out of the traced forward**, which is what
   blocks model compile.
3. **MXFP8**, once a torchao aarch64 CUDA build exists -- upstream reports up to
   28 % on B200. Currently unavailable here (no aarch64 wheel, no nvcc).
