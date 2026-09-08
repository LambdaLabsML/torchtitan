# DeepSeek V4 Pro on 64x GB300 -- baseline result

Job 77, branch `lambda_64xgb300_dsv4_pro_baseline`, config
`deepseek_v4_pro_64xgb300`, 16 nodes x 4 GB300, 50 steps, **exit 0**.

**A 1.573 trillion parameter model trained end to end on 64 GPUs.**

## Headline

| metric | value |
|---|---|
| parameters | 1,572,997,179,491 (1.573 T) |
| GPUs | 64 (16 nodes x 4 GB300) |
| mesh | `dp_shard=64, ep=64, tp=1, cp=1, pp=1` |
| TFLOP/s per GPU (steps 11-50 mean) | **31.48** |
| aggregate | **2.01 PFLOP/s** |
| MFU | **1.26 %** (of 2.5 PFLOP/s per GB300) |
| step time | 42.3 s |
| throughput | 6192 tokens/s (96.8 per GPU) |
| tokens per step | 262144 (64 DP ranks x 4096) |
| peak memory | **244.94 GiB / 276.50 GiB (88.6 %)** |
| loss | 12.39 -> 2.80 |
| grad norm | 53.0 -> 1.09 |

## Step trace

| step | loss | grad_norm | memory | TFLOP/s | MFU |
|---|---|---|---|---|---|
| 1  | 12.39 | 53.0  | 188.05 GiB (68.0 %) | 1.47  | 0.06 % |
| 2  | 7.86  | 293.0 | 233.76 GiB (84.5 %) | 8.61  | 0.34 % |
| 3  | 19.82 | 73.6  | 233.76 GiB (84.5 %) | 32.96 | 1.32 % |
| 5  | 34.16 | 79.9  | 242.04 GiB (87.5 %) | 21.47 | 0.86 % |
| 10 | 9.60  | 28.6  | 241.57 GiB (87.4 %) | 29.39 | 1.18 % |
| 20 | 4.73  | 21.9  | 243.21 GiB (88.0 %) | 31.44 | 1.26 % |
| 30 | 3.11  | 11.0  | 241.57 GiB (87.4 %) | 31.57 | 1.26 % |
| 40 | 2.84  | 4.42  | 241.57 GiB (87.4 %) | 31.68 | 1.27 % |
| 50 | 2.80  | 1.09  | 244.94 GiB (88.6 %) | 31.13 | 1.25 % |

## Memory: the projection held, the headroom did not

Step 1 peaks at **188.05 GiB** against a projected model state of 196.6 GB
(183.1 GiB) -- accurate to about 3 %. AdamW allocates `exp_avg`/`exp_avg_sq` on
the first `.step()`, which is why step 2 jumps to 233.76 GiB and stays there.

What the projection did *not* budget was the rest. Steady state peaks at
**244.94 GiB**, so activations, FSDP all-gather buffers, the flex-attention
workspace and fragmentation take **~56.9 GiB of the 88.4 GiB** left after model
state. The plan fits, with ~11 % spare, but on 65 % of the headroom rather than
the comfortable margin the earlier estimate implied.

This retroactively rules out the 12 B/param alternative. That option projected
to 295 GB/GPU of model state (99 % of HBM) with nothing for activations; with
~57 GiB of real overhead it would have OOM'd outright. `training.dtype=bfloat16`
was not the best of several options -- it was the only one.

## Throughput: attention-kernel bound, not hardware bound

1.26 % MFU is bad, and it is not a statement about GB300. The dominant cause is
the flex-attention tiles this model forces on Blackwell. `pro` uses
`head_dim=512`, which does not fit the shared-memory budget at any normal tile,
so both directions run at the only sizes that launch at all:

```
forward   BLOCK_M=32,  BLOCK_N=32,  num_stages=1, num_warps=4
backward  BLOCK_M1=16, BLOCK_N1=32, BLOCK_M2=32,  BLOCK_N2=16
```

Contributing, in rough order of expected cost:

1. **Attention tiles** -- 32x32 forward and 16x32/32x16 backward are far below
   the 128x128 a healthy kernel would use.
2. **`compile.enable=False`** -- inherited from stock `deepseek_v4_pro`.
   Upstream's `deepseek_v3_671b` enables it for `loss`.
3. **FullAC** -- recompute on all 61 blocks. Needed for memory here, but at
   88.6 % peak there is little room to trade back.
4. **EP=64 all-to-all across 16 nodes with no NVSwitch** -- every MoE layer
   dispatches over IB.

Untangling these is the obvious next piece of work; (2) is nearly free to test.

## Loss is not a convergence signal

The trace is violently unstable early -- 12.39, then 7.86, then 19.82, peaking
at **34.16** on step 5 with a grad norm of 293 on step 2 -- before settling to
2.80. It ends up looking like a curve, but it is not one. Three reasons, all
inherited or forced:

- `lr=8e-4` from stock `deepseek_v4_pro`, a debug-scale LR. Upstream's
  `deepseek_v3_671b` uses `2.2e-4` with 2000 warmup steps; this config warms up
  for 2.
- **bf16 AdamW moments**, required to fit at all (see above).
- `c4_test` is a 4.7 MB local JSON. At 262144 tokens/step for 50 steps the run
  consumes ~13 M tokens, so it loops the dataset roughly 11 times -- the late
  "improvement" is partly memorization.

Treat 31.48 TFLOP/s as the result of this run and the loss as evidence that
nothing diverged. A convergence run needs fp32 moments (and therefore ~110
GPUs), a real LR schedule, and a real dataset.

## Reproducing

```bash
sbatch gb300/dsv4_64xgb300.slurm          # defaults to deepseek_v4_pro_64xgb300
```

Environment and the six upstream defects that had to be worked around are in
`DSV4_PRO_FEASIBILITY.md`.
