# GPT-OSS-120B baseline on 64x GB300

Out-of-the-box baseline for `gpt_oss_120b` across 16 nodes x 4 GB300 on the
yqb01-qa01 cluster, 2026-09-06. Same process as the
[20B baseline](README.md).

```bash
sbatch gb300/gpt_oss_120b_64xgb300.slurm            # 50 steps
STEPS=250 sbatch gb300/gpt_oss_120b_64xgb300.slurm  # what the numbers below are
```

## Result (job 48, 250 steps, stock defaults)

Steady state = steps >= 100.

| | |
| --- | --- |
| per-GPU | **157.6 TFLOP/s** (min 157.3, max 158.0) |
| per-GPU | 3,476 tokens/s |
| **64 GPUs** | **10.09 PFLOP/s**, ~222,000 tokens/s |
| MFU | 6.30% |
| memory | 68.33 GiB/GPU (24.71%) |
| loss | 12.68 -> 5.27 |

Unusually steady: 0.4% spread across the whole measurement window.

## Against the 20B baseline

| | 20B (job 46) | 120B (job 48) | |
| --- | --- | --- | --- |
| per-GPU TFLOP/s | 281.4 | **157.6** | 56% |
| cluster | 18.01 PF | **10.09 PF** | 56% |
| MFU | 11.26% | **6.30%** | |
| tokens/s per GPU | ~9,190 | 3,476 | 38% |
| memory | 18.46 GiB (6.7%) | 68.33 GiB (24.7%) | 3.7x |
| layers / experts | 24 / 32 | 36 / 128 | |

**6x the parameters cost only 3.7x the memory.** Parameters, gradients and
optimizer state all shard 64 ways, so the extra parameters are cheap per GPU;
the 36 layers driving activation memory is what actually moved the number, and
activations do not shard.

**120B is more communication-bound than 20B, not more compute-bound.** Both
models route top-4, so active parameters per token — and therefore FLOPs per
token — are comparable. Yet per-GPU throughput falls to 56%. The extra
parameters have to be all-gathered every step regardless of how few of them each
token touches, and at `local_batch_size=1` there is very little compute to hide
that behind.

That is the same diagnosis as 20B, only worse: **24.7% memory at 6.30% MFU means
three quarters of each GPU is idle.** The 20B sweep showed what fixing that is
worth — no-AC plus a batch large enough to use the freed memory took 20B from
281.4 to 942.3 TFLOP/s (3.35x). The headroom here is comparable, so the same
levers should apply, with the caveat that 120B's larger activations will hit the
memory ceiling at a smaller batch than 20B's bs=6.

See [TUNING_RESULTS.md](TUNING_RESULTS.md) for the 20B sweep, including the
sharp cliff near 99% memory that any 120B batch search has to stay clear of.

## Deviations from stock

Exactly one, and it is the same one 20B needed:
`--training.disable_cuda_graphs`. Varlen attention's `cu_seqlens` changes length
as documents pack differently each step, so CUDA graph capture fails on step 2.
A property of the stock model config, not of this cluster.

Everything else is registry default: `local_batch_size=1`, `seq_len=8192`, FSDP
over all 64 ranks (`dp_shard=64`, everything else 1), `expert_parallel_degree=1`,
FullAC, C4 streamed from HF, AdamW lr 8e-4.

Verified from the run log rather than assumed:

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=64, cp=1, tp=1, ep=1
Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp']
Applied FullAC activation checkpointing to the model
Applied FSDP to the model
Optimizer AdamW (model_part=0): 471 params, lr 0.0008
Trainer is initialized with local batch size 1, global batch size 64, ... sequence length 8192
```

As with 20B, the LR schedule is auto-clamped at 250 steps (warmup 2000 -> 250),
so the whole run is warmup and **the loss curve is not on the real schedule**.
Fine for throughput; do not read the loss descent as meaningful.

## Setup

Nothing new was required. `gpt_oss_120b` was already in the registry, and the
only new artifact was the tokenizer:

```bash
python scripts/download_hf_assets.py --repo_id openai/gpt-oss-120b --assets tokenizer config
```

Every cluster-side prerequisite from the 20B work — IMEX for the 64-GPU NVLink
fabric, `mlx5_ib` from `linux-modules-extra`, the Slurm cpuset fix, FA4 — was
already in place. See
[DEBUG_FOR_BASELINE_64xGB300.md](DEBUG_FOR_BASELINE_64xGB300.md); none of it
needed revisiting for 120B.
