# BF16 + torch.compile batch-size sweep — GPT-OSS-20B and Qwen3-30B-A3B

Results of a `local_batch_size` sweep run on one 8×B200 node, using bf16 + `torch.compile` and FSDP-only sharding. All other knobs are held fixed so each row isolates the effect of changing batch size.

## torchtitan pull

- Upstream: https://github.com/pytorch/torchtitan.git
- Branch: `main`
- Commit at run time: [`1690e0ae`](https://github.com/pytorch/torchtitan/commit/1690e0aeafa0a631251fd522b870112343d8bf77) — *"[cpu-offloading] Encode last consumer as dep arg in ao.wait (#3333)"* (Michael Lazos, 2026-05-13)
- Local uncommitted changes on top of that commit:
  - [torchtitan/models/qwen3/config_registry.py](../torchtitan/models/qwen3/config_registry.py) — added a `qwen3_30b` config registry entry (Qwen3-30B-A3B, `seq_len=8192`, FSDP-only, AC=`full` when compile is on, `lr=3e-4`, `warmup_steps=600`, `steps=3000`).
  - [torchtitan/trainer.py](../torchtitan/trainer.py) — added a `[GRAD-PROBE]` debug print at step 1 and every 50 steps that logs gradient norms for `attn_wq`, `attn_wo`, `expert_mlp1`, `expert_mlp2` on layer 0.

## Common setup

| Setting | Value |
| --- | --- |
| Hardware | 1 node, 8× NVIDIA B200 (192 GB), node `ml-16-b200-node-002` |
| Launcher | `torchrun --nproc_per_node=8`, single-node slurm job |
| Parallelism | `pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1` (pure FSDP) |
| Precision | `--training.dtype=bfloat16` |
| Compile | `--compile.enable` (components: `model`, `loss`) |
| Activation checkpoint | `full` for Qwen3 (set in the local helper when compile is on); gpt_oss uses the upstream `gpt_oss_20b` defaults |
| Sequence length | 8192 |
| Dataset | `c4` (HuggingFace) |
| Loss | `ChunkedCELoss` |
| Optimizer / LR | `lr=3e-4`, `warmup_steps=600`, `total_steps=3000` |
| Alloc env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Run length | 30-min dev slurm allocation per job — jobs are stopped well before 3000 steps; metrics below are steady-state mid-run |

Sbatch scripts: [scripts/run_gpt_oss_20b.sbatch](../scripts/run_gpt_oss_20b.sbatch), [scripts/run_qwen_30b.sbatch](../scripts/run_qwen_30b.sbatch).

`tps` is tokens/sec/GPU (rank-local). `tflops` is per-GPU. `mfu` is fraction of B200 peak bf16 FLOPs. `memory` is reserved CUDA memory on rank 0, with the % being the fraction of the 192 GB B200 HBM (as reported by torchtitan).

## GPT-OSS-20B (20.9B params total / 4.2B active)

Run via `--module gpt_oss --config gpt_oss_20b` (upstream config, no local changes).

| local_bs | global_bs | job id | tps (tok/s/gpu) | tflops/gpu | mfu | memory | last logged step | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 80 | 1424 | ~18,300 | ~573 | ~25.5% | 69.5 GiB (38.96%) | 190 | baseline, longest run |
| 10 | 80 | 1425 | ~18,200 | ~570 | ~25.3% | 69.5 GiB (38.96%) | 60 | rerun of bs=10, matches 1424 |
| 14 | 112 | 1428 | ~17,150 | ~537 | ~23.9% | 84.8 GiB (47.54%) | 50 | |
| 20 | 160 | 1430 | ~16,740 | ~524 | ~23.3% | 104.5 GiB (58.60%) | 140 | largest bs tested |

Observations
- tps/GPU drops monotonically as local bs grows (18.3k → 17.2k → 16.7k), so larger micro-batches are slightly less compute-efficient on this stack even though global_bs more than doubles from bs=10 to bs=20.
- Memory scales close to linearly: +~1.5 GiB per extra sample (bs=10→14: +15.3 GiB; bs=14→20: +19.7 GiB).
- Even at bs=20 the model uses only ~59% of HBM, so plenty of headroom remains.

## Qwen3-30B-A3B (30.5B params total / 3.4B active)

Run via `--module qwen3 --config qwen3_30b` (local registry entry — see the diff above).

| local_bs | global_bs | job id | tps (tok/s/gpu) | tflops/gpu | mfu | memory | last logged step | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 64 | 1423 | ~11,865 | ~446 | ~19.8% | 124.0 GiB (69.56%) | 160 | |
| 10 | 80 | 1426 | ~11,950 | ~449 | ~20.0% | 146.1 GiB (81.92%) | 70 | |
| 12 | 96 | 1429 | ~12,120 | ~455 | ~20.2% | 168.1 GiB (94.27%) | 50 | near memory ceiling |
| 14 | — | 1427 | — | — | — | 176.5 GiB (98.98%) at step 10 | OOM | mapping failed in CUDACachingAllocator with repeated `expandable_segments` failures across all 8 ranks; only step 1 completed normally |

Observations
- Unlike gpt_oss, Qwen3-30B-A3B *gains* tps as local bs grows from 8 → 12 (~11.9k → ~12.1k), suggesting bs=8 is slightly compute-starved for this MoE layout.
- Memory grows ~22 GiB per +2 samples (8→10→12). bs=12 sits at 94.3% reserved, leaving little headroom for activation peaks.
- bs=14 OOMs at the very start of training — peak reserved hit 98.98% before failure; this is the practical ceiling at `seq_len=8192` / FSDP-only on 8×B200.

## Cross-model takeaways

- gpt_oss_20b is roughly 1.5× faster per GPU than qwen3_30b at comparable bs (~17–18k vs ~12k tps), in line with the smaller total parameter count and lower expert capacity.
- mfu is ~5 pp higher on gpt_oss (23–25%) than on qwen3 (~20%) across the swept range; both are well below the ~50% mfu typical of dense compute-bound training, reflecting MoE + activation-checkpointing overheads.
- Largest safe local_bs in this configuration:
  - gpt_oss_20b: ≥ 20 (not pushed to OOM here).
  - qwen3_30b: 12 (bs=14 OOMs).

## How the numbers were extracted

Lines of the form
```
step:  N  loss: ...  grad_norm: ...  memory: X GiB(Y%)  tps: Z  tflops: T  mfu: M%
```
are emitted by [torchtitan/components/metrics.py](../torchtitan/components/metrics.py) every `metrics.log_freq` steps. The tables above use steady-state values reported after compile warmup (step ≥ 20 for short runs, step ≥ 60 for the longer ones); step-1 numbers are excluded since they include compilation cost (typically 7–8× slower than steady state).
