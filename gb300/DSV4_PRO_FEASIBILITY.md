# DeepSeek V4 on 64x GB300 -- what fits, what does not

Branch: `lambda_64xgb300_dsv4_pro_baseline`, fast-forwarded from
`origin/main` (0d2438f9) to upstream `pytorch/torchtitan` main at **f6b9152e**.
The tree is stock upstream -- no Lambda modifications to model or training code.

## Result

`deepseek_v4_pro` **cannot be trained on 64x GB300**, in stock form or with any
stock memory-saving flag. It is a **1.573 T parameter** model and the model
state alone exceeds total cluster memory.

## Measured numbers

Parameter counts, built on meta device from
`torchtitan.models.deepseek_v4.model_registry`:

| flavor              | params    | routed+shared experts |
|---------------------|-----------|-----------------------|
| `debugmodel`        | 0.01 B    | ~0                    |
| `deepseek_v4_flash` | 284.3 B   | 278.1 B               |
| `deepseek_v4_pro`   | 1573.0 B  | 1551.4 B              |

`deepseek_v4_pro` shape: `dim=7168`, `n_layers=61`, `num_experts=384`,
`moe_inter_dim=3072`, `top_k=6`, `num_shared_experts=1`, `vocab=129280`,
`dense_layers=set()` (all 61 layers are MoE). The expert stack alone is
384 x 3 x 7168 x 3072 x 61 = 1.55 T parameters.

Cluster capacity:

- 64 x GB300 @ 284208 MiB = **19.07 TB** HBM total
- 16 nodes x 963905 MiB host RAM = **16.17 TB** host RAM total

Model-state requirement for 1.573 T parameters:

| dtype policy                                        | B/param | needed   | % of 64-GPU HBM | GPUs for state | GPUs w/ 30% headroom |
|-----------------------------------------------------|---------|----------|-----------------|----------------|----------------------|
| torchtitan default (fp32 shard + fp32 grad + fp32 AdamW) | 16 | 25.17 TB | **132 %**       | 84             | **110**              |
| bf16 params+grads, fp32 AdamW                        | 12      | 18.88 TB | 99 %            | 63             | 82                   |
| bf16 params+grads, bf16 AdamW (not stock)            | 8       | 12.58 TB | 66 %            | 42             | 55                   |

torchtitan's default is the 16 B/param row: `mixed_precision_param="bfloat16"`
is an FSDP2 *compute* dtype (`MixedPrecisionPolicy`), so the sharded parameter,
its gradient (`mixed_precision_reduce="float32"`) and both AdamW moments are all
fp32. That is 25.17 TB against 19.07 TB of HBM -- 132 % of the machine, before a
single byte of activations, NCCL buffer, or CUDA context.

Even the theoretical bf16 floor (12 B/param, which no stock config selects)
lands at 99 % of HBM, leaving roughly 3 GB per GPU for everything else.

### CPU offload does not rescue it

`--training.enable-cpu-offload` is a stock flag and offloads params, grads and
optimizer state via FSDP2 `CPUOffloadPolicy`. It does not help here: 25.17 TB of
state against **16.17 TB** of host RAM. HBM + host RAM combined is 35.24 TB, so a
hand-written per-layer hybrid policy could hold the state, but that is not a
stock configuration and step time would be dominated by host transfer.

## The stock `deepseek_v4_pro` config is not a scale-out recipe

Independently of the memory wall, `deepseek_v4_pro()` in
`torchtitan/models/deepseek_v4/config_registry.py` is smoke-test scaffolding,
byte-identical in structure to `deepseek_v4_debugmodel()`:

- `expert_parallel_degree=1` -- no EP at all across 384 experts
- `activation_checkpoint=None`
- `compile=CompileConfig(enable=False)`
- `steps=10`, dataset `c4_test`
- `num_tokens_per_microbatch_per_dp_rank = max_context_length` (4096, one sequence)

Upstream ships **no** multi-node recipe, benchmark config, or slurm script for
`deepseek_v4` at any size -- `grep -rl deepseek_v4` over the tree returns only
the model package and three CPU unit tests. The model README says so directly:

> The debug model has been smoke-tested with 4 GPUs using FSDP2, TP2, EP2 [...]
> Checkpoint compatibility and larger-scale convergence validation should be
> verified before using the larger configs for production training.

So "run it the way upstream runs it" has no 64-GPU answer to copy for `pro`;
upstream's only documented DeepSeek V4 run is the 4-GPU debugmodel.

## What was verified on this branch

- Merge is a clean fast-forward; tree matches upstream f6b9152e exactly.
- Dedicated venv `/mnt/dgxc/venvs/dsv4` built for upstream's new dependency set
  (`grain==0.2.18` replacing `torchdata`, `spmd_types==0.2.5`,
  `attn-gym[linear]==0.0.8`, pinned `torch_remat`, `torch_checkpointing`), on the
  same `torch==2.14.0+cu130` already validated on these nodes.
  `/mnt/dgxc/venvs/torchtitan` was deliberately left untouched -- it installs
  torchtitan editable and is shared by the queued GPT-OSS jobs.
- Upstream's DeepSeek V4 CPU tests pass: `test_deepseek_v4_dsa.py`,
  `test_deepseek_v4_flops.py`, `test_deepseek_v4_mtp.py` -- 4 passed, 4 subtests.

## Options

1. **`deepseek_v4_flash` on 64x GB300** -- 284 B params, 4.55 TB state = 24 % of
   HBM. The largest V4 flavor that fits, runnable fully out-of-the-box today:
   `CONFIG=deepseek_v4_flash sbatch gb300/dsv4_64xgb300.slurm`
2. **`deepseek_v4_pro` on ~28-32 nodes (112-128 GPUs)** -- the smallest count
   that holds stock fp32 AdamW state with activation headroom.
3. **`deepseek_v4_pro` on 64 GPUs with a non-stock memory plan** -- bf16
   optimizer or a per-layer HBM/host hybrid, plus EP + full AC. Deviates from
   upstream defaults, so it stops being a baseline.
