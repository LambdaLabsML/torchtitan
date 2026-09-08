# DeepSeek V4 Pro on 64x GB300 -- fitting a 1.573 T model on 19 TB of HBM

Branch: `lambda_64xgb300_dsv4_pro_baseline`, fast-forwarded from
`origin/main` (0d2438f9) to upstream `pytorch/torchtitan` main at **f6b9152e**,
which is where `torchtitan/models/deepseek_v4/` lives. The model and training
code are stock upstream; the only additions are one derived config and two
slurm launchers under `gb300/`.

## The constraint

`deepseek_v4_pro` is **1.573 T parameters**. Measured on meta device from
`torchtitan.models.deepseek_v4.model_registry`:

| flavor              | params    | routed+shared experts |
|---------------------|-----------|-----------------------|
| `debugmodel`        | 0.01 B    | ~0                    |
| `deepseek_v4_flash` | 284.3 B   | 278.1 B               |
| `deepseek_v4_pro`   | 1573.0 B  | 1551.4 B              |

Shape: `dim=7168`, `n_layers=61`, `num_experts=384`, `moe_inter_dim=3072`,
`top_k=6`, `num_shared_experts=1`, `vocab=129280`, `dense_layers=set()` (all 61
layers MoE). The expert stack alone is 384 x 3 x 7168 x 3072 x 61 = 1.55 T.

Cluster: 64 x GB300 @ 284208 MiB = **19.07 TB** HBM; 16 x 963905 MiB =
**16.17 TB** host RAM.

## Parallelism alone cannot fix this

This is the important part, because it is the intuitive place to look and it is
the wrong one. **Sharding divides model state across ranks; it does not reduce
the total.** Whatever combination of FSDP, TP, EP, PP and CP is chosen, the sum
over all 64 GPUs is fixed by parameter count x bytes-per-parameter, and the
per-GPU floor is that total divided by 64 -- already the best case.

torchtitan's default is `training.dtype="float32"`
(`torchtitan/config/configs.py:89`), which is 16 B/param: fp32 shard, fp32
gradient, and two fp32 AdamW moments. Note that `mixed_precision_param`
defaulting to `bfloat16` does **not** change this -- it is an FSDP2 *compute*
dtype in `MixedPrecisionPolicy`, so the stored shard stays fp32.

| dtype policy                                              | B/param | total    | per GPU (/64) | vs 298 GB HBM |
|-----------------------------------------------------------|---------|----------|---------------|---------------|
| stock: fp32 shard + fp32 grad + fp32 AdamW                | 16      | 25.17 TB | 393 GB        | **132 %**     |
| bf16 shard + bf16 grad, fp32 AdamW                        | 12      | 18.88 TB | 295 GB        | 99 %          |
| full bf16 (`training.dtype="bfloat16"`)                   | 8       | 12.58 TB | **196.6 GB**  | **66 %**      |

The 12 B/param row is why half-measures do not work: it leaves ~3 GB/GPU, which
does not cover activations, NCCL buffers, CUDA context and fragmentation.
`--training.enable-cpu-offload` does not rescue the stock row either -- 25.17 TB
of state against 16.17 TB of host RAM.

So exactly one stock knob moves the floor far enough: `training.dtype`, which is
documented on the field as putting "all parameters, gradients, and optimizer
states in bfloat16, without an extra copy of fp32 weights".

## The config

`deepseek_v4_pro_64xgb300` in `torchtitan/models/deepseek_v4/config_registry.py`
derives from stock `deepseek_v4_pro()` -- the same derive-and-mutate pattern
upstream uses for `deepseek_v3_671b_float8` -- so the delta is auditable and the
model is untouched. Three changes:

| change | from | to | why |
|---|---|---|---|
| `training.dtype` | `float32` | `bfloat16` | the only stock knob that makes it fit: 393 -> 196.6 GB/GPU, 101 GB/GPU headroom |
| `parallelism.expert_parallel_degree` | 1 | 64 | at EP=1 a single layer's expert all-gather is 50.7 GB; at EP=64 each rank owns 6 whole experts and it is 0.79 GB |
| `activation_checkpoint` | `None` | `FullAC` | 61 blocks of dim 7168 with 128 x 512 attention do not fit in the remaining 101 GB otherwise |

`disable_cuda_graphs=True` follows upstream's own `deepseek_v3_671b` recipe.
Optimizer (`default_adamw(lr=8e-4)`), LR schedule, loss, dataloader (`c4_test`),
compile settings and batch shape are all inherited from stock.

EP is genuinely load-bearing here, just not for the reason it first appears: it
does nothing for the *steady-state* floor (196.6 GB/GPU either way) but it
removes the *transient* all-gather spike that would blow the budget during
forward.

Mesh validated with `ParallelDims` at `world_size=64`: `dp_shard=64`, `ep=64`,
`tp=1`, `cp=1`, `pp=1`. EP must divide `dp_shard * cp * tp` (= 64) and 64 | 384,
giving 6 experts per rank.

Budget:

```
per-GPU model state @ 8 B/param   196.6 GB
per-GPU HBM                       298.0 GB
headroom                          101.4 GB
FullAC stored acts (61 blk, 4096 tok, bf16)  3.58 GB
```

TP is left at the stock value of 1. These nodes have **no NVSwitch**, so
sequence-parallel all-gathers every layer would be the wrong thing to add
before knowing whether memory requires it -- and it does not.

## Caveat on the numerics

bf16 AdamW moments are the right trade for a throughput baseline but are not a
convergence-grade choice for a real 1.573 T pretrain. Treat loss curves from
this config as a sanity signal, not as a convergence result. A convergence run
needs either fp32 moments (and therefore ~110 GPUs) or a stochastic-rounding /
Kahan-summation optimizer, neither of which is stock.

## What upstream does and does not give us

Upstream ships **no** multi-node recipe, benchmark config or slurm script for
`deepseek_v4` at any size -- `grep -rl deepseek_v4` returns only the model
package and three CPU unit tests. The model README is explicit:

> The debug model has been smoke-tested with 4 GPUs using FSDP2, TP2, EP2 [...]
> Checkpoint compatibility and larger-scale convergence validation should be
> verified before using the larger configs for production training.

So there is no upstream 64-GPU `pro` invocation to copy. The launch path in
`gb300/dsv4_64xgb300.slurm` is upstream's own
(`srun torchrun ... -m torchtitan.train --module deepseek_v4 --config ...`,
as in `multinode_trainer.slurm`); the additions are this cluster's NCCL
interface names and a worktree/PYTHONPATH guard. The recipe shape (FullAC,
`disable_cuda_graphs`, EP) is modelled on upstream's `deepseek_v3_671b`, the
nearest thing upstream has to a production DeepSeek run.

## Environment

Dedicated venv `/mnt/dgxc/venvs/dsv4` for upstream's new dependency set:
`grain==0.2.18` (replacing `torchdata`), `spmd_types==0.2.5`,
`attn-gym[linear]==0.0.8`, pinned `torch_remat`, `torch_checkpointing==0.1.0` --
on the same `torch==2.14.0+cu130` already validated on these nodes.

`/mnt/dgxc/venvs/torchtitan` was deliberately left untouched: it installs
torchtitan in editable mode and is shared by the queued GPT-OSS jobs, so
upgrading it in place would have changed the code under a running sweep.

## Verified

- Merge is a clean fast-forward; tree matches upstream f6b9152e.
- Upstream's DeepSeek V4 CPU tests pass: `test_deepseek_v4_dsa.py`,
  `test_deepseek_v4_flops.py`, `test_deepseek_v4_mtp.py` (4 passed, 4 subtests).
- `deepseek_v4_pro_64xgb300` builds and passes `Trainer.Config.__post_init__`.
- `ParallelDims` accepts the mesh at `world_size=64`.

Not yet verified on GPU at the time of writing -- gated behind the smoke job:

- `gb300/dsv4_smoke_1node.slurm` runs upstream's documented 4-GPU debugmodel
  test (FSDP2+TP2+EP2) as a cheap gate.
- `gb300/dsv4_64xgb300.slurm` then runs the 16-node pro baseline, chained
  `afterok` on the smoke job so a broken stack cannot waste a 16-node slot.

## Alternative flavor

`deepseek_v4_flash` (284.3 B) fits with room to spare -- 4.55 TB even at stock
fp32, 24 % of HBM -- and needs no config changes at all:

```bash
CONFIG=deepseek_v4_flash sbatch gb300/dsv4_64xgb300.slurm
```
