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

## Two gotchas that block upstream main on GB300

Both were caught by the 1-node smoke gate rather than by the 16-node run, which
is the entire reason the gate exists.

### 1. Upstream main requires a PyTorch nightly on GB300

`torchtitan/distributed/utils.py:enable_fp32_matmul_emulation_with_bf16x9()` is
called from `init_distributed()` on **every** run and hard-fails on any device
with compute capability `>= (10, 0)`. GB300 reports **(10, 3)**, so it always
fires. It sets:

```python
torch.backends.cuda.matmul.fp32_precision = "bfx9"
```

`torch==2.14.0+cu130` -- the build the GPT-OSS sweep runs on -- exposes that
attribute but accepts only `tf32`, `ieee` and `none`, raising
`RuntimeError: Unknown precision: bfx9`. Every run therefore died in trainer
init, before the model was built:

> ValueError: TorchTitan on NVIDIA GPUs with compute capability 10.0 or later
> requires PyTorch with CUDA BFX9 matmul support (pytorch/pytorch#195301) and
> CUDA 12.9 or later.

`torch==2.15.0.dev20260907+cu130` accepts `bfx9` and runs fp32 matmuls on these
GPUs. This matches upstream's README, which expects a nightly for a from-source
install.

**Consequence beyond this branch:** current upstream main cannot run on GB300
with the torch build the rest of this cluster's work depends on. Any future
rebase of the Lambda fork onto upstream inherits the nightly requirement.

### 2. Upstream's own DeepSeek V4 smoke command is broken as written

The command in `torchtitan/models/deepseek_v4/README.md` passes
`--parallelism.expert_parallel_degree 2` but not
`--training.disable-cuda-graphs`. `training.disable_cuda_graphs` defaults to
`False`, and EP with the default `AllToAllTokenDispatcher` synchronizes with the
host during dispatch, so config validation rejects the pair:

> Error parsing Config: CUDA graphs support only expert parallel token
> dispatcher configurations without CPU synchronization. [...] Unsupported
> token dispatcher: AllToAllTokenDispatcher.Config.

`gb300/dsv4_smoke_1node.slurm` adds the flag. `deepseek_v4_pro_64xgb300` sets
`disable_cuda_graphs=True` in the config itself, so it was never affected --
which is also why it passed `Trainer.Config.__post_init__` while the debugmodel
did not.

### 3. deepseek_v4's MoE router is broken under tensor parallelism

`models/common/moe.py:341` does
`topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)`. Under
TP with sequence parallelism the router returns full-length
`topk_expert_ids_TK` while `scores_TE` stays token-sharded, so the gather
fails:

> RuntimeError: Size does not match at dimension 0 expected index
> [131072, 3] to be no larger than self [65536, 4]

The `E`/`K` dims are correct (debugmodel: `num_experts=4`, `top_k=3`); only the
token dim is wrong, by exactly the TP degree. This is what makes upstream's
recommended TP2+EP2 smoke config unrunnable.

`deepseek_v4_pro_64xgb300` uses `tensor_parallel_degree=1`, chosen on hardware
grounds (no NVSwitch, and memory does not require TP), which also avoids this
path entirely.

### 4. mtp_layers used a bare nn.ModuleList

`DeepSeekV4Model.__init__` assigned `torch.nn.ModuleList`, which does not
satisfy torchtitan's `Module` protocol, so `verify_module_protocol()` rejected
**every** deepseek_v4 model at trainer init -- all three flavors, even at
`n_mtp_layers=0`, since the empty list is assigned before the `None` check:

> RuntimeError: The following modules do not satisfy the Module protocol:
> 'mtp_layers' (ModuleList)

torchtitan ships `torchtitan.protocols.module.ModuleList` for exactly this and
the sibling `deepseek_v3/mtp.py:214` already uses it, so this is a port
oversight in V4. **This is the one place this branch modifies upstream model
code** (two lines, matching V3). Without it nothing runs on any hardware. The
CPU unit tests do not catch it because they never call
`verify_module_protocol`.

### 5. head_dim=512 exceeds Blackwell's shared memory in flex attention

Both real V4 flavors use `head_dim=512` (`flash` and `pro`; the debugmodel uses
256). On GB300 the Triton flex-attention kernel wants more shared memory than
the hardware has, and the first forward pass dies:

> No valid triton configs. OutOfMemoryError: out of resource:
> triton_flex_attention Required: 294912 Hardware limit: 232448

Reproduced on a single GB300 with a standalone `flex_attention` call at
`D=512` plus a causal block mask -- the same `No valid triton configs` with
upstream's default (empty) `kernel_options`. Tile search at `D=512`:

| kernel_options | result |
|---|---|
| `{}` (upstream default) | FAIL -- No valid triton configs |
| `BLOCK_M=64, BLOCK_N=64, stages=1, warps=4` | FAIL -- NoValidChoicesError |
| `BLOCK_M=64, BLOCK_N=32, stages=1, warps=4` | FAIL -- NoValidChoicesError |
| `BLOCK_M=128, BLOCK_N=32, stages=1, warps=4` | FAIL -- launch failure |
| `BLOCK_M=32, BLOCK_N=64, stages=1, warps=4` | FAIL -- launch failure |
| `BLOCK_M=32, BLOCK_N=32` (stages/warps unpinned) | FAIL -- launch failure |
| **`BLOCK_M=32, BLOCK_N=32, stages=1, warps=4`** | **OK** |

Inductor filters candidates by shared memory even when `kernel_options` names a
tile explicitly, so larger tiles raise `NoValidChoicesError` rather than
falling back. Exactly one tile runs, and `deepseek_v4_pro_64xgb300` pins it on
all 61 layers.

This is a correctness requirement, not tuning: without it neither real V4
flavor can complete a forward pass on GB300. It is also a *small* tile, so
attention throughput pays for it -- though `pro` is 1.55 T of its 1.573 T
parameters in experts, so MoE, not attention, should dominate the step.

It also explains why upstream validated only the 4-GPU debugmodel: at
`head_dim=256` the default autotune sweep finds a config, so the bug is
invisible there.

## Environment

Venvs, in build order. The runs use **`/mnt/dgxc/venvs/dsv4n`**
(`torch==2.15.0.dev20260907+cu130`, required per the BFX9 gotcha above).
`/mnt/dgxc/venvs/dsv4` is the same dependency set on `torch==2.14.0+cu130`,
kept as a rollback; it cannot run on GB300. Both carry upstream's new set:
`grain==0.2.18` (replacing `torchdata`), `spmd_types==0.2.5`,
`attn-gym[linear]==0.0.8`, pinned `torch_remat`, `torch_checkpointing==0.1.0`.

`/mnt/dgxc/venvs/torchtitan` was deliberately left untouched: it installs
torchtitan in editable mode and is shared by the queued GPT-OSS jobs, so
upgrading it in place would have changed the code under a running sweep.

## Verified

- Merge is a clean fast-forward; tree matches upstream f6b9152e.
- Upstream's DeepSeek V4 CPU tests pass: `test_deepseek_v4_dsa.py`,
  `test_deepseek_v4_flops.py`, `test_deepseek_v4_mtp.py` (4 passed, 4 subtests).
- `deepseek_v4_pro_64xgb300` builds and passes `Trainer.Config.__post_init__`.
- `ParallelDims` accepts the mesh at `world_size=64`.

Verified on GB300 GPUs. Every change `deepseek_v4_pro_64xgb300` makes was
exercised at debug scale first, on 4x GB300 with `pro`'s topology
(FSDP=4, TP=1, EP=4, microbatch 16384 tokens, 3 steps each):

| run | dtype | AC | exit | loss | peak mem/GPU |
|---|---|---|---|---|---|
| baseline    | fp32 | none   | 0 | 8.29 -> 6.94 -> 5.13 | 6.61 GiB |
| + FullAC    | fp32 | FullAC | 0 | 8.10 -> 6.64 -> 4.84 | 3.24 GiB |
| + bf16 (= pro's shape) | bf16 | FullAC | 0 | 8.07 -> 6.71 -> 5.14 | 3.23 GiB |

Together these exercise the torch nightly with `bfx9`, the `ModuleList` fix,
FSDP + EP, `training.dtype=bfloat16`, `FullAC`, and forward/backward/optimizer.
Loss falls in all three and throughput is unchanged by AC (11,016 vs 11,209
tps), so FullAC costs essentially nothing here while halving activation memory.

Two things not to over-read from that table:

- **FullAC's 2.04x memory saving is an activation saving.** It is large at
  debug scale because a 10 M-parameter model is all activations. At `pro` scale
  weights and optimizer state dominate, so AC's job there is narrower: keeping
  the 61-block activation stack inside the ~101 GB left after the weights.
- **bf16 shows no memory win at debug scale** (3.24 -> 3.23 GiB) for the same
  reason -- there are barely any weights to shrink. Its 2x saving is the whole
  reason `pro` fits, and it only shows up once the 1.573 T of parameters
  dominate the budget.

**Partially confirmed on hardware.** Job 75 (16 nodes, 64 GPUs) built the mesh
exactly as designed and materialized the model, then failed in the first
forward pass on the flex-attention tile above:

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=64, cp=1, tp=1, ep=64
Model deepseek_v4 deepseek_v4_pro size: 1,572,997,179,491 total parameters
Applied FullAC activation checkpointing to the model
Applied FSDP to the model
Optimizer AdamW (model_part=0): 1833 params {'fused': True, 'lr': 0.0008, ...}
Trainer is initialized with 4096 tokens per DP rank, 262144 tokens per train step
```

1.573 T parameters materialized and FSDP sharded them 64 ways with no OOM. The
measured parameter count (1,572,997,179,491) matches the meta-device estimate
and the 262144 tokens/step matches 64 x 4096.

**What is not yet validated is the full 196.6 GB/GPU.** Job 76 reports:

```
CUDA capacity: NVIDIA GB300 with 276.50GiB memory
CUDA memory usage for model: 54.24GiB(19.62%)
Peak FLOPS used for computing MFU: 2.500e+15
```

54.24 GiB is the bf16 *parameter* shard only (1.573 T x 2 / 64 = 49 GB, plus
overhead). The optimizer is constructed at init -- 1833 param groups -- but
PyTorch AdamW allocates `exp_avg` and `exp_avg_sq` lazily on the first
`.step()`, which has not yet run. The projected total is params 49 GB +
grads 49 GB + two bf16 moments 98 GB = ~196 GB, and only a completed step
proves it.

Note also that torchtitan reports **276.50 GiB** usable per GB300, not the
277.55 GiB of raw HBM, which trims the headroom slightly.

Still unverified: a completed step, and therefore the 196.6 GB/GPU total,
TFLOP/s, expert all-to-all throughput over IB at EP=64, and whether
activations fit in what remains.

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
