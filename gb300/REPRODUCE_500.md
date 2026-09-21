# DeepSeek-V4-flash, 8k seq, 32x GB300: 587.5 TFLOP/s balanced routing (job 1015) / 519.1 collapsed (job 1006)

Branch `dsv4_te_mhc` = the full stack. Everything below default-off is a knob.

## Two regimes, two numbers
20-step runs from random init collapse the router onto ~6 experts by step 2;
the EP-rank token skew then costs 8% of the step in barrier waits. With forced
load-balanced routing (`config.debug.moe_force_load_balance`, the `_balanced`
config variants; Megatron's `--moe-router-force-load-balancing`) the barrier
vanishes and the same kernels run +14% faster. Balanced is the proxy for
trained routing; collapsed is what step 20 from scratch really does. Name the
regime with every number.

## The balanced run (587.5, job 1015)
```
RACK=r02 WORKTREE=<this checkout> \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TE_REDUCE_AMAX=0 TORCHTITAN_TE_MHC=1 \
TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0 FSDP_PREFETCH_DEPTH=2 \
MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4 \
CONFIG=deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense_7x_balanced STEPS=20 TAG=best_balanced \
/mnt/dgxc/sbatch_rack.sh --parsable --nodes=8 --time=00:50:00 gb300/dsv4_64xgb300.slurm
```
No offload: in the balanced regime 7x beats 10x+offload (572.3) and 13x+offload
(549.7). `MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4` is the
EP row-copy launch geometry (16 rows per CTA instead of 4, 4 warps instead of
8): -17% per dispatch/combine copy in the 2-GPU microbench, 578.5 -> 587.5 on
the 8-node run, bitwise.

## The collapsed run (519.1, job 1006)
```
RACK=r02 WORKTREE=<this checkout> \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TE_REDUCE_AMAX=0 TORCHTITAN_TE_MHC=1 \
TORCHTITAN_BLOCK_INPUT_OFFLOAD=1 TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0 FSDP_PREFETCH_DEPTH=2 \
MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4 \
CONFIG=deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense_13x STEPS=20 TAG=best \
/mnt/dgxc/sbatch_rack.sh --parsable --nodes=8 --time=00:50:00 gb300/dsv4_64xgb300.slurm
```
8 nodes x 4 GB300 in ONE rack (inter-rack IB is slow); the config is 13 sequences
of 8192 per rank (224.6 GiB peak; 12x = 509.4-513.9, 7x without the offload = 506.1), EP=2, FSDP over the rest, FullAC, MinimalAsyncEP dispatcher
with 4 receive slots (2 slots corrupt expert weight grads: see ledger).

## What is in the recipe (in stacking order, each measured in the ledger)
cuDNN DSA sparse-attention forward+backward (cudnn frontend 1.29 in
`/mnt/dgxc/pydeps-cudnn`) -> TF32 for the fp32 matmuls -> fused output inverse
RoPE (Triton) -> FSDP `dense-never` reshard policy -> cuDNN fused indexer
top-k (`cudnn_indexer`) -> Transformer Engine 2.19 fp8 dense linears
(`torchtitan/quantization/te_linear.py`, `TE_DENSE_RECIPE=delayed`) -> copy-free
grouped output projection + in-place indexer RoPE (`attention.py`,
`compressor.py`) -> TE fused mHC kernels (`TORCHTITAN_TE_MHC=1`, residual
stream kept as [T, D, n]) -> 7x microbatch (500.3, job 880) -> `TE_REDUCE_AMAX=0`
(skips TE's per-module synchronous amax all-reduce, 520 per step; 506.1, job
940) -> `TORCHTITAN_BLOCK_INPUT_OFFLOAD=1` (FullAC block inputs to pinned host,
copy at block entry, one-layer-deep restore in backward; -68 GiB at 7x) which
makes 12x fit -> 12x microbatch (513.9, job 942) -> 13x on the merged branch
(bounded SwiGLU on by default; `TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0`, the
persistent workspace costs 16 GiB and -1.5% at 32 GPUs and is for 128) = 517.6
(job 996). 14x needs `OPT_STATE_OFFLOAD=1 OPT_STATE_OFFLOAD_LAYERWISE=1` too (503.8, loses).
`FSDP_PREFETCH_DEPTH=2` (explicit two-block-ahead all-gather prefetch) is
neutral-to-+0.3% and on in both runs.

## External pieces not in this repo (paths on the yqb01 cluster)
* venv: `/mnt/dgxc/venvs/dsv4n` (torch 2.15 nightly cu130, aarch64).
* cuDNN frontend with DSA + indexer wrappers: `/mnt/dgxc/pydeps-cudnn`.
* Transformer Engine 2.19: `/mnt/dgxc/pydeps-te` -- core from the aarch64
  `transformer-engine-cu13` wheel, torch extension built from sdist with
  `--no-build-isolation`, `CUDA_HOME=/mnt/dgxc/cuda13`, the torch wheel's
  cuDNN/NCCL headers on CPATH; needs `onnxscript`, `pydantic`, the
  `transformer-engine` meta package. pip `--target` drops the extension's
  `.so` when `transformer_engine/` exists: extract `wheel_lib/*.so` by hand.
* Python headers for nodes 1-8 (`/mnt/dgxc/pyinclude`, CPATH in the launcher).
* Optional: cuBLAS 13.8 preload (`CUBLAS_NEW=1`) only for TE grouped experts.

## Knobs that are OFF in the best run (measured negative or neutral)
`OPT_STATE_OFFLOAD`(+`_LAYERWISE`), `TE_EXPERTS`, `DUAL_MB` (other branch), DeepEP (`DEEPEP=1`), selective AC
configs, EP=1 configs, `parallelism.fp8_expert_all_gather`. The ledger
`gb300/FLASH_8K_RESULTS.md` (branch `dsv4_flash_64xgb300`) has every number.
