# DeepSeek-V4-flash, 8k seq, 32x GB300: 517.6 TFLOP/s (job 996)

Branch `dsv4_te_mhc` = the full stack. Everything below default-off is a knob.

## The run
```
RACK=r02 WORKTREE=<this checkout> \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TE_REDUCE_AMAX=0 TORCHTITAN_TE_MHC=1 \
TORCHTITAN_BLOCK_INPUT_OFFLOAD=1 TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0 \
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
(job 996). 14x needs `OPT_STATE_OFFLOAD=1 OPT_STATE_OFFLOAD_LAYERWISE=1` too.

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
