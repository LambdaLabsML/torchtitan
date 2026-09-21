# DeepSeek-V4-flash, 8k seq, 32x GB300: 500.3 TFLOP/s (job 880)

Branch `dsv4_te_mhc` = the full stack. Everything below default-off is a knob.

## The run
```
RACK=r02 WORKTREE=<this checkout> \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TORCHTITAN_TE_MHC=1 \
CONFIG=deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense_7x STEPS=20 TAG=best \
/mnt/dgxc/sbatch_rack.sh --parsable --nodes=8 --time=00:50:00 gb300/dsv4_64xgb300.slurm
```
8 nodes x 4 GB300 in ONE rack (inter-rack IB is slow); the config is 7 sequences
of 8192 per rank, EP=2, FSDP over the rest, FullAC, MinimalAsyncEP dispatcher
with 4 receive slots (2 slots corrupt expert weight grads: see ledger).

## What is in the recipe (in stacking order, each measured in the ledger)
cuDNN DSA sparse-attention forward+backward (cudnn frontend 1.29 in
`/mnt/dgxc/pydeps-cudnn`) -> TF32 for the fp32 matmuls -> fused output inverse
RoPE (Triton) -> FSDP `dense-never` reshard policy -> cuDNN fused indexer
top-k (`cudnn_indexer`) -> Transformer Engine 2.19 fp8 dense linears
(`torchtitan/quantization/te_linear.py`, `TE_DENSE_RECIPE=delayed`) -> copy-free
grouped output projection + in-place indexer RoPE (`attention.py`,
`compressor.py`) -> TE fused mHC kernels (`TORCHTITAN_TE_MHC=1`, residual
stream kept as [T, D, n]) -> 7x microbatch.

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
`OPT_STATE_OFFLOAD`(+`_LAYERWISE`), `TORCHTITAN_BLOCK_INPUT_OFFLOAD`,
`TE_EXPERTS`, `DUAL_MB` (other branch), DeepEP (`DEEPEP=1`), selective AC
configs, EP=1 configs, `parallelism.fp8_expert_all_gather`. The ledger
`gb300/FLASH_8K_RESULTS.md` (branch `dsv4_flash_64xgb300`) has every number.
