# DeepSeek-V4-flash, 8k seq, 32x GB300: 733.1 TFLOP/s balanced routing (job 1104) / 519.1 collapsed (job 1006)

Branch `dsv4_te_mhc` = the full stack. Everything below default-off is a knob.

## Two regimes, two numbers
20-step runs from random init collapse the router onto ~6 experts by step 2;
the EP-rank token skew then costs 8% of the step in barrier waits. With forced
load-balanced routing (`config.debug.moe_force_load_balance`, the `_balanced`
config variants; Megatron's `--moe-router-force-load-balancing`) the barrier
vanishes and the same kernels run +14% faster. Balanced is the proxy for
trained routing; collapsed is what step 20 from scratch really does. Name the
regime with every number.

## The balanced run (733.1, job 1104)
```
RACK=r02 WORKTREE=<this checkout> \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TE_REDUCE_AMAX=0 TORCHTITAN_TE_MHC=1 \
TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0 FSDP_PREFETCH_DEPTH=2 \
MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4 \
MINIMAL_ASYNC_EP_POOL_FACTOR=1.25 MOE_PACKED_EXPERT_WEIGHTS=1 FSDP_DIRECT_GATHER=1 TE_MHC_BF16_GRAD_PHI=1 \
TORCHTITAN_DSA_DETERMINISTIC=0 FSDP_DIRECT_REDUCE_SCATTER=1 \
CONFIG=deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense_9x_balanced STEPS=20 TAG=best_balanced \
/mnt/dgxc/sbatch_rack.sh --parsable --nodes=8 --time=00:50:00 gb300/dsv4_64xgb300.slurm
```
9 sequences per rank with no offload (243.2 GiB). `FSDP_DIRECT_REDUCE_SCATTER=1`
feeds the packed expert gradient's own flat view to the reduce-scatter instead
of a staged `chunk_cat` copy (723.6 -> 733.1). Needs branch commit
`b8672e291` or later: the direct gather's first version waited on the compute
stream and serialized every prefetched expert gather (685.7 -> 723.6 fixed). The last three knobs are the
2026-09-22 additions: `MINIMAL_ASYNC_EP_POOL_FACTOR=1.25` bounds MinimalAsyncEP's
receive pool and capacity-padded routed activation at 1.25x the expected
receive instead of ep_size x (balanced regime only: imbalanced routing beyond
it overflows; 589.1 at 7x, +0.3%); `MOE_PACKED_EXPERT_WEIGHTS=1` +
`FSDP_DIRECT_GATHER=1` pack w1/w2/w3 into one [3E, F*D] parameter and gather
it straight into its unsharded storage, skipping FSDP2's copy-out (591.9 at 7x,
+0.5%, -17 GiB); the freed memory fits the 8th sequence (+2.6%); `TE_MHC_BF16_GRAD_PHI=1`
(607.2 -> 614.2, +1.2%) is a one-line patch in the side-installed TE copy (see
below), not in this repo. `TORCHTITAN_DSA_DETERMINISTIC=0` runs the cuDNN
sparse-attention backward with fp32-atomic dK/dV accumulation instead of the
deterministic per-CTA shards + fold: 614.2 -> 677.9 at 8x (+10.4%) and 20 GiB
less scratch, which fits 9x (685.7). Gradients then vary run to run in
summation order only, like any flash-attention backward; set it to 1 for
bitwise reproducibility at -10%. In the
balanced regime the offload variants lose: 7x no offload beat 10x+offload
(572.3) and 13x+offload (549.7). `MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4` is the
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
* Transformer Engine 2.19: `/mnt/dgxc/pydeps-te` -- **locally patched**:
  `transformer_engine/pytorch/triton/mhc.py`, `mHCProjectionOp.backward`, casts
  `grad_H` down to x's dtype instead of x up to fp32 when `TE_MHC_BF16_GRAD_PHI=1`
  (default off = stock TE). Re-apply after any TE reinstall. Core from the aarch64
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
