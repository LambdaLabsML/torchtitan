# DeepSeek V4 Pro / 64x GB300 -- config optimization results

Branch `deepseek_v4_config_opt`, from the
`lambda_64xgb300_dsv4_pro_baseline` result (31.48 TFLOP/s/GPU, 1.26 % MFU,
peak 244.94 of 276.50 GiB). All runs 25 steps.

## Never-reshard: does not fit, by ~3 GiB

Job 78, `deepseek_v4_pro_64xgb300_nevershard`
(`fsdp_reshard_after_forward="never"`), **exit 1, OOM at step 2**.

| | baseline | never-reshard |
|---|---|---|
| step 1 peak | 188.05 GiB (68.0 %) | **212.90 - 213.85 GiB (77.0 - 77.3 %)** |
| step 2 (AdamW moments allocate) | 233.76 GiB | **OOM** |
| step 1 TFLOP/s | 1.47 | 4.77 |

At the failure:

```
CUDA out of memory. Tried to allocate 3.18 GiB. GPU 3 has a total capacity of
276.50 GiB of which 2.80 GiB is free. this process has 273.68 GiB memory in
use. Of the allocated memory 246.14 GiB is allocated by PyTorch
```

It died wanting **0.38 GiB more than existed**.

Two things this corrects in the pre-run analysis:

- The predicted cost of never-reshard, ~43 GB/GPU, was too pessimistic. The
  measured cost is **~+25 GiB** at step 1 (188.05 -> ~213). FSDP does not hold
  the full ~21.6 B non-expert parameter set unsharded, or holds it more cheaply
  than the estimate assumed.
- The prediction that it would OOM was right, but for the wrong reason. It is
  not that never-reshard alone exceeds the budget -- it is that ~25 GiB of extra
  residency leaves too little for the step-2 AdamW moment allocation.

So never-reshard is viable on this model with roughly **3 GiB more headroom per
GPU**. Nothing in this config can return that: the microbatch is already at the
4096-token minimum (one sequence at `max_context_length`). It would need a
different EP/FSDP split leaving fewer non-expert parameters resident, or more
GPUs. Given the baseline is attention-kernel bound at 1.26 % MFU rather than
all-gather bound, this is a low-value direction to keep pushing.

## MXFP8: not possible on this cluster

Both MXFP8 paths require torchao's compiled SM100 CUDA kernels, and neither is
installable here.

| path | failure on GB300 (capability 10.3) |
|---|---|
| `MXFP8Linear` | `NotImplementedError: Could not run 'torchao::mxfp8_quantize' with arguments from the 'CUDA' backend` |
| grouped expert GEMM | `AssertionError: SM100 kernels not available. Please use torchao CUDA 12.8+ build on SM100/100a device(s)` |

The chain:

1. `MXFP8LinearConverter` demands a torchao **nightly** -- the 32x32 swizzled
   cast kernels are in no release up to v0.18.0.
2. There is **no aarch64 torchao wheel**, on PyPI or the nightly cu130 index.
   PyPI ships `manylinux_2_24_x86_64` plus a `py3-none-any` pure-Python wheel;
   the nightly index ships x86_64 only. GB300 is Grace/aarch64.
3. The `py3-none-any` wheel and a `USE_CPP=0` source build of 0.19.0 both
   produce **zero compiled `.so` files**, so the custom ops never register.
4. Building the extension needs **nvcc, which is not installed** -- this cluster
   has no CUDA toolkit.

`KernelPreference.EMULATED` exists but is documented as "without efficient
kernels", so it would be slower than bf16 and the resulting number would
mislead. Not run for that reason.

`nvidia-cutlass-dsl==4.7.1` and `apache-tvm-ffi==0.1.13.post3` did install, and
`torch._scaled_grouped_mm` is present in PyTorch, so only the torchao CUDA
extension is missing. Enabling MXFP8 means installing a CUDA toolkit (or
`nvidia-cuda-nvcc-cu13`) and building torchao for aarch64 with `USE_CPP=1` --
an environment project, not a config change.

`deepseek_v4_pro_64xgb300_mxfp8` and `..._mxfp8_nevershard` are written,
correct, and left ready for that. **They build fine and fail at runtime**, so do
not queue them until a torchao aarch64 CUDA build exists.
