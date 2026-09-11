# MXFP8 for GPT-OSS on GB300: unblocking what the sweep recorded as impossible

`gb300/TUNING_RESULTS.md` lists MXFP8 as blocked for both GPT-OSS models:

> torchao publishes CUDA kernels for **x86_64 only**; the aarch64 wheel is
> `py3-none-any` with no compiled `.so`, so SM100 kernels are absent. No aarch64
> build on cu128/cu129/cu130 or either nightly index.

That is accurate as a statement about published wheels and wrong as a statement
about the hardware. The kernel source supports sm_103 fine. The DeepSeek-V4 work
on this same cluster built it from source and ran MXFP8 for +5.7%, documented in
`MXFP8_SETUP.md` on the dsv4 branches. This note records only what is different
for GPT-OSS.

## What was different

The dsv4 build went into `/mnt/dgxc/venvs/dsv4n`, which runs **torch
2.15.0.dev20260907+cu130**. The GPT-OSS sweep runs `/mnt/dgxc/venvs/torchtitan`
on **torch 2.14.0+cu130**, and that venv cannot simply be swapped for the other
one:

- `dsv4n` has no `flash-attn-4`, and GPT-OSS attention uses the `varlen`
  backend, which is flash-attn.
- torchtitan's upstream `main` (what `dsv4n` is pinned to) hard-fails on GB300
  in `init_distributed` via `enable_fp32_matmul_emulation_with_bf16x9()`, which
  needs the nightly. The GPT-OSS fork sits at an older upstream base and does
  not make that call, so it is torch 2.14-compatible and should stay there.

torchao's `_C_mxfp8` extension links against libtorch, so the `dsv4n` build
cannot be borrowed across a torch minor version. It was rebuilt against torch
2.14 instead.

## The build

`/mnt/dgxc/ao-src` is already at torchao `3efa9ed` with both dsv4 patches
applied (`setup.py` sm_103 gencodes, and the non-power-of-two `num_groups` fix
in `prototype/moe_training/kernels/mxfp8/quant.py`). Both are needed here too:
sm_103 for the same reason, and the group fix because GPT-OSS-120B has **128
experts** -- a power of two, but at `expert_parallel_degree` 8 or 16 the local
count is 16 or 8, and the padding path is exercised regardless.

nvcc did not need reinstalling: the `dsv4n` venv already has the correctly
pinned 13.0.88 toolchain with the unversioned-library symlinks, and it works as
a `CUDA_HOME` for a build driven by a different interpreter.

```bash
cd /mnt/dgxc/ao-src
export CUDA_HOME=/mnt/dgxc/venvs/dsv4n/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
USE_CPP=1 TORCHAO_MXFP8_ONLY=1 MAX_JOBS=48 TORCH_CUDA_ARCH_LIST="10.0;10.3" \
  /mnt/dgxc/venvs/torchtitan/bin/pip wheel --no-build-isolation --no-deps \
  -w /mnt/dgxc/overlays/wheels .
```

Produces `torchao-0.19.0+git3efa9ed-cp310-abi3-linux_aarch64.whl` containing
`torchao/_C_mxfp8.cpython-312-aarch64-linux-gnu.so` (3.6 MB). A wheel without
that `.so` is the failure mode to watch for -- it installs cleanly, imports
cleanly, and then every MXFP8 op falls back or raises at first use.

## Why it is an overlay, not an install

```bash
pip install --no-deps --target /mnt/dgxc/overlays/torchao-tt <wheel>
```

and `MXFP8=1` in `gb300/run_1k.slurm` prepends that directory to `PYTHONPATH`.

The `torchtitan` venv is shared by every GPT-OSS job, and installing into it
would upgrade torchao 0.18.0 -> 0.19.0 underneath any queued run. An overlay
makes the change opt-in per job and reversible by deleting a directory, and it
keeps the bf16 arms of this comparison running against exactly the torchao the
earlier sweep measured.

## Guard rails

A silent fallback here does not error -- it measures emulation or plain bf16 and
reports a plausible number. Two checks stand in the way:

1. `run_1k.slurm` refuses to start when `MXFP8=1` unless
   `_SM100_KERNELS_AVAILABLE` is true, printing the torchao version and path it
   resolved.
2. `gpt_oss_debugmodel_1k_mxfp8` is a 4-GPU, 5-step gate on one node. MXFP8
   needed six separate build/kernel fixes to run on sm_103 at all and every one
   of them failed at model-build or first-forward time, so the gate costs three
   minutes and can save a 16-node slot.

The gate converts the same fqn surface as the 120B configs. That detail is
load-bearing: the dsv4 work had a gate whose fqn list was narrower than the real
config's, it passed, and the real run then died on a Linear the gate never
converted.
