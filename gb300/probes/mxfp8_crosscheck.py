"""Is the Triton activation cast faithful to the CuTeDSL one it replaces?

The K=2880 probe showed the patched path runs and lands ~3.8% from bf16, which
is the expected mxfp8 quantization error -- but it cannot distinguish "correct"
from "consistently wrong", because there is no CuTeDSL result to compare against
at K=2880 (that is the whole reason for the patch).

So compare the two casts at K where BOTH are legal (K % 128 == 0) and check they
agree. If they do, the only thing the patch changes at K=2880 is which kernel
produces an identical-in-intent quantization.
"""
import torch
import torchao.prototype.moe_training.mxfp8_grouped_mm as M

dev = "cuda"
torch.manual_seed(0)
print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}")

import os
E = int(os.environ.get("PROBE_E", "128"))
M_PER = int(os.environ.get("PROBE_M_PER", "1024"))
orig = M._quantize_2d_1x32_blocked


def force(which):
    def f(x, scaling_mode, offs, block_size=32):
        if which == "triton":
            # Exactly what the patch does, so this compares the shipped path.
            from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
            from torchao.prototype.moe_training.kernels.mxfp8.quant import (
                triton_mx_block_rearrange_2d_M_groups,
            )
            q, s = triton_to_mxfp8_dim0(x.contiguous(), block_size, scaling_mode)
            return q, triton_mx_block_rearrange_2d_M_groups(s, offs)
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mxfp8_quantize_2d_1x32_cutedsl,
        )
        return mxfp8_quantize_2d_1x32_cutedsl(x, scaling_mode=scaling_mode, offs=offs)
    return f


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


for K in (2944, 2816):
    assert K % 128 == 0
    A0 = torch.randn(M_PER * E, K, device=dev, dtype=torch.bfloat16)
    B_nk = torch.randn(E, K, K, device=dev, dtype=torch.bfloat16) / (K ** 0.5)
    offs = torch.arange(1, E + 1, device=dev, dtype=torch.int32) * M_PER
    outs, grads = {}, {}
    for which in ("cutedsl", "triton"):
        M._quantize_2d_1x32_blocked = force(which)
        A = A0.detach().clone().requires_grad_(True)
        B_t = B_nk.transpose(-2, -1).detach().clone().requires_grad_(True)
        o = M._to_mxfp8_then_scaled_grouped_mm(A, B_t, offs)
        g = torch.ones_like(o)
        o.backward(g)
        torch.cuda.synchronize()
        outs[which], grads[which] = o.detach(), A.grad.detach()
    M._quantize_2d_1x32_blocked = orig
    print(f"  groups={E} K={K:5d}  out rel {rel(outs['triton'], outs['cutedsl']):.8f}   "
          f"d(A) rel {rel(grads['triton'], grads['cutedsl']):.8f}")
print("done")
