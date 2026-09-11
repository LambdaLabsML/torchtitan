"""The leading hypothesis for the 64-GPU MXFP8 NaN.

GptOssGroupedExperts.forward builds

    offsets_E = cumsum(num_tokens_per_expert_E)          # real tokens only
    tail_slack = x_RD.shape[0] - offsets_E[-1]           # > 0

so the activation buffer it hands the grouped mm has MORE rows than the last
offset covers. Every probe so far constructed `offs` that covered exactly all
rows, so none of them reproduced this.

It matters because mxfp8_quantize_2d_1x32_cutedsl takes `offs` and is
offset-aware, while triton_to_mxfp8_dim0 -- what the patch substitutes -- has no
`offs` argument. It casts every row, and the swizzle then sizes itself from the
full row count rather than from offs[-1]. If the grouped GEMM expects a scale
layout sized by the groups, the scales it reads are displaced.

Compares patched vs CuTeDSL at K=2944 (where both are legal) WITH tail slack
present, so a disagreement isolates the bug to slack handling rather than to K.
"""
import os
import torch
import torchao.prototype.moe_training.mxfp8_grouped_mm as M

dev = "cuda"
torch.manual_seed(0)
print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}")
orig = M._quantize_2d_1x32_blocked


def force(which):
    def f(x, scaling_mode, offs, block_size=32):
        if which == "triton":
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


E = int(os.environ.get("PROBE_E", "16"))
K = int(os.environ.get("PROBE_K", "2944"))      # %128==0 so CuTeDSL is legal too
for slack in (0, 512):
    torch.manual_seed(1)
    sizes = ((torch.randint(3000, 4000, (E,), device=dev) + 127) // 128) * 128
    offs = torch.cumsum(sizes, 0).to(torch.int32)
    rows = int(offs[-1]) + slack               # slack rows beyond the last offset
    A0 = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
    B = (torch.randn(E, K, K, device=dev, dtype=torch.bfloat16) / K ** 0.5)
    res = {}
    for which in ("cutedsl", "triton"):
        M._quantize_2d_1x32_blocked = force(which)
        A = A0.detach().clone().requires_grad_(True)
        B_t = B.transpose(-2, -1).detach().clone().requires_grad_(True)
        o = M._to_mxfp8_then_scaled_grouped_mm(A, B_t, offs)
        o.backward(torch.ones_like(o))
        torch.cuda.synchronize()
        res[which] = (o.detach(), A.grad.detach(), B_t.grad.detach())
    M._quantize_2d_1x32_blocked = orig
    valid = int(offs[-1])
    print(f"\n  tail_slack={slack:4d}  rows={rows}  offs[-1]={valid}")
    print(f"    out  (rows covered by offs) rel {rel(res['triton'][0][:valid], res['cutedsl'][0][:valid]):.8f}")
    print(f"    d(A) (rows covered by offs) rel {rel(res['triton'][1][:valid], res['cutedsl'][1][:valid]):.8f}")
    print(f"    d(B)                        rel {rel(res['triton'][2], res['cutedsl'][2]):.8f}")
    for nm, t in (("out", res['triton'][0]), ("d(A)", res['triton'][1]), ("d(B)", res['triton'][2])):
        if not torch.isfinite(t).all():
            print(f"    !! patched {nm} has non-finite values")
print("\ndone")
