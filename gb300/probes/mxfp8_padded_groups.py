"""Does the Triton cast survive the zero-padded token groups the model feeds it?

Job 286 produced a normal step-1 loss (12.729) and an immediately NaN
grad_norm, so the forward is fine and the backward is not. The suspect is
padding: TorchAOTokenDispatcher pads each expert's token group up to
pad_multiple=128, so `A` arrives with zero rows in every group's tail. The
kernel this patch replaced, mxfp8_quantize_2d_1x32_cutedsl, takes `offs` and is
offset-aware. triton_to_mxfp8_dim0 has no `offs` argument at all -- it casts
every row, including the padding, and an all-zero 32-element block has amax 0,
whose e8m0 scale is exactly where an inf/NaN can be minted.

Forward would not notice: padded rows only produce output rows that get
discarded. wgrad would, because dW = A^T @ dY contracts OVER the token
dimension, so a NaN in a padded row lands directly in the weight gradient.

This reproduces that shape -- unequal group sizes, zero tails -- and reports
where the non-finites first appear.
"""
import torch
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    triton_mx_block_rearrange_2d_M_groups,
)
import torchao.prototype.moe_training.mxfp8_grouped_mm as M

dev = "cuda"
torch.manual_seed(0)
print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}")

# 1. the narrow question: what does the Triton cast do with an all-zero row?
x = torch.randn(256, 2880, device=dev, dtype=torch.bfloat16)
x[128:, :] = 0.0                                   # second half is padding
q, s = triton_to_mxfp8_dim0(x.contiguous(), 32, "rceil")
sv = s.view(torch.uint8) if s.dtype != torch.uint8 else s
print(f"\nzero-row cast: qdata finite={bool(torch.isfinite(q.float()).all())}  "
      f"scale e8m0 max={int(sv.max())} (255 == NaN encoding)  "
      f"zero-row scales unique={sorted(set(sv[128:].flatten().tolist()))[:4]}")

# 2. the real shape: unequal groups, each zero-padded up to a multiple of 128
import os
# 128 groups AND unequal padded sizes -- the combination none of the earlier
# probes covered. 284 had 128 equal groups (no padding), 288 had 16 unequal ones.
E = int(os.environ.get("PROBE_E", "128"))
K = 2880
torch.manual_seed(1)
real = torch.randint(3000, 4000, (E,), device=dev)  # unequal -> every group pads
padded = ((real + 127) // 128) * 128
offs = torch.cumsum(padded, 0).to(torch.int32)
total = int(offs[-1])
A = torch.zeros(total, K, device=dev, dtype=torch.bfloat16)
start = 0
for e in range(E):
    n = int(real[e])
    A[start:start + n] = torch.randn(n, K, device=dev, dtype=torch.bfloat16)
    start = int(offs[e])
print(f"\ngroups={E}  padded rows = {total - int(real.sum())} of {total}")

A.requires_grad_(True)
N = int(os.environ.get("PROBE_N", "2880"))   # 5760 reproduces w13 (gate+up fused)
B_t = (torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) / K ** 0.5).transpose(
    -2, -1).detach().clone().requires_grad_(True)
print(f"  B_t=({E},{K},{N})  N%128={N % 128}")

out = M._to_mxfp8_then_scaled_grouped_mm(A, B_t, offs)
print(f"  forward  finite={bool(torch.isfinite(out).all())}")
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
print(f"  d(A)     finite={bool(torch.isfinite(A.grad).all())}")
print(f"  d(B)     finite={bool(torch.isfinite(B_t.grad).all())}")
if not torch.isfinite(B_t.grad).all():
    bad = (~torch.isfinite(B_t.grad)).sum(dim=(1, 2))
    print(f"  d(B) non-finite per expert: {bad.tolist()}")
print("done")
