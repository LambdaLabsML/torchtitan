"""Does MXFP8 grouped mm run, and is it correct, at GPT-OSS's K=2880?

MXFP8 grouped GEMMs are known to work on sm_103 -- the DeepSeek-V4 sweep on
this cluster ran them for +5.7% -- but only at K % 128 == 0 (dsv4's dim is
7168). GPT-OSS contracts over 2880, which the CuTeDSL activation cast rejects.
This tests the patched path that routes those casts through Triton instead.

Checks both that it runs and that it is right: forward AND backward against a
bf16 reference. A forward-only check would not be evidence about training, and
loss curves cannot serve as the equivalence test on this cluster because the
seed is not fixed by default.
"""
import torch

torch.manual_seed(0)
dev = "cuda"
print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}")

from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm as MX,
)

E, K, N = 8, 2880, 2880          # gpt-oss-120b: dim 2880 -> dim 2880 (w2)
M_PER = 4096                     # bs=16 -> 16*8192*4/128 = 4096 tokens/expert
print(f"shapes: A=({M_PER*E},{K})  B_t=({E},{K},{N})   K%128={K % 128}  K%32={K % 32}")

A = torch.randn(M_PER * E, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
B_nk = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) / (K ** 0.5)
B_t = B_nk.transpose(-2, -1).detach().clone().requires_grad_(True)
offs = (torch.arange(1, E + 1, device=dev, dtype=torch.int32) * M_PER)

# bf16 reference through the same op family, so only precision differs.
A_ref = A.detach().clone().requires_grad_(True)
B_ref = B_t.detach().clone().requires_grad_(True)

try:
    out = MX(A, B_t, offs)
    torch.cuda.synchronize()
    print(f"  forward   OK   out={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
except Exception as e:
    print(f"  forward   FAIL {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    raise SystemExit(1)

ref = torch._grouped_mm(A_ref, B_ref, offs=offs, out_dtype=torch.bfloat16)
g = torch.randn_like(ref)

out.backward(g)
ref.backward(g)
torch.cuda.synchronize()
print("  backward  OK")


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


print("\nrelative error vs bf16 grouped mm (mxfp8 has ~2^-8 mantissa, so ~1e-2 is expected):")
print(f"  output   {rel(out, ref):.5f}")
print(f"  d(A)     {rel(A.grad, A_ref.grad):.5f}")
print(f"  d(B)     {rel(B_t.grad, B_ref.grad):.5f}")
allf = all(bool(torch.isfinite(t).all()) for t in (out, A.grad, B_t.grad))
print(f"  all finite: {allf}")
print("done")
