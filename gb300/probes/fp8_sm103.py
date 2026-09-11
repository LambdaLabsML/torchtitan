"""Which fp8 matmul paths actually run on sm_103 (GB300)?

The 120B fp8 arms die with CUTLASS's "Arch conditional MMA instruction used
without targeting appropriate compute capability". That is the sm_103 problem
already documented for torchao's MXFP8 extension, but here it is inside torch's
own kernels, so it needs isolating per op rather than per library: bf16
_grouped_mm demonstrably works on this hardware (it is 49.7% of the reference
step), so the question is which *scaled* entry points do.

Shapes are GPT-OSS-120B's: dim = 2880, one expert group of 4096 rows (bs=16
gives 16*8192*4/128 = 4096 tokens per expert).
"""
import torch

D = 2880
M = 4096
E = 8
dev = "cuda"
cap = torch.cuda.get_device_capability()
print(f"device {torch.cuda.get_device_name()}  capability {cap}  torch {torch.__version__}")


def run(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  {name:46s} OK      out={tuple(out.shape)} {out.dtype}")
    except Exception as e:
        msg = str(e).split("\n")[0][:90]
        print(f"  {name:46s} FAIL    {type(e).__name__}: {msg}")


def rowwise(x, dim):
    amax = x.abs().amax(dim=dim, keepdim=True).float().clamp(min=1e-12)
    scale = amax / 448.0
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale.reciprocal()


print("\n--- 2D scaled_mm (the Float8Linear path) ---")
A = torch.randn(M, D, device=dev, dtype=torch.bfloat16)
B = torch.randn(D, D, device=dev, dtype=torch.bfloat16)
Aq, As = rowwise(A, 1)
Bq, Bs = rowwise(B.t().contiguous(), 1)
run("torch._scaled_mm rowwise", lambda: torch._scaled_mm(
    Aq, Bq.t(), scale_a=As, scale_b=Bs.t(), out_dtype=torch.bfloat16))

print("\n--- bf16 grouped mm (known-good control) ---")
Ag = torch.randn(M * E, D, device=dev, dtype=torch.bfloat16)
Bg = torch.randn(E, D, D, device=dev, dtype=torch.bfloat16).transpose(-2, -1).contiguous().transpose(-2, -1)
offs = torch.arange(1, E + 1, device=dev, dtype=torch.int32) * M
run("torch._grouped_mm bf16", lambda: torch._grouped_mm(Ag, Bg, offs=offs, out_dtype=torch.bfloat16))

print("\n--- fp8 scaled grouped mm (the expert-GEMM path) ---")
Agq, Ags = rowwise(Ag, 1)
Bgq = torch.empty(E, D, D, device=dev, dtype=torch.float8_e4m3fn)
Bgs = torch.empty(E, 1, D, device=dev, dtype=torch.float32)
for e in range(E):
    q, s = rowwise(Bg[e].t().contiguous(), 1)
    Bgq[e] = q.t()
    Bgs[e] = s.reshape(1, D)
run("torch._scaled_grouped_mm rowwise", lambda: torch._scaled_grouped_mm(
    Agq, Bgq, Ags.squeeze(-1).float(), Bgs.squeeze(1).float(),
    offs=offs, out_dtype=torch.bfloat16))

print("\n--- torchao's wrapper, exactly what the converter installs ---")
try:
    from torchao.prototype.moe_training.fp8_grouped_mm import (
        _to_fp8_rowwise_then_scaled_grouped_mm,
    )
    run("torchao _to_fp8_rowwise_then_scaled_grouped_mm",
        lambda: _to_fp8_rowwise_then_scaled_grouped_mm(
            Ag, Bg, offs, torch.bfloat16, torch.float8_e4m3fn, False))
except Exception as e:
    print(f"  import failed: {e}")
print("\ndone")
