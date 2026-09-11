"""Does an fp8 GROUPED mm exist that runs on sm_103, in either venv?

v1 established: torch._scaled_mm (2D) works on GB300, bf16 _grouped_mm works,
and torchao's fp8 grouped path dies with CUTLASS's "Arch conditional MMA
instruction used without targeting appropriate compute capability".

v1's own direct _scaled_grouped_mm call was invalid (mat2 was row-major), so it
proved nothing about torch's operator. This fixes the layout -- mat2 must be
column-major, stride (K*N, 1, K) -- and is meant to be run under both venvs:
torch 2.14 (the GPT-OSS sweep) and the 2.15 nightly.
"""
import torch

D, M, E = 2880, 4096, 8
dev = "cuda"
print(f"torch {torch.__version__}  capability {torch.cuda.get_device_capability()}")


def rowwise(x):
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    s = amax / 448.0
    return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s.squeeze(-1).float()


A = torch.randn(M * E, D, device=dev, dtype=torch.bfloat16)
Aq, As = rowwise(A)

# B_t logical shape (E, K, N) and column-major: build (E, N, K) contiguous then transpose.
B_nk = torch.randn(E, D, D, device=dev, dtype=torch.bfloat16)
Bq_nk = torch.empty(E, D, D, device=dev, dtype=torch.float8_e4m3fn)
Bs = torch.empty(E, D, device=dev, dtype=torch.float32)
for e in range(E):
    q, s = rowwise(B_nk[e])
    Bq_nk[e], Bs[e] = q, s
Bq_t = Bq_nk.transpose(-2, -1)           # (E, K, N), stride (K*N, 1, K)
print(f"mat2 shape {tuple(Bq_t.shape)} stride {Bq_t.stride()}  col_major={Bq_t.stride(-2)==1}")

offs = (torch.arange(1, E + 1, device=dev, dtype=torch.int32) * M)


def run(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  {name:44s} OK    out={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    except Exception as e:
        print(f"  {name:44s} FAIL  {type(e).__name__}: {str(e).splitlines()[0][:80]}")


run("torch._scaled_grouped_mm (correct layout)", lambda: torch._scaled_grouped_mm(
    Aq, Bq_t, As, Bs, offs=offs, out_dtype=torch.bfloat16))

try:
    from torchao.prototype.moe_training.fp8_grouped_mm import (
        _to_fp8_rowwise_then_scaled_grouped_mm as f,
    )
    Bbf_t = B_nk.transpose(-2, -1)
    run("torchao fp8 grouped wrapper", lambda: f(
        A, Bbf_t, offs, torch.bfloat16, torch.float8_e4m3fn, False))
except ImportError as e:
    print(f"  torchao wrapper unavailable: {e}")
print("done")
