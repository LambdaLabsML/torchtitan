# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""SwiGLU over the valid prefix of a capacity-padded routed activation.

Under an async EP dispatcher the experts' input is the full receive-capacity
buffer (EP size x max tokens per rank x top-k rows: 1,179,648 rows at 12x/8k
on DSv4 flash) and the grouped GEMMs compute only the rows below the last
group offset, which the routing fills to ~50% on average. The activation in
between runs over every row: on the 8-node 12x profile that is ~0.4 s of SiLU
forward/backward per step on rows no GEMM reads (2.5% of the step).

Slicing needs the row count on the host, i.e. a device sync per MoE layer,
which the dispatcher exists to avoid. These kernels take the count as a
device scalar (the last group offset) and exit for row blocks past it, so the
launch stays static and only the valid rows are touched. Rows past the count
are left uninitialised in the outputs, exactly as the grouped GEMMs leave
them; nothing downstream reads them (the combine copies valid rows only).

fp32 math on bf16 storage with one rounding at the end, like the Inductor
fusion this replaces (`_swiglu` in activation.py).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr, up_ptr, h_ptr, nvalid_ptr, F, stride_r,
    BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr,
):
    """Grid (cdiv(R, BLOCK_R), cdiv(F, BLOCK_F)). h = silu(gate) * up on rows < *nvalid."""
    pid_r = tl.program_id(0)
    n = tl.load(nvalid_ptr)
    r0 = pid_r * BLOCK_R
    if r0 >= n:
        return
    rows = r0 + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (rows[:, None] < n) & (cols[None, :] < F)
    offs = rows[:, None] * stride_r + cols[None, :]
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h = g * tl.sigmoid(g) * u
    tl.store(h_ptr + offs, h.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    dh_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, nvalid_ptr, F, stride_r,
    BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr,
):
    """dgate = dh * up * dsilu(gate), dup = dh * silu(gate) on rows < *nvalid."""
    pid_r = tl.program_id(0)
    n = tl.load(nvalid_ptr)
    r0 = pid_r * BLOCK_R
    if r0 >= n:
        return
    rows = r0 + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (rows[:, None] < n) & (cols[None, :] < F)
    offs = rows[:, None] * stride_r + cols[None, :]
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))
    tl.store(dgate_ptr + offs, (dh * u * dsilu).to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offs, (dh * silu).to(dup_ptr.dtype.element_ty), mask=mask)


_BLOCK_R = 8
_BLOCK_F = 1024
# TORCHTITAN_SWIGLU_BOUNDED=zero allocates the outputs zeroed (a memset per
# tensor) so rows past the count are 0 rather than uninitialised. Job 963
# (skip mode, uninitialised tail) went non-finite at step 4.
_ZERO_TAIL = os.environ.get("TORCHTITAN_SWIGLU_BOUNDED", "1") == "zero"
_alloc = torch.zeros_like if _ZERO_TAIL else torch.empty_like


def _check(gate: torch.Tensor, up: torch.Tensor, num_valid_rows: torch.Tensor) -> None:
    if gate.shape != up.shape or gate.dim() != 2:
        raise ValueError(f"swiglu_bounded: gate {tuple(gate.shape)} and up {tuple(up.shape)} must be equal 2D shapes")
    if num_valid_rows.numel() != 1 or num_valid_rows.device != gate.device:
        raise ValueError("swiglu_bounded: num_valid_rows must be a one-element tensor on the activation's device")


class _SwiGLUBounded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, num_valid_rows: torch.Tensor) -> torch.Tensor:
        gate = gate.contiguous()
        up = up.contiguous()
        rows, cols = gate.shape
        h = _alloc(gate)
        grid = (triton.cdiv(rows, _BLOCK_R), triton.cdiv(cols, _BLOCK_F))
        _swiglu_fwd_kernel[grid](
            gate, up, h, num_valid_rows, cols, gate.stride(0),
            BLOCK_R=_BLOCK_R, BLOCK_F=_BLOCK_F,
        )
        ctx.save_for_backward(gate, up, num_valid_rows)
        return h

    @staticmethod
    def backward(ctx, dh: torch.Tensor):
        gate, up, num_valid_rows = ctx.saved_tensors
        dh = dh.contiguous()
        rows, cols = gate.shape
        dgate = _alloc(gate)
        dup = _alloc(up)
        grid = (triton.cdiv(rows, _BLOCK_R), triton.cdiv(cols, _BLOCK_F))
        _swiglu_bwd_kernel[grid](
            dh, gate, up, dgate, dup, num_valid_rows, cols, gate.stride(0),
            BLOCK_R=_BLOCK_R, BLOCK_F=_BLOCK_F,
        )
        return dgate, dup, None


def swiglu_bounded(gate: torch.Tensor, up: torch.Tensor, num_valid_rows: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` on rows ``< num_valid_rows`` (a device int scalar);
    rows at or past it are left uninitialised."""
    _check(gate, up, num_valid_rows)
    return _SwiGLUBounded.apply(gate, up, num_valid_rows)
