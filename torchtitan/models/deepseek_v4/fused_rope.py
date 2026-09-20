# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modelled on NVIDIA Megatron-LM's fused_mla_yarn_rope_apply.py (Copyright (c)
# 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0), adapted to this model's
# adjacent-pair complex layout.

"""Tail-only RoPE rotation kernels for fusing the output inverse RoPE
(Megatron-LM #7036).

DSv4 applies RoPE to the last ``rd`` of each 512-wide head and, because the
attention's K doubles as V, the attention output carries the rotation and
must be un-rotated. Done as its own op, Inductor copies the whole
``[T, H, 512]`` output to rotate a 64-wide tail -- 2.33% + 0.90% (bwd) of GPU
time in the 8-node profile. These kernels touch only the tail (1/8 of the
bytes) and can work in place on a tensor the caller owns, which is what lets
the rotation live inside the sparse-attention autograd Function: no separate
rotated activation is retained for backward.

Layout matches ``_rotate_tail`` in attention.py exactly: adjacent pairs
``(x[2i], x[2i+1])`` rotated by ``(c, d) = cache_ri[t, 0, i]`` (the
``view_as_real`` of the complex cache), ``inverse`` negating ``d``. fp32
math on bf16 storage, as the compiled path does.

No autograd here on purpose: ``rotate_tail_`` and ``rotate_tail`` are raw
kernels for a Function that handles its own backward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_H": bh}, num_warps=nw) for bh in (4, 8, 16, 32) for nw in (2, 4)],
    key=["RD", "H"],
    # The autotuner benchmarks every config on the REAL arguments. In place,
    # that rotates the tail once per trial and leaves garbage behind on the
    # first call for a given (RD, H) -- the whole cuDNN-forward branch came
    # out uncorrelated with its reference while the second branch was exact.
    # restore_value snapshots the mutated buffer between trials (Megatron's
    # kernel carries the same guard, restore_value=["DO"]).
    restore_value=["out_ptr"],
)
@triton.jit
def _rotate_tail_kernel(
    x_ptr, out_ptr, cache_ptr, T, H: tl.constexpr, D: tl.constexpr, RD: tl.constexpr,
    s_x_t, s_x_h, s_o_t, s_o_h, s_c_t,
    INVERSE: tl.constexpr, COPY_HEAD: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Grid (T, cdiv(H, BLOCK_H)). Rotates x[t, h, D-RD:] pairs into out; with
    COPY_HEAD also copies x[t, h, :D-RD] so out is a complete out-of-place
    result (the in-place caller passes out == x and skips the copy)."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    half = tl.arange(0, RD // 2)
    c = tl.load(cache_ptr + t * s_c_t + half * 2).to(tl.float32)
    d = tl.load(cache_ptr + t * s_c_t + half * 2 + 1).to(tl.float32)
    if INVERSE:
        d = -d
    base_in = x_ptr + t * s_x_t + offs_h[:, None] * s_x_h + (D - RD)
    base_out = out_ptr + t * s_o_t + offs_h[:, None] * s_o_h + (D - RD)
    a_off = half[None, :] * 2
    m2 = mask_h[:, None] & (half[None, :] >= 0)
    a = tl.load(base_in + a_off, mask=m2, other=0.0).to(tl.float32)
    b = tl.load(base_in + a_off + 1, mask=m2, other=0.0).to(tl.float32)
    ra = a * c[None, :] - b * d[None, :]
    rb = a * d[None, :] + b * c[None, :]
    tl.store(base_out + a_off, ra.to(out_ptr.dtype.element_ty), mask=m2)
    tl.store(base_out + a_off + 1, rb.to(out_ptr.dtype.element_ty), mask=m2)
    if COPY_HEAD:
        for c0 in range(0, D - RD, 64):
            offs = c0 + tl.arange(0, 64)
            mk = mask_h[:, None] & (offs[None, :] < D - RD)
            v = tl.load(x_ptr + t * s_x_t + offs_h[:, None] * s_x_h + offs[None, :], mask=mk, other=0.0)
            tl.store(out_ptr + t * s_o_t + offs_h[:, None] * s_o_h + offs[None, :], v, mask=mk)


def _launch(x, out, cache_ri, rd, inverse, copy_head):
    T, H, D = x.shape
    assert x.stride(-1) == 1 and out.stride(-1) == 1, "head dim must be contiguous"
    assert rd % 2 == 0 and (rd & (rd - 1)) == 0, "rd must be a power of two"
    # cache_ri: [T, 1, rd//2, 2] fp32 view_as_real of the complex cache
    cache = cache_ri.reshape(T, rd)
    assert cache.stride(-1) == 1
    grid = lambda META: (T, triton.cdiv(H, META["BLOCK_H"]))
    _rotate_tail_kernel[grid](
        x, out, cache, T, H, D, rd,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1), cache.stride(0),
        INVERSE=inverse, COPY_HEAD=copy_head,
    )
    return out


def rotate_tail_(x: torch.Tensor, cache_ri: torch.Tensor, *, rd: int, inverse: bool) -> torch.Tensor:
    """In place: rotate the last ``rd`` features of ``x`` [T, H, D]."""
    return _launch(x, x, cache_ri, rd, inverse, copy_head=False)


def rotate_tail(x: torch.Tensor, cache_ri: torch.Tensor, *, rd: int, inverse: bool) -> torch.Tensor:
    """Out of place, one pass: a fresh tensor with the head copied and the
    tail rotated. For gradients that may alias other buffers."""
    out = torch.empty_like(x)
    return _launch(x, out, cache_ri, rd, inverse, copy_head=True)
