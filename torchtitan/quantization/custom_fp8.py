# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A torchao-free fp8 dense linear (tensorwise current scaling).

Why this exists: on GB300 the fp8 GEMM itself is 1.4-1.8x faster than bf16,
but torchao's Float8Linear spends 2-3.5x the bf16 time per linear on
per-call amax/scale/casts and transposed re-casts for the backward GEMMs
(probes 816/817), so the 8-node run was -18%. This module does what
Transformer Engine does in Megatron: one fused cast kernel emits both the
row-major and column-major e4m3 copies plus a per-tensor scale, the weight
cast is cached across forward, recompute and backward of a step, and the
three GEMMs are plain ``torch._scaled_mm`` calls (probe 821: 14.95 vs 19.97
ms per layer for the five converted dsv4 linears).

Numerics: per-tensor dynamic scaling of activations, weights and output
grads to e4m3 -- the coarsest fp8 recipe (~3.7e-2 rel error vs bf16 on the
probe). Block or row scaling would need a different GEMM entry point.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear

_E4M3_MAX = 448.0


@torch.compile(dynamic=False)
def _cast_both(x: torch.Tensor):
    """e4m3 copies of ``x`` in both layouts plus the fp32 per-tensor scale."""
    scale = (x.abs().amax().float() / _E4M3_MAX).clamp_min(1e-12)
    x8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x8, x8.t().contiguous(), scale


@torch.compile(dynamic=False)
def _cast_one(x: torch.Tensor):
    scale = (x.abs().amax().float() / _E4M3_MAX).clamp_min(1e-12)
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


class _Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, w8, w8t, sw):
        # x: [M, K] bf16; w8: [N, K] e4m3 row-major; w8t: [K, N] e4m3 row-major
        x8, x8t, sx = _cast_both(x)
        y = torch._scaled_mm(
            x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=x.dtype
        )
        ctx.save_for_backward(x8t, sx, w8t, sw)
        ctx.w_dtype = weight.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        x8t, sx, w8t, sw = ctx.saved_tensors
        dy8, dy8t, sdy = _cast_both(dy.contiguous())
        # dx[M, K] = dy[M, N] @ W[N, K]; B must be column-major -> w8t.t()
        dx = torch._scaled_mm(
            dy8, w8t.t(), scale_a=sdy, scale_b=sw, out_dtype=dy.dtype
        )
        # dW[N, K] = dy^T[N, M] @ X[M, K]; B column-major -> x8t.t()
        dw = torch._scaled_mm(
            dy8t, x8t.t(), scale_a=sdy, scale_b=sx, out_dtype=ctx.w_dtype
        )
        return dx, dw, None, None, None


class CustomFloat8Linear(Linear):
    """``Linear`` whose GEMMs run in tensorwise-scaled fp8 (see module doc)."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds CustomFloat8Linear."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._w8_key = None
        self._w8_cache = None

    def _weight_fp8(self, w: torch.Tensor):
        # FSDP2 hands us the unsharded (bf16) weight during compute; it is
        # rewritten once per step, which bumps ``_version``. Forward, the
        # FullAC recompute and backward all see the same key.
        key = (w.data_ptr(), w._version, w.shape)
        if self._w8_key != key:
            with torch.no_grad():
                w8, sw = _cast_one(w.detach())
                self._w8_cache = (w8, w8.t().contiguous(), sw)
            self._w8_key = key
        return self._w8_cache

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if not input.is_cuda or input.dtype != torch.bfloat16:
            return F.linear(input, w, self.bias)
        w8, w8t, sw = self._weight_fp8(w)
        x2 = input.reshape(-1, input.shape[-1])
        y = _Fp8LinearFn.apply(x2, w, w8, w8t, sw)
        y = y.view(*input.shape[:-1], w.shape[0])
        if self.bias is not None:
            y = y + self.bias
        return y


def convert_linear_config(linear_config: Linear.Config) -> CustomFloat8Linear.Config:
    """Rebuild a Linear.Config as a CustomFloat8Linear.Config with the same fields."""
    kwargs = {
        f.name: getattr(linear_config, f.name)
        for f in dataclasses.fields(linear_config)
        if f.init
    }
    return CustomFloat8Linear.Config(**kwargs)
