# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import os

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module
from torchtitan.tools.leaf_compile import leaf_compile


# ---------------------------------------------------------------------------
# The HC math as plain tensor functions, compiled individually.
#
# Why here and not through the trainer's ``compile.enable``: compiling the
# whole transformer block never fused anything on the flash recipe -- Dynamo
# hit a graph break inside an SPMD typecheck context manager in the attention
# path and left the block body eager (four attempts, profile-verified). These
# functions contain nothing but tensor ops, so each is its own small graph:
# the fp32 upcast, RMS statistic, mixing linear, sinkhorn and the
# broadcast-multiply-sum become a handful of fused kernels instead of ~20
# separate passes over a [T, hc_mult*D] fp32 tensor per call.
#
# ``fullgraph=True`` makes a graph break an error rather than a silent
# fallback to eager. ``dynamic=False``: shapes are fixed per rank.
# HC_COMPILE=0 / TORCHTITAN_LEAF_COMPILE=0 run the same functions eagerly.
# ---------------------------------------------------------------------------


def _sinkhorn_split(mixes, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = (
        torch.sigmoid(
            pre * hc_scale[0]
            + hc_base[:hc_mult].view(*([1] * (pre.ndim - 1)), hc_mult)
        )
        + eps
    )
    post = 2 * torch.sigmoid(
        post * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult].view(*([1] * (post.ndim - 1)), hc_mult)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(
        *([1] * (comb.ndim - 2)), hc_mult, hc_mult
    )

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _hc_pre_math(x, hc_fn, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps, norm_eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre, post, comb = _sinkhorn_split(
        mixes.float(), hc_scale.float(), hc_base.float(),
        hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype), post, comb


def _hc_post_math(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def _hc_head_math(x, hc_fn, hc_scale, hc_base, *, norm_eps, eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype)


# --- Transformer Engine fused mHC (TORCHTITAN_TE_MHC=1) -----------------------
# TE 2.19 ships the DeepSeek mHC Triton kernels (projection+RMS, scale,
# log-space sinkhorn, aggregate, expand+combine) with autograd. They want the
# stream axis innermost: x is (s, b, C, n). With the knob on, the decoder keeps
# its residual stream as [T, D, hc_mult] (see model.py) and HcPre/HcPost call
# TE; HcHead permutes back once per step. Column order of the projection
# weight differs (TE flattens (C, n), torchtitan flattens (n, D)), so hc_fn is
# re-laid-out per call (a 1.5 MB copy). Semantics: TE's post step is the
# paper's residual mixing sum_j H_res[i, j] x_j; torchtitan's eager HcPost
# reduces to (sum_j comb[i, j]) x_i (no mixing) -- see the ledger.
_TE_MHC = os.environ.get("TORCHTITAN_TE_MHC", "0") == "1"


def te_mhc_enabled() -> bool:
    return _TE_MHC


def _te_mhc():
    import transformer_engine.pytorch  # noqa: F401
    from transformer_engine.pytorch.triton import mhc as te_mhc

    return te_mhc


_hc_pre = leaf_compile(_hc_pre_math, group="hc")
_hc_post = leaf_compile(_hc_post_math, group="hc")
_hc_head = leaf_compile(_hc_head_math, group="hc")





class HcSplitSinkhorn(Module):
    """Convert HC mix logits into pre, post, and combination weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        sinkhorn_iters: int = 20
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.eps = config.eps

    def forward(self, mixes, hc_scale, hc_base):
        """Split and normalize HC mixing logits.

        Args:
            mixes: HC logits of shape ``[B, L, (2 + hc_mult) * hc_mult]``.
            hc_scale: Scale tensor of shape ``[3]``.
            hc_base: Bias tensor of shape ``[(2 + hc_mult) * hc_mult]``.

        Returns:
            ``pre`` and ``post`` tensors of shape ``[T, hc_mult]`` and
            ``comb`` of shape ``[T, hc_mult, hc_mult]``.
        """
        return _sinkhorn_split(
            mixes, hc_scale, hc_base,
            hc_mult=self.hc_mult, sinkhorn_iters=self.sinkhorn_iters, eps=self.eps,
        )


# TORCHTITAN_MHC_GRAD_CHAIN=1: fuse the residual stream's gradient accumulation.
# The residual x feeds HcPre (TE: projection + aggregate, two internal grad_x
# contributions) and HcPost (as the residual). Autograd sums all three with two
# full [T, D, n] passes per half-block (~1.75% of the step). TE's kernels can
# instead read-modify-write a provided buffer. This wrapper makes x's only
# consumer the wrapper itself: it returns a pass-through view of x that HcPost
# uses as the residual, so HcPost's residual gradient arrives here as the
# pass-through's gradient and becomes the accumulate buffer for the inner TE
# backward. bf16 buffer, fp32 math in-kernel (patched side-installed TE).
_TE_GRAD_CHAIN = os.environ.get("TORCHTITAN_MHC_GRAD_CHAIN", "0") == "1"


class _GradHolder:
    __slots__ = ("tensor",)

    def __init__(self):
        self.tensor = None


class _HcPreChain(torch.autograd.Function):
    """Forward: TE's mHC pre-mix without a graph, plus a pass-through view of x.
    Only the inputs are saved (through save_for_backward, so FullAC's saved-
    tensor hooks free them in the forward and restore them at recompute --
    keeping the inner graph in ctx pinned every block's residual and OOMed,
    job 1124). Backward: rebuild TE's small forward graph, set the holder to
    the pass-through gradient (HcPost's residual gradient) and replay TE's
    backward, whose kernels accumulate into it. One extra mHC forward per
    half-block (~0.5 ms) against two fewer full [T, D, n] passes."""

    @staticmethod
    def forward(ctx, x, phi, hc_scale, hc_base_row):
        te = _te_mhc()
        t, d, n = x.shape
        out, h_post, h_res = te.mhc_generate_mix_and_aggregate(
            x.view(t, 1, d, n), phi, hc_scale, hc_base_row
        )
        ctx.save_for_backward(x, phi, hc_scale, hc_base_row)
        return out, h_post, h_res, x.view_as(x)

    @staticmethod
    def backward(ctx, g_out, g_post, g_res, g_xpass):
        te = _te_mhc()
        x, phi, hc_scale, hc_base_row = ctx.saved_tensors
        t, d, n = x.shape
        holder = _GradHolder()
        with torch.enable_grad():
            x_d = x.detach().requires_grad_(True)
            phi_d = phi.detach().requires_grad_(ctx.needs_input_grad[1])
            scale_d = hc_scale.detach().requires_grad_(ctx.needs_input_grad[2])
            base_d = hc_base_row.detach().requires_grad_(ctx.needs_input_grad[3])
            out, h_post, h_res = te.mhc_generate_mix_and_aggregate(
                x_d.view(t, 1, d, n), phi_d, scale_d, base_d, fused_grad_x_acc_buffer=holder
            )
            if g_xpass is None:
                g_xpass = torch.zeros_like(x)
            elif not g_xpass.is_contiguous():
                g_xpass = g_xpass.contiguous()
            holder.tensor = g_xpass  # HcPost's residual gradient; TE accumulates into it
            outs, grads = [], []
            for o, g in ((out, g_out), (h_post, g_post), (h_res, g_res)):
                if g is not None:
                    outs.append(o)
                    grads.append(g.reshape(o.shape))
            leaves = [t_ for t_ in (x_d, phi_d, scale_d, base_d) if t_.requires_grad]
            torch.autograd.backward(outs, grads, inputs=leaves)
        return (
            holder.tensor,
            phi_d.grad if ctx.needs_input_grad[1] else None,
            scale_d.grad if ctx.needs_input_grad[2] else None,
            base_d.grad if ctx.needs_input_grad[3] else None,
        )


class HcPre(Module):
    """Reduce HC branches before attention or FFN computation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_mult = config.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_mult = config.hc_mult
        self.norm_eps = config.norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.sinkhorn = HcSplitSinkhorn.Config(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            eps=config.eps,
        ).build()

    def forward(self, x):
        """Project multi-branch hidden states into a single branch.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Tuple ``(y, post, comb)`` where ``y`` has shape ``[T, D]`` and
            ``post``/``comb`` are consumed by ``HcPost``.
        """
        if _TE_MHC and x.is_cuda:
            te = _te_mhc()
            t, d, n = x.shape  # TE layout [T, D, n]
            phi = self.hc_fn.view(-1, n, d).transpose(1, 2).reshape(-1, n * d)
            if _TE_GRAD_CHAIN:
                out, h_post, h_res, x_pass = _HcPreChain.apply(x, phi, self.hc_scale, self.hc_base.view(1, -1))
                return out.view(t, d), h_post.view(t, n), h_res.view(t, n, n), x_pass
            out, h_post, h_res = te.mhc_generate_mix_and_aggregate(
                x.view(t, 1, d, n), phi, self.hc_scale, self.hc_base.view(1, -1)
            )
            return out.view(t, d), h_post.view(t, n), h_res.view(t, n, n)
        return _hc_pre(
            x, self.hc_fn, self.hc_scale, self.hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn.sinkhorn_iters,
            eps=self.sinkhorn.eps,
            norm_eps=self.norm_eps,
        )


class HcPost(Module):
    """Expand a single-branch output back to HC branches with residual mixing."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x, residual, post, comb):
        """Apply HC post mixing.

        Args:
            x: Single-branch output of shape ``[T, D]``.
            residual: Residual branches of shape ``[T, hc_mult, D]``.
            post: Post weights of shape ``[T, hc_mult]``.
            comb: Branch combination weights of shape ``[T, hc_mult, hc_mult]``.

        Returns:
            Hidden states of shape ``[T, hc_mult, D]``.
        """
        if _TE_MHC and x.is_cuda:
            te = _te_mhc()
            t, d, n = residual.shape  # TE layout [T, D, n]
            out = te.mhc_fused_expand_combine(
                x.view(t, 1, d), None, post.view(t, 1, n),
                residual.view(t, 1, d, n), comb.view(t, 1, n, n), n,
            )
            return out.view(t, d, n)
        return _hc_post(x, residual, post, comb)


class HcHead(Module):
    """Merge final HC branches before the output norm and LM head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        norm_eps: float = 1e-6
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_dim = config.hc_mult * config.dim
        self.norm_eps = config.norm_eps
        self.eps = config.eps
        self.hc_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x):
        """Merge HC branches.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Hidden states of shape ``[T, D]``.
        """
        if _TE_MHC and x.is_cuda:
            x = x.transpose(1, 2).contiguous()  # [T, D, n] -> [T, n, D], once per step
        return _hc_head(
            x, self.hc_fn, self.hc_scale, self.hc_base,
            norm_eps=self.norm_eps, eps=self.eps,
        )
