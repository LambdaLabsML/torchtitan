# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""cuDNN's DSA backward kernel in place of the FlexAttention backward.

Why. On 64x GB300 the FlexAttention backward is the single largest cost in
every profile of DeepSeek-V4 flash -- about 40 % of kernel time, roughly four
times the forward -- because it is a generic Triton template running at
head_dim=512, register- and shared-memory-bound, with its tiles capped by the
sparse block size. Retuning those tiles was worth 11.6 %, which is the most a
generic template gives. cuDNN ships a CuTe-DSL kernel written for exactly this
shape (DeepSeek Sparse Attention: one shared KV head, an explicit per-query
index list, a per-head sink) and it is what Megatron-LM uses for this model.

What this does NOT change: the forward. The flex forward stays, so the block
mask, the selection and the numerics of the forward are untouched; only the
gradient computation is swapped. That keeps the change small and lets the two
backwards be compared directly on identical forward state.

The sink. Upstream computes attention with the sink logit inside the softmax.
This model applies it outside, as ``out * sigmoid(lse - sink)`` -- algebraically
the same thing (see ``apply_attention_sink_rescale``), which means the
kernel's inputs can be reconstructed exactly:

    out_with_sink = out_no_sink * sigmoid(lse_no_sink - sink)
    lse_with_sink = logaddexp(lse_no_sink, sink)

Layout. The kernel is flat and unbatched: ``q [T, H, D]``, ``kv [N, D]`` (one
KV head shared by all query heads, K and V being the same tensor here),
``topk_idxs [T, K]`` of int32 global indices with -1 padding. A batched
``[B, L, ...]`` stream is folded by flattening the KV to ``[B*N, D]`` and
offsetting each sequence's indices by ``b * N``, which keeps every query
inside its own sequence.

Install: ``pip install nvidia-cudnn-frontend[cutedsl]``. Requires SM90+; on
SM100+ the query-head count must divide 128 and TopK must be a multiple of 64.
"""

from __future__ import annotations

import torch

_DSA = None
_TOPK_ALIGN_SM100 = 64
_TOPK_ALIGN_SM90 = 128
_HEAD_ALIGN_SM100 = 128
_HEAD_ALIGN_SM90 = 64


def _dsa_namespace():
    global _DSA
    if _DSA is None:
        try:
            from cudnn import DSA
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "cuDNN's DSA namespace is required for the fused DSA backward. "
                "Install with `pip install nvidia-cudnn-frontend[cutedsl]`."
            ) from e
        _DSA = DSA
    return _DSA


def cudnn_dsa_available() -> bool:
    """True if the kernel can run on this device."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        _dsa_namespace()
    except ImportError:
        return False
    return True


def _alignments() -> tuple[int, int]:
    """(topk alignment, head alignment) for the current device."""
    major = torch.cuda.get_device_capability()[0]
    if major >= 10:
        return _TOPK_ALIGN_SM100, _HEAD_ALIGN_SM100
    return _TOPK_ALIGN_SM90, _HEAD_ALIGN_SM90


class _CudnnDsaBackward(torch.autograd.Function):
    """Flex forward, cuDNN backward.

    The forward runs under ``no_grad`` precisely because its graph would be
    dead weight: this Function supplies the gradients itself.
    """

    @staticmethod
    def forward(ctx, q_THD, kv_ND, attn_sink_H, topk_idxs_TK, softmax_scale, flex_fwd):
        with torch.no_grad():
            out_no_sink, lse_no_sink = flex_fwd(q_THD, kv_ND)
            # Reconstruct what a sink-inside-the-softmax kernel would produce.
            sink = attn_sink_H.to(lse_no_sink.dtype).view(
                *([1] * (lse_no_sink.ndim - 1)), -1
            )
            out = out_no_sink * torch.sigmoid(lse_no_sink - sink).unsqueeze(-1).to(
                out_no_sink.dtype
            )
            lse = torch.logaddexp(lse_no_sink, sink.expand_as(lse_no_sink))
        ctx.save_for_backward(q_THD, kv_ND, out, lse, attn_sink_H, topk_idxs_TK)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, kv, out, lse, attn_sink, topk_idxs = ctx.saved_tensors
        dsa = _dsa_namespace()
        result = dsa.sparse_attention_backward_wrapper(
            q.contiguous(),
            kv.contiguous(),
            out.contiguous(),
            d_out.contiguous(),
            lse.float().contiguous(),
            attn_sink.float().contiguous(),
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=None,
            # Bitwise-reproducible gradients: this model's MoE router flips on
            # near-ties, and a non-deterministic backward would reintroduce the
            # forward-vs-recompute mismatch that activation checkpointing
            # rejects.
            deterministic=True,
        )
        dq = result["dq"].to(q.dtype)
        dkv = result["dkv"].to(kv.dtype)
        d_sink = result["d_sink"].to(attn_sink.dtype)
        return dq, dkv, d_sink, None, None, None


def fused_dsa_attention(
    q_THD: torch.Tensor,
    kv_ND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    softmax_scale: float,
    flex_fwd,
) -> torch.Tensor:
    """Sink-scaled DSA output whose backward is cuDNN's fused kernel.

    Args:
        q_THD: queries ``[T, H, D]`` (already folded across any batch).
        kv_ND: the concatenated KV stream ``[N, D]``, shared by all heads.
        attn_sink_H: per-head sink logit ``[H]``.
        selected_indices: ``[T, K]`` int64/int32 global KV positions, -1 padded.
        softmax_scale: the QK scale (the sink logit is NOT scaled).
        flex_fwd: callable ``(q, kv) -> (out_no_sink, lse_no_sink)``.
    """
    topk_align, head_align = _alignments()
    n_heads = q_THD.size(1)
    if head_align % n_heads != 0:
        raise ValueError(
            f"cuDNN DSA needs the query-head count to divide {head_align}, got {n_heads}"
        )
    k = selected_indices.size(-1)
    if k % topk_align:
        pad = topk_align - k % topk_align
        selected_indices = torch.nn.functional.pad(selected_indices, (0, pad), value=-1)
    return _CudnnDsaBackward.apply(
        q_THD,
        kv_ND,
        attn_sink_H,
        selected_indices.to(torch.int32).contiguous(),
        softmax_scale,
        flex_fwd,
    )


def flatten_batched_indices(
    selected_indices: torch.Tensor, kv_len: int
) -> torch.Tensor:
    """``[B, L, K]`` per-sequence indices -> ``[B*L, K]`` global ones.

    Each sequence's KV stream occupies its own ``kv_len`` slice of the
    flattened stream, so sequence ``b``'s indices shift by ``b * kv_len``.
    Padding (-1) stays -1.
    """
    if selected_indices.ndim == 2:
        return selected_indices
    bsz = selected_indices.size(0)
    offsets = (
        torch.arange(bsz, device=selected_indices.device).view(bsz, 1, 1) * kv_len
    )
    shifted = torch.where(selected_indices >= 0, selected_indices + offsets, -1)
    return shifted.flatten(0, 1)
