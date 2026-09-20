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

so the kernel gets ``out_with_sink`` together with the raw ``lse_no_sink`` --
its ``lse`` argument is defined as the KV-only LSE, sink excluded.

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


def compact_topk_indices(idx_TK: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each row's valid indices ascending into a prefix, with its length.

    Two things the kernel wants that our selection does not provide:

    * **Compaction.** It reads the first ``topk_length`` entries of a row, so
      the -1 holes in our selection (causally masked compressed positions, and
      the short window at the start of a sequence) must be squeezed out.
    * **Ascending order.** The compressed half of the selection comes back in
      top-k score order, not key order. Attention is permutation-invariant over
      the key set, so sorting is semantically free, but it measurably improves
      the kernel's accuracy: on an fp64 reference, dq went from 2.0e-2 with
      shuffled indices to 9.8e-3 with sorted ones.

    Sorting with the invalid entries mapped to ``INT32_MAX`` does both at once:
    valid indices come out ascending at the front, -1 padding at the back.
    """
    sentinel = torch.iinfo(torch.int32).max
    key = torch.where(idx_TK >= 0, idx_TK.int(), sentinel)
    ordered = key.sort(dim=-1).values
    lengths = (ordered != sentinel).sum(dim=-1).int()
    ordered = torch.where(ordered == sentinel, -1, ordered)
    return ordered.int().contiguous(), lengths.contiguous()


class _CudnnDsaBackward(torch.autograd.Function):
    """cuDNN backward, with the forward on either flex or cuDNN.

    The forward runs under ``no_grad`` precisely because its graph would be
    dead weight: this Function supplies the gradients itself.

    With ``flex_fwd=None`` the forward is cuDNN's ``sparse_attention_forward``,
    which takes ``attn_sink`` directly and returns the sink-included output
    alongside the KV-only LSE -- exactly the pair the backward wants, so no
    rescale is applied on top. That also retires the flex block mask, whose
    construction is pure overhead once no flex kernel consumes it.
    """

    @staticmethod
    def forward(
        ctx,
        q_THD,
        kv_ND,
        attn_sink_H,
        topk_idxs_TK,
        topk_length_T,
        softmax_scale,
        flex_fwd,
        out_rope_cache=None,
        out_rope_rd=0,
    ):
        from torchtitan.models.common.attention import apply_attention_sink_rescale

        if flex_fwd is None:
            with torch.no_grad():
                res = _dsa_namespace().sparse_attention_forward_wrapper(
                    q_THD.contiguous(),
                    kv_ND.contiguous(),
                    topk_idxs_TK,
                    attn_sink=attn_sink_H.float().contiguous(),
                    topk_length=topk_length_T,
                    softmax_scale=softmax_scale,
                )
                # out already carries the sink; lse excludes it.
                out, lse = res["out"].to(q_THD.dtype), res["lse"]
                out = _apply_out_rope_(out, out_rope_cache, out_rope_rd)
            ctx.save_for_backward(
                q_THD, kv_ND, out, lse, attn_sink_H, topk_idxs_TK, topk_length_T,
                *(() if out_rope_cache is None else (out_rope_cache,)),
            )
            ctx.softmax_scale = softmax_scale
            ctx.out_rope_rd = out_rope_rd if out_rope_cache is not None else 0
            return out

        with torch.no_grad():
            out_no_sink, lse_no_sink = flex_fwd(q_THD, kv_ND)
            # Exactly the rescale the unfused path applies -- the same
            # (compiled) function, so the two forwards agree bitwise and any
            # difference in a comparison is attributable to the backward.
            out = apply_attention_sink_rescale(out_no_sink, lse_no_sink, attn_sink_H)
            # The kernel wants the KV-only LSE, EXCLUDING the sink (see
            # sparse_attention_backward/_interface_sm100.py:200): it folds
            # attn_sink in itself. Passing logaddexp(lse, sink) double-counts
            # the sink and corrupts dq/d_sink in proportion to the sink's share
            # of the softmax mass -- worst on short rows, where it dominates.
            lse = lse_no_sink
            out = _apply_out_rope_(out, out_rope_cache, out_rope_rd)
        ctx.save_for_backward(
            q_THD, kv_ND, out, lse, attn_sink_H, topk_idxs_TK, topk_length_T,
            *(() if out_rope_cache is None else (out_rope_cache,)),
        )
        ctx.softmax_scale = softmax_scale
        ctx.out_rope_rd = out_rope_rd if out_rope_cache is not None else 0
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, kv, out, lse, attn_sink, topk_idxs, topk_length, *rest = ctx.saved_tensors
        if ctx.out_rope_rd:
            # The saved `out` is the ROTATED output the block consumed (and
            # whose consumers have already run their backward, so mutating it
            # now is safe). Un-rotate it in place to recover the raw attention
            # output the kernel needs, and push d_out through the rotation's
            # VJP -- a forward rotation of its tail -- out of place, since
            # d_out may alias other gradient buffers. Megatron-LM #7036/#5526.
            (cache,) = rest
            from .fused_rope import rotate_tail, rotate_tail_

            rotate_tail_(out, cache, rd=ctx.out_rope_rd, inverse=False)
            d_out = rotate_tail(d_out.contiguous(), cache, rd=ctx.out_rope_rd, inverse=False)
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
            topk_length=topk_length,
            # Bitwise-reproducible gradients: this model's MoE router flips on
            # near-ties, and a non-deterministic backward would reintroduce the
            # forward-vs-recompute mismatch that activation checkpointing
            # rejects.
            deterministic=True,
        )
        dq = result["dq"].to(q.dtype)
        dkv = result["dkv"].to(kv.dtype)
        d_sink = result["d_sink"].to(attn_sink.dtype)
        return dq, dkv, d_sink, None, None, None, None, None, None


def _apply_out_rope_(out, cache_ri, rd):
    """Inverse-rotate the rope tail of the Function-owned output in place.

    Fusing the output inverse RoPE here (Megatron-LM #7036) means no separate
    rotated [T, H, D] activation is retained for backward, and the rotation
    touches only the rd-wide tail instead of copying the whole output -- the
    standalone op was 2.33% + 0.90% of GPU time in the 8-node profile.
    """
    if cache_ri is None:
        return out
    from .fused_rope import rotate_tail_

    return rotate_tail_(out.contiguous(), cache_ri, rd=rd, inverse=True)


def fused_dsa_attention(
    q_THD: torch.Tensor,
    kv_ND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    softmax_scale: float,
    flex_fwd=None,
    out_rope=None,
) -> torch.Tensor:
    """Sink-scaled DSA output whose backward is cuDNN's fused kernel.

    Args:
        q_THD: queries ``[T, H, D]`` (already folded across any batch).
        kv_ND: the concatenated KV stream ``[N, D]``, shared by all heads.
        attn_sink_H: per-head sink logit ``[H]``.
        selected_indices: ``[T, K]`` int64/int32 global KV positions, -1 padded.
        softmax_scale: the QK scale (the sink logit is NOT scaled).
        flex_fwd: callable ``(q, kv) -> (out_no_sink, lse_no_sink)``, or
            None to run cuDNN's sparse-attention forward instead.
        out_rope: optional ``(cache_ri, rd)`` -- apply the output inverse RoPE
            in place inside the Function (see ``_apply_out_rope_``).
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
    compact_idx, topk_length = compact_topk_indices(selected_indices)
    cache, rd = (None, 0) if out_rope is None else out_rope
    return _CudnnDsaBackward.apply(
        q_THD,
        kv_ND,
        attn_sink_H,
        compact_idx,
        topk_length,
        softmax_scale,
        flex_fwd,
        cache,
        rd,
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
