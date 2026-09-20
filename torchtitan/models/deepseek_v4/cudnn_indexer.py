# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""cuDNN fused DSA indexer scoring + top-k selection.

torchtitan's ``Indexer.select`` materialises the dense ``[T, T/ratio]`` index
score with an einsum, masks it, and takes a stable descending sort per row
(``aten::sort`` is ~2.3% of kernel time at 8K, the score GEMM and its
elementwise tail sit on top). cuDNN frontend >= 1.24 ships a CuTe-DSL kernel
that computes ``sum_h w_h * relu(q_h . k)`` and selects the top-k per query
without ever writing the dense score (Megatron tracks its integration as
NVIDIA/Megatron-LM#5992). The indexer receives no gradient in torchtitan
(its auxiliary loss is dropped), so this is a pure forward replacement.

Contract kept with the eager path: returns compressed-key ids in
``[0, T/ratio)`` per query, shape ``[T, k]`` (folded) or ``[B, L, k]``
(batched), ``k = min(topk, L // ratio)``. Entries that are not valid for a
query (padding for early rows) come back as ``-1``; the caller already maps
anything outside the causal limit to ``-1``, and also has to treat ``-1`` as
invalid (the eager path never produced negatives).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
_wrapper = None
_warned = False


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        from cudnn.deepseek_sparse_attention.indexer_forward.api import (
            indexer_forward_top_k_wrapper,
        )

        _wrapper = indexer_forward_top_k_wrapper
    return _wrapper


@torch.no_grad()
def cudnn_indexer_select(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    *,
    seqlen: int,
    ratio: int,
    topk: int,
) -> torch.Tensor:
    batched = idx_q.ndim == 4
    n_heads = idx_q.shape[-2]
    if n_heads not in (32, 64):
        # The bf16 cuDNN indexer kernel is built for 32 or 64 index heads
        # (DSv4 flash has 64); smaller debug models fall back to the eager path.
        global _warned
        if not _warned:
            logger.warning(
                "cudnn_indexer: %d index heads unsupported (needs 32/64); using eager select",
                n_heads,
            )
            _warned = True
        from .compressor import Indexer

        return Indexer.select(idx_q, idx_k, idx_w, seqlen=seqlen, ratio=ratio, topk=topk)
    if batched:
        q, k, w = idx_q, idx_k, idx_w
    else:
        q, k, w = idx_q.unsqueeze(0), idx_k.unsqueeze(0), idx_w.unsqueeze(0)
    k_eff = min(topk, seqlen // ratio)
    out = _get_wrapper()(
        q.contiguous().to(torch.bfloat16),
        k.unsqueeze(2).contiguous().to(torch.bfloat16),  # (B, S_k, H_kv=1, D)
        w.contiguous().to(torch.bfloat16),
        top_k=k_eff,
        ratio=ratio,
        sm_scale=1.0,  # idx_w already carries softmax_scale * H^-0.5
        return_softmax=False,
        deterministic=True,  # ties at the k-th boundary -> lowest key id
    )
    indices = out["indices"].to(torch.int64)  # (B, S_q, k)
    return indices if batched else indices.squeeze(0)
