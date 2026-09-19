# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class Compressor(Module):
    """Compress local hidden states into lower-rate KV tokens.

    The compressor scores each token inside a compression group, forms a
    weighted sum, normalizes the result, and applies RoPE to the rope slice.
    For ``compress_ratio == 4`` it also includes the previous group's value as
    the overlapping candidate used by CSA.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4
        seq_len: int = 0
        """Length of one packed sequence. 0 means "the whole token stream is one
        sequence" (the original behaviour). When set and the stream is longer,
        the overlap shift below stays inside each sequence instead of feeding
        sequence n's first compressed group from sequence n-1's tail."""

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.seq_len = cfg.seq_len
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()

        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features))

    def _overlap_transform(self, tensor, value=0):
        """Append previous-token overlap candidates along the ratio dimension.

        Args:
            tensor: Grouped tensor of shape ``[B, L // R, R, D]``.
            value: Fill value for the first previous group.

        Returns:
            Tensor of shape ``[B, L // R, 2 * R, D]``.
        """
        d = self.head_dim
        groups_per_seq = self._groups_per_seq(tensor.size(0))
        if groups_per_seq is None:
            prev = torch.cat(
                [
                    torch.full_like(tensor[:1, :, :d], value),
                    tensor[:-1, :, :d],
                ],
                dim=0,
            )
        else:
            # Shift by one group WITHIN each packed sequence: the first group of
            # every sequence gets ``value``, not the tail of the previous one.
            g = tensor.unflatten(0, (-1, groups_per_seq))[..., :d]
            prev = torch.cat(
                [torch.full_like(g[:, :1], value), g[:, :-1]], dim=1
            ).flatten(0, 1)
        curr = tensor[:, :, d:]
        return torch.cat([prev, curr], dim=1)

    def _groups_per_seq(self, n_groups: int) -> int | None:
        """Compressed groups per packed sequence, or None for a single sequence."""
        if not self.seq_len:
            return None
        per_seq = self.seq_len // self.compress_ratio
        if per_seq <= 0 or n_groups % per_seq != 0:
            raise ValueError(
                f"compressed groups ({n_groups}) must be a multiple of "
                f"seq_len // compress_ratio ({per_seq})"
            )
        return None if n_groups == per_seq else per_seq

    def forward(self, x, positions):
        """Compress hidden states into compressed KV states.

        Args:
            x: Hidden states of shape ``[B, L, D_model]``.
            positions: Position IDs of shape ``[B, L]``.

        Returns:
            Compressed KV tensor of shape ``[B, L // compress_ratio, head_dim]``.
        """
        seqlen = x.size(0)
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x)
            score = self.wgate(x)
        if seqlen % ratio != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
            )
        comp_positions = positions[::ratio] if positions is not None else None
        kv = kv.unflatten(0, (-1, ratio))
        score = score.unflatten(0, (-1, ratio)) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(kv_rope.unsqueeze(1), positions=comp_positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)
        return kv


class Indexer(Module):
    """Produce low-dimensional query/key features for CSA top-k selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: "Compressor.Config"
        num_index_heads: int = 64
        index_head_dim: int = 128
        rope_head_dim: int = 64

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x):
        dim = x.size(-1)
        hadamard_mat = _hadamard(dim, dtype=x.dtype, device=x.device)
        if isinstance(x, DTensor):
            hadamard_mat = DTensor.from_local(
                hadamard_mat,
                x.device_mesh,
                [Replicate()] * x.device_mesh.ndim,
                run_check=False,
            )
        return F.linear(x, hadamard_mat) * (dim**-0.5)

    def forward(
        self,
        x,
        qr,
        *,
        positions,
    ):
        """Project raw indexer queries, keys, and per-head weights."""
        seqlen = x.size(0)
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        q_rope = self.rope(q_rope, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = self._rotate_activation(q)
        k = self.compressor(x, positions=positions)
        k = self._rotate_activation(k)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.num_index_heads**-0.5
        )
        return q, k, weights

    @staticmethod
    def select(
        idx_q,
        idx_k,
        idx_w,
        *,
        seqlen: int,
        ratio: int,
        topk: int,
    ) -> torch.Tensor:
        """Select top-k compressed positions per query token.

        Accepts the folded single-sequence layout (``idx_q`` ``[T, H, D]``,
        ``idx_k`` ``[T // ratio, D]``, ``idx_w`` ``[T, H]`` -> ``[T, topk]``) or
        a batched one (``[B, L, H, D]``, ``[B, L // ratio, D]``, ``[B, L, H]``
        -> ``[B, L, topk]``). Batched, the score tensor is ``B * L * (L/ratio)``
        instead of ``(B*L) * (B*L/ratio)`` -- linear in tokens rather than
        quadratic -- and a query can only select keys from its own sequence.
        """
        batched = idx_q.ndim == 4
        eq = "bshd,btd->bsht" if batched else "shd,td->sht"
        index_score = torch.einsum(eq, idx_q, idx_k)
        # Out of place: the einsum lowers to aten.bmm, whose output selective
        # activation checkpointing caches, and an in-place relu on it trips the
        # SAC mutation guard.
        index_score = index_score.relu() * idx_w.unsqueeze(-1)
        index_score = index_score.sum(dim=-2)

        compress_causal_limit = (
            torch.arange(1, seqlen + 1, device=idx_q.device).unsqueeze(1) // ratio
        )
        compress_causal_mask = (
            torch.arange(seqlen // ratio, device=idx_q.device).repeat(seqlen, 1)
            >= compress_causal_limit
        )
        if batched:
            compress_causal_mask = compress_causal_mask.unsqueeze(0)
        index_score = index_score.masked_fill(
            compress_causal_mask, torch.finfo(index_score.dtype).min
        )
        # Deterministic selection. bf16 index scores are quantised (~0.1 apart
        # near their top), so the k-th boundary is frequently a tie, and
        # torch.topk's order among tied values is arbitrary -- and on the code
        # path taken for wide fp32 rows, not even stable call to call. That
        # made the FORWARD non-deterministic: the same inputs selected
        # different keys on a re-run, which breaks the activation-checkpoint
        # recompute (the MoE routed-token counts then differ) and makes any
        # A/B comparison of two backward kernels meaningless. A stable
        # descending sort breaks ties toward the lower key index; where there
        # is no boundary tie the selection is identical to topk's.
        k = min(topk, seqlen // ratio)
        order = torch.sort(index_score, dim=-1, descending=True, stable=True).indices
        return order[..., :k]
