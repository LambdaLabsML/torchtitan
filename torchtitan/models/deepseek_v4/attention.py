# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    apply_attention_sink_rescale,
    BaseAttention,
    FlexInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.leaf_compile import leaf_compile

from .cudnn_dsa import flatten_batched_indices, fused_dsa_attention
from .cudnn_indexer import cudnn_indexer_select

from .compressor import Compressor, Indexer


def _assert_spmd_attention_type(tensor, *, tp):
    if spmd.is_type_checking():
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


# ---------------------------------------------------------------------------
# Leaf-compiled attention pre/post-processing (see torchtitan/tools/leaf_compile.py).
#
# Eager, the q path is: per-head RMS normalisation (3 passes over the
# [T, H, 512] bf16 q), split, fp32 upcast + complex rotation of the rope tail,
# cast back, and a cat that copies the whole q again; the output path repeats
# the split / inverse rotation / cat on o. In the 8-node profile the cat and
# split copies alone were 5.6 % of kernel time. Each function below is one
# graph, so Inductor writes q (or o) exactly once. The rotation is the
# complex product written in real arithmetic -- (a + bi)(c + di) -- on the
# same adjacent-pair layout ComplexRoPE.apply_rotary_emb uses, so the math is
# unchanged; the complex cache is passed as its real view because Inductor
# does not lower complex tensors.
# ---------------------------------------------------------------------------


def _rotate_tail(x, cache_ri, *, rd, inverse):
    """Rotate the last ``rd`` features of ``x`` [T, H, D] by the RoPE cache
    given as ``view_as_real`` of the complex cache, shape [T, 1, rd // 2, 2]."""
    head, tail = x[..., :-rd], x[..., -rd:]
    pairs = tail.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    c, d = cache_ri[..., 0], cache_ri[..., 1]
    if inverse:
        d = -d
    rotated = torch.stack((a * c - b * d, a * d + b * c), dim=-1)
    return torch.cat([head, rotated.flatten(-2).type_as(tail)], dim=-1)


@leaf_compile(group="attn")
def _q_norm_rope(q, cache_ri, *, rd, norm_eps):
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + norm_eps)
    return _rotate_tail(q, cache_ri, rd=rd, inverse=False)


@leaf_compile(group="attn")
def _o_rope_inverse(o, cache_ri, *, rd):
    return _rotate_tail(o, cache_ri, rd=rd, inverse=True)


class DSV4FlexInnerAttention(FlexInnerAttention):
    """DeepSeek sparse attention core for DeepSeek-V4.

    The core attends over the concatenated KV sequence ``[0, L + n_cmp)``,
    where the first ``L`` positions are the uncompressed sliding-window KV
    (``swa_k``) and the next ``n_cmp`` positions are the compressed KV
    (``cmp_k``):

    - sliding window: fixed pattern over ``swa_k``, expressed as a
      ``mask_mod`` predicate (no indices);
    - compressed blocks: for HCA (``compress_ratio=128``) all causal blocks
      are attendable, also a fixed ``mask_mod`` pattern; for CSA
      (``compress_ratio=4``) each query attends only its top-k selected
      compressed positions, which is the only dynamic (index-based) part.

    The ``mask_mod`` is evaluated at token granularity inside flex_attention;
    the per-query-block KV block listing (``BlockMask.from_kv_blocks``) only
    restricts which blocks the kernel loads.

    Overrides can replace ``_build_block_mask`` (e.g. NPU varlen kernels) or
    the whole ``forward`` (e.g. fused SMLA/CSA kernels, which consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors). Under context
    parallelism, all-gathering ``idx_k`` and ``cmp_k`` at this module boundary
    enables global sparse selection.

    TODO: the indexer auxiliary loss is intentionally dropped for now; it will
    be re-added as a carrier-injected aux loss (see the NPU fork) once the
    general aux-loss mechanism lands.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        fused_dsa_backward: bool = False
        fused_dsa_forward: bool = False
        """Compute the DSA gradients with cuDNN's fused kernel instead of the
        FlexAttention backward. The forward is unchanged. The flex backward is
        ~40 % of kernel time on GB300 at head_dim=512 -- a generic Triton
        template whose tiles are capped by the sparse block size -- while
        cuDNN ships a CuTe-DSL kernel written for this exact shape. Needs
        ``nvidia-cudnn-frontend[cutedsl]`` and SM90+."""

        cudnn_indexer: bool = False
        """Select the CSA top-k with cuDNN's fused indexer kernel instead of
        the eager einsum + stable sort (see ``cudnn_indexer.py``). Forward
        only; the indexer has no gradient path in torchtitan."""

        seq_len: int = 0
        """Length of one packed sequence. 0 (the original behaviour) treats the
        whole per-rank token stream as a single sequence, which makes the DSA
        cost quadratic in the microbatch: the indexer scores every query
        against every compressed key of the whole stream and the selection mask
        is a dense [T, T + n_cmp]. With it set, the stream is split into
        ``T // seq_len`` independent sequences and both scale with ``seq_len``
        instead -- so a larger microbatch costs proportionally more rather than
        quadratically more, and queries stop attending across packed document
        boundaries (their RoPE positions restart per sequence, so those scores
        were meaningless)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.seq_len = config.seq_len
        self.fused_dsa_backward = config.fused_dsa_backward
        self.fused_dsa_forward = config.fused_dsa_forward
        self.cudnn_indexer = config.cudnn_indexer
        self.block_size = config.block_size

    def get_window_topk_idxs(self, *, bsz, seqlen, device):
        """Cached: the index build depends only on (seqlen, window), yet ran
        86 times a step (eager clamp_min + arange, 1.2% of GPU time)."""
        cache = self.__dict__.setdefault("_window_idx_cache", {})
        key = (seqlen, str(device))
        if key not in cache:
            cache[key] = self._get_window_topk_idxs_uncached(
                bsz=1, seqlen=seqlen, device=device
            )[0]
        return cache[key].unsqueeze(0).expand(bsz, -1, -1)

    def _get_window_topk_idxs_uncached(
        self,
        *,
        bsz: int,
        seqlen: int,
        device,
    ) -> torch.Tensor:
        """Return sliding-window KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, W]``. Valid entries are uncompressed KV
            positions in ``[0, L)`` and padded entries are ``-1``.
        """
        window = min(seqlen, self.window_size)
        q_idx = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (q_idx - window + 1).clamp_min(0) + torch.arange(window, device=device)
        idxs = torch.where(idxs <= q_idx, idxs, -1)
        return idxs.unsqueeze(0).expand(bsz, -1, -1)

    def get_compress_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        n_cmp: int,
        device,
    ) -> torch.Tensor:
        """Return causal compressed KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            n_cmp: Number of compressed KV tokens.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, n_cmp]``. Valid entries are compressed KV
            positions offset by ``seqlen`` and padded entries are ``-1``.
        """
        if n_cmp == 0:
            return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)

        cmp_idx = torch.arange(n_cmp, device=device).repeat(seqlen, 1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
        causal_limit = causal_limit // self.compress_ratio
        cmp_idx = torch.where(cmp_idx < causal_limit, seqlen + cmp_idx, -1)
        return cmp_idx.unsqueeze(0).expand(bsz, -1, -1)

    def _build_block_mask(
        self,
        bsz: int,
        seqlen: int,
        kv_len: int,
        selected_indices: torch.Tensor,
        device,
    ) -> BlockMask:
        """Build a FlexInnerAttention block mask from selected KV indices.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length.
            kv_len: Length of the concatenated KV sequence.
            selected_indices: Tensor of shape ``[B, L, K]`` containing final KV
                positions in ``[0, kv_len)``; ``-1`` entries are ignored.
            device: Device used for mask tensors.

        Returns:
            ``BlockMask`` whose block list and token-level predicate encode
            exactly the selected KV positions.
        """
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        assert (
            seqlen % bq == 0
        ), f"seqlen ({seqlen}) must be divisible by Q block size ({bq})"
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq

        valid = selected_indices >= 0
        safe_indices = selected_indices.clamp(0, kv_len - 1)

        selected_blocks = (safe_indices // bk).reshape(
            bsz, n_q_blocks, bq * selected_indices.size(-1)
        )
        block_values = valid.reshape(selected_blocks.shape).to(torch.int32)
        bm = torch.zeros(
            bsz, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device
        )
        bm[:, 0].scatter_add_(-1, selected_blocks, block_values)
        bm = (bm > 0).to(torch.int32)
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        selected_count = torch.zeros(
            bsz, seqlen, kv_len, dtype=torch.int32, device=device
        )
        selected_count.scatter_add_(2, safe_indices, valid.to(torch.int32))
        selected_mask = selected_count > 0

        def dsa_mask_mod(b, h, q_idx, kv_idx):
            return selected_mask[b, q_idx, kv_idx]

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=dsa_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    @torch.no_grad()
    def selected_kv_indices(
        self, *, bsz, seqlen, n_cmp, idx_q, idx_k, idx_w, device
    ) -> torch.Tensor:
        """The KV positions each query attends, ``[B, L, K]``, -1 padded.

        Window positions live in ``[0, L)`` and compressed ones are offset by
        ``L``, so the indices address the concatenated per-sequence KV stream
        directly. Exposed as a method because the block mask, the fused
        backward and the tests all need exactly this selection.
        """
        selected_indices = [
            self.get_window_topk_idxs(bsz=bsz, seqlen=seqlen, device=device)
        ]
        if self.compress_ratio == 4:
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError(
                    "DSV4FlexInnerAttention requires idx_q, idx_k, "
                    "and idx_w when compress_ratio=4"
                )
            select = cudnn_indexer_select if self.cudnn_indexer else Indexer.select
            cmp_topk = select(
                idx_q,
                idx_k,
                idx_w,
                seqlen=seqlen,
                ratio=self.compress_ratio,
                topk=self.index_topk,
            )
            if cmp_topk.ndim == 2:  # folded layout -> add the batch dim
                cmp_topk = cmp_topk.unsqueeze(0)
            causal_limit = (
                torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
                // self.compress_ratio
            )
            # The cuDNN path pads invalid slots with -1; the eager path never
            # produces negatives, so the extra test is free for it.
            valid = (cmp_topk >= 0) & (cmp_topk < causal_limit.unsqueeze(0))
            cmp_topk = torch.where(valid, seqlen + cmp_topk, -1)
            selected_indices.append(cmp_topk)
        elif self.compress_ratio > 1:
            selected_indices.append(
                self.get_compress_topk_idxs(
                    bsz=bsz, seqlen=seqlen, n_cmp=n_cmp, device=device
                )
            )
        selected_indices = torch.cat(selected_indices, dim=-1)
        return selected_indices

    def _forward_impl(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attention_masks=None,
        out_rope=None,
    ) -> torch.Tensor:
        """Run DSV4 sparse attention over a folded token stream."""
        if attention_masks is not None:
            raise ValueError(
                "DSV4FlexInnerAttention does not accept attention_masks; "
                "the DSA block mask is built internally."
            )
        if attn_sink is None:
            raise ValueError("DSV4FlexInnerAttention requires attn_sink")

        bsz, seqlen = self._batch_shape(q.size(0))
        n_cmp = 0 if cmp_k is None else cmp_k.size(0) // bsz

        if bsz == 1:
            kv = swa_k.unsqueeze(1)
            if cmp_k is not None:
                kv = torch.cat([kv, cmp_k.unsqueeze(1)], dim=0)
            kv = kv.expand(-1, q.size(1), -1)
            kv_len = kv.size(0)
            q_in = q
        else:
            # One KV stream per sequence: [B, L (+ n_cmp), H, D]. Window and
            # compressed indices are per-sequence, so no query can reach
            # another sequence's keys.
            n_heads, head_dim = q.size(1), q.size(2)
            kv = swa_k.view(bsz, seqlen, head_dim)
            if cmp_k is not None:
                kv = torch.cat([kv, cmp_k.view(bsz, n_cmp, head_dim)], dim=1)
            kv_len = kv.size(1)
            kv = kv.unsqueeze(2).expand(-1, -1, n_heads, -1)
            q_in = q.view(bsz, seqlen, n_heads, head_dim)
            if idx_q is not None:
                idx_q = idx_q.view(bsz, seqlen, idx_q.size(1), idx_q.size(2))
                idx_k = idx_k.view(bsz, n_cmp, idx_k.size(1))
                idx_w = idx_w.view(bsz, seqlen, idx_w.size(1))

        with spmd.no_typecheck():
            selected_indices = self.selected_kv_indices(
                bsz=bsz,
                seqlen=seqlen,
                n_cmp=n_cmp,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_w=idx_w,
                device=q.device,
            )

            # The block mask exists only to drive a flex kernel. With both
            # halves of the attention on cuDNN nothing reads it, and building
            # it is a dense [B, n_q_blocks, n_kv_blocks] scatter per layer.
            block_mask = (
                None
                if self.fused_dsa_forward
                else self._build_block_mask(
                    bsz, seqlen, kv_len, selected_indices, q.device
                )
            )

            def apply_sink(out_THV, lse_TH):
                return apply_attention_sink_rescale(out_THV, lse_TH, attn_sink)

            if self.fused_dsa_backward:
                return self._fused_dsa(
                    q_in, kv, attn_sink, selected_indices, block_mask, bsz, kv_len,
                    out_rope=out_rope,
                )
            assert out_rope is None, "output RoPE fusion needs fused_dsa_backward"

            return super().forward(
                q_in,
                kv,
                kv,
                attention_masks=block_mask,
                scale=self.softmax_scale,
                out_transform=apply_sink,
            )

    def _fused_dsa(
        self, q_in, kv, attn_sink, selected_indices, block_mask, bsz, kv_len, *, out_rope=None
    ):
        """cuDNN backward (and optionally forward) over the flat layout.

        The kernel takes one KV stream and global indices, so a batched
        ``[B, L, ...]`` input is flattened and each sequence's indices are
        offset into its own slice -- which is exactly the isolation the
        batched path already guarantees.
        """

        def flex_fwd(q_flat, kv_flat):
            del q_flat, kv_flat  # the flex kernel wants the original layout
            out, lse = None, None

            def capture(out_THV, lse_TH):
                nonlocal out, lse
                out, lse = out_THV, lse_TH
                return out_THV

            super(DSV4FlexInnerAttention, self).forward(
                q_in,
                kv,
                kv,
                attention_masks=block_mask,
                scale=self.softmax_scale,
                out_transform=capture,
            )
            return out, lse

        # [B, L, H, D] -> [B*L, H, D] and [B, N, H, D] -> [B*N, D]. All query
        # heads share one KV stream, so head 0 is the whole stream and dropping
        # the head axis is a view. Batched kv is [B, N, H, D] against
        # [N, H, D] unbatched, so ndim tells the two layouts apart.
        if q_in.ndim == 4:
            q_flat = q_in.flatten(0, 1)
            kv_flat = kv[..., 0, :].flatten(0, 1)
        else:
            q_flat = q_in
            kv_flat = kv[:, 0, :]
        kv_flat = kv_flat.contiguous()
        assert kv_flat.shape[0] == bsz * kv_len, (
            f"kv_flat has {kv_flat.shape[0]} rows, expected {bsz * kv_len}"
        )
        idx_flat = flatten_batched_indices(selected_indices, kv_len)
        # Both kernels return the flat [B*L, H, V] stream, which is also what
        # FlexInnerAttention.forward hands back, so no reshape is needed here.
        return fused_dsa_attention(
            q_flat,
            kv_flat,
            attn_sink,
            idx_flat,
            softmax_scale=self.softmax_scale,
            flex_fwd=None if self.fused_dsa_forward else flex_fwd,
            out_rope=out_rope,
        )

    def _batch_shape(self, num_tokens: int) -> tuple[int, int]:
        """Split the per-rank token stream into ``(bsz, seq_len)``.

        ``seq_len`` unset, or a stream exactly one sequence long, keeps the
        original single-sequence behaviour.
        """
        if not self.seq_len or num_tokens == self.seq_len:
            return 1, num_tokens
        if num_tokens % self.seq_len != 0:
            raise ValueError(
                f"token stream ({num_tokens}) must be a multiple of "
                f"seq_len ({self.seq_len})"
            )
        return num_tokens // self.seq_len, self.seq_len


class SlidingWindowAttention(DSV4FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_masks=None,
        out_rope=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            attention_masks=attention_masks,
            out_rope=out_rope,
        )


class HeavilyCompressedAttention(DSV4FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_masks=None,
        out_rope=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            attention_masks=attention_masks,
            out_rope=out_rope,
        )


class CompressedSparseAttention(DSV4FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        idx_q,
        idx_k,
        idx_w,
        attn_sink,
        *,
        attention_masks=None,
        out_rope=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            attention_masks=attention_masks,
            out_rope=out_rope,
        )


class _GroupedLowRankProj(torch.autograd.Function):
    """``einsum("tgd,grd->tgr", o, w)`` as strided bmm with no permute copies.

    Forward writes ``[T, G, R]`` directly through a transposed view; backward
    does the same for ``d_o``. The weight gradient is a plain bmm.
    """

    @staticmethod
    def forward(ctx, o, w):
        ctx.save_for_backward(o, w)
        t, g, _ = o.shape
        r = w.shape[1]
        out = torch.empty(t, g, r, device=o.device, dtype=o.dtype)
        torch.bmm(o.transpose(0, 1), w.transpose(1, 2), out=out.transpose(0, 1))
        return out

    @staticmethod
    def backward(ctx, grad):
        o, w = ctx.saved_tensors
        g3 = grad.transpose(0, 1)  # [G, T, R] view
        d_o = d_w = None
        if ctx.needs_input_grad[0]:
            d_o = torch.empty_like(o)
            torch.bmm(g3, w, out=d_o.transpose(0, 1))
        if ctx.needs_input_grad[1]:
            d_w = torch.bmm(g3.transpose(1, 2), o.transpose(0, 1))
        return d_o, d_w


class Attention(BaseAttention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4FlexInnerAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: DSV4FlexInnerAttention.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()

        # Build all sub-modules from their configs.
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build()

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()

    def forward(self, x, attention_masks=None, positions=None):
        """Apply one DeepSeek V4 attention layer over folded tokens."""
        num_tokens = x.size(0)
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        # Complex RoPE cache for these positions as a real [T, 1, rd/2, 2]
        # view; shared by the q path here and the inverse rotation of o below.
        rope_cache_ri = torch.view_as_real(
            self.rope._reshape_cache(q[..., -rd:], positions)
        )
        q = _q_norm_rope(q, rope_cache_ri, rd=rd, norm_eps=self.norm_eps)

        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(kv_rope.unsqueeze(1), positions=positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        # With the DSA backward on cuDNN the sparse-attention Function applies
        # the output inverse RoPE itself, in place on the output it owns.
        fuse_out_rope = bool(getattr(self.inner_attention, "fused_dsa_backward", False))
        out_rope = (rope_cache_ri.contiguous(), rd) if fuse_out_rope else None
        if self.compress_ratio == 4:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                idx_q,
                idx_k,
                idx_w,
                attn_sink_param,
                attention_masks=attention_masks,
                out_rope=out_rope,
            )
        elif self.compress_ratio > 1:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                attn_sink_param,
                attention_masks=attention_masks,
                out_rope=out_rope,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_masks=attention_masks,
                out_rope=out_rope,
            )

        if not fuse_out_rope:
            o = _o_rope_inverse(o, rope_cache_ri, rd=rd)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        # Grouped low-rank projection without the einsum's permute copies: the
        # einsum copied ``o`` ([T, G, D], 3.2 GiB at 8k/6x) into batch-major
        # layout and copied its output back to token-major -- ~4.5 ms per
        # layer-step of pure copies in the job 680 trace. Strided bmm reads
        # ``o`` in place and writes straight into a token-major buffer
        # (probe 848: 15.21 -> 11.75 ms fwd+bwd, bitwise identical).
        o = _GroupedLowRankProj.apply(o, wo_a)
        with spmd.local():
            o = o.reshape(num_tokens, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        return self.wo_b(o)
