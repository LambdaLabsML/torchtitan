# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import ModuleList

from .mhc import HcHead, HcPost, HcPre, te_mhc_enabled

if TYPE_CHECKING:
    from .attention import Attention
    from .mtp import MTPBlock


class DeepSeekV4TransformerBlock(TransformerBlock):
    """Transformer block with HC pre/post mixing around attention and FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: "Attention.Config"  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        # Split the block's tokens into two microbatches and overlap one
        # half's EP dispatch/combine (comm stream) with the other half's
        # attention/experts (compute stream). See ``_forward_dual_microbatch``.
        dual_microbatch: bool = False

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        if cfg.moe is not None:
            assert cfg.moe is not None
            self.moe = cfg.moe.build()
            self.moe_enabled = True
        else:
            assert cfg.feed_forward is not None
            self.moe = None
            self.feed_forward = cfg.feed_forward.build()
            self.moe_enabled = False

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()
        self.dual_microbatch = cfg.dual_microbatch

    def forward(
        self,
        x: torch.Tensor,
        input_ids_T: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        """Run one DeepSeek V4 decoder block.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.
            input_ids_T: Token IDs of shape ``[T]`` used by hash routing.
            attention_masks: Optional decoder mask handle; sparse attention may
                ignore it and build masks internally.
            positions: Optional position IDs of shape ``[T]``.

        Returns:
            Hidden states of shape ``[T, hc_mult, D]``.
        """
        if self.dual_microbatch and self.moe_enabled and x.shape[0] % 2 == 0:
            return self._forward_dual_microbatch(
                x, input_ids_T, attention_masks, positions, padding_mask
            )
        residual = x
        pre = self.hc_attn_pre(x)
        if len(pre) == 4:  # TORCHTITAN_MHC_GRAD_CHAIN: use the pass-through view as the residual
            x, post, comb, residual = pre
        else:
            x, post, comb = pre
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        pre = self.hc_ffn_pre(x)
        if len(pre) == 4:
            x, post, comb, residual = pre
        else:
            x, post, comb = pre
        if self.moe_enabled:
            assert self.moe is not None
            ffn_input = self.ffn_norm(x)
            if getattr(self.moe.router, "hash", False):
                x = self.moe(
                    ffn_input,
                    padding_mask_T=padding_mask,
                    input_ids_T=input_ids_T,
                )
            else:
                x = self.moe(ffn_input, padding_mask_T=padding_mask)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x

    def _attn_and_ffn_pre(self, x, attention_masks, positions):
        """Attention sub-block plus the FFN's HC pre-mix and norm."""
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        return self.ffn_norm(x), residual, post, comb

    def _forward_dual_microbatch(
        self, x, input_ids_T, attention_masks, positions, padding_mask
    ):
        """Two-microbatch schedule that hides the EP dispatch/combine.

        Profiles of the GB300 recipe show MinimalAsyncEP's dispatch copy and EP
        barrier fully exposed (~10% of GPU time): the dispatcher is a
        synchronous custom-op boundary, so on one stream nothing runs while
        rows are copied to peers and the group barrier spins. Megatron's
        combined-1F1B fine-grained schedule (``overlap_moe_expert_parallel_comm``)
        hides it by interleaving two microbatches; this is that idea inside one
        block:

            compute stream: attn(A) | attn(B)   | experts(A) | experts(B) | post(A) post(B)
            comm stream:            | dispatch(A)| dispatch(B)| combine(A) | combine(B)

        Every rank issues the four comm ops in the same order, and the receive
        pool holds four slots (``set_microbatch_split(2)``) so a slot is never
        rewritten before its consumer has run. Autograd replays each op on the
        stream it ran on, so backward (and FullAC's recompute) overlap the same
        way. The halves are whole sequences (``T/2`` must be a multiple of the
        sequence length), so attention, routing and HC mixing are unchanged
        per token and the result matches the single-microbatch forward.
        """
        from torchtitan.models.common.moe import maybe_set_sparse_mesh

        moe = self.moe
        assert moe is not None
        half = x.shape[0] // 2

        def split(t):
            return (None, None) if t is None else (t[:half], t[half:])

        xs, ids, pos, pad = split(x), split(input_ids_T), split(positions), split(padding_mask)
        use_hash = bool(getattr(moe.router, "hash", False))
        main = torch.cuda.current_stream()
        comm = _dual_mb_comm_stream()

        def dispatch(ffn_in, i):
            kw = {"input_ids_T": ids[i]} if use_hash else {}
            scores, eids, rmap = moe.router(
                ffn_in, moe.expert_bias_E, padding_mask_T=pad[i], **kw
            )
            counts = rmap.sum(dim=0)
            return moe.routed_experts.token_dispatcher.dispatch(
                ffn_in, scores, eids, counts
            )

        experts_mod = moe.routed_experts.inner_experts
        # Under dense-never the experts are their own FSDP unit. Two forward and
        # two backward passes through it per block would all-gather and
        # reduce-scatter the expert weights twice (job 792: 374 vs 401 TFLOP/s).
        # Keep them unsharded between the halves and let only the second
        # backward (half A, which autograd reaches last) reduce-scatter; FSDP2
        # accumulates the first half's grads in the reduce dtype meanwhile.
        fsdp = experts_mod if hasattr(experts_mod, "set_requires_gradient_sync") else None
        if fsdp is not None:
            fsdp.set_reshard_after_forward(False, recurse=False)
            fsdp.set_requires_gradient_sync(False, recurse=False)
            fsdp.set_reshard_after_backward(False, recurse=False)

        def experts(routed_in, n_e):
            with maybe_set_sparse_mesh():
                return experts_mod(routed_in, n_e)

        def combine(routed_out, meta, ffn_in):
            return moe.routed_experts.token_dispatcher.combine(routed_out, meta, ffn_in)

        def finish(out, ffn_in, residual, post, comb):
            if moe.shared_experts is not None:
                out = out + moe.shared_experts(ffn_in)
            return self.hc_post(out, residual, post, comb)

        def on_comm(fn, *args):
            ev = torch.cuda.Event()
            ev.record(main)
            with torch.cuda.stream(comm):
                comm.wait_event(ev)
                for a in args:
                    _record_stream(a, comm)
                out = fn(*args)
            done = torch.cuda.Event()
            done.record(comm)
            return out, done

        def join(done, *tensors):
            main.wait_event(done)
            for t in tensors:
                _record_stream(t, main)

        f_a = self._attn_and_ffn_pre(xs[0], attention_masks, pos[0])
        d_a, ev_da = on_comm(dispatch, f_a[0], 0)
        f_b = self._attn_and_ffn_pre(xs[1], attention_masks, pos[1])
        d_b, ev_db = on_comm(dispatch, f_b[0], 1)
        join(ev_da, d_a[0], d_a[1])
        e_a = experts(d_a[0], d_a[1])
        if fsdp is not None:
            # Backward reaches this before half A's expert backward: turn the
            # gradient sync and the post-backward reshard back on for it.
            e_a = _FsdpSyncOnBackward.apply(e_a, fsdp)
        c_a, ev_ca = on_comm(combine, e_a, d_a[2], f_a[0])
        join(ev_db, d_b[0], d_b[1])
        e_b = experts(d_b[0], d_b[1])
        if fsdp is not None:
            fsdp.reshard()  # free the unsharded experts after the second forward
        c_b, ev_cb = on_comm(combine, e_b, d_b[2], f_b[0])
        join(ev_ca, c_a)
        out_a = finish(c_a, *f_a)
        join(ev_cb, c_b)
        out_b = finish(c_b, *f_b)
        return torch.cat([out_a, out_b], dim=0)


class _FsdpSyncOnBackward(torch.autograd.Function):
    """Identity in forward; in backward re-enables gradient sync and the
    post-backward reshard on the experts' FSDP unit (see _forward_dual_microbatch).
    It sits on half A's expert output, so it runs after half B's post-backward
    (which only accumulates) and before half A's (which reduce-scatters)."""

    @staticmethod
    def forward(ctx, x, fsdp_module):
        ctx.fsdp_module = fsdp_module
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.fsdp_module.set_requires_gradient_sync(True, recurse=False)
        ctx.fsdp_module.set_reshard_after_backward(True, recurse=False)
        return grad, None


_DUAL_MB_COMM_STREAM = None


def _dual_mb_comm_stream():
    """The comm stream; DUAL_MB_SAME_STREAM=1 runs the same schedule on the
    compute stream (no overlap) to separate split-math from stream effects."""
    global _DUAL_MB_COMM_STREAM
    if os.environ.get("DUAL_MB_SAME_STREAM", "0") == "1":
        return torch.cuda.current_stream()
    if _DUAL_MB_COMM_STREAM is None:
        _DUAL_MB_COMM_STREAM = torch.cuda.Stream()
    return _DUAL_MB_COMM_STREAM


def _record_stream(t, stream) -> None:
    """Tell the caching allocator ``t`` is used on ``stream`` (cross-stream use)."""
    if torch.is_tensor(t) and t.is_cuda:
        try:
            t.record_stream(stream)
        except RuntimeError:
            pass  # symmetric-memory buffers are not caching-allocator tensors


class DeepSeekV4Model(Decoder):
    """DeepSeek V4 decoder model with HC branches and sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int
        vocab_size: int
        hc_mult: int = 4
        n_mtp_layers: int = 0
        compress_ratios: tuple[int, ...] = (1, 1, 4, 4)
        n_layers: int = 4
        norm_eps: float = 1e-6
        hc_head: HcHead.Config
        mtp_layers: list["MTPBlock.Config"] | None = None

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if self.mtp_layers is not None and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "DeepSeek V4 MTP does not support pipeline parallelism yet."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(
                            f"n_heads ({n_heads}) must be divisible by tp ({tp})"
                        )
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(
                            f"n_groups ({n_groups}) must be divisible by tp ({tp})"
                        )

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not yet supported for DeepSeek V4 sparse attention."
                )

            from .sharding import set_deepseek_v4_sharding_config

            set_deepseek_v4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Estimate DeepSeek V4 training FLOPs from the final model config."""
            deepseek_v4_model = cast(DeepSeekV4Model, model)
            nparams, active_nparams = get_nparams_and_active_nparams(deepseek_v4_model)

            attention_op_flops = 0
            for layers in (self.layers, self.mtp_layers or ()):
                for layer in layers:
                    attention = layer.attention
                    inner_attention = attention.inner_attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=seq_len,
                        sliding_window_size=inner_attention.window_size,
                    )

                    if attention.compress_ratio > 1:
                        compressed_seq_len = seq_len // attention.compress_ratio
                        if attention.compress_ratio == 4:
                            attention_op_flops += (
                                6
                                * attention.index_n_heads
                                * attention.index_head_dim
                                * compressed_seq_len
                            )
                            compressed_seq_len = min(
                                compressed_seq_len, inner_attention.index_topk
                            )
                        attention_op_flops += quadratic_attention_flops_per_token(
                            num_heads=attention.n_heads,
                            qk_head_dim=attention.head_dim,
                            v_head_dim=attention.head_dim,
                            seq_len=compressed_seq_len,
                        )

            active_nparams += len(deepseek_v4_model.mtp_layers) * sum(
                param.numel() for param in deepseek_v4_model.lm_head.parameters()
            )
            active_nparams += (self.hc_mult - 1) * sum(
                param.numel()
                for mtp_layer in deepseek_v4_model.mtp_layers
                for param in cast("MTPBlock", mtp_layer).h_proj.parameters()
            )

            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.hc_mult = cfg.hc_mult
        self.n_mtp_layers = cfg.n_mtp_layers
        self.compress_ratios = list(cfg.compress_ratios)[: cfg.n_layers]
        self.n_main_layers = cfg.n_layers

        self.hc_head = cfg.hc_head.build()
        self.mtp_layers = ModuleList()
        if cfg.mtp_layers is not None:
            self.mtp_layers = ModuleList(
                mtp_layer.build() for mtp_layer in cfg.mtp_layers
            )

    def get_attention_masks(
        self,
        positions,
        *,
        padding_mask=None,
        max_num_documents=None,
        max_context_length=None,
    ):
        del positions, padding_mask, max_num_documents, max_context_length
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """Run the DeepSeek V4 decoder."""
        if len(self.mtp_layers) > 0 and self.tok_embeddings is None:
            raise ValueError("DeepSeek V4 MTP forward requires token embeddings.")
        if len(self.mtp_layers) > 0 and self._skip_lm_head:
            raise ValueError(
                "DeepSeek V4 MTP cannot skip the LM head because chunked "
                "cross entropy is not supported."
            )

        input_ids_T = tokens.detach().long()
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        if te_mhc_enabled():
            # TE fused mHC keeps the residual stream as [T, D, hc_mult].
            if len(self.mtp_layers) > 0:
                raise ValueError("TORCHTITAN_TE_MHC=1 does not support MTP layers yet")
            h = h.unsqueeze(-1).expand(-1, -1, self.hc_mult).contiguous()
        else:
            h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for i in range(self.n_main_layers):
            layer = self.layers[str(i)]
            h = layer(
                h,
                input_ids_T,
                attention_masks,
                positions,
                padding_mask=padding_mask,
            )

        prev_hc_hidden = h
        main_hidden = self.hc_head(h)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        if len(self.mtp_layers) == 0:
            if self._skip_lm_head or self.lm_head is None:
                return main_hidden
            return self.lm_head(main_hidden)

        outputs = [main_hidden] + self.mtp_forward(
            prev_hc_hidden,
            tokens,
            attention_masks,
            positions,
            padding_mask,
        )
        return [
            self.lm_head(item) if self.lm_head is not None else item for item in outputs
        ]

    def mtp_forward(
        self,
        prev_hc_hidden: torch.Tensor,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run auxiliary MTP depths and return prediction hidden states."""
        mtp_outputs = []
        for depth, mtp_block in enumerate(self.mtp_layers, 1):
            mtp_tokens, valid_mask = roll_mtp_sequence(
                tokens,
                shift=depth,
                fill_value=0,
                positions=positions,
                padding_mask=padding_mask,
                return_valid_mask=True,
            )
            prev_hc_hidden, prediction_hidden = mtp_block(
                self.tok_embeddings(mtp_tokens),
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                attention_masks,
                positions,
                padding_mask=padding_mask,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs
