"""MXFP8 grouped expert GEMMs through Transformer Engine's functional grouped-tensor API.

Why: torchao's fp8 grouped path aborts on sm_103 and its MXFP8 path needed the
padded TorchAO dispatcher (-7.5%). TE's fused cuBLASLt grouped GEMM with
device-side splits measured 1.12x torch's bf16 ``_grouped_mm`` balanced and
1.34x collapsed on the real per-rank shapes once weight quantization is cached
(jobs 833-836).

Constraints handled here:
* the fused path takes at most 64 groups per kernel -> chunks of 64 experts,
  one host sync per chunk boundary for the row cut;
* TE grouped quantize needs every group's row count to be a multiple of 128 -> routed rows are
  padded once per layer to 128-row groups (index_copy into a zeroed buffer), the whole w1/w3/w2
  chain runs in the padded layout, and the output is un-padded once;
* weight quantization (~4 ms per layer-call) is cached per step, keyed on the
  unsharded weight's ``data_ptr``/``_version``, so the FullAC recompute and
  the backward reuse the forward's fp8 weights.

Knob: ``TE_EXPERTS=1`` on the debugmodel policy config; production config
``..._tedense_7x_teexperts``.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from torchtitan.models.common.moe import GroupedExperts


def _tex():
    import transformer_engine_torch as tex

    return tex


def _offsets(splits, in_f, out_f):
    tex = _tex()
    return tex.splits_to_offsets_multi(
        splits,
        splits.device,
        strides=[1, in_f, out_f],
        include_leading_zero=[True, True, True],
        dtypes=[torch.int64] * 3,
        bulk_allocate=True,
    )


def _grouped(data2d, n, split_sizes, offs, last_dim):
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear

    return _GroupedLinear._make_grouped_tensor(
        data2d, num_gemms=n, split_sizes=split_sizes, tensor_offsets=offs,
        last_dim=last_dim, dtype=data2d.dtype,
    )


def _quantizer():
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    q = MXFP8Quantizer(fp8_dtype=_tex().DType.kFloat8E4M3, rowwise=True, columnwise=True)
    q.optimize_for_gemm = True
    return q


class _TEGroupedMM(torch.autograd.Function):
    """y[R,O] = grouped_mm(x[R,I], w[E,O,I]); ``splits`` device int64 [E<=64], each % 32 == 0."""

    @staticmethod
    def forward(ctx, x, w, splits, w_cache):
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor

        tex = _tex()
        E, O, I = w.shape
        split_sizes, (_, in_off, out_off) = _offsets(splits, I, O)
        gx = tex.group_quantize(x, _quantizer(), E, split_sizes, tensor_offsets=in_off)
        key = (w.data_ptr(), w._version, w.shape)
        if w_cache is not None and w_cache.get("key") == key:
            gw = w_cache["gw"]
        else:
            gw = tex.group_quantize(w.reshape(E * O, I), _quantizer(), E, None)
            if w_cache is not None:
                w_cache["key"] = key
                w_cache["gw"] = gw
        y = torch.empty(x.shape[0], O, device=x.device, dtype=x.dtype)
        general_grouped_gemm_for_grouped_tensor(gw, gx, _grouped(y, E, split_sizes, out_off, O), layout="TN")
        ctx.gx, ctx.gw = gx, gw
        ctx.meta = (split_sizes, in_off, out_off, E, O, I, x.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor

        tex = _tex()
        split_sizes, in_off, out_off, E, O, I, dtype = ctx.meta
        gdy = tex.group_quantize(dy.contiguous(), _quantizer(), E, split_sizes, tensor_offsets=out_off)
        dx = torch.empty(dy.shape[0], I, device=dy.device, dtype=dtype)
        general_grouped_gemm_for_grouped_tensor(ctx.gw, gdy, _grouped(dx, E, split_sizes, in_off, I), layout="NN")
        dw = torch.empty(E, O, I, device=dy.device, dtype=dtype)
        general_grouped_gemm_for_grouped_tensor(ctx.gx, gdy, [dw[i] for i in range(E)], layout="NT")
        return dx, dw, None, None


def te_grouped_mm(x, w, splits, cache=None, chunk=64):
    E = w.shape[0]
    if E <= chunk:
        return _TEGroupedMM.apply(x, w, splits, cache)
    cuts = torch.cumsum(splits, 0)[chunk - 1::chunk].tolist()  # host sync per chunk boundary
    outs, row = [], 0
    for i, e0 in enumerate(range(0, E, chunk)):
        e1 = min(E, e0 + chunk)
        r1 = cuts[i] if i < len(cuts) else x.shape[0]
        c = None if cache is None else cache.setdefault(i, {})
        outs.append(_TEGroupedMM.apply(x[row:r1], w[e0:e1], splits[e0:e1], c))
        row = r1
    return torch.cat(outs, 0)


def pad_plan(splits):
    """Rows per expert -> (padded splits % 32, destination row of every source row, padded row count)."""
    # TE's grouped quantize needs EVERY group's rows % 128 == 0 (probe 867), not just 32.
    padded = (splits + 127) // 128 * 128
    # TE grouped tensors also need the TOTAL row count % 128 == 0: grow the last
    # group's zero padding (its extra rows carry zero dy, so wgrad is unaffected).
    padded[-1] += (-padded.sum()) % 128
    off = torch.cumsum(splits, 0) - splits
    poff = torch.cumsum(padded, 0) - padded
    shift = torch.repeat_interleave(poff - off, splits)
    dst = torch.arange(shift.numel(), device=splits.device) + shift
    return padded, dst, int(padded.sum().item())  # one host sync per layer


class TEGroupedExperts(GroupedExperts):
    """``GroupedExperts`` whose three grouped GEMMs run in MXFP8 through TE."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        """Drop-in replacement for GroupedExperts.Config that builds TEGroupedExperts."""

    def __init__(self, config: Config):
        super().__init__(config)
        object.__setattr__(self, "_te_caches", [{}, {}, {}])

    def forward(self, x_RD, num_tokens_per_expert_E):
        if not x_RD.is_cuda or x_RD.dtype != torch.bfloat16:
            return super().forward(x_RD, num_tokens_per_expert_E)
        splits = num_tokens_per_expert_E.to(torch.int64)
        padded, dst, rp = pad_plan(splits)
        # MinimalAsyncEP hands over its full-capacity receive slot; only the
        # first sum(counts) rows are valid, and combine expects an output with
        # the same capacity row count.
        r_cap = x_RD.shape[0]
        r_valid = dst.numel()
        xv = x_RD[:r_valid] if r_valid != r_cap else x_RD
        xp = xv.new_zeros(rp, xv.shape[1]).index_copy(0, dst, xv)
        c = self.__dict__["_te_caches"]
        w1, w3, w2 = self.w1_EFD, self.w3_EFD, self.w2_EDF
        gate = te_grouped_mm(xp, w1.bfloat16(), padded, c[0])
        up = te_grouped_mm(xp, w3.bfloat16(), padded, c[1])
        h = self.activation_fn(gate, up)
        yp = te_grouped_mm(h, w2.bfloat16(), padded, c[2])
        y = yp.index_select(0, dst).type_as(x_RD)
        if r_valid != r_cap:
            y = F.pad(y, (0, 0, 0, r_cap - r_valid))
        return y


def convert_experts_config(cfg: GroupedExperts.Config) -> TEGroupedExperts.Config:
    kwargs = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.init}
    return TEGroupedExperts.Config(**kwargs)
