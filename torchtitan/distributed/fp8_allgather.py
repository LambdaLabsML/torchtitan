# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 all-gather for MoE expert weights (the idea of Megatron-LM #5470).

Under FullAC each MoE block's expert weights are all-gathered twice per step
(forward, and again for the backward recompute); on dsv4_flash at 6x/EP=2
that is ~1 s/step of bf16 bytes and, with gather COUNT already minimal
(dense-never), the largest remaining communication cost. This halves the
bytes on the wire: FSDP2's tensor-subclass extension casts each rank's shard
to e4m3 with row-wise fp32 scales *before* the gather, gathers the fp8 data
and the (tiny) scales, and dequantizes back to bf16 *after* -- so the bf16
grouped GEMM, MinimalAsyncEP and the optimizer are untouched. Master weights
stay bf16 (safer than #5470's FP8 primaries; same wire effect).

What changes numerically: the GEMMs see weights rounded to e4m3 with a
per-output-row scale (``[E, O, 1]`` for ``_grouped_mm``'s ``A @ W^T``), the
standard row-wise fp8 weight recipe. The quantize runs on the 1/16 shard; the
dequantize on the gathered weight is one compiled pass (~1.3 ms per 6 GiB
layer, ~110 ms/step) against ~0.5 s/step of gather removed.

FSDP2 contract used: ``fsdp_pre_all_gather(mesh) -> (inputs, metadata)`` with
inputs of mixed dtypes gathered along dim 0; ``fsdp_post_all_gather`` returns
the unsharded tensor and the inner tensors whose storage FSDP frees on reshard
and re-allocates before calling again with ``out=``; the release flag lets FSDP
free the raw fp8/scale gather buffers immediately, since nothing aliases them.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils import _pytree as pytree

logger = logging.getLogger(__name__)

_E4M3_MAX = 448.0

# Ops whose outputs stay wrapped so the subclass survives to_empty(), init and
# FSDP's own slicing/copying of the sharded parameter (torchao's list).
_PRESERVE = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


def _preserve(func) -> bool:
    if func in _PRESERVE:
        return True
    # torchtitan's SPMD layout sharding (Module._spmd_distribute_state ->
    # spmd_types.shard) takes the local expert chunk through a custom op,
    # not aten.split; without this the sharded parameter comes back as a plain
    # tensor and FSDP never sees the fp8 all-gather hooks.
    return getattr(func, "namespace", None) == "spmd_types"


@torch.compile(dynamic=False)
def _quantize_rowwise(w: torch.Tensor):
    """[E, O, I] -> (e4m3 [E, O, I], fp32 scale [E, O, 1]); row = output channel."""
    wf = w.float()
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / _E4M3_MAX
    q = (wf / scale).to(torch.float8_e4m3fn)
    return q, scale


@torch.compile(dynamic=False)
def _dequantize_rowwise(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return (q.to(torch.float32) * scale).to(dtype)


class ExpertWeightFP8AllGather(torch.Tensor):
    """A bf16 master weight whose FSDP all-gather travels as row-wise e4m3."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return ExpertWeightFP8AllGather(args[0]._tensor)
        args, kwargs = pytree.tree_map_only(
            ExpertWeightFP8AllGather, lambda t: t._tensor, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if not _preserve(func):
            return out
        return pytree.tree_map_only(torch.Tensor, ExpertWeightFP8AllGather, out)

    def __tensor_flatten__(self):
        return ["_tensor"], {}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return ExpertWeightFP8AllGather(inner_tensors["_tensor"])

    def __repr__(self):
        return f"ExpertWeightFP8AllGather({self._tensor!r})"

    # ---- FSDP2 extension ---------------------------------------------------
    _logged = {"pre": False, "post": False}

    def fsdp_pre_all_gather(self, mesh):
        if not ExpertWeightFP8AllGather._logged["pre"]:
            ExpertWeightFP8AllGather._logged["pre"] = True
            logger.info(
                "fp8 expert all-gather: pre hook engaged (shard %s %s -> e4m3 + fp32 scales)",
                tuple(self._tensor.shape), self._tensor.dtype,
            )
        q, scale = _quantize_rowwise(self._tensor)
        return (q, scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        q, scale = all_gather_outputs
        if not ExpertWeightFP8AllGather._logged["post"]:
            ExpertWeightFP8AllGather._logged["post"] = True
            logger.info(
                "fp8 expert all-gather: post hook engaged (gathered %s %s, scales %s)",
                tuple(q.shape), q.dtype, tuple(scale.shape),
            )
        if out is not None:
            # FSDP re-allocated our bf16 storage; refill it in place.
            with torch.no_grad():
                out.copy_(_dequantize_rowwise(q, scale, out.dtype))
            return None
        w = _dequantize_rowwise(q, scale, param_dtype)
        return w, (w,)

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        return True  # the bf16 result aliases nothing in the raw gather buffers


torch.serialization.add_safe_globals([ExpertWeightFP8AllGather])

_EXPERT_WEIGHTS = ("w1_EFD", "w2_EDF", "w3_EFD")


def wrap_expert_weights_for_fp8_all_gather(model: nn.Module) -> int:
    """Wrap every GroupedExperts weight so FSDP gathers it as fp8. Call before
    ``model.parallelize`` / FSDP, while the model is still on the meta device.
    Returns the number of parameters wrapped."""
    from torchtitan.models.common.moe import GroupedExperts

    n = 0
    for _, module in model.named_modules():
        if not isinstance(module, GroupedExperts):
            continue
        for name in _EXPERT_WEIGHTS:
            p = getattr(module, name)
            if isinstance(p, ExpertWeightFP8AllGather):
                continue
            wrapped = nn.Parameter(
                ExpertWeightFP8AllGather(p.detach()), requires_grad=p.requires_grad
            )
            setattr(module, name, wrapped)
            n += 1
    logger.info("fp8 expert all-gather: wrapped %d expert weights", n)
    return n


def debug_log_fsdp_expert_storage(model: nn.Module) -> None:
    """After FSDP: what does FSDP actually hold as the sharded local tensor of
    each expert weight, and did it register the all-gather extension?"""
    import torch.distributed as dist
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    from torchtitan.models.common.moe import GroupedExperts

    if dist.is_initialized() and dist.get_rank() != 0:
        return
    seen = 0
    for name, module in model.named_modules():
        state = _get_module_fsdp_state(module)
        if state is None or state._fsdp_param_group is None:
            continue
        for fp in state._fsdp_param_group.fsdp_params:
            pname = getattr(fp._module_info, "param_name", "?")
            if pname not in _EXPERT_WEIGHTS:
                continue
            lt = fp._sharded_local_tensor
            logger.info(
                "fp8 debug: %s.%s sharded_local=%s has_hook=%s extension=%s",
                name, pname, type(lt).__name__, hasattr(lt, "fsdp_pre_all_gather"),
                getattr(fp, "_extensions_data", None) is not None,
            )
            seen += 1
            if seen >= 3:
                return
