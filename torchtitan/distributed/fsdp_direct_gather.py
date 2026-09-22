"""FSDP2 direct all-gather (FSDP_DIRECT_GATHER=1).

FSDP2 all-gathers into a staging buffer and then copies it out into each
parameter's unsharded storage (``torch.ops.fsdp.split_with_sizes_copy``) on the
compute stream. For the DSv4-flash expert weights that copy-out is 6.4 GB per
layer per gather, 1.85% of the step, fully exposed. When an FSDP group holds a
single parameter sharded on dim 0 with no padding and no dtype cast, the
gathered layout *is* the unsharded parameter, so this patch gathers straight
into the parameter's own storage and turns the copy-out into a wait. Groups
that do not qualify fall through to the stock path. Pair with
``MOE_PACKED_EXPERT_WEIGHTS=1`` (w1/w2/w3 as one parameter).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _C
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _G
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    _get_device_handle,
)

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("FSDP_DIRECT_GATHER", "0") == "1"
STATS = {"direct": 0, "fallback": 0}
_orig_gather = _C.foreach_all_gather
_orig_copy_out = _C.foreach_all_gather_copy_out
_logged = False
_logged_fallback = False


_reasons: dict[str, int] = {}


def _reject(why: str) -> bool:
    _reasons[why] = _reasons.get(why, 0) + 1
    return False


def _eligible(fsdp_params, world_size: int) -> bool:
    if len(fsdp_params) != 1:
        return _reject(f"{len(fsdp_params)} params in group")
    p = fsdp_params[0]
    if hasattr(p._sharded_local_tensor, "fsdp_pre_all_gather"):
        return _reject("extension param")
    if p.fsdp_placement.dim != 0:
        return _reject("placement dim != 0")
    if p.param_dtype is not None and p.param_dtype != p.orig_dtype:
        return _reject(f"dtype cast {p.orig_dtype}->{p.param_dtype}")
    inputs = p.all_gather_inputs
    if len(inputs) != 1 or not inputs[0].is_contiguous():
        return _reject("non-contiguous/multi input")
    if inputs[0].numel() * world_size != p._orig_size.numel():
        return _reject("padded shard")
    return True


def foreach_all_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device, all_gather_comm):
    world_size = group.size()
    if not _eligible(fsdp_params, world_size):
        STATS["fallback"] += 1
        return _orig_gather(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device, all_gather_comm)
    p = fsdp_params[0]
    dh = _get_device_handle(device.type)
    inp = p.all_gather_inputs[0]
    numels, dtypes = [inp.numel()], [inp.dtype]
    # Ordering: unshard() already made the copy-in stream wait on the previous
    # reshard's event (the point where this storage was freed and its last
    # readers had been issued), so waiting on the copy-in stream is the whole
    # dependency -- the same one stock FSDP2 uses. Do NOT wait on the compute
    # stream itself: under FullAC the CPU is a block ahead, that wait made every
    # prefetched gather start only after the current block's attention had run
    # (job 1098 profile: 92% of all-gather time exposed, -11% of the step).
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with dh.stream(all_gather_stream):
        # allocate the unsharded storage on the stream that writes it, so the
        # caching allocator's per-stream pool ordering covers its reuse
        p.init_all_gather_outputs(numels, dtypes, world_size, device)
        p.alloc_all_gather_outputs()
        out = p.all_gather_outputs[0]
        out._fsdp_direct = True
        work = all_gather_comm(output_tensor=out, input_tensor=inp, group=group, async_op=async_op)
        event = all_gather_stream.record_event()
    STATS["direct"] += 1
    return AllGatherResult(out, event, work, [dtypes], [numels], numels)


def foreach_all_gather_copy_out(all_gather_result, fsdp_params, group):
    global _logged
    out = all_gather_result.all_gather_output
    if getattr(out, "_fsdp_direct", False):
        dh = _get_device_handle(out.device.type)
        if all_gather_result.all_gather_event is not None:
            dh.current_stream().wait_event(all_gather_result.all_gather_event)
        if isinstance(all_gather_result.all_gather_work, dist.distributed_c10d.Work):
            all_gather_result.all_gather_work.wait()
        if not _logged:
            _logged = True
            logger.info("FSDP direct all-gather active (%d direct so far, %d stock)", STATS["direct"], STATS["fallback"])
        return
    global _logged_fallback
    if not _logged_fallback and STATS["fallback"] >= 50:
        _logged_fallback = True
        logger.info("FSDP direct all-gather: %d direct, %d stock; stock reasons %s", STATS["direct"], STATS["fallback"], _reasons)
    return _orig_copy_out(all_gather_result, fsdp_params, group)


def apply() -> None:
    _C.foreach_all_gather = foreach_all_gather
    _C.foreach_all_gather_copy_out = foreach_all_gather_copy_out
    _G.foreach_all_gather = foreach_all_gather
    _G.foreach_all_gather_copy_out = foreach_all_gather_copy_out
    logger.info("FSDP direct all-gather patch installed")


if ENABLED:
    apply()
