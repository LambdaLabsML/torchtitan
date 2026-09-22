"""FSDP2 direct reduce-scatter (FSDP_DIRECT_REDUCE_SCATTER=1).

FSDP2 stages every reduce-scatter: it allocates a flat input buffer and
``chunk_cat``s the unsharded gradients into it *on the compute stream* before
the reduce-scatter stream takes over. For a group holding one parameter sharded
on dim 0 with no padding and no dtype change, the gradient already is that
input layout (chunk r = rows of shard r), so the gradient's flat view can be
the collective's input directly: no staging buffer (6.4 GB per expert layer at
DSv4-flash), no copy (1.8% of the step on the critical path). Twin of
``fsdp_direct_gather.py``; stock path for every other group.
"""

from __future__ import annotations

import logging
import os

import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _C
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _G

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("FSDP_DIRECT_REDUCE_SCATTER", "0") == "1"
STATS = {"direct": 0, "stock": 0}
_orig_reduce = _C.foreach_reduce
_orig_copy_in = _C.foreach_reduce_scatter_copy_in
_logged = False


class _DirectInputComm:
    """Proxy over the ReduceScatter comm: the first ``allocate`` (the input
    buffer, sized ``grad.numel()``) returns the gradient's own flat view; every
    other call (the output buffer, the collective itself) is delegated."""

    def __init__(self, inner, grad_flat: torch.Tensor):
        self._inner = inner
        self._grad_flat = grad_flat
        self._served_input = False

    def allocate(self, size, *, dtype, device):
        numel = int(size[0]) if len(size) == 1 else int(torch.Size(size).numel())
        if not self._served_input and numel == self._grad_flat.numel() and dtype == self._grad_flat.dtype:
            self._served_input = True
            return self._grad_flat
        return self._inner.allocate(size, dtype=dtype, device=device)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size):
    if len(unsharded_grads) == 1 and unsharded_grads[0].data_ptr() == reduce_scatter_input.data_ptr():
        return  # the input *is* the gradient: nothing to copy
    return _orig_copy_in(unsharded_grads, reduce_scatter_input, world_size)


def _eligible(fsdp_params, unsharded_grads, group, reduce_dtype) -> bool:
    if len(fsdp_params) != 1 or len(unsharded_grads) != 1 or group is None or group.size() <= 1:
        return False
    p, g = fsdp_params[0], unsharded_grads[0]
    if p.fsdp_placement.dim != 0 or not g.is_contiguous():
        return False
    if (reduce_dtype or g.dtype) != g.dtype:
        return False  # the dtype change happens in the copy-in
    if g.size(0) % group.size() != 0:
        return False  # padded shard
    return True


def foreach_reduce(fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_stream, reduce_scatter_comm, orig_dtype, reduce_dtype, device, gradient_divide_factor, all_reduce_group, all_reduce_stream, all_reduce_grads, partial_reduce_output, all_reduce_hook, force_sum_reduction_for_comms=False):
    global _logged
    if _eligible(fsdp_params, unsharded_grads, reduce_scatter_group, reduce_dtype):
        STATS["direct"] += 1
        comm = _DirectInputComm(reduce_scatter_comm, unsharded_grads[0].view(-1))
    else:
        STATS["stock"] += 1
        comm = reduce_scatter_comm
    if not _logged and STATS["direct"] + STATS["stock"] >= 130:
        _logged = True
        logger.info("FSDP direct reduce-scatter: %d direct, %d stock", STATS["direct"], STATS["stock"])
    return _orig_reduce(fsdp_params, unsharded_grads, reduce_scatter_group, reduce_scatter_stream, comm, orig_dtype, reduce_dtype, device, gradient_divide_factor, all_reduce_group, all_reduce_stream, all_reduce_grads, partial_reduce_output, all_reduce_hook, force_sum_reduction_for_comms)


def apply() -> None:
    _C.foreach_reduce_scatter_copy_in = foreach_reduce_scatter_copy_in
    _C.foreach_reduce = foreach_reduce
    _G.foreach_reduce = foreach_reduce
    logger.info("FSDP direct reduce-scatter patch installed")


if ENABLED:
    apply()
