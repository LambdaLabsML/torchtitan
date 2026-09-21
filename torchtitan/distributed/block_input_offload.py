"""Offload FullAC block inputs to pinned host memory (TORCHTITAN_BLOCK_INPUT_OFFLOAD=1).

Under non-reentrant FullAC the only activation kept per decoder block is its
input, but on DSv4 the residual stream is four streams wide: [T, 4, 4096]
bf16 is 268 MB per block per 8k sequence, 11.5 GiB per sequence over 43
layers -- 60% of the ~19.4 GiB each extra sequence of microbatch costs on
this recipe. This wrapper releases that storage during the forward and
restores it one layer ahead of use in the backward:

* forward: run the (FSDP-wrapped, checkpointed) block; copy the input to a
  persistent pinned buffer on a D2H stream; at the NEXT block's forward, once
  that copy has landed, free the input's GPU storage (``resize_(0)`` on the
  very tensor the checkpoint closure holds, so the recompute later sees the
  same object).
* backward: a hook on the block output's gradient (which fires before the
  block's recompute needs its input) waits for this layer's input to be
  restored and issues the H2D restore of the previous layer's input, so the
  restore of layer i-1 hides under the recompute + backward of layer i.

Traffic is 0.033 MB per token per block each way (50x less than offloading
whole-block activations), ~1.2 s per direction per step at 9x against a ~13 s
step, and both directions overlap compute. Pinned memory: 43 x 268 MB x
microbatch per rank (115 GB at 10x; nodes have 942 GB).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

ENABLED = os.environ.get("TORCHTITAN_BLOCK_INPUT_OFFLOAD", "0") == "1"


class _Streams:
    d2h = None
    h2d = None

    @classmethod
    def get(cls):
        if cls.d2h is None:
            cls.d2h = torch.cuda.Stream()
            cls.h2d = torch.cuda.Stream()
        return cls.d2h, cls.h2d


class BlockInputOffload(nn.Module):
    """Wraps one decoder block (outermost, after FSDP). ``prev`` links layers."""

    def __init__(self, inner: nn.Module, index: int):
        super().__init__()
        self.inner = inner
        self.index = index
        self.prev: BlockInputOffload | None = None
        self.next: BlockInputOffload | None = None
        self.host = None          # pinned mirror of the current input
        self.x = None             # the GPU input tensor object (storage freed between fwd and bwd)
        self.nbytes = 0
        self.copy_done = None     # D2H event
        self.restore_done = None  # H2D event
        self.freed = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None or name == "inner":
                raise
            return getattr(inner, name)

    # --- forward side ---
    def _offload(self, x: torch.Tensor) -> None:
        d2h, _ = _Streams.get()
        if self.host is None or self.host.numel() != x.numel() or self.host.dtype != x.dtype:
            self.host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        d2h.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(d2h):
            self.host.copy_(x, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(d2h)
        x.record_stream(d2h)
        self.x, self.nbytes, self.copy_done, self.freed = x, x.untyped_storage().nbytes(), ev, False

    def _release_prev(self) -> None:
        p = self.prev
        if p is not None and p.x is not None and not p.freed and p.copy_done is not None:
            torch.cuda.current_stream().wait_event(p.copy_done)
            p.x.untyped_storage().resize_(0)
            p.freed = True

    # --- backward side ---
    def _restore(self) -> None:
        if self.x is None or not self.freed:
            return
        _, h2d = _Streams.get()
        self.x.untyped_storage().resize_(self.nbytes)
        h2d.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(h2d):
            self.x.copy_(self.host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(h2d)
        self.restore_done = ev
        self.freed = False

    def _on_output_grad(self, grad):
        main = torch.cuda.current_stream()
        if self.restore_done is None:
            self._restore()  # not prefetched (e.g. last layer): restore now
        main.wait_event(self.restore_done)
        if self.prev is not None:
            self.prev._restore()  # one layer ahead
        return grad

    def forward(self, x, *args, **kwargs):
        if not (self.training and torch.is_grad_enabled() and x.is_cuda):
            return self.inner(x, *args, **kwargs)
        self._release_prev()
        out = self.inner(x, *args, **kwargs)
        self._offload(x)
        self.restore_done = None
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.register_hook(self._on_output_grad)
        return out


def apply_block_input_offload(model: nn.Module) -> int:
    """Wrap ``model.layers`` children (call after FSDP). Returns the count."""
    layers = getattr(model, "layers", None)
    if layers is None:
        return 0
    wrapped: list[BlockInputOffload] = []
    for key, child in list(layers.named_children()):
        w = BlockInputOffload(child, int(key))
        setattr(layers, key, w)
        wrapped.append(w)
    wrapped.sort(key=lambda w: w.index)
    for a, b in zip(wrapped, wrapped[1:]):
        a.next, b.prev = b, a
    return len(wrapped)
