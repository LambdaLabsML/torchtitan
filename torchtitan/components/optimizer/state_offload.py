"""Chunked offload of Adam moments to pinned host memory (Megatron #6244's idea).

Why: on the GB300 DSv4-flash recipe the two bf16 Adam moments are ~33 GiB per
rank of the ~66 GiB static footprint; parameters and gradients are bf16 too, so
the moments are the whole offloadable optimizer state. Each extra sequence of
microbatch is worth ~+2.5% and ~19 GiB, and 7x sits at 238 GiB of 276.

How: the first step runs torch's fused AdamW normally (the bf16 state hook
creates the moments on the GPU); the moments are then moved to pinned host
tensors and the GPU copies freed. Every later step walks the parameters in
chunks of ~CHUNK_ELEMS elements through two GPU staging buffers on three
streams: H2D copy of chunk k+1 overlaps the fused AdamW of chunk k, whose D2H
write-back overlaps chunk k+1. The write-back of the last chunks drains into
the next forward. Registered step pre-/post-hooks (bf16 state init, MoE
expert-bias update) still run. State dict readers see CPU tensors.

Knob: ``OPT_STATE_OFFLOAD=1`` (env); ``OPT_STATE_OFFLOAD_CHUNK_GIB`` (default 1).
"""

from __future__ import annotations

import os

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

ENABLED = os.environ.get("OPT_STATE_OFFLOAD", "0") == "1"
_DEBUG = os.environ.get("OPT_STATE_OFFLOAD_DEBUG", "0") == "1"
_CHUNK_ELEMS = int(float(os.environ.get("OPT_STATE_OFFLOAD_CHUNK_GIB", "1")) * 2**30 // 2)  # bf16 elems per moment


def _local(t):
    return t._local_tensor if isinstance(t, DTensor) else t


class _Offloader:
    def __init__(self):
        self.h2d = torch.cuda.Stream()
        self.d2h = torch.cuda.Stream()
        self.staging = None  # 2 x (exp_avg buf, exp_avg_sq buf)
        self.migrated = False
        self.last_d2h_events = [None, None]

    def _ensure_staging(self, device, dtype):
        if self.staging is None:
            self.staging = [
                (torch.empty(_CHUNK_ELEMS, device=device, dtype=dtype), torch.empty(_CHUNK_ELEMS, device=device, dtype=dtype))
                for _ in range(2)
            ]

    def migrate(self, optim: Optimizer) -> int:
        """Move exp_avg/exp_avg_sq of every param to pinned host memory."""
        n = 0
        for group in optim.param_groups:
            for p in group["params"]:
                st = optim.state.get(p)
                if not st or "exp_avg" not in st or not st["exp_avg"].is_cuda:
                    continue
                for k in ("exp_avg", "exp_avg_sq"):
                    src = _local(st[k])
                    host = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
                    host.copy_(src)
                    st[k] = host
                n += 1
        torch.cuda.synchronize()
        return n

    def step(self, optim: Optimizer) -> None:
        for hook in optim._optimizer_step_pre_hooks.values():
            hook(optim, (), {})
        for group in optim.param_groups:
            plist = [p for p in group["params"] if p.grad is not None]
            if not plist:
                continue
            lr = group["lr"]
            lr = float(lr.item()) if torch.is_tensor(lr) else float(lr)
            beta1, beta2 = group["betas"]
            wd, eps = float(group["weight_decay"]), float(group["eps"])
            assert not group.get("amsgrad", False), "amsgrad not supported with state offload"
            device = _local(plist[0]).device
            dtype = optim.state[plist[0]]["exp_avg"].dtype
            self._ensure_staging(device, dtype)
            # chunk by cumulative numel (a single huge param gets its own chunk)
            chunks, cur, cur_n = [], [], 0
            for p in plist:
                n = _local(p).numel()
                if cur and cur_n + n > _CHUNK_ELEMS:
                    chunks.append(cur); cur, cur_n = [], 0
                cur.append(p); cur_n += n
            if cur:
                chunks.append(cur)
            main = torch.cuda.current_stream()
            # Order the side streams after everything main has issued so far (grads,
            # the trainer's async finiteness assert, the previous step's kernels) and
            # after the previous step's write-backs before anything is reused.
            self.h2d.wait_stream(main)
            self.h2d.wait_stream(self.d2h)
            for k, chunk in enumerate(chunks):
                buf_m, buf_v = self.staging[k % 2]
                total = sum(_local(p).numel() for p in chunk)
                if total > _CHUNK_ELEMS:  # oversized single param: dedicated temp buffers
                    with torch.cuda.stream(self.h2d):  # allocate on the stream that writes first
                        buf_m = torch.empty(total, device=device, dtype=dtype)
                        buf_v = torch.empty(total, device=device, dtype=dtype)
                    buf_m.record_stream(main); buf_v.record_stream(main)
                    buf_m.record_stream(self.d2h); buf_v.record_stream(self.d2h)
                # the staging pair is free once chunk k-2's write-back finished
                if self.last_d2h_events[k % 2] is not None:
                    self.h2d.wait_event(self.last_d2h_events[k % 2])
                with torch.cuda.stream(self.h2d):
                    off = 0
                    views = []
                    for p in chunk:
                        st = optim.state[p]; n = _local(p).numel()
                        shp = _local(p).shape
                        m = buf_m[off:off + n].view(shp); v = buf_v[off:off + n].view(shp)
                        m.copy_(_local(st["exp_avg"]), non_blocking=True); v.copy_(_local(st["exp_avg_sq"]), non_blocking=True)
                        views.append((m, v)); off += n
                h2d_done = torch.cuda.Event(); h2d_done.record(self.h2d)
                main.wait_event(h2d_done)
                params = [_local(p) for p in chunk]
                grads = [_local(p.grad) for p in chunk]
                steps = [optim.state[p]["step"] for p in chunk]
                if _DEBUG:
                    for p_, g_ in zip(params, grads):
                        assert p_.is_contiguous() and g_.is_contiguous(), (p_.shape, p_.stride(), g_.stride())
                        assert p_.dtype == g_.dtype == dtype, (p_.dtype, g_.dtype, dtype)
                # torch.optim's fused path increments the step counters BEFORE the kernel
                torch._foreach_add_(steps, 1.0)
                torch._fused_adamw_(
                    params, grads, [m for m, _ in views], [v for _, v in views], [],
                    steps,
                    lr=lr, beta1=beta1, beta2=beta2, weight_decay=wd, eps=eps,
                    amsgrad=False, maximize=False,
                )
                if _DEBUG:
                    torch.cuda.synchronize()
                    bad = [(tuple(p_.shape), str(p_.dtype)) for p_, (m, v) in zip(params, views)
                           if not (torch.isfinite(p_).all() and torch.isfinite(m).all() and torch.isfinite(v).all())]
                    if bad:
                        import logging
                        logging.getLogger(__name__).error("state offload: non-finite after chunk %d: %s (lr=%s steps=%s)", k, bad[:4], lr, steps[0].item())
                upd = torch.cuda.Event(); upd.record(main)
                self.d2h.wait_event(upd)
                with torch.cuda.stream(self.d2h):
                    for p, (m, v) in zip(chunk, views):
                        st = optim.state[p]
                        _local(st["exp_avg"]).copy_(m, non_blocking=True); _local(st["exp_avg_sq"]).copy_(v, non_blocking=True)
                ev = torch.cuda.Event(); ev.record(self.d2h)
                self.last_d2h_events[k % 2] = ev
            # main must not run ahead into work that could free/reuse anything the
            # write-back still reads; the copies themselves keep draining.
            main.wait_stream(self.h2d)
        for hook in optim._optimizer_step_post_hooks.values():
            hook(optim, (), {})


_offloaders: dict[int, _Offloader] = {}


def offload_step(optim: Optimizer) -> None:
    """Drop-in for ``optim.step()`` with the moments kept on the host between steps."""
    off = _offloaders.get(id(optim))
    if off is None:
        off = _offloaders[id(optim)] = _Offloader()
    if not off.migrated:
        optim.step()  # first step: fused AdamW creates the (bf16) moments on the GPU
        if os.environ.get("OPT_STATE_OFFLOAD_NO_MIGRATE", "0") == "1":
            n = 0  # bisect aid: keep moments on the GPU, still use the chunked step (D2D copies)
        else:
            n = off.migrate(optim)
        off.migrated = True
        if os.environ.get("OPT_STATE_OFFLOAD_NO_EMPTY_CACHE", "0") != "1":
            torch.cuda.empty_cache()
        if _DEBUG:
            torch.cuda.synchronize()
            badp = [tuple(p.shape) for g in optim.param_groups for p in g["params"] if not torch.isfinite(_local(p)).all()]
            import logging
            logging.getLogger(__name__).info("state offload debug: after first step+migrate, non-finite params: %s", badp[:5])
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            import logging

            logging.getLogger(__name__).info(
                "optimizer state offload: moved %d params' Adam moments to pinned host memory", n
            )
        return
    mode = os.environ.get("OPT_STATE_OFFLOAD_BISECT", "")
    if mode == "native":  # wrapper only: native step every step
        optim.step()
        return
    if mode == "roundtrip":  # bring moments back to the GPU, native step, re-offload
        for g in optim.param_groups:
            for p in g["params"]:
                st = optim.state.get(p)
                if st and "exp_avg" in st and not st["exp_avg"].is_cuda:
                    st["exp_avg"] = st["exp_avg"].to(_local(p).device); st["exp_avg_sq"] = st["exp_avg_sq"].to(_local(p).device)
        optim.step()
        off.migrate(optim)
        return
    off.step(optim)
