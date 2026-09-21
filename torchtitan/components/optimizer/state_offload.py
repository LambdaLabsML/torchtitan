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

Knob: ``OPT_STATE_OFFLOAD=1`` (env); ``OPT_STATE_OFFLOAD_CHUNK_GIB`` (default 1);
``OPT_STATE_OFFLOAD_DEBUG=1`` adds finiteness/contiguity checks (synchronizing).

Stream ordering matters here: the H2D stream must wait for main before its
first copy. Without that, its copies could run before the trainer's
asynchronous finiteness assert had executed and, through the caching
allocator, overwrite a just-freed main-stream tensor (jobs 882-890: NaN at
"step 2", clean under CUDA_LAUNCH_BLOCKING=1). Debugmodel with the bookends
matches the native optimizer bitwise over 4 steps (jobs 893/894).
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
            # chunk by cumulative numel within a (param dtype, grad dtype) class: the
            # fused kernel needs uniform dtypes per call (fp32 hc weights live next to
            # bf16 ones here; fp32 params + bf16 moments is its mixed-precision path).
            chunks = []
            by_dtype: dict[tuple, list] = {}
            for p in plist:
                by_dtype.setdefault((_local(p).dtype, _local(p.grad).dtype), []).append(p)
            for group_params in by_dtype.values():
                cur, cur_n = [], 0
                for p in group_params:
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
        n = off.migrate(optim)
        off.migrated = True
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
    off.step(optim)


# --- Layer-wise deferred step, interleaved with the next forward ---------------
# ``OPT_STATE_OFFLOAD_LAYERWISE=1`` (implies OPT_STATE_OFFLOAD=1). The plain
# chunked step above exposes the whole moment transfer (~0.4 s/step at ~89 GB/s)
# because nothing runs inside the optimizer step to hide it. Here ``step()``
# only captures the work (grad references, lr, betas) per decoder layer and
# updates the non-layer parameters (embedding, final norm, head) at once; every
# layer's update -- H2D moments, fused AdamW, D2H write-back -- then runs on
# side streams during the NEXT forward, two layers ahead of use. Each block's
# forward pre-hook (registered with prepend=True, so it runs before FSDP2's own
# pre-forward hook) makes the main stream wait on the update events of its own
# layer and of the next one, because FSDP2's implicit prefetch all-gathers layer
# i+1 during layer i; FSDP2's all-gather stream waits on main, so the gathered
# shards are the updated ones. Grad references are held until the layer's
# update event has been waited on, so ``zero_grad(set_to_none=True)`` at the
# start of the next step cannot free them early.
LAYERWISE = os.environ.get("OPT_STATE_OFFLOAD_LAYERWISE", "0") == "1"
_AHEAD = int(os.environ.get("OPT_STATE_OFFLOAD_AHEAD", "2"))


class _LayerwiseScheduler:
    def __init__(self, container, optim: Optimizer):
        self.optim = optim
        self.off = _offloaders.setdefault(id(optim), _Offloader())
        self.opt_stream = torch.cuda.Stream()
        self.layer_of: dict[int, int] = {}  # id(param) -> layer (-1 = non-layer)
        self.blocks: dict[int, torch.nn.Module] = {}
        import re

        pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
        for model in container.model_parts:
            for name, p in model.named_parameters():
                m = pat.search(name)
                self.layer_of[id(p)] = int(m.group(1)) if m else -1
            layers = getattr(model, "layers", None)
            if layers is not None:
                for key, block in layers.named_children():
                    i = int(key)
                    self.blocks[i] = block
                    block.register_forward_pre_hook(self._make_hook(i), prepend=True)
        self.n_layers = max(self.blocks) + 1 if self.blocks else 0
        self.pending: dict[int, list] = {}
        self.hyper = None
        self.done: dict[int, torch.cuda.Event] = {}
        self.launched: set[int] = set()

    # --- capture at step() ---
    def capture(self):
        optim = self.optim
        for hook in optim._optimizer_step_pre_hooks.values():
            hook(optim, (), {})
        self.pending = {}
        self.hyper = []
        for gi, group in enumerate(optim.param_groups):
            lr = group["lr"]
            lr = float(lr.item()) if torch.is_tensor(lr) else float(lr)
            self.hyper.append((lr, *group["betas"], float(group["weight_decay"]), float(group["eps"])))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = _local(p.grad)
                g.record_stream(self.opt_stream)
                self.pending.setdefault(self.layer_of.get(id(p), -1), []).append((gi, p, g))
        for hook in optim._optimizer_step_post_hooks.values():
            hook(optim, (), {})
        self.done = {}
        self.launched = set()
        main = torch.cuda.current_stream()
        # non-layer params (embedding, norm, head): update now; the root FSDP unit
        # gathers them at the very start of the forward.
        if -1 in self.pending:
            self._launch(-1)
            main.wait_event(self.done[-1])
        for i in range(min(_AHEAD, self.n_layers)):
            self._launch(i)

    def _make_hook(self, i):
        def hook(module, args, kwargs):
            main = torch.cuda.current_stream()
            for j in (i, i + 1):  # own layer, and the one FSDP prefetches during this layer
                if j in self.pending and j not in self.launched:
                    self._launch(j)
                if j in self.done:
                    main.wait_event(self.done[j])
                    self.pending.pop(j, None)  # release grad references
            nxt = i + _AHEAD
            if nxt in self.pending and nxt not in self.launched:
                self._launch(nxt)
        return hook

    # --- the update of one layer on the side streams ---
    def _launch(self, layer):
        items = self.pending.get(layer)
        self.launched.add(layer)
        if not items:
            return
        off, optim = self.off, self.optim
        h2d, d2h, opt = off.h2d, off.d2h, self.opt_stream
        main = torch.cuda.current_stream()
        h2d.wait_stream(main)  # grads and the previous step's kernels are issued on main
        by_class: dict[tuple, list] = {}
        for gi, p, g in items:
            by_class.setdefault((gi, _local(p).dtype, g.dtype), []).append((p, g))
        for (gi, pdtype, gdtype), plist in by_class.items():
            lr, beta1, beta2, wd, eps = self.hyper[gi]
            device = _local(plist[0][0]).device
            sdtype = optim.state[plist[0][0]]["exp_avg"].dtype
            off._ensure_staging(device, sdtype)
            chunks, cur, cur_n = [], [], 0
            for p, g in plist:
                n = _local(p).numel()
                if cur and cur_n + n > _CHUNK_ELEMS:
                    chunks.append(cur); cur, cur_n = [], 0
                cur.append((p, g)); cur_n += n
            if cur:
                chunks.append(cur)
            for k, chunk in enumerate(chunks):
                buf_m, buf_v = off.staging[k % 2]
                total = sum(_local(p).numel() for p, _ in chunk)
                if total > _CHUNK_ELEMS:
                    with torch.cuda.stream(h2d):
                        buf_m = torch.empty(total, device=device, dtype=sdtype)
                        buf_v = torch.empty(total, device=device, dtype=sdtype)
                    buf_m.record_stream(opt); buf_v.record_stream(opt)
                    buf_m.record_stream(d2h); buf_v.record_stream(d2h)
                ev = off.last_d2h_events[k % 2]
                if ev is not None:
                    h2d.wait_event(ev)
                with torch.cuda.stream(h2d):
                    o, views = 0, []
                    for p, _ in chunk:
                        st = optim.state[p]; n = _local(p).numel(); shp = _local(p).shape
                        m = buf_m[o:o + n].view(shp); v = buf_v[o:o + n].view(shp)
                        m.copy_(_local(st["exp_avg"]), non_blocking=True); v.copy_(_local(st["exp_avg_sq"]), non_blocking=True)
                        views.append((m, v)); o += n
                    h2d_done = torch.cuda.Event(); h2d_done.record(h2d)
                opt.wait_event(h2d_done)
                with torch.cuda.stream(opt):
                    steps = [optim.state[p]["step"] for p, _ in chunk]
                    torch._foreach_add_(steps, 1.0)
                    torch._fused_adamw_(
                        [_local(p) for p, _ in chunk], [g for _, g in chunk],
                        [m for m, _ in views], [v for _, v in views], [], steps,
                        lr=lr, beta1=beta1, beta2=beta2, weight_decay=wd, eps=eps,
                        amsgrad=False, maximize=False,
                    )
                    upd = torch.cuda.Event(); upd.record(opt)
                d2h.wait_event(upd)
                with torch.cuda.stream(d2h):
                    for (p, _), (m, v) in zip(chunk, views):
                        st = optim.state[p]
                        _local(st["exp_avg"]).copy_(m, non_blocking=True); _local(st["exp_avg_sq"]).copy_(v, non_blocking=True)
                    ev2 = torch.cuda.Event(); ev2.record(d2h)
                off.last_d2h_events[k % 2] = ev2
        done = torch.cuda.Event(); done.record(opt)
        self.done[layer] = done


_layerwise: dict[int, _LayerwiseScheduler] = {}


def layerwise_step(container, optim: Optimizer) -> None:
    """``offload_step`` variant whose per-layer updates run during the next forward."""
    off = _offloaders.setdefault(id(optim), _Offloader())
    if not off.migrated:
        offload_step(optim)  # first step native + migration
        return
    sched = _layerwise.get(id(optim))
    if sched is None:
        sched = _layerwise[id(optim)] = _LayerwiseScheduler(container, optim)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            import logging

            logging.getLogger(__name__).info(
                "optimizer state offload: layer-wise deferred step over %d layers, %d ahead", sched.n_layers, _AHEAD
            )
    sched.capture()
