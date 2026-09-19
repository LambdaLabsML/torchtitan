# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""cuDNN's fused DSA backward against the FlexAttention backward.

The contract is not "the two agree to some tolerance" -- both round P and dS
to bf16 inside their kernels, so they legitimately differ by their own error.
It is the same contract used for the gather-CSA work earlier: measure BOTH
against an fp64 reference built from the same bf16 inputs and the same
selection, and require the fused kernel to be no worse than the flex backward
it replaces.

The forward is shared (flex computes it either way), so ``out`` must match
exactly; only the gradients are at issue.
"""

import unittest

import torch

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.cudnn_dsa import cudnn_dsa_available

_HAS_GPU = torch.cuda.is_available()
_HAS_CUDNN_DSA = _HAS_GPU and cudnn_dsa_available()

# Required on GB300 at head_dim=512; also what production pins.
_TILES = {
    "BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4,
    "BLOCK_M1": 16, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 16,
}
HEAD_DIM, WINDOW, RATIO, TOPK, IDX_D = 512, 128, 4, 64, 128
GRADS = ("q", "swa_k", "cmp_k", "attn_sink")


def _cfg(fused: bool, seq_len: int, fused_fwd: bool = False):
    return CompressedSparseAttention.Config(
        block_size=32,
        kernel_options=dict(_TILES),
        window_size=WINDOW,
        compress_ratio=RATIO,
        softmax_scale=HEAD_DIM**-0.5,
        index_topk=TOPK,
        seq_len=seq_len,
        max_autotune=False,
        fused_dsa_backward=fused,
        fused_dsa_forward=fused_fwd,
    )


def _inputs(seqlen, n_heads, n_idx_heads, seed, microbatch=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_cmp = microbatch * (seqlen // RATIO)
    tokens = microbatch * seqlen

    def r(*shape, std=1.0, grad=False):
        t = (torch.randn(*shape, generator=g, device="cuda") * std).to(torch.bfloat16)
        return t.requires_grad_(grad)

    return dict(
        q=r(tokens, n_heads, HEAD_DIM, grad=True),
        swa_k=r(tokens, HEAD_DIM, grad=True),
        cmp_k=r(n_cmp, HEAD_DIM, grad=True),
        idx_q=r(tokens, n_idx_heads, IDX_D),
        idx_k=r(n_cmp, IDX_D),
        idx_w=r(tokens, n_idx_heads, std=0.1),
        attn_sink=r(n_heads, std=0.5, grad=True),
        cot=r(tokens, n_heads, HEAD_DIM),
    )


def _clone(inp):
    out = {}
    for k, v in inp.items():
        c = v.detach().clone()
        out[k] = c.requires_grad_(True) if v.requires_grad else c
    return out


def _run(module, inp):
    out = module(inp["q"], inp["swa_k"], inp["cmp_k"], inp["idx_q"], inp["idx_k"],
                 inp["idx_w"], inp["attn_sink"])
    (out.float() * inp["cot"].float()).sum().backward()
    return out.detach(), {k: inp[k].grad.detach() for k in GRADS}


def _reference(module, inp, seqlen, microbatch=1):
    """fp64 attention over exactly the positions the module selects.

    Written against the flattened stream so one code path covers both the
    single-sequence and the packed-batch layouts: sequence b's KV occupies its
    own ``kv_len`` slice, which is the isolation the batched path guarantees.
    """
    from torchtitan.models.deepseek_v4.cudnn_dsa import flatten_batched_indices

    q, swa_k, cmp_k, sink = inp["q"], inp["swa_k"], inp["cmp_k"], inp["attn_sink"]
    bsz = microbatch
    n_cmp = cmp_k.size(0) // bsz
    kv_len = seqlen + n_cmp
    # selected_kv_indices takes the batched views, as the module's forward does
    idx_q, idx_k, idx_w = inp["idx_q"], inp["idx_k"], inp["idx_w"]
    if bsz > 1:
        idx_q = idx_q.view(bsz, seqlen, idx_q.size(1), idx_q.size(2))
        idx_k = idx_k.view(bsz, n_cmp, idx_k.size(1))
        idx_w = idx_w.view(bsz, seqlen, idx_w.size(1))
    with torch.no_grad():
        idx = module.selected_kv_indices(
            bsz=bsz, seqlen=seqlen, n_cmp=n_cmp, idx_q=idx_q,
            idx_k=idx_k, idx_w=idx_w, device=q.device,
        )
        idx = flatten_batched_indices(idx, kv_len) if bsz > 1 else idx[0]
    kv = torch.cat(
        [swa_k.view(bsz, seqlen, HEAD_DIM), cmp_k.view(bsz, n_cmp, HEAD_DIM)], dim=1
    ).reshape(bsz * kv_len, HEAD_DIM).double()          # differentiable
    qd = q.double()
    sel = idx.clamp_min(0)
    gathered = kv[sel]                                  # [T, K, D]
    s = torch.einsum("thd,tkd->thk", qd, gathered) * module.softmax_scale
    s = s.masked_fill(~(idx >= 0).unsqueeze(1), float("-inf"))
    s = torch.cat([s, sink.double().view(1, -1, 1).expand(s.size(0), s.size(1), 1)], -1)
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("thk,tkd->thd", p[..., :-1], gathered)
    (out * inp["cot"].double()).sum().backward()
    return out.detach(), {k: inp[k].grad.detach() for k in GRADS}


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()


@unittest.skipUnless(_HAS_CUDNN_DSA, "requires a GPU with cuDNN's DSA kernel")
class TestCudnnDsaBackward(unittest.TestCase):
    def _check(self, seqlen, n_heads, n_idx_heads, seed, microbatch=1, fused_fwd=False):
        flex = _cfg(False, seqlen).build().cuda()
        fused = _cfg(True, seqlen, fused_fwd).build().cuda()

        base = _inputs(seqlen, n_heads, n_idx_heads, seed, microbatch)
        i_flex, i_fused, i_ref = _clone(base), _clone(base), _clone(base)
        out_flex, g_flex = _run(flex, i_flex)
        out_fused, g_fused = _run(fused, i_fused)
        out_ref, g_ref = _reference(flex, i_ref, seqlen, microbatch)

        print(f"\n  T={seqlen} H={n_heads} D={HEAD_DIM} topk={TOPK} "
              f"seed={seed} microbatch={microbatch}")
        print(f"    {'tensor':<10} {'flex-vs-fp64':>14} {'cudnn-vs-fp64':>15} {'cudnn-vs-flex':>15}")
        if fused_fwd:
            # A different forward kernel: hold it to the fp64 reference the
            # flex forward is held to, rather than to bitwise equality.
            rf_o, rc_o = _rel(out_flex, out_ref), _rel(out_fused, out_ref)
            print(f"    {'out':<10} {rf_o:>14.3e} {rc_o:>15.3e} "
                  f"{_rel(out_fused, out_flex):>15.3e}")
            self.assertLessEqual(rc_o, max(1.5 * rf_o, 1e-3), "forward too far from fp64")
        else:
            # the forward is shared, so outputs must be identical
            self.assertTrue(torch.equal(out_flex, out_fused), "forward differs")
        for k in GRADS:
            rf, rc = _rel(g_flex[k], g_ref[k]), _rel(g_fused[k], g_ref[k])
            print(f"    {k:<10} {rf:>14.3e} {rc:>15.3e} {_rel(g_fused[k], g_flex[k]):>15.3e}")
            # No worse than the backward it replaces. attn_sink gets more slack:
            # it is one value per head reduced over every token, so it
            # accumulates tie-breaking noise the per-token grads do not.
            slack, floor = (2.0, 2e-3) if k == "attn_sink" else (1.5, 1e-3)
            self.assertLessEqual(
                rc, max(slack * rf, floor),
                f"{k}: cudnn rel-err {rc:.3e} worse than flex {rf:.3e}",
            )

    def test_csa_small(self):
        self._check(seqlen=512, n_heads=64, n_idx_heads=8, seed=1)

    def test_csa_longer(self):
        self._check(seqlen=1024, n_heads=64, n_idx_heads=8, seed=2)

    def test_csa_packed_batch(self):
        # Two packed sequences per rank. The kernel is flat and unbatched, so
        # this is the case where the [B, kv_len, D] KV stream has to be
        # flattened and each sequence's indices offset into its own slice.
        self._check(seqlen=512, n_heads=64, n_idx_heads=8, seed=3, microbatch=2)

    def test_csa_packed_batch_4x(self):
        self._check(seqlen=512, n_heads=64, n_idx_heads=8, seed=4, microbatch=4)

    def _check_variant(self, cls, ratio, seqlen, seed, microbatch=1):
        """HCA and SWA share ``_forward_impl``, so the fused path changes them
        too -- 22 of the model's 43 attention layers. Neither uses the indexer:
        the selection branches on compress_ratio (128 -> fixed compressed
        blocks, 1 -> window only), so they exercise index shapes the
        CompressedSparseAttention cases never produce.
        """
        def cfg(fused_bwd, fused_fwd):
            return cls.Config(
                block_size=32, kernel_options=dict(_TILES), window_size=WINDOW,
                compress_ratio=ratio, softmax_scale=HEAD_DIM**-0.5,
                index_topk=TOPK, seq_len=seqlen, max_autotune=False,
                fused_dsa_backward=fused_bwd, fused_dsa_forward=fused_fwd,
            )

        tokens = microbatch * seqlen
        n_cmp = microbatch * max(seqlen // ratio, 1) if ratio > 1 else 0
        g = torch.Generator(device="cuda").manual_seed(seed)

        def r(*shape, std=1.0, grad=False):
            t = (torch.randn(*shape, generator=g, device="cuda") * std).to(torch.bfloat16)
            return t.requires_grad_(grad)

        base = dict(
            q=r(tokens, 64, HEAD_DIM, grad=True),
            swa_k=r(tokens, HEAD_DIM, grad=True),
            attn_sink=r(64, std=0.5, grad=True),
            cot=r(tokens, 64, HEAD_DIM),
        )
        if ratio > 1:
            base["cmp_k"] = r(n_cmp, HEAD_DIM, grad=True)

        names = ("q", "swa_k", "attn_sink") + (("cmp_k",) if ratio > 1 else ())

        def run(module, inp):
            args = ([inp["q"], inp["swa_k"]]
                    + ([inp["cmp_k"]] if ratio > 1 else [])
                    + [inp["attn_sink"]])
            out = module(*args)
            (out.float() * inp["cot"].float()).sum().backward()
            return out.detach(), {k: inp[k].grad.detach() for k in names}

        flex = cfg(False, False).build().cuda()
        fused = cfg(True, True).build().cuda()
        out_flex, g_flex = run(flex, _clone(base))
        out_fused, g_fused = run(fused, _clone(base))

        print(f"\n  {cls.__name__} ratio={ratio} T={seqlen} microbatch={microbatch}")
        print(f"    {'tensor':<10} {'cudnn-vs-flex':>15}")
        for k in ("out",) + names:
            a, b = (out_fused, out_flex) if k == "out" else (g_fused[k], g_flex[k])
            d = _rel(a, b)
            print(f"    {k:<10} {d:>15.3e}")
            # Both are bf16 kernels over the same selection; the CSA cases put
            # flex and cuDNN within 4.6e-3 of each other against fp64, so hold
            # these to the same order rather than to bitwise equality.
            self.assertLessEqual(d, 2e-2, f"{cls.__name__} {k} diverges")

    def test_heavily_compressed(self):
        self._check_variant(HeavilyCompressedAttention, 128, 512, seed=7)

    def test_heavily_compressed_packed_batch(self):
        self._check_variant(HeavilyCompressedAttention, 128, 512, seed=8, microbatch=4)

    def test_sliding_window(self):
        self._check_variant(SlidingWindowAttention, 1, 512, seed=9)

    def test_sliding_window_packed_batch(self):
        self._check_variant(SlidingWindowAttention, 1, 512, seed=10, microbatch=4)

    def test_fused_forward(self):
        self._check(seqlen=512, n_heads=64, n_idx_heads=8, seed=5, fused_fwd=True)

    def test_fused_forward_packed_batch(self):
        self._check(seqlen=512, n_heads=64, n_idx_heads=8, seed=6, microbatch=4,
                    fused_fwd=True)


if __name__ == "__main__":
    unittest.main()
