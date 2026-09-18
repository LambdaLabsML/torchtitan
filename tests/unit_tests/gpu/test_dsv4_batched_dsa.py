# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The batched DSA path must equal independent single-sequence runs.

``DSV4FlexInnerAttention`` with ``seq_len = L`` splits a ``[B*L]`` token stream into
B independent sequences. The contract this pins down is exactly that: running
B sequences together produces, for every sequence, what running that sequence
alone produces -- outputs and gradients. That is the property the old folded
path violated (a query could select keys from an earlier document, and the
compressor's overlap fed sequence n's first group from sequence n-1's tail),
and it is what makes a microbatch > 1 meaningful rather than merely affordable.

Tolerances: not bitwise. The batched kernel reduces over a per-sequence KV
stream of length ``L + n_cmp`` while the single-sequence run uses the same
length, so the flex block geometry matches -- but Inductor's chosen tile
schedule and the bf16 rounding of P/dS differ once the batch dim is present.
The bound is therefore relative to the single-sequence result's own magnitude.
"""

import unittest
from functools import partial

import torch

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.compressor import Compressor

_HAS_GPU = torch.cuda.is_available()

# Required on GB300 at head_dim=512 (see _pin_gb300_flex_tiles); harmless here.
_TILES = {
    "BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4,
    "BLOCK_M1": 16, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 16,
}
HEAD_DIM, WINDOW, RATIO, TOPK, IDX_D = 128, 64, 4, 32, 64


def _cfg(cls, *, seq_len, ratio=RATIO):
    return cls.Config(
        block_size=32,
        kernel_options=dict(_TILES),
        window_size=WINDOW,
        compress_ratio=ratio,
        softmax_scale=HEAD_DIM**-0.5,
        index_topk=TOPK,
        seq_len=seq_len,
    )


def _rel(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()


@unittest.skipUnless(_HAS_GPU, "requires a CUDA device")
class TestBatchedDSAMatchesSingleSequence(unittest.TestCase):
    def _inputs(self, *, bsz, seqlen, n_heads, n_idx_heads, seed, ratio):
        g = torch.Generator(device="cuda").manual_seed(seed)
        n_cmp = seqlen // ratio
        T = bsz * seqlen

        def r(*shape, std=1.0, grad=False):
            t = (torch.randn(*shape, generator=g, device="cuda") * std).to(torch.bfloat16)
            return t.requires_grad_(grad)

        return dict(
            q=r(T, n_heads, HEAD_DIM, grad=True),
            swa_k=r(T, HEAD_DIM, grad=True),
            cmp_k=r(bsz * n_cmp, HEAD_DIM, grad=True),
            idx_q=r(T, n_idx_heads, IDX_D),
            idx_k=r(bsz * n_cmp, IDX_D),
            idx_w=r(T, n_idx_heads, std=0.1),
            attn_sink=r(n_heads, std=0.5, grad=True),
            cot=r(T, n_heads, HEAD_DIM),
        )

    def _slice(self, inp, b, *, seqlen, n_cmp):
        """Sequence b of a packed batch, as its own single-sequence input."""
        out = {}
        for k, v in inp.items():
            if k == "attn_sink":
                sl = v
            elif k in ("cmp_k", "idx_k"):
                sl = v[b * n_cmp : (b + 1) * n_cmp]
            else:
                sl = v[b * seqlen : (b + 1) * seqlen]
            out[k] = sl.detach().clone().requires_grad_(v.requires_grad)
        return out

    def _run(self, module, inp, keys):
        out = module(*[inp[k] for k in keys])
        (out.float() * inp["cot"].float()).sum().backward()
        grads = {k: inp[k].grad for k in inp if inp[k].requires_grad}
        return out.detach(), grads

    def _check(self, cls, keys, *, bsz, seqlen, n_heads, n_idx_heads, seed, ratio=RATIO):
        n_cmp = seqlen // ratio
        batched = _cfg(cls, seq_len=seqlen, ratio=ratio).build().cuda()
        single = _cfg(cls, seq_len=seqlen, ratio=ratio).build().cuda()

        inp = self._inputs(bsz=bsz, seqlen=seqlen, n_heads=n_heads,
                           n_idx_heads=n_idx_heads, seed=seed, ratio=ratio)
        out_b, grad_b = self._run(batched, inp, keys)

        print(f"\n  {cls.__name__} B={bsz} L={seqlen} H={n_heads} ratio={ratio}")
        for b in range(bsz):
            sub = self._slice(inp, b, seqlen=seqlen, n_cmp=n_cmp)
            out_s, grad_s = self._run(single, sub, keys)
            sl = slice(b * seqlen, (b + 1) * seqlen)
            csl = slice(b * n_cmp, (b + 1) * n_cmp)

            pairs = [("out", out_b[sl], out_s)]
            for name, gb, gs in (
                ("dq", grad_b.get("q"), grad_s.get("q")),
                ("dswa_k", grad_b.get("swa_k"), grad_s.get("swa_k")),
            ):
                if gb is not None:
                    pairs.append((name, gb[sl], gs))
            if grad_b.get("cmp_k") is not None:
                pairs.append(("dcmp_k", grad_b["cmp_k"][csl], grad_s["cmp_k"]))

            for name, a, s_ in pairs:
                err = _rel(a, s_)
                print(f"    seq {b} {name:8s} rel-err {err:.3e}")
                self.assertLess(
                    err, 2e-2, f"{cls.__name__} seq {b} {name}: batched vs single {err:.3e}"
                )
            # the sink gradient is summed over all sequences, so only check it is live
            if grad_b.get("attn_sink") is not None:
                self.assertGreater(grad_b["attn_sink"].abs().max().item(), 0.0)

    def test_csa_batch2(self):
        self._check(
            CompressedSparseAttention,
            ("q", "swa_k", "cmp_k", "idx_q", "idx_k", "idx_w", "attn_sink"),
            bsz=2, seqlen=512, n_heads=4, n_idx_heads=4, seed=1,
        )

    def test_csa_batch3_longer(self):
        self._check(
            CompressedSparseAttention,
            ("q", "swa_k", "cmp_k", "idx_q", "idx_k", "idx_w", "attn_sink"),
            bsz=3, seqlen=256, n_heads=2, n_idx_heads=4, seed=2,
        )

    def test_sliding_window_batch2(self):
        self._check(
            SlidingWindowAttention, ("q", "swa_k", "attn_sink"),
            bsz=2, seqlen=512, n_heads=4, n_idx_heads=4, seed=3, ratio=1,
        )

    def test_heavily_compressed_batch2(self):
        self._check(
            HeavilyCompressedAttention, ("q", "swa_k", "cmp_k", "attn_sink"),
            bsz=2, seqlen=512, n_heads=4, n_idx_heads=4, seed=4, ratio=8,
        )

    def test_batch1_is_the_folded_path(self):
        """seq_len set but only one sequence present must take the old path."""
        m = _cfg(CompressedSparseAttention, seq_len=512).build().cuda()
        self.assertEqual(m._batch_shape(512), (1, 512))
        self.assertEqual(m._batch_shape(1536), (3, 512))
        m0 = _cfg(CompressedSparseAttention, seq_len=0).build().cuda()
        self.assertEqual(m0._batch_shape(1536), (1, 1536))
        with self.assertRaises(ValueError):
            m._batch_shape(700)


@unittest.skipUnless(_HAS_GPU, "requires a CUDA device")
class TestCompressorOverlapIsPerSequence(unittest.TestCase):
    """The compressor's overlap must not feed sequence n from sequence n-1."""

    def _compressor(self, seq_len):
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.common.nn_modules import RMSNorm
        from torchtitan.models.common.rope import ComplexRoPE

        dim, hd, rd = 64, HEAD_DIM, 32
        cfg = Compressor.Config(
            rope=ComplexRoPE.Config(dim=rd, max_context_length=4096, theta=1e4, scaling="none"),
            wkv=Linear.Config(in_features=dim, out_features=2 * hd, bias=False),
            wgate=Linear.Config(in_features=dim, out_features=2 * hd, bias=False),
            norm=RMSNorm.Config(normalized_shape=hd),
            head_dim=hd, rope_head_dim=rd, compress_ratio=RATIO, seq_len=seq_len,
            param_init={"ape": partial(torch.nn.init.trunc_normal_, std=0.02)},
        )
        with torch.device("cuda"):
            m = cfg.build()
        torch.manual_seed(0)
        m.init_states(buffer_device=torch.device("cuda"))
        for p in m.parameters():
            if not torch.isfinite(p).all():
                with torch.no_grad():
                    p.normal_(0, 0.02)
        return m.cuda()

    def test_per_sequence_overlap(self):
        L, B, dim = 256, 2, 64
        m = self._compressor(L)
        torch.manual_seed(1)
        x = torch.randn(B * L, dim, device="cuda").to(torch.bfloat16)
        pos = torch.arange(L, device="cuda").repeat(B)
        out = m(x, positions=pos)
        for b in range(B):
            sub = m(x[b * L : (b + 1) * L], positions=pos[b * L : (b + 1) * L])
            err = _rel(out[b * (L // RATIO) : (b + 1) * (L // RATIO)], sub)
            print(f"\n  compressor seq {b}: rel-err vs standalone {err:.3e}")
            self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
