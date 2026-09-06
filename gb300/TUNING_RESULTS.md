# GPT-OSS-20B tuning sweep — 64x GB300

All runs: 16 nodes x 4 GB300, `seq_len=8192`, FSDP over all 64 ranks,
`expert_parallel_degree=1`, AdamW lr 8e-4, C4 streamed from HF,
`--training.disable_cuda_graphs`. 250 steps; steady state = steps >= 100.

Peak is GB300 dense bf16 ~2503 TFLOP/s/GPU, which is what MFU is against.

## Round 1 — measured

| job | config | bs | AC | compile | reshard | TFLOP/s/GPU | cluster | MFU | mem | vs base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46 | `gpt_oss_20b` (baseline) | 1 | Full | - | default | 281.4 | 18.01 PF | 11.3% | 6.7% | — |
| 46 | `..._compile_loss_only` | 1 | Full | loss | default | 282.1 | 18.05 PF | 11.3% | 6.7% | +0.2% |
| 46 | `..._noreshard` | 1 | Full | - | never | 322.2 | 20.62 PF | 12.9% | 20.4% | +14.5% |
| 46 | `..._noac` | 4 | none | - | default | 569.2 | 36.43 PF | 22.8% | 86.5% | +102% |
| 46 | `..._noac_compile_noreshard` | 4 | none | full | never | 917.0 | 58.68 PF | 36.7% | 66.3% | +226% |
| 46 | **`..._noac_compile`** | 6 | none | full | default | **942.3** | **60.30 PF** | **37.7%** | 77.0% | **+235%** |
| 46 | `..._noac_compile_maxbs` | 8 | none | full | default | 160.6 | 10.28 PF | 6.4% | 98.7% | **-43%** |

Best: **942.3 TFLOP/s/GPU, 60.30 PFLOP/s across 64 GPUs, 37.7% MFU** — 3.35x the
out-of-the-box baseline.

### Blocked

| config | why |
| --- | --- |
| `..._compile` | `AssertionError: Node add_21 was invalid, but is output` in AOTAutograd's `min_cut_rematerialization_partition`. FullAC + compiling the *model* is the broken pair, not compile itself — both escapes (drop AC, or compile only the loss) work. |
| `..._mxfp8`, `..._noac_compile_noreshard_mxfp8` | torchao publishes CUDA kernels for **x86_64 only**; the aarch64 wheel is `py3-none-any` with no compiled `.so`, so SM100 kernels are absent. No aarch64 build on cu128/cu129/cu130 or either nightly index. `KernelPreference.EMULATED` would run but measures emulation, not mxfp8. |

## What the numbers say

**Memory utilisation is the whole game — until it isn't.** The baseline recomputes
activations while sitting in 6.7% of the GPU. Spending that headroom is worth
2-3.3x. But bs=8 at 98.7% is **43% slower than the baseline** and 5.9x slower than
bs=6 at 77%: the allocator thrashes, and wall clock shows it (~20 min for that one
config vs ~5-11 for the others). There is a cliff between 77% and 98.7%.

**compile on its own is worth nothing** (+0.2%). Its entire value is second-order:
it cuts activation memory enough to raise the no-AC batch from 4 to 6, and *that*
is worth +65% on top of no-AC alone (569 -> 942). Judging compile by
`compile_loss_only` would have led straight to the wrong conclusion.

**Dropping AC at the stock batch is a regression.** 248 TFLOP/s at bs=1 vs 281.4
for the baseline (6-step probe). No-AC only pays once the freed memory is spent.
The two halves are not separable, which is why there is no `noac`-at-bs=1 row.

**never-reshard is quietly efficient.** At bs=4 it reaches 917.0 in 66.3% of
memory — within 3% of the best result while using 11 points less memory. It buys
back most of two steps of batch. Nothing has yet tried it at bs=6.

## Round 2 — configs added, not yet run

Chosen from the above.

| config | hypothesis |
| --- | --- |
| `..._noac_compile_bs7` | Bisects the cliff (~88% mem). 942 at bs=6, 161 at bs=8 — nothing pins where the edge is, or whether 942 is the peak. |
| `..._noac_compile_noreshard_bs6` | never-reshard reached 917 at bs=4/66.3%. That spare memory is the point; it has never been tried at bs=6. Highest upside of the four. |
| `..._sac_compile` | Round 1 measured only the ends of the AC spectrum. Per-op SelectiveAC stores strictly less than no-AC, so it should buy batch without full-AC's recompute bill. bs=8 is an **estimate**, not measured. |
| `..._noac_compile_hsdp` | Shard 4 within-node, replicate 16 across. Every config so far all-gathers across the whole rack; 20B is small for 64-way sharding, so the collective is likely latency-bound — the regime where narrowing the shard group wins. never-reshard helping is evidence communication matters. |

`MemoryBudgetAC` was considered and dropped: it overlaps SelectiveAC's space, and
the never-reshard result was stronger evidence for where to spend a slot.

## Supporting jobs

| job | purpose | outcome |
| --- | --- | --- |
| 38 | baseline, `log_freq=1` | 277.4 — measurement artifact, superseded |
| 39 | baseline, stock defaults | 287.8 (50 steps) |
| 40 | attribution: `log_freq=10`, `OMP=8` | 287.0 — isolates `log_freq` as the cause |
| 41 | no-AC batch probe | bs=4 fits (86.5%), 6 and 8 OOM |
| 42 | validate 6 configs | compile fails; mxfp8 missing `Python.h` |
| 43 | mxfp8 + compile retry | mxfp8 -> SM100 kernels absent; compile fails again |
| 44 | compile isolation | `noac_compile` and `compile_loss_only` both work |
| 45 | no-AC+compile batch probe | bs=6 77%, bs=8 98.8%, bs=10 OOM |
| 46 | **the sweep** | 7 configs x 250 steps |

Baseline is quoted as 281.4 (job 46, 250 steps) rather than 287.8 (job 39, 50
steps) so every row in the round-1 table is measured identically.
