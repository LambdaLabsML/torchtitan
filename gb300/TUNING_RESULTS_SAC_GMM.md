# SelectiveAC + the MoE expert GEMM — 64x GB300

16 nodes x 4 GB300, `seq_len=8192`, FSDP over all 64 ranks,
`expert_parallel_degree=1`, AdamW lr 8e-4, `c4_local`,
`--training.disable_cuda_graphs`. 250 steps, steady state = steps >= 100.

## The bug

`SelectiveAC`'s save set is seeded from torch's
`_functorch.partitioners.get_default_op_list().compute_intensive_ops` — `mm`,
`bmm`, `addmm`, `convolution`, `sdpa*`, `_scaled_mm` — plus torchtitan's own
additions. Grouped mm is in neither list, and the policy sends every unlisted op
to `PREFER_RECOMPUTE`. Checked against the run venv:

```
torch_attn._varlen_attn.default        saved=True
aten.linear.default                    saved=True
aten._grouped_mm.default               saved=False   <-- the expert GEMM
```

So per-op SAC recomputed every expert GEMM in backward while faithfully saving
the far cheaper attention projections. `aten.index` and the pointwise ops are
recomputable too, so **SelectiveAC degenerated to full recompute inside the
MoE** — which is where GPT-OSS spends 54% (20B) / 56% (120B) of its real FLOPs.

That predicted both SAC rows already in the sweep: the low memory was
activations being discarded, and the gap to no-AC was the recompute bill
(+18% FLOPs in theory, 11% observed).

## Measured — GPT-OSS-20B

| job | config | bs | TFLOP/s/GPU | cluster | MFU | mem |
| --- | --- | --- | --- | --- | --- | --- |
| — | `..._sac_compile` (prior) | 8 | 859.6 | 55.01 PF | 34.4% | 27.17% |
| 61 | `..._sac_gmm_bs8` | 8 | **979.5** | 62.69 PF | 39.18% | 61.31% |
| — | `..._noac_compile_bs7_local` (prior best) | 7 | 987.8 | 63.22 PF | 39.5% | 88.90% |
| 63 | same, re-measured this session | 7 | 988.6 | 63.27 PF | 39.54% | 88.91% |
| **65** | **`..._sac_gmm_bs12`** | **12** | **1012.5** | **64.80 PF** | **40.50%** | 89.90% |

All exit 0, 250/250 steps, zero dataloader retries.

**+13.9% over SelectiveAC at an identical batch** (job 61 vs the prior
`sac_compile`), which is far outside the cluster's ~2% noise floor. The gap
between SAC and no-AC was the expert-GEMM recompute, and one op in the save set
removes it.

**Job 63 is the control that makes the rest comparable.** It re-ran the prior
best config from a worktree and reproduced it to within 0.1% (988.6 vs 987.8),
which validates the worktree harness and the session.

**New best is 1012.5 TFLOP/s (job 65), but read it carefully.** Against job 63's
same-session 988.6 that is +2.4% — only marginally above the ~2% noise floor, so
treat "bs=12 beats no-AC" as suggestive, not settled. The solid result is job 61:
equal throughput to no-AC on **27.6 points less memory**, which is what buys the
batch headroom in the first place.

### Memory model (measured, not estimated)

bs=8 came in at 169.52 GiB against `sac_compile`'s 75.12 GiB, so saving
mlp1 `(R, 2F)` + mlp2 `(R, D)` costs `(169.52 - 75.12) / 8 = 11.80 GiB` per batch
unit, against 12.7 predicted. With SAC's own ~7.69 that is ~19.49 GiB/bs-unit
over ~13.6 GiB fixed — and the fixed term reproduces independently from both
configs, so the model holds:

| bs | 8 | 11 | 12 | 13 |
| --- | --- | --- | --- | --- |
| memory | 61.3% (measured) | ~82.5% | 89.9% (measured) | ~96.6% |

`bs11` is retained as the fallback below the cliff; `bs13` was not attempted.

## GPT-OSS-120B — does not work, unbisected

`gpt_oss_120b_gb300_sac_gmm_bs5` is **known broken** and marked as such in the
registry. Two runs, neither reached step 2:

| job | timeout | SeqNum | NumelIn | |
| --- | --- | --- | --- | --- |
| 62 | 300 s | 333 | 579,133,440 | lm_head gradient |
| 67 | 1800 s | 338 | 3,213,080,192 | one layer's expert-weight gradient |

Both `_REDUCE_SCATTER_BASE` in step 1's backward. Ruled out by log inspection on
both runs: no OOM, no `expandable_segments` / `memory mapping failed` /
`cudaMalloc`, no dataloader retries, all 16 nodes healthy afterwards.

Job 62's diagnosis — that `init_timeout_seconds` (300 s), not
`train_timeout_seconds`, governs step 1, so round 4's bump of the latter never
protected it — is correct and worth keeping. It was not the whole cause: job 67
raised the init timeout to 1800 s and hung the full 30 minutes anyway. This is a
genuine deadlock in `save_grouped_mm` at 36 layers, not compile skew, and 20B
runs the identical code path clean for 250 steps twice.

Job 62 also drained `yqb01-qa01-mgx-00063` with "Kill task failed" on the way
down, blocking the queue until the node was resumed — worth a node check after
any watchdog abort.

Next steps, cheapest first: `save_grouped_mm=True` at bs=16 to separate the save
set from the batch change; then eager, to separate the partitioner from FSDP.

## Caveat on every number here

These are on the **uncorrected** FLOP count, so they are comparable to the rest
of `gb300/*.md` and to nothing else. GPT-OSS puts a 128-token sliding window on
every even layer and `get_moe_model_nparams_and_flops` charges all of them full
`O(L^2)`; against a corrected count the 20B figures scale by 0.851, making the
best run **~861 TFLOP/s / 34.45% MFU** rather than 1012.5 / 40.50%.
Config-to-config comparisons are unaffected — it is a constant factor at fixed
seq_len. The fix is on `perf/flops-sliding-window`, deliberately not in this
branch so these numbers stay comparable to the existing tables.
