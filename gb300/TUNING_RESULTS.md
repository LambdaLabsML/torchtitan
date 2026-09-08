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

### Chart — per-GPU TFLOP/s (250 steps, steady state)

`R1`/`R2` = round. `|` marks the 281.4 baseline.

```
                                                    baseline
                                                       |
R2 noac_compile_bs7        bs7  89% ############################################### 963.5
R1 noac_compile            bs6  77% ############################################## 942.3
R1 noac_compile_noreshard  bs4  66% ############################################ 917.0
R2 noac_compile_noreshard  bs6  90% ########################################### 880.7
R2 noac_compile_hsdp       bs4  81% ########################################## 864.3
R2 sac_compile             bs8  27% ######################################### 859.6
R1 noac                    bs4  87% ############################ 569.2
R1 noreshard               bs1  20% ###############| 322.2
R1 compile_loss_only       bs1   7% #############| 282.1
R1 BASELINE                bs1   7% #############| 281.4
R1 noac_compile_maxbs      bs8  99% #######| 160.6
                                    0    200   400   600   800  1000
```

Note the memory column next to the bars: `sac_compile` reaches within 11% of the
best result using **a quarter of the memory** the leaders need.

### Blocked

| config | why |
| --- | --- |
| `..._compile` | `AssertionError: Node add_21 was invalid, but is output` in AOTAutograd's `min_cut_rematerialization_partition`. FullAC + compiling the *model* is the broken pair, not compile itself — both escapes (drop AC, or compile only the loss) work. |
| `..._mxfp8`, `..._noac_compile_noreshard_mxfp8` | torchao publishes CUDA kernels for **x86_64 only**; the aarch64 wheel is `py3-none-any` with no compiled `.so`, so SM100 kernels are absent. No aarch64 build on cu128/cu129/cu130 or either nightly index. `KernelPreference.EMULATED` would run but measures emulation, not mxfp8. |

## What the numbers say

**Memory utilisation is the whole game — until it isn't.** The baseline recomputes
activations while sitting in 6.7% of the GPU. Spending that headroom is worth
2-3.5x. But `noac_compile` at bs=8/98.7% is **43% slower than the baseline** and
5.9x slower than bs=6 at 77%: the allocator thrashes, and wall clock shows it
(~20 min for that one config vs ~5-11 for the others).

**Corrected by the 120B sweep:** this is NOT a simple "stay under ~90%" rule. On
120B, SelectiveAC runs at 98.7% with no penalty whatsoever (786.1 vs 786.9 at
84.2%). What thrashes the allocator is **large, no-AC-sized allocations near the
ceiling**, not the percentage itself — SelectiveAC's per-step allocations are ~5x
smaller and sit at 99% safely. 20B SAC at 93.9% also ran fine, which fits. See
[TUNING_RESULTS_120B.md](TUNING_RESULTS_120B.md).

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

## Round 2 — measured (job 49, 250 steps)

| config | bs | AC | reshard | shard | TFLOP/s/GPU | cluster | MFU | mem | vs base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`..._noac_compile_bs7`** | 7 | none | default | 64 | **963.5** | **61.66 PF** | **38.5%** | 88.9% | **+242%** |
| `..._noac_compile_noreshard_bs6` | 6 | none | never | 64 | 880.7 | 56.36 PF | 35.2% | 90.2% | +213% |
| `..._noac_compile_hsdp` | 4 | none | default | 4x16 | 864.3 | 55.32 PF | 34.6% | 80.8% | +207% |
| `..._sac_compile` | 8 | Selective | default | 64 | 859.6 | 55.01 PF | 34.4% | **27.2%** | +206% |

All four ran; none OOMed, so both estimated batch sizes were viable.

**New best: 963.5 TFLOP/s/GPU, 61.66 PFLOP/s, 38.5% MFU** — 3.42x baseline.

### What round 2 changed

**The peak is bs=7, and it is a narrow ridge.** 942.3 at 77% -> 963.5 at 88.9%
-> 160.6 at 98.7%. Only +2.3% was left above bs=6, and the drop beyond it is a
factor of six. The cliff sits between 88.9% and 98.7%; bs=7 is close enough to it
that a longer run or a different dataset shuffle could plausibly tip over. bs=6 at
77% is the setting to hand someone who needs it to finish.

**never-reshard stops helping under memory pressure.** It gave 917.0 at bs=4/66%
in round 1; at bs=6/90% it gives 880.7 — worse than its own smaller-batch version
and worse than plain FSDP at bs=7. Its benefit is real only while there is slack;
near the ceiling, holding parameters gathered competes with the activations that
are actually earning throughput.

**HSDP did not help.** 864.3 against 963.5 for full 64-way sharding at a
comparable batch. The hypothesis was that 20B is small enough for the all-gather
to be latency-bound, so narrowing the shard group to one node would win. It does
not, and the reason is the hardware: all 64 GPUs are one NVLink fabric
(`cliqueSize 64`), so a 64-way all-gather is not paying a cross-node penalty
worth avoiding. Useful negative result — on an IB-connected cluster this would
likely go the other way.

**SelectiveAC is the efficiency winner and the biggest remaining lever.** 859.6 —
within 11% of the best — using **27.2% of memory**, where every other config near
that throughput needs 80-90%. Per-op SAC recomputes so cheaply that bs=8 barely
touches the GPU. Nothing has tried it at a batch that actually fills memory: on
these numbers there is room for roughly 3x the batch before it reaches the ~89%
where bs=7 peaks. That is the obvious next experiment.

## Round 3 — SelectiveAC at large batch: hit a DATA limit, not a compute one

Round 2 left SelectiveAC as the open lever: 859.6 TFLOP/s in 27.2% of memory,
where every other config near that throughput needed 80-90%. A probe (job 50, 8
steps) confirmed the headroom — per-op SAC costs **~7.7 GiB per batch unit against
no-AC's ~33**, about 4.4x cheaper, and nothing OOMed up to bs=32:

| bs | 16 | 24 | 28 | 32 |
| --- | --- | --- | --- | --- |
| memory | 48.4% | 70.8% | 82.8% | 93.9% |

All three 250-step runs (job 51) then **failed**, and the reason is worth reading
carefully because it is not about SelectiveAC at all.

| config | outcome | peak seen | mem |
| --- | --- | --- | --- |
| `..._sac_compile_bs24` | died at step ~50 | 922.3 TF/GPU (59.03 PF) | 70.8% |
| `..._sac_compile_bs28` | died before step 1 | — | — |
| `..._sac_compile_bs32` | died at step ~100 | 927.7 TF/GPU (59.37 PF) | 93.9% |

**Those peaks are not results.** They are the throughput reached before the run
fell over, on runs whose data pipeline was already degrading. They are recorded
because they are suggestive, not because they are measurements.

### Cause

```
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=8697,
OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=100000)
ran for 100023 milliseconds before timing out.
```

A **one-element** allreduce timed out after 100 s — the per-step sync. One rank
never arrived. It is not memory (bs=24 died at 70.8% while running at 749-922
TFLOP/s) and not SelectiveAC. It is the dataloader:

| run | Retry | ConnectionError | HTTP 429 |
| --- | --- | --- | --- |
| baseline (bs1) | 0 | 0 | 0 |
| `noac_compile` (bs6) | 0 | 0 | 1 |
| `noac_compile_bs7` (bs7) | 0 | 1 | 0 |
| `sac_compile` (bs8) | 0 | 0 | 20 |
| `sac_compile_bs24` | **142** | **87** | 3 |
| `sac_compile_bs32` | **99** | **32** | 3 |

C4 is **streamed from HuggingFace at train time** (`streaming=True` in
`hf_datasets/text_datasets.py`), unauthenticated. Token demand scales with batch:
the bs=1 baseline pulls ~0.5M tokens/step across 64 ranks; bs=24 pulls ~12.6M.
Somewhere above bs=8 that exceeds what the Hub will serve, ranks starve waiting
for data, one misses the 100 s window, and the NCCL watchdog aborts the job.

### What this means for the rest of the table

**Everything at bs<=8 stands.** Those runs show 0-1 retries and no connection
errors — 963.5 for `noac_compile_bs7` is a clean measurement. The throttling only
appears above bs=8, and only the round-3 runs were affected.

**Above bs=8 this cluster cannot currently benchmark at all**, regardless of
config. That is an infrastructure ceiling, not a property of any tuning.

### To get past it

1. **Stage C4 locally.** 250 steps at bs=24 needs ~3.1B tokens, roughly 20 of
   C4's 1024 shards (~7 GB gzipped) onto `/mnt/dgxc`, which has 3.3 TB free.
   Needs a local dataset entry in `hf_datasets/text_datasets.py` alongside `c4`
   and `c4_test` — a code change, not a config one.
2. **Set `HF_TOKEN`.** The logs carry the Hub's own warning about unauthenticated
   rate limits. Cheapest fix, but it makes the benchmark depend on a network path
   that is outside the cluster and not reproducible run to run.

Option 1 is the right one for a benchmark: a throughput number should not depend
on someone else's rate limiter.

## Round 4 — the data fix worked (job 54, 250 steps)

Round 3's failures were HuggingFace throttling, not SelectiveAC. Fix: 64 C4
shards staged to `/mnt/dgxc/data/c4_local` (one per rank, 20.4 GB) plus a
`c4_local` dataset entry, and `train_timeout_seconds` 100 -> 600 as a safety net.

| config | bs | TFLOP/s/GPU | cluster | MFU | mem | steps | Retry/ConnErr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`..._noac_compile_bs7_local`** | 7 | **987.8** | **63.22 PF** | **39.5%** | 88.9% | 250/250 | 0 / 0 |
| `..._sac_compile_bs24` | 24 | 922.0 | 59.01 PF | 36.9% | 70.8% | 250/250 | 0 / 0 |

Both ran to completion with **zero retries and zero connection errors**, where the
same SAC batch previously died at step ~50 with 142 retries and 87 connection
errors. The fix is confirmed at the source.

**The control validates the dataset change.** `noac_compile_bs7` measured 963.5 on
Hub-streamed C4; on local shards it measures **987.8** — same config, same 88.9%
memory footprint, +2.5%. Local is neutral-to-slightly-better, so large-batch
numbers are directly comparable to the rest of the table. The +2.5% is most
plausibly the last traces of throttling in the original run (which logged 1
connection error).

**New overall best: 987.8 TFLOP/s/GPU, 63.22 PFLOP/s, 39.5% MFU — 3.51x the
281.4 baseline.**

**SelectiveAC at bs=24 reaches 922.0 in 70.8% of memory.** That is 6.7% below the
best while using 18 points less memory — the same shape as its bs=8 result, and it
confirms SAC is the memory-efficient option rather than the fastest. Round 3's
"peak seen" of 922.3 before the crash turned out to be almost exactly right, which
is luck rather than vindication: it was sampled from a degrading run.

Job 54 was cancelled after these two runs, deliberately. It packed 9 runs into one
allocation, and cancelling to reach any one of them discards the rest. Remaining
runs now go through `gb300/run_one.slurm` - one config per job.

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
| 46 | round 1 sweep | 7 configs x 250 steps |
| 47 | 120B validation | ran clean, 6 steps |
| 48 | **120B baseline** | 157.6 TFLOP/s, 250 steps - see README_120B.md |
| 49 | round 2 sweep | 4 configs x 250 steps, all exit 0 |
| 50 | SAC batch probe | bs 16/24/28/32 all fit; 93.9% at bs=32 |
| 51 | SAC 250-step sweep | all 3 killed by NCCL watchdog - HF rate limiting, see round 3 |
| 53 | 120B probes + c4_local validation | no-AC ceiling bs=3 (76.6%); SAC bs=20 (95.0%); c4_local OK |
| 54 | round 4 (cancelled after 2 of 9) | bs7_local 987.8; sac_bs24 922.0; both 0 retries |
| 56-60 | **120B tuning** | see TUNING_RESULTS_120B.md - best 786.9 (4.99x) |

Baseline is quoted as 281.4 (job 46, 250 steps) rather than 287.8 (job 39, 50
steps) so every row in the round-1 table is measured identically.
