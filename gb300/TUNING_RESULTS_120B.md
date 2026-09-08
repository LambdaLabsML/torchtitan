# GPT-OSS-120B tuning sweep — 64x GB300

16 nodes x 4 GB300, `seq_len=8192`, FSDP over all 64 ranks,
`expert_parallel_degree=1`, AdamW lr 8e-4, `--training.disable_cuda_graphs`.
250 steps, steady state = steps >= 100. Baseline is
[157.6 TFLOP/s](README_120B.md) (job 48).

Peak is GB300 dense bf16 ~2503 TFLOP/s/GPU, which is what MFU is against.

The configs were **reasoned from the 20B sweep rather than copied from it**, and
that turned out to matter: the 20B winner is the loser here.

## Measured

| job | config | bs | AC | reshard | TFLOP/s/GPU | cluster | MFU | mem | vs base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | `gpt_oss_120b` (baseline) | 1 | Full | default | 157.6 | 10.09 PF | 6.3% | 24.7% | — |
| 57 | `..._noac_compile` | 3 | none | default | 455.3 | 29.14 PF | 18.2% | 76.6% | +189% |
| 60 | `..._sac_compile_maxbs` | 20 | Selective | default | 786.1 | 50.31 PF | 31.4% | 98.7% | +399% |
| **56** | **`..._sac_compile`** | 18 | Selective | default | **786.9** | **50.36 PF** | **31.5%** | 84.2% | **+399%** |

**Best: 786.9 TFLOP/s/GPU, 50.36 PFLOP/s across 64 GPUs, 31.5% MFU — 4.99x the
out-of-the-box baseline.**

```
                                            baseline
                                               |
sac_compile        bs18  84% ##################################### 786.9
sac_compile_maxbs  bs20  99% ##################################### 786.1
noac_compile       bs3   77% ##################### 455.3
BASELINE           bs1   25% #######| 157.6
                             0     200    400    600    800
```

## Failed

| job | config | bs | why |
| --- | --- | --- | --- |
| 58 | `..._noreshard` | 1 | Never produced a step in 90 min. Allocator log: `expandable_segments: memory mapping failed with OOM` repeatedly, on many ranks. Killed by the job time limit rather than by a clean OOM. |
| 59 | `..._sac_compile_noreshard` | 12 | Clean OOM: "Tried to allocate 1.98 GiB. GPU 0 has a total capacity of 276.50 GiB of which 1.08 GiB is free." |

**Both failures are the same cause, and it refutes a prediction I made.** I
expected never-reshard to help *more* on 120B than on 20B, reasoning that 120B is
communication-bound and never-reshard removes a collective. That reasoning ignored
what never-reshard costs: it keeps parameters gathered after forward, and 120B
unsharded is ~240 GB of bf16 parameters against 276 GB of GPU. There is no batch
size at which that fits alongside optimizer state and activations. It is not that
never-reshard is a bad trade on 120B — it is mechanically unavailable, and the
model size alone was enough to know that in advance.

## What this tells you

**The 20B recipe does not transfer, and copying it would have cost 42%.**
`noac_compile` was the 20B winner at 987.8 TFLOP/s; on 120B it manages 455.3
against SelectiveAC's 786.9.

The reason is memory cost per batch unit, measured in job 53:

| | 20B | 120B |
| --- | --- | --- |
| no-AC | ~33 GiB | **~55 GiB** |
| SelectiveAC | ~7.7 GiB | **~10.8 GiB** |
| max batch, no-AC | 7 | **3** |
| max batch, SelectiveAC | 32 | **20** |

Batch is what pays, and on 120B only SelectiveAC can buy it. no-AC stalls at bs=3
because 36 layers of unsaved activations are simply large.

**120B tunes better than 20B in relative terms: 4.99x vs 3.51x.** The baseline was
wasting more — 24.7% memory at 6.30% MFU — so there was more to reclaim. The
absolute number stays lower (786.9 vs 987.8) because 120B has more parameters to
all-gather per token of useful work.

**The memory cliff is about the allocation pattern, not the percentage.** This is
a correction to the 20B write-up. There, `noac_compile` at 98.7% collapsed to
160.6 TFLOP/s — 43% *below* baseline — and it was recorded as a cliff between 89%
and 99%. But on 120B, `sac_compile_maxbs` runs at **98.7% with no penalty at all**
(786.1 vs 786.9 at 84.2%, a 0.1% difference). Two things separate them: the
successful case is SelectiveAC, whose per-step allocations are ~5x smaller, and
120B's steps carry more compute to amortise allocator work against. So "stay under
~90%" is the wrong lesson. The right one is that **large, no-AC-sized allocations
near the ceiling are what thrash the allocator**; SelectiveAC can sit at 99%
safely. 20B SAC at 93.9% also ran fine, which fits.

Practically: `sac_compile` at bs=18 is the config to use. bs=20 buys nothing
(-0.1%) and leaves 1.3% of memory spare, so there is no reason to run that close
to the edge.

## Not tested, deliberately

**HSDP.** 20B settled it: 864.3 vs 963.5 for full 64-way sharding, because all 64
GPUs are one NVLink fabric (`cliqueSize 64`) so a 64-way all-gather pays no
cross-node penalty worth avoiding. That conclusion is a property of the
interconnect, not the model, so repeating it on 120B would spend a slot to
re-learn it.

**MXFP8.** Blocked on aarch64 for both models: torchao publishes CUDA kernels for
x86_64 only, and the aarch64 wheel is `py3-none-any` with no compiled `.so`, so
SM100 kernels are absent. See [TUNING_RESULTS.md](TUNING_RESULTS.md).

**compile without AC changes.** On 20B, compile alone was worth +0.2%; its value
is entirely that it frees activation memory and so buys batch. Every config here
already has it enabled.

## Data pipeline

All runs above bs=8 stream C4 from **locally staged shards**
(`/mnt/dgxc/data/c4_local`, 64 shards, 20.4 GB), not the HuggingFace Hub. Hub
streaming cannot feed 64 GB300s at these batches — see round 3 in
[TUNING_RESULTS.md](TUNING_RESULTS.md) for what that failure looks like, including
that it strands nodes in DRAIN with wedged GPUs.
