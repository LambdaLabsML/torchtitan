# `deepseek_v4_flash` on 64x GB300 — 8k optimization round

Branch `dsv4_flash_64xgb300`, worktree `/mnt/dgxc/worktrees/dsv4-flash`.
All numbers are 30-step runs at `seq_len=8192`, 64 ranks, mean TFLOP/s over
steps 15+ (skipping warmup/compile), peak memory from the last step line.

## Headline

| | config | TFLOP/s | peak |
|---|---|---:|---:|
| baseline (stock recipe, 4096) | `deepseek_v4_flash_64xgb300` | 24.14 | 56.14 GiB (20.3 %) |
| **8k reference** (same recipe at 8192) | `deepseek_v4_flash_8k` | **18.29** | 95.14 GiB (34.4 %) |
| **best** | `deepseek_v4_flash_8k_ep4_blk32` | **25.45** | 76.58 GiB (27.7 %) |

**+39.1 % over the 8k reference**, and above the 4096 baseline while doing 2x
the context. Everything below is measured against the 8k reference, not the
4096 baseline: doubling context changes attention work per token, so a
cross-length comparison would not be clean.

**The replicate spread is 1.3 %**, measured the hard way -- three runs that were
meant to be different configs turned out identical (see the AC bug below) and
returned 24.74 / 24.62 / 24.42. Treat anything under ~1.5 % as noise. This is
tighter than the ~2 % assumed from earlier rounds.

## What worked

Two levers, and they compose almost exactly multiplicatively (1.170 x 1.158 =
1.355 predicted, 1.353 observed) because they attack disjoint costs: MoE
dispatch scope and attention score area.

### 1. Expert parallel degree 64 -> 4 (+20.5 % alone)

Stock is EP=64, which spreads every layer's dispatch across all 16 nodes. The
sweep, at `block_size` 128:

| EP | TFLOP/s | peak |
|---:|---:|---:|
| 64 (stock) | 18.29 | 95.14 GiB |
| 16 | 21.39 | 65.70 GiB |
| 8 | 21.84 | 67.69 GiB |
| **4** | **22.04** | 76.02 GiB |
| 1 | 20.80 | 128.98 GiB |

Composed with `block_size` 32:

| EP | TFLOP/s | peak |
|---:|---:|---:|
| 16 | 24.74 | 72.40 GiB |
| **4** | **25.45** | **76.58 GiB** |
| 2 | 25.22 | 94.66 GiB |

EP=4 is the optimum, and the reason is the node shape: 4 GPUs per node means an
EP=4 group fits entirely inside one node's NVLink mesh, so expert dispatch
never crosses InfiniBand. EP=2 ties it on throughput (0.9 %, inside noise) but
costs 18 GiB more, and EP=1 collapses -- with no expert parallelism every rank
holds all 256 routed experts for FSDP to all-gather, which is both slower and
129 GiB.

### 2. FlexAttention `block_size` 128 -> 32 (+15.8 % alone)

`deepseek_v4_flash_8k_blocksize32`: 21.18 TFLOP/s, and it *frees* 19.3 GiB.
This is the DSA block-mask granularity: at 128 the mask keeps whole 128-token
blocks that top-k only partly selected, so the kernel computes score area it
then discards. Transfers almost exactly from `pro` (+19 % there), which
confirms the mechanism is mask granularity and not flavor-specific.

`block_size` 16 was never measured -- the one untested tuning question left.

## What did not work

| lever | result | why it is a dead end, not a tuning problem |
|---|---|---|
| no activation checkpointing | **OOM** | Wants ~265 GiB at the 8k reference; still OOMs on the composed base. FullAC is load-bearing at 8192 (it was optional at 4096). |
| `SelectiveAC` | **hard error** | `Tensor cached during selective activation checkpoint has been mutated` -- a correctness guard in the AC machinery, not a memory limit. Some op in this model mutates a tensor SAC cached. |
| microbatch 2x and 4x | **hang** | Three attempts, none reached step 1. See below -- the cause is NOT settled. |
| `mixed_precision_reduce=bfloat16` | **noise** | 18.58 vs 18.29 = +1.6 %, under the 1.3-1.5 % floor. Frees 20.7 GiB though, so it is a memory lever to spend elsewhere, not a speed lever. |

### The batch hang (unresolved)

The most valuable open problem here. Peak memory at the best config is 76.58 GiB
of ~276.5 GiB -- **~200 GiB unused** -- and FullAC keeps the *stored* activation
term small by construction, so a doubled microbatch should cost tens of GiB.
Memory is not the constraint. Yet:

| job | config | batch | outcome |
|---|---|---|---|
| 330 | `ep16_blk32_batch4` | 4x | 3 h wall clock, step 0 |
| 341 | `ep16_blk32_batch2` | 2x | cancelled, 30 min static, step 0 |
| 365 | `ep4_blk32_batch2` | 2x | 6 h clock, cancelled after 43 min static, step 0 |

**The cause is not established, and an earlier diagnosis in this file was
wrong.** This was first recorded as a FlexAttention autotune hang, on the
reasoning that 341 stopped on an autotune line and its log held 64 AUTOTUNE
events "one per rank". Both halves were mistaken: the launcher passes
`--local-ranks-filter 0`, so those 64 events are 64 separate *kernel*
autotunings on rank 0 alone, not one per rank. And job 365 -- the same lever on
the best config, with double the wall clock -- froze with **zero** autotune
blocks, wedged in the DTensor redistribute / weight-materialization phase of
model init, long before any compilation.

So the failure is in init at 2x, not in autotune. Note the tiles genuinely are
pinned: all 7 reported choices carry identical `BLOCK_M=32, BLOCK_M1=16, ...`
and times within 0.2 ms of one another (311.77-311.86 ms), so autotuning is
re-benchmarking each flex kernel rather than searching tile space.

One caveat against calling it a hard deadlock: job 346 (EP=2, 1x) sat at
*exactly* the same 27389-byte log offset for over 10 minutes and then recovered
and finished at 25.22. That phase is legitimately slow. 365 sat there 43 minutes.

Next step is a stack dump from a wedged rank (`py-spy dump`, or
`PYTORCH_DISTRIBUTED_DEBUG=DETAIL` plus the NCCL flight recorder) to see which
collective is blocked -- not another long run at a different batch size.

## Configs added

In `torchtitan/models/deepseek_v4/config_registry.py`:

- `deepseek_v4_flash_8k` — the 8k reference
- `deepseek_v4_flash_8k_ep{1,2,4,8,16}` — EP sweep at `block_size` 128
- `deepseek_v4_flash_8k_blocksize32`, `_bf16reduce`, `_no_ac`, `_sac`
- `_flash_8k_ep_blk(ep, block_size)` — the composition helper
- **`deepseek_v4_flash_8k_ep4_blk32`** — the best config
- `deepseek_v4_flash_8k_ep{2,8,16}_blk32`, `_ep16_blk16`
- `_ep16_blk32_{sac,no_ac,batch2,batch4}`

Supporting changes outside the registry:

- `models/deepseek_v4/__init__.py`: raised the `deepseek_v4_flash` context cap
  from 4096 to 65536 so 8192 is expressible at all.
- `config/configs.py`: widened `mixed_precision_reduce` to accept `bfloat16`.

## A measurement bug worth not repeating

The first SAC and no-AC configs assigned `config.model_spec.ac`, which is not a
dataclass field on `model_spec`. Python accepted it as a stray attribute on a
non-slots dataclass and it controlled nothing, so both runs silently executed
plain FullAC and looked like flat results with slightly different memory. The
operative field is `config.activation_checkpoint`. Two jobs were wasted and
`no_ac` was briefly reported as "fits and is flat" when it had never run.

Assert on the field you think you set, in-process, before queuing 64 ranks --
every config in this round is now verified by constructing it and printing the
values that matter.
