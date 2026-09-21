# 2026-09-21 session plan: scale-out, profile-guided optimization, hero run

Recipe under test: `gb300/REPRODUCE_500.md` (12x microbatch, EP=2, FullAC,
TE fp8 dense, block-input offload). All jobs launched from this checkout
(`/mnt/dgxc/chelsea/torchtitan`, branch `dsv4_te_mhc`). Logs in
`/mnt/dgxc/runs/dsv4_<TAG>.log`, slurm output in `/mnt/dgxc/runs/dsv4-<job>.out`.
NOTE: the login node's NFS view of a running job's log lags by minutes; read
it from a compute node (`srun --jobid=<j> --overlap -N1 -n1 -w <node> ...`)
or wait for the `###### END` marker in the `.out`.

Cluster: 38 usable nodes (r01 9-10, r02 11-14, r03 18; mgx-00057 is the
login node, 00006/12/14/61/62/66/68/71/72 drained). Session window ends
~16:00 UTC.

## Results so far (mean TFLOP/s per GPU over steps 11-20)

| job | GPUs | layout | TFLOP/s | peak mem | notes |
|---|---|---|---|---|---|
| 942 | 32 | r02, 8 nodes | 507.9 (513.9 @ step 20) | 209.8 GiB | yesterday's best |
| 945 | 32 | r02, same nodes | 503.3 (506.9 @ step 20) | 209.6 GiB | reproduce: within noise |
| 946 | 32 | r01, profiler on | ~490 | 210.5 GiB | traces: `outputs/profiling/traces/iteration_10` (32 ranks) |
| 947 | 64 | r03, 16 nodes, FSDP over 64 | 485.2 | 177 GiB | -3.6% vs 32; 30 GiB freed by wider sharding |
| 948 | 96 | 8+8+8, FSDP over 96 | crash | | shared-expert fused w13 init: 2048 pairs not divisible by 96 |
| 949 | 96 | 8+8+8, HSDP shard=32 replicate=3 | 502.5 (501.9 @ step 20) | 207.6 GiB | HSDP holds the 32-GPU number across 3 racks |
| 950 | 128 | 16 (r03) + 8 + 8, HSDP shard=32 replicate=4 | 499.2 (505.8 @ step 20) | 207.2 GiB | |
| 951 | 128 | same + persistent DSA workspace (commit 7fc610c5) | 509.0 (502.6 @ step 20) | 223.8 GiB | +2.0%, step spread 499-514 (950: 444-513) |
| 952 | 128 | hero: 951 recipe, 100 steps, real C4 (`c4_local`) | 505 @ step 20, running | 225.7 GiB | TAG `hero_128_12x_c4`; loss 3.54 @ 20 on C4 |
| 953 | 128 | reference: bf16 dense, bfx9 matmuls, eager mHC, 100 steps, C4 | queued after 952 | | TAG `ref_128_12x_c4_bf16` |

Noise floor: two runs with bitwise-identical numerics (950 vs 951, c4_test)
differ in per-step loss by 0.2 on average after step 10 (max 1.1 early);
942 vs 945 differ 0.08 at step 20. Kernel and collective ordering is not
deterministic, so the hero-vs-reference verdict is "curves overlap within
this band", not an exact match. TFLOP/s noise is ~1% on the 10-step mean.

## Scale-out queue

- [x] 1. reproduce 513.9 on 32 GPUs (945: 503.3 mean, within noise)
- [x] 2. profile the 32-GPU recipe (job 946)
- [x] 3. analyze the profile (findings below)
- [x] 4. 64 GPUs, one rack (job 947): 485.2 (plain FSDP; HSDP retest is B0)
- [x] 5. 96 GPUs, 8 nodes per rack, HSDP (job 949): 502.5
- [x] 6. 128 GPUs: 16 on r03 + 8 on r01 + 8 on r02, HSDP shard=32 replicate=4
      (job 950): 499.2. Two shard groups on r03. Slurm orders ranks by
      hostname and the shard axis is the inner mesh axis, so pick exactly 8
      or 16 nodes per rack. Launch: `--parallelism.data_parallel_replicate_degree 4`
      plus `--nodelist`, partition `all`.
- [x] 7a. first optimization applied and measured at 128 (A below, +2.0%)
- [ ] 8. hero run on 128 GPUs (job 952), 100 steps, loss curve on real C4.
      Reference (job 953) chained behind it on the same 32 nodes: same recipe
      minus TE fp8 dense (bf16 `Linear`), `TORCHTITAN_FP32_MATMUL_PRECISION=bfx9`,
      `TORCHTITAN_TE_MHC=0`. Both use the c4_local dataset (64 staged C4 shards
      streamed locally, commit 5320d553) and the same schedule (warmup 2,
      linear decay over the last 80 steps). ETA: hero ~12:50, reference ~13:40.
      Compare with `scratchpad/curves.py` (loss, grad_norm, TFLOP/s overlay).
- [ ] 7b. after the reference: experiment batch, 4 concurrent 8-node jobs
      (~15 min): control = 12x + workspace fix at 32 GPUs; `NCCL_PROTO=Simple`;
      HybridEP (`..._tedense_12x_hybridep`, `HYBRIDEP=1`); 14x microbatch
      (`..._tedense_14x`, memory now 224 GiB at 12x). Then B0 (64 GPUs HSDP)
      and stack whatever wins into a final 128-GPU run.

## Profile findings (job 946, rank 0, step 9, 18.4 s step, GPU 99.6% busy)

Main compute stream, by item:

| item | ms | share |
|---|---|---|
| dense + expert GEMMs | 3818 | 21% |
| EP dispatcher barrier (`multimem_barrier_kernel`, waiting for the EP peer) | 2163 | 12% |
| cuDNN DSA backward (deterministic) | 1899 | 10% |
| EP dispatcher row copies (`_copy_rows_to_peer_ptrs_kernel`) | 1322 | 7% |
| unfused elementwise / copies / casts | ~1200 | 7% |
| deterministic-backward workspace clear + dKV fold | ~480 | 3% |

Side streams (fully overlapped, ~50 ms exposed total): FSDP reduce-scatter
3144 ms, all-gather 2287 ms, offload DtoH 1882 ms, restore HtoD 1805 ms.
Every NCCL kernel in the trace is `RING_LL`.

Root cause of the 850 ms per-step stall (item A): each layer's cuDNN DSA
backward called `torch.empty` for a 26-32 GB scratch workspace and freed it
after the layer (42x per step). Near the memory cap the caching allocator had
to release cached blocks (cudaFree + device synchronize) to place it: every
rank shows ~15 such syncs per profiled window, 0.1-1.2 s each, and the
slowest rank (rank 27 in step 9: 1.22 s) stalls all 32 through the next
expert all-gather/reduce-scatter and the EP barrier. The 457.6-recipe
profile (job 847, 6x) has no such stall; it appeared with 12x pushing memory
to the edge. Analysis scripts: `scratchpad/prof_summary.py`, `straggler.py`,
`slow_empty.py` (kept in the session scratchpad, not the repo).

## Optimization candidates, in order

- [x] A. **Persistent cuDNN backward workspace** (commit 7fc610c5): one
      grow-only uint8 buffer shared by all layers, passed as the wrapper's
      `workspace=`; `TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0` restores the old
      path. Single-GPU check: outputs and all grads bitwise equal. 128 GPUs:
      509.0 vs 499.2 (+2.0%), memory +16.6 GiB (the buffer no longer shares
      its region with per-layer activations). Less than the 4.6% the stall
      cost, so some allocator churn may remain; a re-profile would show.
- [ ] A2. `NCCL_PROTO=Simple`: every collective runs LL today; the ledger
      measured +0.85% on an old recipe and the dedicated protocol test (job
      783) never ran. Env-only, in the 7b batch.
- [ ] B0. **64 GPUs as HSDP shard=32 replicate=2** (one 16-node rack): plain
      FSDP over 64 gave 485; HSDP at 96/128 gave 502-509, so the 64 number is
      likely recoverable the same way. Trades the 30 GiB memory saving back.
- [ ] B. **14x microbatch** (`..._tedense_14x`, 114688 tokens/rank). Never
      measured (943 was cancelled, not OOM). Probably OOM now: the device sits
      at 278 of 284 GB at 12x with the persistent workspace. Low priority.
- [ ] C. **Unfused expert-path elementwise** (~0.6 s, 3%): the two dgrad
      grouped GEMMs (w1 and w3 are separate `_grouped_mm` calls in
      `torchtitan/models/common/moe.py`) leave 688 adds of [1179648, 4096] per
      step, plus 950 bf16->fp32 copies and 688 copies of a [T, 16384] tensor
      (mHC residual stream `.contiguous()`). Fusing w1/w3 into one grouped GEMM
      means a `w13_E(2F)D` parameter layout: checkpoint mapping, FSDP
      sharding, init. Not a same-day change; sized for a follow-up.
- [ ] D. EP=1 + CUDA graphs (user request). EP=1 removes the symmetric-memory
      dispatcher (barrier + copies = 19% of the step) and makes shapes static
      so `training.disable_cuda_graphs=False` becomes possible, but every rank
      then all-gathers all 256 experts per layer (12.9 GB/layer; ledger EP=1
      results were a collapse on the old recipe). `_ep1_variant` configs exist.
- [ ] D2. HybridEP dispatcher (user request): config
      `..._tedense_12x_hybridep` (commit b066f2c4), launch with `HYBRIDEP=1`
      (build at `/mnt/dgxc/DeepEP-hybrid`; it forces CUDA graphs). Earlier
      sweep (jobs 784-795, 1k recipe): 2.9% below MinimalAsyncEP. Retest now
      that the dispatcher barrier is 12% of the step. In the 7b batch.
- [x] E. (explained) The even/odd barrier asymmetry was item A: odd ranks'
      main streams show 840 ms less kernel time in an identical step, i.e.
      the straggler stall was absorbed as barrier spinning on even ranks and
      as a stream gap on odd ranks. Residual barrier (1.6 s/rank) is genuine
      peer sync; even ranks carry ~220 ms more expert GEMM (experts 0-127
      receive more tokens than 128-255).
- [ ] I. **Elementwise over padded EP capacity** (~3-4%, half a day). The
      expert-path elementwise kernels (SiLU, gate*up, dgrad add) run over the
      dispatcher's full receive capacity (1,179,648 rows) while the grouped
      GEMMs respect the real offsets; expected fill is ~50% with top-k 6 over
      2 EP ranks, so about half of the 2.5 s of Triton + plain elementwise is
      wasted. Slicing needs the row count on the host (which MinimalAsyncEP
      avoids); the fix is a masked / offset-aware fused kernel.
- [ ] J. **Re-profile at 128 GPUs after the fix** (15 min): residual allocator
      churn (the fix recovered 2.0 of the 4.6% and the card sits at 278 of
      284 GB) and whether the cross-rack replica all-reduce is exposed.
- [ ] K. **Re-measure the offload's own cost at 7x**: this morning's pair
      (937/938, profiler on) put it at 4.8%. If that was mostly the allocator
      stall it is now free; otherwise there is a second offload issue.
- [x] L. (dead end) cuDNN DSA backward launch granularity: 33k launches/step
      at ~77 us. The wrapper's `block_tile` is overridden by the SM100 backend
      selector (tested 32/64/128 locally: identical time, bitwise equal).
      Needs a cuDNN-side change.
- [x] M. (keep) the per-layer index sort (0.4 s, 2%) is intentional: sorted
      key sets measurably improve the kernel's numerics (cudnn_dsa.py:135).
- [ ] N. FSDP2 flat-buffer copy-in/out for expert params: 0.33 s (1.8%) on
      the main stream, inherent to FSDP2's all-gather layout. Low priority.
- [x] F. (closed) TE MXFP8 grouped experts: ledger jobs 871-878. MXFP8 grouped
      GEMMs are 1.8x faster in isolation but fp8 weights cannot be cached
      across the step under FSDP-sharded bf16 experts (277 GB), so weights are
      requantized every pass (~3% of the step) and the chain holds ~27 GiB more
      at the block backward peak (OOM at 6x). Net: no gain. Would pay only with
      fp8 primary expert weights all-gathered in MXFP8 (Megatron's design,
      1-2 days of FSDP2 work).
- [x] G. (closed) deterministic DSA backward (~0.5 s): required, the router
      flips on near-ties and FullAC rejects the forward/recompute mismatch.
- [x] H. (answered) Not compiled: `compile.enable=False` everywhere; only
      leaf-compiled helpers (silu, rmsnorm, mHC pieces, 6 small Inductor
      graphs) run compiled. TE `te.Linear` is not torch.compiled and does not
      need to be (cast + cuBLASLt fp8 GEMM are fused inside TE). The unfused
      work in C is outside TE.

## Bugs found

- [ ] `fused_gate_up_param_init` (`torchtitan/models/common/config_utils.py`)
      unflattens the *sharded* DTensor: fails when dp_shard does not divide
      2048 (96 ranks, job 948). Fix like the fused QKV init: build the
      replicated tensor and `copy_` into the shard. Not hit with HSDP shard=32.
- [x] Login-node NFS view of a running job's log lags by minutes (looked like
      a hang at 11:17; py-spy showed all ranks training). Read from a compute
      node instead.

## Commits this session (branch `dsv4_te_mhc`)

- 98e9f936 config: 12x profile variant
- 7fc610c5 cudnn_dsa: persistent backward workspace
- 5320d553 data: c4_local dataset + 12x hero/reference configs
- 229e1450 TODOS: 128-GPU results
- b066f2c4 config: 12x HybridEP variant
