# 2026-09-21 session plan: scale-out, profile-guided optimization, hero run

Recipe under test: `gb300/REPRODUCE_500.md` (12x microbatch, EP=2, FullAC,
TE fp8 dense, block-input offload). All jobs launched from this checkout
(`/mnt/dgxc/chelsea/torchtitan`, branch `dsv4_te_mhc`). Logs in
`/mnt/dgxc/runs/dsv4_<TAG>.log`, slurm output in `/mnt/dgxc/runs/dsv4-<job>.out`.
NOTE: the login node's NFS view of a running job's log lags by minutes; read
it from a compute node (`srun --jobid=<j> --overlap -N1 -n1 -w <node> ...`)
or wait for the `###### END` marker in the `.out`.

## Results so far (mean TFLOP/s per GPU over steps 11-20)

| job | GPUs | layout | TFLOP/s | peak mem | notes |
|---|---|---|---|---|---|
| 942 | 32 | r02, 8 nodes | 507.9 (513.9 @ step 20) | 209.8 GiB | yesterday's best |
| 945 | 32 | r02, same nodes | 503.3 (506.9 @ step 20) | 209.6 GiB | reproduce: within noise |
| 946 | 32 | r01, profiler on | ~490 | 210.5 GiB | traces: `outputs/profiling/traces/iteration_10` (32 ranks) |
| 947 | 64 | r03, 16 nodes, FSDP over 64 | 485.2 | 177 GiB | -3.6% vs 32; 30 GiB freed by wider sharding |
| 948 | 96 | 8+8+8, FSDP over 96 | crash | | shared-expert fused w13 init: 2048 pairs not divisible by 96 |
| 949 | 96 | 8+8+8, HSDP shard=32 replicate=3 | 502.5 (501.9 @ step 20) | 207.6 GiB | HSDP holds the 32-GPU number across 3 racks |
| 950 | 128 | 16 (r03) + 8 + 8, HSDP shard=32 replicate=4 | running | | |

## Scale-out queue

- [x] 1. reproduce 513.9 on 32 GPUs
- [x] 2. profile the 32-GPU recipe (job 946)
- [x] 3. analyze the profile (findings below)
- [x] 4. 64 GPUs, one rack (job 947)
- [x] 5. 96 GPUs, 8 nodes per rack, HSDP (job 949): 502.5
- [ ] 6. 128 GPUs: 16 on r03 + 8 on r01 + 8 on r02, HSDP shard=32 replicate=4 (job 950)
      (two shard groups on r03). Slurm orders ranks by hostname and the shard
      axis is the inner mesh axis, so pick exactly 8 or 16 nodes per rack.
- [ ] 7. optimize the best of 4/5/6 (profile + PGO, list below)
- [ ] 8. hero run, >= 100 steps, loss curve captured (W&B is on by default).
      Budget: ~18 s/step at every scale so far -> 100 steps ~30 min + ~5 min
      startup; reserve 45 min. A reference run at the same GPU count and
      microbatch (numerics knobs off) is another ~70 min, can run on another
      rack in parallel. Decide: what is the curve compared against?

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

## Optimization candidates, in order

- [ ] A. **The 850 ms stall at the forward->backward transition** (4.6% of
      the step). On every odd rank the compute stream idles ~875 ms waiting
      for a 1.04 s all-gather that runs alongside a 0.96 s reduce-scatter;
      steady-state per-layer all-gathers take ~60 ms. The even rank then waits
      for its odd EP peer in the barrier (even ranks: 2.25 s barrier/step,
      odd: 1.63 s). Cheapest experiment: `NCCL_PROTO=Simple` (ledger: +0.85%
      on an old recipe; the dedicated protocol test, job 783, never ran).
      Second: find which parameter group's collectives these are (output
      projection / embedding grads, 1 GB each, at backward start) and split or
      reorder them.
- [ ] B0. **64 GPUs as HSDP shard=32 replicate=2** (one 16-node rack): plain FSDP
      over 64 gave 485; HSDP at 96 gave 502, so the 64 number is likely
      recoverable the same way. Trades the 30 GiB memory saving back.
- [ ] B. **14x microbatch at 64+ GPUs (plain FSDP only).** Wider FSDP sharding freed 30 GiB
      (177 GiB at 64 GPUs vs 210 at 32). 14x was never measured (943 was
      cancelled, not OOM). Config exists: `..._tedense_14x`.
- [ ] C. **Unfused expert-path elementwise** (~0.6 s, 3%): in-place add of
      the two dgrad grouped GEMMs (w1 and w3 are separate `_grouped_mm`
      calls, 688 adds of [1179648, 4096] per step), 950 bf16->fp32 copies,
      688 copies of a [T, 16384] tensor (mHC residual stream `.contiguous()`).
      Fusing w1/w3 into one grouped GEMM removes the add and one input pass.
- [ ] D. EP=1 + CUDA graphs (user request). EP=1 removes the symmetric-memory
      dispatcher (barrier + copies = 19% of the step) and makes shapes static
      so `training.disable_cuda_graphs=False` becomes possible, but every rank
      then all-gathers all 256 experts per layer (12.9 GB/layer; ledger EP=1
      results were a collapse on the old recipe). Try after A-C.
- [ ] D2. HybridEP dispatcher (user request): launcher `HYBRIDEP=1`, configs
      `..._hep4/6/8` from the earlier sweep (jobs 784-795: HybridEP measured
      2.9% below MinimalAsyncEP on the 1k recipe, and it forces CUDA graphs).
      Retest on the 12x recipe with `_swap_ep_backend(config, "hybridep")`
      since the dispatcher barrier is now 12% of the step.
- [ ] E. Remaining EP barrier waits (1.6 s/rank, 100 barriers of 1-90 ms) are
      routing imbalance between the two EP peers. No cheap fix.
- [x] F. (closed) TE MXFP8 grouped experts: ledger jobs 871-878. MXFP8 grouped
      GEMMs are 1.8x faster in isolation but fp8 weights cannot be cached
      across the step under FSDP-sharded bf16 experts (277 GB), so weights are
      requantized every pass (~3% of the step) and the chain holds ~27 GiB more
      at the block backward peak (OOM at 6x). Net: no gain. Would pay only with
      fp8 primary expert weights all-gathered in MXFP8 (Megatron's design,
      1-2 days of FSDP2 work).
- [x] G. (closed) deterministic DSA backward (~0.5 s): required, the router
      flips on near-ties and FullAC rejects the forward/recompute mismatch.
- [ ] H. Not compiled: `compile.enable=False` everywhere; only leaf-compiled
      helpers (silu, rmsnorm, mHC pieces, 6 small Inductor graphs) run
      compiled. TE `te.Linear` is not torch.compiled and does not need to be
      (cast + cuBLASLt fp8 GEMM are fused inside TE). The unfused work in C is
      outside TE.

## Bugs found

- [ ] `fused_gate_up_param_init` (`torchtitan/models/common/config_utils.py`)
      unflattens the *sharded* DTensor: fails when dp_shard does not divide
      2048 (96 ranks). Fix like the fused QKV init: build the replicated
      tensor and `copy_` into the shard. Not needed with HSDP shard=32.
