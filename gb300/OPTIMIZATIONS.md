# DeepSeek-V4-flash on 32x GB300: every optimization, labeled for PR splitting

Branch `dsv4_te_mhc` (LambdaLabsML/torchtitan), 160 commits over upstream base
`6857b67b6` ("DSv4 Flash: Batched DSA Patched via Claude (#22)"). Numbers are
TFLOP/s per GPU, 8 nodes x 4 GB300 in one NVL72 rack, seq 8192, 20-step runs;
the regime (collapsed = raw router from random init, balanced = forced load
balancing, Megatron's `--moe-router-force-load-balancing`) is stated on every
number because they differ by ~14% on the same code (job 940 vs 1008). Full
evidence: `gb300/FLASH_8K_RESULTS.md` on branch `dsv4_flash_64xgb300`.
Exact reproduction command: `gb300/REPRODUCE_500.md`. Current best: 756.1 TFLOP/s at step 20 (job 1126) / 756.8 40-step mean (job 1129), 9x balanced; Megatron-LM reference 748.

Status legend: **ON** = in the 685.7 recipe; **OFF** = merged, default off,
measured neutral/negative or regime-specific; **CLOSED** = measured negative,
code kept only where it documents the measurement; **FIX** = correctness.

Two sessions worked this branch. Units 20-23 are the second session's (commits
authored `clowman`, 2026-09-21) and are labeled as such; everything else is
from this campaign.

---

## Part A -- units in the 685.7 TFLOP/s recipe (in stacking order)

### 1. Full-bf16 training recipe + GB300 launcher  **ON**
- Commits: `61c080063`, `a9eeca59a`, `81df0e70e`, `1737fcfa4`, `fcc371a4b`, `937de5bf4`, `2cbe50db2`
- Files: `torchtitan/models/deepseek_v4/config_registry.py` (gb300 configs), `gb300/dsv4_64xgb300.slurm`, `/mnt/dgxc/sbatch_rack.sh` (cluster-side)
- What: bf16 params/grads/optimizer states (the lever that makes 8k fit; fp32 default costs ~80 GiB/rank), fused bf16 Adam, the Slurm launcher (CPATH for header-less nodes, `CUBLAS_NEW=1` cuBLAS 13.8 preload, `DEEPEP`/`HYBRIDEP` PYTHONPATH handling, single-rack submission).
- Effect: enabling. Numerics: bf16 optimizer states (deliberate recipe choice).
- PR: config + launcher; no library code.

### 2. cuDNN fused DSA sparse-attention backward and forward  **ON**
- Commits: `6eacaec50`, `4c92a64e5`, `df7daf6e0`, `d7340953a`, `ca4f61ea2`, `42319144a`, `c537579c7`, `738f6cf16`, `49f4122dd`, `d52d4157e`, `ec91517d6`
- Files: `torchtitan/models/deepseek_v4/cudnn_dsa.py` (new), `attention.py`, `compressor.py` (stable-sort deterministic selection), tests
- Knob: config `cudnn_dsa` (flag on the attention config)
- Effect: 170 -> 368 (collapsed, early recipe); the largest single step of the campaign. Requires cuDNN frontend 1.29 with the DSA wrappers in `/mnt/dgxc/pydeps-cudnn` (external).
- Numerics: bitwise-checked against the eager path in the A/B test.

### 3. `TORCHTITAN_FP32_MATMUL_PRECISION` (TF32 for the fp32 matmuls)  **ON**
- Commit: `09fad6a97`. File: `torchtitan/models/deepseek_v4/config_registry.py` / model glue.
- Knob: `TORCHTITAN_FP32_MATMUL_PRECISION=tf32` (default `bfx9`).
- Effect: part of the 500 recipe (router/mHC fp32 matmuls). Numerics: TF32 in fp32 matmuls.

### 4. Fused output inverse RoPE in the DSA Function  **ON**
- Commit: `9a0a58d4f` (+ `754bdca2f` FIX). Files: `torchtitan/models/deepseek_v4/fused_rope.py` (Triton), `cudnn_dsa.py`
- Effect: ~+3%. FIX `754bdca2f`: int64 row offsets -- the int32 index overflowed past 65,536 rows of [T,64,512], which is exactly the 9x microbatch (step-1 NaN, probes 923/924).

### 5. FSDP `dense-never` reshard policy  **ON**
- Commits: `7c0f1e79e`, `1b0e35498`. File: `torchtitan/distributed/fsdp.py`
- Knob: `FSDP_POLICY=dense-never` (dense params stay unsharded between forward and backward; experts still reshard).
- Effect: positive in the 400-450 era (ledger); 13.6 GiB of dense params resident.

### 6. MinimalAsyncEP receive pool: 4 slots  **FIX / ON**
- Commit: `cb2205745`. File: `torchtitan/distributed/minimal_async_ep/api.py`
- Knob: `MINIMAL_ASYNC_EP_SLOTS` (default 4). Experts save a raw alias of the receive slot; with 2 slots the backward comm ops rewrote it before the wgrad GEMM read it (0.12-0.13% gradient error, jobs 754/768). 8 for the dual-microbatch schedule.

### 7. cuDNN fused indexer top-k  **ON**
- Commits: `942577746`, `d3585c435`, `aa46bd50c`. File: `torchtitan/models/deepseek_v4/cudnn_indexer.py` (new)
- Knob: config `cudnn_indexer`. Effect: 401 -> 441 (+10%), -11 GiB (collapsed). Local (per-sequence) ids, -1 pads, 32/64 heads only (eager fallback otherwise).

### 8. Transformer Engine fp8 dense linears (`TELinear`)  **ON**
- Commits: `864e56fbe`, `460a9d861`, `f07adb168` (+ superseded torchao/custom fp8: `b54d2ca98`, `d2ed70184`, `0c37e573a`, `cd3496b4c`, `5338ddf21`, `116cb32ba`, `50bf0d07e`, `5df0f8c32`, `33fe4297f`)
- Files: `torchtitan/quantization/te_linear.py` (new), `config_registry.py` (`_apply_te_dense`), `torchtitan/quantization/*` (custom tensorwise fp8, kept as the torchao-free fallback)
- Knobs: `TE_DENSE=1`, `TE_DENSE_RECIPE=delayed|mxfp8|current|nvfp4`, `TE_DENSE_EXCLUDE`; `FP8_DENSE*` for the non-TE path.
- Effect: 450.8 (custom fp8) -> 457.6 (TE delayed scaling), 449 for MXFP8/NVFP4. Pattern: keep torchtitan's parameter, hold an unregistered `te.Linear`, re-point its weight each forward (FSDP2-compatible). External: TE 2.19 in `/mnt/dgxc/pydeps-te`.
- Numerics: fp8 dense GEMMs (delayed scaling); caveat: FullAC recompute uses newer amax history than the forward (routing can flip on near-ties; MinimalAsyncEP's fixed buffers tolerate it, DeepEP's did not).

### 9. `TE_REDUCE_AMAX=0`  **ON**
- Commit: `3624a0dc0`. File: `te_linear.py`
- What: TE reduces fp8 amaxes with a synchronous all-reduce at every outermost autocast exit -- one per TELinear per pass, 520 tiny fp32 all-reduces per step, 4.4% exposed. FSDP gathers identical weights on every rank, so the reduction is unnecessary.
- Effect: 500.3 -> 506.1 (collapsed). Only +1.2% of the 4.4%: the rest was rank skew the barrier had been absorbing. Numerics: per-rank fp8 scales.

### 10. Eager-site fusions: copy-free grouped low-rank projection, in-place indexer RoPE  **ON**
- Commits: `88ff93074`, `674a1b93d`. Files: `attention.py` (`_GroupedLowRankProj` strided bmm), `compressor.py`
- Effect: 457.6 -> 471.0. Numerics: bitwise.

### 11. TE fused mHC kernels  **ON**
- Commit: `c3d0a85aa`. Files: `torchtitan/models/deepseek_v4/mhc.py`, `model.py` (residual stream kept as [T, D, n])
- Knob: `TORCHTITAN_TE_MHC=1`. Effect: 482.8 -> 500.3. Also fixes upstream torchtitan's HcPost, whose comb term equals the residual (does not mix). MTP unsupported on this path.
- External patch (NOT in git): `/mnt/dgxc/pydeps-te/transformer_engine/pytorch/triton/mhc.py`, `mHCProjectionOp.backward`: `TE_MHC_BF16_GRAD_PHI=1` casts the [M,32] grad down to bf16 instead of the [T, n*C] activation up to fp32 (fp32 accumulation kept). 607.2 -> 614.2 (balanced). Re-apply after any TE reinstall; a PR against TE is the right home.

### 12. Microbatch variants and balanced-routing variants  **ON (9x balanced)**
- Commits: `8c168fe60`, `d3e87c3fb`, `7f4ddc202`, `e99ec2c3a`, `664ae74dd`, `401e75b58`, `88ff72894`, `c477ba5fe`, `4f7189a15` (profile), `10994d078`, `daf630cd4`, `f30d6d05c`, `73084a41b` (EP variants)
- File: `config_registry.py`. Knob: config name (`_7x` ... `_16x`, `_Nx_balanced`, `_ep8/_ep32`, `_profile`). `config.debug.moe_force_load_balance` for the balanced regime.
- Effect: 7x is the fill point without offload (587.5 balanced); 9x with units 15-19 (685.7).

### 13. `FSDP_PREFETCH_DEPTH`  **ON (=2)**
- Commit: `4f7189a15`. File: `torchtitan/distributed/fsdp.py`
- Effect: 517.6 -> 519.1 (collapsed 13x), neutral within spread, +0.8 GiB. Kept on. Interaction: see unit 17 FIX.

### 14. MinimalAsyncEP row-copy launch geometry  **ON**
- Commit: `882bb741e`. Files: `minimal_async_ep/api.py`, `kernels.py`
- Knobs: `MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4` (`_BLOCK_N` 2048). Microbench: 2.70 -> 2.24 ms per dispatch leg (`/mnt/dgxc/bench_ep_copy.py`).
- Effect: 578.5 -> 587.5 (balanced 7x). Numerics: bitwise.

### 14b. MinimalAsyncEP TMA-store row copy  **ON**
- Commit: `04064469a` (branch `dsv4_tma_copy`, merged). File: `minimal_async_ep/kernels.py` (`_tma_copy_rows_to_peer_ptrs_kernel`)
- Knob: `MINIMAL_ASYNC_EP_COPY_TMA=1`. Ordinary vectorized stores into peer memory saturate near 620 GB/s per direction; TMA bulk stores (Triton 3.8 device-side tensor descriptors, one per peer, each bf16 row as a [cols/256, 256] box because TMA boxes cap at 256 per dim) get closer to line rate. 2-GPU bench: 2.25 -> 2.00 ms per leg. EP=2, bf16, cols % 256 only; stock kernel otherwise. Needs `triton.set_allocator` (set lazily).
- Effect: 738.3 -> **743.8** at 9x balanced (job 1112, plateau 745-747). Numerics: bitwise.

### 15. `MINIMAL_ASYNC_EP_POOL_FACTOR`  **ON (=1.25, balanced only)**
- Commits: `73084a41b`, `987fa5996`. Files: `minimal_async_ep/api.py`, `torchtitan/models/common/token_dispatcher.py`
- What: bounds the receive pool and the capacity-padded routed activation at factor x the expected receive instead of ep_size x. Required for EP>2 (90 GB/slot at EP=32 otherwise). Overflow trips a device-side assert.
- Effect: 587.5 -> 589.1, -3.6 GiB (balanced). **Balanced-regime knob**: collapsed routing overflows it. The 3 hash-routed layers are not forced-balanced (25% headroom).

### 16. Packed expert weights + FSDP2 direct all-gather  **ON**
- Commits: `2e1331d0f`, `56cbfc6b7`, `1f2dd2e05`, `222d24885`, `4902b9c54`, **`b8672e291`** (ordering fix: never wait on the compute stream before a prefetched gather; allocate the unsharded storage on the gather stream)
- Files: `torchtitan/distributed/fsdp_direct_gather.py` (new; patches `_fsdp_collectives.foreach_all_gather` / `foreach_all_gather_copy_out`), `torchtitan/models/common/moe.py` (GroupedExperts packed layout), `torchtitan/models/deepseek_v4/sharding.py`, `torchtitan/models/deepseek_v3/parallelize.py` (import hook)
- Knobs: `MOE_PACKED_EXPERT_WEIGHTS=1 FSDP_DIRECT_GATHER=1`
- What: FSDP2 copies every all-gather out of a staging buffer on the compute stream (1.85% of the step for expert weights, ~17 GiB in-flight staging). A one-parameter group sharded on dim 0 gathers straight into its unsharded storage. Experts packed as [3E, F*D] dim-0 chunks (NOT [E,3,N]: batch-strided GEMM weight + three zero-filled select_backward passes = -9%, job 1080).
- Effect: 589.1 -> 591.9 at 7x, -17 GiB; the memory fit 8x without offload: 607.2 (balanced). With the ordering fix, 685.7 -> **723.6** at 9x (job 1100): the first version's compute-stream wait had turned prefetched expert gathers into just-in-time gathers (92% of all-gather time exposed). Numerics: init is no longer bitwise with the 3-parameter layout (DTensor random init keyed to the global shape; each region keeps its std); state-dict key changes (`w_3EN`).
- PR notes: the FSDP patch is generic torch-internals monkeypatching; upstream FSDP2 could take the single-param fast path natively. Checkpoint adapters for the packed key are not written.

### 16b. FSDP2 direct reduce-scatter  **ON**
- Commit: `e4c076165` (branch `dsv4_direct_rs`, merged). File: `torchtitan/distributed/fsdp_direct_reduce_scatter.py` (new), `parallelize.py` import hook
- Knob: `FSDP_DIRECT_REDUCE_SCATTER=1`. Twin of unit 16: for a one-parameter dim-0 group with an unpadded contiguous gradient already in the reduce dtype, the gradient's flat view is the collective's input; the staging buffer (6.4 GB per expert layer) and the `chunk_cat` on the compute stream (173 ms of idle per two steps) disappear. Implemented as a proxy over the ReduceScatter comm's `allocate` plus a no-op copy-in when the input already is the gradient; stock path otherwise.
- Effect: 723.6 -> **733.1** at 9x balanced (job 1104). Numerics: bitwise (debugmodel 1103).

### 16c. cuBLAS 13.8 preload for the dense GEMMs  **ON**
- Commit: `937de5bf4` (launcher `CUBLAS_NEW=1`, originally for TE grouped GEMMs). External: `/mnt/dgxc/cublas-new/nvidia/cu13/lib` (`nvidia-cublas==13.8.0.4`); LD_PRELOAD because torch loads its bundled 13.1 by absolute path.
- Effect: 733.1 -> **738.3** at 9x balanced (job 1106), plateaus disjoint. Numerics: library version only.

### 16d. Compressor wkv/wgate as bf16-in/fp32-out GEMMs  **ON**
- Commits: `2b870a0cc` (+ `2e8cfcebb`, `e50ce267d`, `ffa577140`: the same idea in `CastLinear.forward` behind `LINEAR_BF16_FP32OUT`, inert for this model since the compressor uses the base `Linear`; kept as a general option). Branch `dsv4_linear_bf16out`, cherry-picked.
- Files: `torchtitan/models/deepseek_v4/compressor.py`, `torchtitan/models/common/linear.py`
- Knob: `COMPRESSOR_BF16_GEMM=1`. The compressor ran its two projections inside `torch.autocast(float32)`, casting x [T, 4096] and both weights up to fp32 on every call (forward and FullAC recompute) -- the [T, 4096] `copy_` family in every profile. bf16 is exact in TF32, so a bf16 GEMM with fp32 accumulation/output (the router-gate Function) computes the same products.
- Effect: 738.3 -> 744.0 alone (job 1118), -6 GiB; with the TMA copy **747.1** at 9x balanced (job 1120, plateau 747-753). Numerics: accumulation order (1e-5 on the debugmodel).

### 16e. Residual-stream gradient chain through TE's mHC kernels  **ON (small)**
- Commits: `acc062d1c`, `b73c16aca` (branch `dsv4_mhc_gradchain`, cherry-picked). Files: `torchtitan/models/deepseek_v4/mhc.py` (`_HcPreChain`, `_GradHolder`), `model.py` (pass-through residual). External TE patch (side-installed copy): `fused_grad_x_acc_buffer` may be bf16 and may be a holder resolved at backward.
- Knob: `TORCHTITAN_MHC_GRAD_CHAIN=1`. x's two consumers (HcPre's TE projection+aggregate, HcPost's residual) become one: a wrapper Function returns a pass-through view of x for HcPost, so HcPost's residual gradient arrives as the pass-through's gradient and is the buffer TE's kernels read-modify-write into. Saves inputs only (FullAC-safe; the first version pinned every block's residual and OOMed) and rebuilds TE's small forward graph in backward.
- Effect: paired 40-step 9x balanced: **756.8 vs 753.9** mean over steps 21-40 (job 1129 vs 1130), +0.4%; the 20-step A/B (1126: 756.1 vs 747.1) overstated it because the control itself ran ~1% faster that day. Numerics: debugmodel tracks within 7e-5 over 20 steps; 40-step losses 3.186 vs 3.188.

### 17. Optimizer-state offload (chunked Adam moments, plain and layer-wise)  **OFF**
- Commits: `78f8a5794`, `b2bf7c07b`, `095ecf8a9`, `00a248724`, `83e17b9b6`, `95e826040`, `c904332aa`, `a74bcd0f7`, `afce197bc`, FIX `70476fdc9`
- Files: `torchtitan/components/optimizer/state_offload.py` (new), `optimizer.py`
- Knobs: `OPT_STATE_OFFLOAD=1`, `OPT_STATE_OFFLOAD_LAYERWISE=1`, `_AHEAD`, `_CHUNK_GIB`, `_DEBUG`
- Effect: plain -5.6% (8x collapsed); layer-wise -0.6% collapsed but -4.1% balanced (side-stream traffic hid in the EP barrier's idle time, which balance removes). Frees 33 GiB. FIX `70476fdc9`: the layer-wise wait/launch distance must follow `FSDP_PREFETCH_DEPTH` (depth 2 let a layer's all-gather read half-updated weights: silent bad loss, job 1036).

### 18. FullAC block-input offload  **OFF**
- Commits: `e412f6b7a`, `8c0e697fb`, `857c3b14b`, `534194475`, `03556702a`, `7d00c041c`, `c50ef5f74`
- File: `torchtitan/distributed/block_input_offload.py` (new), `parallelize.py`
- Knob: `TORCHTITAN_BLOCK_INPUT_OFFLOAD=1`. Bitwise. -68 GiB at 7x. Lessons in commits: release restored inputs after the block above's backward; issue D2H at block entry.
- Effect: collapsed regime +2.7% (513.9 at 13x); balanced regime every offload point loses to 7x without it. Kept for memory-constrained use.

### 19. cuDNN DSA backward: `TORCHTITAN_DSA_DETERMINISTIC=0`  **ON**
- Commit: `93f7014a0`. File: `cudnn_dsa.py`
- What: deterministic mode forces the generic M64 kernel with per-CTA dKV shards (20.0 GiB scratch at 8x) plus a fold kernel; the atomic path needs 0.19 GiB and runs ~2x faster on the backward kernel.
- Effect: 614.2 -> 677.9 at 8x (+10.4%), -13 GiB; 9x then fits: **685.7**. Numerics: dK/dV summation order varies run to run (as in every flash-attention backward). Default stays deterministic.

---

## Part B -- second session's units (author `clowman`, 2026-09-21), on the same branch

### 20. Persistent DSA backward workspace  **OFF at 32 GPUs, ON at 128**
- Commit: `7fc610c56`. Knob: `TORCHTITAN_DSA_PERSISTENT_WORKSPACE` (default 1; the 685.7 run sets 0). +2.0% at 128 GPUs (allocator stalls), -1.5% and +16 GiB at 32. Largely moot once unit 19 shrinks the scratch to 0.19 GiB -- re-measure.

### 21. Bounded SwiGLU  **ON by default**
- Commits: `54cca37cf`, `d23fb9a55`, `f7f66b262`. Files: `torchtitan/models/common/swiglu_bounded.py`, `activation.py`, `moe.py`. Knob `TORCHTITAN_SWIGLU_BOUNDED` (default on). +2.6% at 32 GPUs on r03 in the collapsed regime; not visible on r02 (995 vs 942). Changed the debugmodel's bitwise reference at the 1e-5 level.

### 22. C4 local streaming dataset + 12x hero configs  **infra**
- Commit: `5320d5534`. 128-GPU hero runs: 515.0 (job 970), logs in `gb300/logs/`.

### 23. HybridEP dispatcher variant  **CLOSED**
- Commits: `b066f2c47`, `937c20622`. Crashes at init (NVLink domain 4 vs EP=2) and on the FullAC recompute shape mismatch.

---

## Part C -- measured and closed (code retained as documentation of the measurement)

| unit | commits | result |
|---|---|---|
| fp8 all-gather of expert weights (Megatron #5470) | `9b3798776`, `f13f059c6`, `b415d0786` | -0.9%: gathers already 95% overlapped, dequant inline HBM-bound. Contains a real fix: torchtitan's SPMD shard op strips tensor subclasses (`_preserve` namespace). |
| TE MXFP8 grouped experts (`te_grouped.py`) | `ba8be8c19` .. `66a0f00e4` | 1.04-1.22x per layer at kernel level, but fp8 weight cache is 277 GB and the chain holds +27 GiB at the backward peak: OOM at 6x. Needs cuBLASLt >= 13.3 (`CUBLAS_NEW=1`). |
| DeepEP dispatcher variant | `b4914341c` | -5.6% (host-synced compact layout). |
| Two-microbatch EP overlap (`dual_microbatch`) | `016e6aefc`, `9271189b0`, `fdc082753`, `c3bc027a2` | -6% collapsed, -20% balanced: the split costs more than the ~6% copy it hides. Knob `DUAL_MB=1`, default off. |
| Shared-expert overlap, both stream placements | branches `dsv4_shared_expert_overlap`, `dsv4_dispatch_overlap` | 0 gain: nothing runs "beside" the SM-resident copy kernel. |
| FP8 dispatch / copy engines for the EP copy | benches `/mnt/dgxc/bench_ep_copy.py`, `bench_ce_copy.py` | ~0.3% / loses after the gather (772 GB/s CE ceiling). |
| EP=8 / EP=32 | unit 12 variants | 537.7 / ~523 vs 587.5: dispatch copy goes 1/2 -> 31/32 remote. |
| Selective AC configs | `bb45c44eb`, `e374c3234`, `9e029ef9f`, `2cbf22f2d` | slower than FullAC at every batch. |
| MXFP8 expert GEMMs via torchao path | bench `/mnt/dgxc/bench_mxfp8_experts.py` | 1.10x fwd+bwd, 1.0x fwd: ~1.7% ceiling under FullAC, not built. |
| Residual-gradient accumulation fusion | analysis | byte-neutral without a chained fp32 grad_output (TE kernels take bf16). |

## Part D -- tooling (small PRs or keep local)
- `6f225c524` profiler `PROFILER_WITH_STACK`, debugmodel `DEBUG_PROFILE` knob.
- Cluster-side, not in repo: `/mnt/dgxc/attrib_trace.py`, `/mnt/dgxc/profiles/*/analyze_trace.py`, `exposed_comm.py`, `memcpy_stalls.py`, the `bench_*.py` microbenchmarks, the OOM-crawl guard script.

## Suggested PR order (by dependency)
1. Unit 1 (recipe/launcher) -> 2. Units 2+4 (cuDNN DSA + fused RoPE, with the int64 fix) -> 3. Unit 7 (indexer) -> 4. Unit 8+9 (TE dense + amax) -> 5. Unit 10 (eager fusions) -> 6. Unit 11 (TE mHC; TE-side patch as a separate upstream PR) -> 7. Units 5+13 (FSDP policies) -> 8. Units 6+14+14b+15 (MinimalAsyncEP: slots, geometry, TMA copy, pool factor) -> 9. Units 16+16b (packed experts + direct gather + direct reduce-scatter) -> 10. Unit 19 (DSA determinism knob) -> 11. Units 17+18 (offloads, default off) -> 12. Unit 12 configs -> 13. Part C as one "measured experiments" PR or dropped.

---

## This session's optimizations, in order, with config / branch / TFLOP/s

All configs are functions in `torchtitan/models/deepseek_v4/config_registry.py`
with the prefix `deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense`
(written `…tedense` below); the optimizations themselves are env knobs, so the
config only sets microbatch (`_Nx`), routing regime (`_balanced`) and EP degree.
"Branch" is where the change was developed; everything marked merged is also
on `dsv4_te_mhc`. Numbers are step-20 TFLOP/s per GPU, 8 nodes x 4 GB300,
seq 8192, before -> after, same day and rack unless noted. The regime switch
(collapsed -> balanced routing) is a measurement change, not an optimization,
and is listed where it happened.

| # | optimization (knob) | config | branch | TFLOP/s | job |
|---|---|---|---|---:|---|
| 1 | TE fp8 dense linears, delayed scaling (`TE_DENSE=1 TE_DENSE_RECIPE=delayed`) | `…tedense` (6x) | `dsv4_te_experts` -> `dsv4_te_mhc` | 450.8 -> 457.6 | 840 |
| 2 | eager fusions: strided-bmm low-rank projection, in-place indexer RoPE | `…tedense` | `dsv4_eager_fusions` | 457.6 -> 471.0 | 852 |
| 3 | 7x microbatch | `…tedense_7x` | `dsv4_eager_fusions` | 471.0 -> 482.8 | 853 |
| 4 | TE fused mHC kernels (`TORCHTITAN_TE_MHC=1`) | `…tedense_7x` | `dsv4_te_mhc` | 482.8 -> 500.3 | 880 |
| 5 | fused-RoPE int64 fix (9x+ NaN) (FIX) | `…tedense_9x` | `dsv4_te_mhc` | enables 9x+ | 923/924 |
| 6 | `TE_REDUCE_AMAX=0` | `…tedense_7x` | `dsv4_te_mhc` | 500.3 -> 506.1 | 940 |
| 7 | block-input offload (`TORCHTITAN_BLOCK_INPUT_OFFLOAD=1`) + 12x | `…tedense_12x` | `dsv4_te_mhc` | 506.1 -> 513.9 (collapsed) | 942 |
| 8 | 13x on the merged branch | `…tedense_13x` | `dsv4_te_mhc` | 513.9 -> 517.6 (collapsed) | 996 |
| — | **regime: forced balanced routing** (`_balanced` configs; Megatron's benchmark mode); EP barrier 8% -> 0 | `…tedense_7x_balanced` | `dsv4_te_mhc` | 506.1 -> 578.5 (7x, no offload) | 1008 |
| 9 | `FSDP_PREFETCH_DEPTH=2` | `…tedense_13x` | `dsv4_te_mhc` | 517.6 -> 519.1 (neutral, kept) | 1006 |
| 10 | MinimalAsyncEP copy geometry (`…COPY_BLOCK_M=16 …COPY_WARPS=4`) | `…tedense_7x_balanced` | `dsv4_te_mhc` | 578.5 -> 587.5 | 1015 |
| 11 | `MINIMAL_ASYNC_EP_POOL_FACTOR=1.25` | `…tedense_7x_balanced` | `dsv4_te_mhc` | 587.5 -> 589.1 | 1075 |
| 12 | packed expert weights + FSDP direct all-gather (`MOE_PACKED_EXPERT_WEIGHTS=1 FSDP_DIRECT_GATHER=1`) | `…tedense_7x_balanced` | `dsv4_te_mhc` | 589.1 -> 591.9, -17 GiB | 1084 |
| 13 | 8x on the freed memory | `…tedense_8x_balanced` | `dsv4_te_mhc` | 591.9 -> 607.2 | 1085 |
| 14 | TE grad_phi down-cast (`TE_MHC_BF16_GRAD_PHI=1`, TE side patch) | `…tedense_8x_balanced` | `dsv4_te_mhc` + `/mnt/dgxc/pydeps-te` | 607.2 -> 614.2 | 1091 |
| 15 | DSA backward atomic dKV (`TORCHTITAN_DSA_DETERMINISTIC=0`) | `…tedense_8x_balanced` | `dsv4_te_mhc` | 614.2 -> 677.9, -13 GiB | 1095 |
| 16 | 9x on the freed memory | `…tedense_9x_balanced` | `dsv4_te_mhc` | 677.9 -> 685.7 | 1097 |
| 17 | direct-gather ordering fix (no compute-stream wait) (FIX) | `…tedense_9x_balanced` | `dsv4_te_mhc` | 685.7 -> 723.6 | 1100 |
| 18 | FSDP direct reduce-scatter (`FSDP_DIRECT_REDUCE_SCATTER=1`) | `…tedense_9x_balanced` | `dsv4_direct_rs` (merged) | 723.6 -> 733.1 | 1104 |
| 19 | cuBLAS 13.8 preload for dense GEMMs (`CUBLAS_NEW=1`) | `…tedense_9x_balanced` | `dsv4_te_mhc` (launcher) | 733.1 -> 738.3 | 1106 |
| 20 | TMA-store EP row copy (`MINIMAL_ASYNC_EP_COPY_TMA=1`) | `…tedense_9x_balanced` | `dsv4_tma_copy` (merged) | 738.3 -> 743.8 | 1112 |
| 21 | compressor bf16-in/fp32-out GEMMs (`COMPRESSOR_BF16_GEMM=1`) | `…tedense_9x_balanced` | `dsv4_linear_bf16out` (merged) | 738.3 -> 744.0 alone; with 20: 747.1 | 1118 / 1120 |
| 22 | mHC residual-gradient chain (`TORCHTITAN_MHC_GRAD_CHAIN=1`, TE side patch) | `…tedense_9x_balanced` | `dsv4_mhc_gradchain` (merged) | 747.1 -> 756.1 (20-step); 753.9 -> 756.8 paired 40-step | 1126 / 1129 vs 1130 |

Measured and closed this session (code kept where it documents the measurement):
fp8 expert all-gather (-0.9%, `dsv4_fused_output_rope`), TE MXFP8 grouped experts
(OOM at 6x), DeepEP (-5.6%), NVFP4/MXFP8 dense (449), optimizer-state offload
plain/layer-wise (-5.6% / -4.1% balanced), selective AC (slower at every batch),
EP=1 (365.7), dual-microbatch overlap (-6% / -20% balanced, `dsv4_dual_microbatch`),
shared-expert overlap both placements (0, `dsv4_shared_expert_overlap`,
`dsv4_dispatch_overlap`), FP8 dispatch (~0.3%), copy engines (772 GB/s ceiling),
EP=8/32 (537.7 / ~523), MXFP8 expert GEMMs (1.10x), pool factor 1.1 (0),
prefetch depth 3 (0), MXFP8 dense under cuBLAS 13.8 (-0.6%), 14x (OOM), 10x (OOM).

Collapsed-regime headline for reference: 519.1 (13x + block-input offload,
job 1006). Balanced headline: 756.1 (job 1126) / 756.8 40-step mean (job 1129);
Megatron-LM's reference for this model is 748.
