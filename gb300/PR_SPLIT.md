# PR split of `dsv4_te_mhc`

`dsv4_te_mhc` (164 commits over upstream `6857b67b`, tip `1457b7ba`) has been
cut into 47 stacked branches `split/01-*` .. `split/47-*`, based on
`origin/main` (`f5d6cb07`, which already carries the num_stages=2 configs as
#23). Each branch contains exactly one optimization, fix, experiment or
config family from `gb300/OPTIMIZATIONS.md`; each branch's base is the branch
before it, so a PR opened with that base shows only its own change. Merge
bottom-up. Every cherry-picked commit carries a `Cherry-picked-from:` (or
`cherry picked from commit`) trailer pointing at the original commit on
`dsv4_te_mhc`; commits that had to be split by path or reworded say so.

The tree at `split/47-docs` is identical to `dsv4_te_mhc` except for:

1. `deepseek_v4_flash_8k_gb300_batched_stages2` is main's copy (#23) and
   `_batched_stages2_profile` comes from main; the branch's duplicate
   definition, which shadowed main's, is dropped (`split/07`).
2. `..._tedense_10x_balanced` gets its `return config` back; the
   dual-microbatch commit 016e6aef had swallowed it, so on `dsv4_te_mhc` that
   config returns `None` (`split/35`).
3. `..._densenever_tedense_9x` (commit 9b668566, a NaN-bisect config) is
   dropped: it calls `..._cudnn_full_ep2_densenever_tedense`, which was never
   defined, so it raised `NameError` on `dsv4_te_mhc`. The NaN it bisected was
   fixed by the int64 fused-RoPE offsets (`split/10`).
4. Top-level function order in `config_registry.py` differs (function-level
   merge); no semantic change.

Not in this repo and therefore not on any branch: the two Transformer Engine
patches in the side-installed copy (`TE_MHC_BF16_GRAD_PHI`, and the
bf16/holder `fused_grad_x_acc_buffer` used by `split/46`); the cluster-side
`sbatch_rack.sh`, benches and trace-analysis scripts; the ledger
`FLASH_8K_RESULTS.md` (branch `dsv4_flash_64xgb300`). Branch
`dsv4_dispatch_overlap` (`TORCHTITAN_DISPATCH_OVERLAP`, measured 0 gain) was
never merged into `dsv4_te_mhc` and is not included.

| # | branch | base | commits | +/- | OPTIMIZATIONS.md unit | status | contents |
|---|---|---|---|---|---|---|---|
| 01 | `split/01-flex-max-autotune-per-layer` | `main` | 1 | +48/-2 | -- | infra | flex autotune knob (was bundled into the MinimalAsyncEP restore commit) |
| 02 | `split/02-restore-minimal-async-ep` | `split/01-flex-max-autotune-per-layer` | 1 | +2304/-13 | -- | ON | MinimalAsyncEP dispatcher, deprecated upstream in #4627, restored |
| 03 | `split/03-gb300-launcher` | `split/02-restore-minimal-async-ep` | 3 | +174/-0 | 1 | ON | Slurm launcher (base + EXTRA_PYTHONPATH + CPATH) |
| 04 | `split/04-gb300-bf16-recipe` | `split/03-gb300-launcher` | 5 | +106/-2 | 1 | ON | full-bf16 recipe, fused bf16 Adam, all-16 backward tiles, fp32-params/fastdata/free-tiles variants |
| 05 | `split/05-selective-ac-sweep-config` | `split/04-gb300-bf16-recipe` | 2 | +44/-2 | C | CLOSED | selective-AC sweep config + MinimalAsyncEP accepting SelectiveAC |
| 06 | `split/06-indexer-deterministic-select` | `split/05-selective-ac-sweep-config` | 1 | +15/-4 | 2 | FIX | deterministic Indexer.select (stable sort); prerequisite for the cuDNN A/B test |
| 07 | `split/07-cudnn-dsa-backward` | `split/06-indexer-deterministic-select` | 9 | +617/-34 | 2 | ON | cuDNN fused DSA backward + A/B test + configs (main's #23 stages2 kept, duplicate dropped) |
| 08 | `split/08-cudnn-dsa-forward` | `split/07-cudnn-dsa-backward` | 3 | +186/-13 | 2 | ON | cuDNN DSA forward (both halves off flex) + test coverage + 5x control |
| 09 | `split/09-fp32-matmul-precision-knob` | `split/08-cudnn-dsa-forward` | 1 | +14/-2 | 3 | ON | TORCHTITAN_FP32_MATMUL_PRECISION knob (bfx9 default) |
| 10 | `split/10-fused-output-rope` | `split/09-fp32-matmul-precision-knob` | 3 | +214/-9 | 4 | ON | fused output inverse RoPE in the DSA Function, incl. the int64 offset FIX |
| 11 | `split/11-fsdp-dense-never` | `split/10-fused-output-rope` | 3 | +122/-9 | 5 | ON | FSDP dense-never reshard policy + debugmodel pair config |
| 12 | `split/12-fp8-expert-allgather` | `split/11-fsdp-dense-never` | 4 | +270/-0 | C | CLOSED | fp8 all-gather of expert weights (-0.9%); contains the spmd shard subclass fix |
| 13 | `split/13-maep-4-receive-slots` | `split/12-fp8-expert-allgather` | 1 | +7/-1 | 6 | FIX | MinimalAsyncEP 4 receive slots (MINIMAL_ASYNC_EP_SLOTS) |
| 14 | `split/14-cudnn-indexer` | `split/13-maep-4-receive-slots` | 3 | +130/-4 | 7 | ON | cuDNN fused indexer top-k |
| 15 | `split/15-fp8-dense-torchao-custom` | `split/14-cudnn-indexer` | 9 | +292/-0 | 8 | superseded | torchao rowwise / custom tensorwise fp8 dense linears |
| 16 | `split/16-te-dense-linear` | `split/15-fp8-dense-torchao-custom` | 3 | +155/-0 | 8 | ON | Transformer Engine fp8 dense linears (TELinear, TE_DENSE_RECIPE, TE_DENSE_EXCLUDE) |
| 17 | `split/17-grouped-lowrank-proj-bmm` | `split/16-te-dense-linear` | 1 | +36/-1 | 10 | ON | copy-free grouped low-rank output projection (strided bmm) |
| 18 | `split/18-indexer-inplace-rope` | `split/17-grouped-lowrank-proj-bmm` | 1 | +17/-3 | 10 | ON | indexer query RoPE in place with the fused Triton kernel |
| 19 | `split/19-microbatch-variant-configs` | `split/18-indexer-inplace-rope` | 7 | +71/-36 | 12 | configs | microbatch variants 7x..16x and profile variants |
| 20 | `split/20-deepep-variant` | `split/19-microbatch-variant-configs` | 2 | +32/-1 | C | CLOSED | DeepEP dispatcher variant (launcher + _swap_ep_backend config) |
| 21 | `split/21-cublas-13.8-preload` | `split/20-deepep-variant` | 1 | +6/-0 | 16c | ON | CUBLAS_NEW=1 cuBLAS 13.8 preload (launcher) |
| 22 | `split/22-te-grouped-experts` | `split/21-cublas-13.8-preload` | 9 | +326/-0 | C | CLOSED | TE MXFP8 grouped experts (te_grouped.py) |
| 23 | `split/23-te-fused-mhc` | `split/22-te-grouped-experts` | 1 | +52/-2 | 11 | ON | TE fused mHC kernels (TORCHTITAN_TE_MHC) |
| 24 | `split/24-optimizer-state-offload` | `split/23-te-fused-mhc` | 10 | +390/-1 | 17 | OFF | optimizer-state offload, plain and layer-wise, incl. the prefetch-depth race FIX |
| 25 | `split/25-block-input-offload` | `split/24-optimizer-state-offload` | 7 | +173/-0 | 18 | OFF | FullAC block-input offload |
| 26 | `split/26-sac-ep1-experiment-configs` | `split/25-block-input-offload` | 4 | +55/-0 | C | CLOSED | selective-AC, FullAC-4x reference and EP=1 experiment configs |
| 27 | `split/27-te-reduce-amax` | `split/26-sac-ep1-experiment-configs` | 1 | +17/-6 | 9 | ON | TE_REDUCE_AMAX=0 |
| 28 | `split/28-dsa-persistent-workspace` | `split/27-te-reduce-amax` | 1 | +41/-2 | 20 | OFF@32 | persistent cuDNN DSA backward workspace |
| 29 | `split/29-c4-local-dataset` | `split/28-dsa-persistent-workspace` | 1 | +45/-0 | 22 | infra | c4_local streaming dataset + 12x hero/reference configs |
| 30 | `split/30-hybridep-variant` | `split/29-c4-local-dataset` | 2 | +11/-1 | 23 | CLOSED | HybridEP dispatcher variant (launcher + config) |
| 31 | `split/31-bounded-swiglu` | `split/30-hybridep-variant` | 3 | +160/-2 | 21 | ON | bounded SwiGLU over the valid prefix |
| 32 | `split/32-fsdp-prefetch-depth` | `split/31-bounded-swiglu` | 1 | +26/-25 | 13 | ON | FSDP_PREFETCH_DEPTH |
| 33 | `split/33-balanced-routing-configs` | `split/32-fsdp-prefetch-depth` | 6 | +85/-0 | 12 | configs | balanced-routing variants and _force_balanced_routing helper |
| 34 | `split/34-maep-copy-geometry` | `split/33-balanced-routing-configs` | 1 | +13/-1 | 14 | ON | MinimalAsyncEP row-copy launch geometry knobs |
| 35 | `split/35-dual-microbatch` | `split/34-maep-copy-geometry` | 5 | +260/-2 | C | CLOSED | two-microbatch EP overlap (dual_microbatch); also restores the dropped return in _10x_balanced |
| 36 | `split/36-maep-pool-factor` | `split/35-dual-microbatch` | 2 | +21/-0 | 15 | ON | MINIMAL_ASYNC_EP_POOL_FACTOR |
| 37 | `split/37-ep-degree-configs` | `split/36-maep-pool-factor` | 3 | +43/-0 | 12 | configs | EP=8 / EP=32 balanced variants |
| 38 | `split/38-packed-expert-weights` | `split/37-ep-degree-configs` | 4 | +89/-17 | 16 | ON | packed expert weights [3E, F*D] (MOE_PACKED_EXPERT_WEIGHTS) |
| 39 | `split/39-fsdp-direct-gather` | `split/38-packed-expert-weights` | 3 | +123/-0 | 16 | ON | FSDP2 direct all-gather for single-parameter groups, incl. the ordering FIX |
| 40 | `split/40-profiler-with-stack` | `split/39-fsdp-direct-gather` | 1 | +7/-0 | D | tooling | PROFILER_WITH_STACK, debugmodel DEBUG_PROFILE |
| 41 | `split/41-dsa-deterministic-knob` | `split/40-profiler-with-stack` | 1 | +11/-6 | 19 | ON | TORCHTITAN_DSA_DETERMINISTIC knob (atomic dKV path) |
| 42 | `split/42-fsdp-direct-reduce-scatter` | `split/41-dsa-deterministic-knob` | 1 | +96/-0 | 16b | ON | FSDP2 direct reduce-scatter |
| 43 | `split/43-maep-tma-copy` | `split/42-fsdp-direct-reduce-scatter` | 1 | +97/-0 | 14b | ON | MinimalAsyncEP TMA-store row copy |
| 44 | `split/44-linear-bf16-fp32out` | `split/43-maep-tma-copy` | 3 | +28/-0 | 16d | ON (inert here) | LINEAR_BF16_FP32OUT on CastLinear |
| 45 | `split/45-compressor-bf16-gemm` | `split/44-linear-bf16-fp32out` | 1 | +26/-3 | 16d | ON | COMPRESSOR_BF16_GEMM |
| 46 | `split/46-mhc-grad-chain` | `split/45-compressor-bf16-gemm` | 2 | +86/-2 | 16e | ON | mHC residual-gradient chain (needs the patched side-installed TE) |
| 47 | `split/47-docs` | `split/46-mhc-grad-chain` | 35 | +13556/-0 | -- | docs | REPRODUCE_500.md, OPTIMIZATIONS.md, TODOS.md, hero logs and plot |
