# Perf optimization plan — building on the bf16+compile baseline

Companion to [bs_sweep_results.md](bs_sweep_results.md). This is a quick triage of perf knobs we can pull next, ordered by effort, with **rough** estimated per-GPU tflops on 8×B200 starting from those baselines.

## Where we are today

Steady-state, 8×B200, FSDP-only, bf16 + `torch.compile(components=["model","loss"])`, seq_len=8192:

| Model | bs | tflops/gpu | mfu | headroom to bf16 peak (~2.25 PF/s) | headroom to fp8 peak (~4.5 PF/s) |
| --- | --- | --- | --- | --- | --- |
| gpt_oss_20b | 10 | ~573 | 25.5% | 4.0× | 7.9× |
| qwen3_30b (Qwen3-30B-A3B) | 12 | ~455 | 20.2% | 5.0× | 9.9× |

Estimates below are **back-of-envelope only**, calibrated against the relative cost of GEMM vs comm/AC/launch in MoE LLMs at this scale. Treat the `+%` columns as "expected order of magnitude, real result needs to be measured."

## 1. Free wins (no torchao, ~hours of work)

### 1.1 Loosen activation checkpointing for Qwen3

Right now the local `qwen3_30b` helper hard-codes `mode="full"` whenever compile is on. `full` AC recomputes ~all transformer block forward in backward (~+33% compute). gpt_oss_20b already uses `selective` at bs<full-ac threshold in upstream, which is why its mfu is ~5 pp higher.

- **Mechanism**: switch to `mode="selective"` with `selective_ac_option="op"` (recompute only attention sdpa + a few cheap ops), keep `full` only as a fallback.
- **Catch**: At qwen3 bs=12 we are at 94% HBM — selective AC will OOM. Must combine with either reduced bs (bs=8–10) **or** fp8 (item 2) to free memory for the activations we no longer recompute.
- **Estimated lift (Qwen3 at bs=8, selective AC)**: 455 → **520–560** tflops/gpu (+15–22%), mfu 23–25%.
- **Estimated lift (gpt_oss)**: already at selective in upstream; negligible.

### 1.2 Inductor max-autotune for GEMMs

- **Mechanism**: set `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` (or `--compile.mode=max-autotune` if we add it as a knob — currently CompileConfig only exposes `enable / components / backend`). Inductor will template-tune the matmul kernels for the actual shapes seen at runtime.
- **Cost**: ~5–15 min added to first-step compile; cache persists in `inductor_cache/`.
- **Estimated lift**: +3–7% on both models. gpt_oss: 573 → **590–615**. qwen3: 455 → **470–490**.

### 1.3 Compile lr-scheduler / optimizer step

The trainer currently only compiles `model` and `loss`. The fused-Adam-style optimizer step is many small kernels; on B200 it's launch-overhead-bound.

- **Mechanism**: extend [torchtitan/config/configs.py](../torchtitan/config/configs.py) `CompileConfig.components` to accept `"optimizer"` and wrap `clip_grad_norm_` + `optimizer.step` in `torch.compile(fullgraph=True)`. Trainer wrappers already use `sl.log_trace_span("optim")` so the boundary is clean.
- **Estimated lift**: +2–5%. gpt_oss: 573 → **585–600**. qwen3: 455 → **465–480**.
- **Risk**: low. Optimizer compile is well-trodden in torchao examples.

Stacking 1.1+1.2+1.3 (Qwen3 at bs=8, selective AC, max-autotune, compiled optim):
- Qwen3-30B-A3B: 455 → **~570–610** tflops/gpu (+25–34%, mfu ~25–27%).
- gpt_oss-20b: 573 → **~610–650** tflops/gpu (+6–13%).

## 2. torchao fp8 (medium effort)

`torchtitan/components/quantization/float8.py` already exposes `Float8LinearConverter` and `Float8GroupedExpertsConverter` — see [float8.md](../torchtitan/components/quantization/float8.md). The local `_qwen3_30b_base` even imports `Float8LinearConverter` but never plumbs it through to `model_registry(..., quantization=...)`. So this is mostly a config-wiring change.

### 2.1 Float8LinearConverter on dense linears (attn proj + lm_head/embed routing)

- **Mechanism**: rowwise dynamic scaling on attention Q/K/V/O and the shared dense MLP. Recipe: `recipe_name="rowwise"`, `filter_fqns=["output", "router.gate", "auto_filter_small_kn"]`, `model_compile_enabled=True`.
- **Compute share affected**: ~15–25% of MoE training time (attention + lm_head; experts excluded).
- **Estimated lift**: +5–10% overall. gpt_oss: 573 → **610–630**. qwen3: 455 → **485–500**.
- **Risk**: low–medium. Rowwise fp8 has converged-loss runs in torchao for llama3-405B; need a short loss-vs-bf16 check on c4 per [.claude/CLAUDE.md](../torchtitan/CLAUDE.md) "Validating Numerics".

### 2.2 Float8GroupedExpertsConverter on MoE experts (big lever)

- **Mechanism**: stack `Float8GroupedExpertsConverter.Config(model_compile_enabled=True)` after the linear converter. Uses torchao scaled grouped GEMMs for the expert path. This is the optimization deepseek_v3's config registry already opts into.
- **Compute share affected**: 55–70% of training time (experts dominate at these sizes).
- **Estimated lift**: +25–40% on both models — but bigger relative win on qwen3 since its `active=3.4B` ratio is more expert-heavy.
- **gpt_oss-20b**: 573 → **720–800** tflops/gpu, mfu 32–36%.
- **Qwen3-30B-A3B**: 455 → **570–640** tflops/gpu, mfu 25–28%.
- **Memory bonus**: fp8 weights for experts halve their resident memory → frees ~30–40 GiB on Qwen3, finally making bs=14 feasible and re-enabling looser AC.
- **Risk**: medium. Grouped GEMM fp8 is newer than per-linear fp8; need stricter loss check.

Stacking 2.1+2.2 with the 1.x wins (selective AC freed by fp8 memory savings):
- **gpt_oss-20b**: 573 → **~820–900** tflops/gpu (+43–57%, mfu ~37–40%).
- **Qwen3-30B-A3B (bs ≥ 12 now safe)**: 455 → **~680–760** tflops/gpu (+50–67%, mfu ~30–34%).

## 3. DeepGEMM grouped fp8 (larger effort)

torchao's grouped scaled GEMM is reasonable but not state-of-art. DeepSeek's [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) ships hand-tuned fp8 grouped GEMMs targeted at Hopper-class hardware and B200. Reported ~1.3–1.7× over best alternatives on the relevant shapes.

- **Mechanism**: write a `DeepGEMMGroupedExpertsConverter` that mirrors `Float8GroupedExpertsConverter` but routes expert forward through `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`. Lives in `torchtitan/experiments/` per [.claude/CLAUDE.md](../torchtitan/CLAUDE.md) experiments rule (third-party kernel dependency).
- **Estimated incremental lift over 2.2**: +5–15% on expert compute, i.e. another +3–10% overall.
- **gpt_oss-20b (on top of 2.x)**: 820 → **870–950** tflops/gpu.
- **Qwen3-30B-A3B (on top of 2.x)**: 680 → **730–820** tflops/gpu, mfu ~32–36%.
- **Risk**: higher. Custom expert-forward needs autograd backward via the same kernel path; need a numerics + perf microbench before integration. Also adds a non-PyTorch dep, so it must stay under `torchtitan/experiments/` (core principle #1 in CLAUDE.md).

## 4. graph_trainer (aggressive, experimental)

`torchtitan/experiments/graph_trainer/` already has [qwen3](../torchtitan/experiments/graph_trainer/qwen3/config_registry.py) and deepseek_v3 entries. It captures forward + loss + backward (and optionally optim) as a single FX graph and applies passes for fusion, comm overlap, and CUDAGraph capture.

- **Mechanism**: switch `--module` from `qwen3` to `graph_trainer.qwen3` (and similarly for gpt_oss once an entry is added). Inherits SimpleFSDP, regional CUDAGraph, async expert comm.
- **Estimated lift over the stack in (3)**: +5–15% additional, mainly from launch-overhead reduction (CUDAGraph capture on a B200 dominated by lots of small MoE kernels) and overlap of expert all-to-all with compute.
- **Risk**: experimental. Not all parallelism combos work; loss must be re-validated.

## Summary

Estimated path from baseline to a stacked configuration (per-GPU tflops, mfu in parens, B200 dense bf16 peak ≈ 2.25 PF/s):

| Stage | gpt_oss-20b | qwen3-30B-A3B | Confidence |
| --- | --- | --- | --- |
| Baseline (today) | 573 (25.5%) | 455 (20.2%) | measured |
| + 1.1–1.3 free wins | 610–650 (+10%) | 570–610 (+30%) | high |
| + 2.x torchao fp8 | 820–900 (+50%) | 680–760 (+60%) | medium |
| + 3 DeepGEMM | 870–950 (+60%) | 730–820 (+70%) | low |
| + 4 graph_trainer | 920–1050 (+70%) | 770–900 (+80%) | low |

## Recommended order

1. **First**: 1.1 selective AC + 1.2 max-autotune on Qwen3 (one-day; no third-party deps; validates the memory model for fp8 next).
2. **Second**: 2.1 → 2.2 — wire `Float8LinearConverter` then `Float8GroupedExpertsConverter` through `_qwen3_30b_base` and the gpt_oss config. Land each independently with a loss-vs-bf16 check on c4.
3. **Third**: 1.3 compiled optimizer step (small, orthogonal — can land any time after 2.x).
4. **Fourth (experimental)**: DeepGEMM converter under `torchtitan/experiments/deepgemm_moe/` with a microbench harness.
5. **Optional**: graph_trainer once 2.x and 3 are stable, as a side-by-side comparison rather than a replacement.

## Sanity checks before each step

Per [.claude/CLAUDE.md](../torchtitan/CLAUDE.md):
- Non-computation changes (1.1, 1.3, partly 4): bitwise-identical loss & grad_norm under `--debug.seed=42 --debug.deterministic`.
- Computation changes (1.2 with new autotune kernels, all of 2.x, 3, 4): short c4 run, compare loss curve vs bf16 baseline via `scripts/loss_compare.py`.
