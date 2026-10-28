# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import torch
from torch.distributed.tensor import Shard

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    default_adamw,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.components.validate import Validator
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import (
    FullAC,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    ComputeLayout,
    MuonComputeShardingConfig,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry


def _gpt_oss_debugmodel(attn_backend: str = "varlen") -> Trainer.Config:
    model_spec = model_registry("debugmodel", attn_backend=attn_backend)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def gpt_oss_debugmodel() -> Trainer.Config:
    return _gpt_oss_debugmodel()


def gpt_oss_debugmodel_flex() -> Trainer.Config:
    return _gpt_oss_debugmodel(attn_backend="flex")


def gpt_oss_20b() -> Trainer.Config:
    model_spec = model_registry("20b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def gpt_oss_120b() -> Trainer.Config:
    model_spec = model_registry("120b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def gptoss20b_bs3() -> Trainer.Config:
    """gpt_oss_20b with local_batch_size 1 -> 3. Measured: 18.29% MFU,
    411.58 TFLOPs/GPU, 61.88GiB peak (job 818).

    Kept as its own config so the bs=3 row of the ladder in
    RESULTS_GPTOSS20B.md stays reproducible. This was the single best
    MFU-per-GiB point on the whole sweep: +4.53 points over the 13.76% stock
    baseline for +8.7GiB, where every later batch increase bought ~1 point for
    tens of GiB. Use this one, not gptoss20b_workers, when the goal is a cheap
    config that leaves memory free for another variable.

    See gptoss20b_bs16 for why disable_cuda_graphs is set.
    """
    config = gpt_oss_20b()
    config.training.local_batch_size = 3
    config.training.disable_cuda_graphs = True
    return config


def gptoss20b_bs16() -> Trainer.Config:
    """gpt_oss_20b with local_batch_size 1 -> 16. Batch-size scaling probe; the
    question this one answers is whether it OOMs.

    Measured points on this ladder (8x B200, otherwise-stock gpt_oss_20b):
        bs=1  (job 806)  13.76% MFU, 309.61 TFLOPs/GPU, 53.19GiB peak
        bs=3  (job 818)  18.29% MFU, 411.58 TFLOPs/GPU, 61.88GiB peak
    Batch size is the right lever because expert_parallel_degree=1 makes every
    rank run grouped GEMMs across all 32 experts; at bs=1 top-4 routing leaves
    only ~1024 tokens per expert, so the GEMMs are launch/latency-bound rather
    than tensor-core-bound. More tokens per step grows the M dimension of every
    expert GEMM and amortizes the optimizer step and the FSDP all-gathers, which
    do not get more expensive.

    Memory model fitted to those two points: total = fixed + a + b*bs, where
    fixed = 20.91e9 * 16 B/param / 8 ranks = ~41.8GiB of fp32 params + grads +
    the two AdamW moments (independent of batch size). 53.19 and 61.88 give
    b = ~4.35GiB per unit of batch and a = ~7.0GiB, so bs=16 projects to
    ~41.8 + 7.0 + 69.5 = ~118GiB against the 178.35GiB limit. That says it
    should fit with ~60GiB of headroom -- but the bs=1 -> bs=3 step already came
    in 14GiB under a linear-in-tokens projection, so the extrapolation is doing
    real work over a 5x jump and OOM is a live outcome. If it OOMs, walk back
    to 8; if it fits, MFU vs bs is likely flattening and the next lever is
    EP=8 or torch.compile rather than more batch.

    disable_cuda_graphs is baked in here rather than passed on the command line.
    Stock gpt_oss leaves CUDA graphs on, but GPT-OSS uses the varlen attention
    backend whose cu_seqlens change shape every step, so graph replay dies at
    step 2 with "CUDA graph tensor inputs must keep the same shape" (reproduced
    at 20b in job 800, and at 120b before that). The baselines this is compared
    against were measured with the same flag, so setting it keeps batch size the
    only variable.

    FullAC, expert_parallel_degree=1, dtype=float32 and no compile are all left
    at their stock values on purpose.
    """
    config = gpt_oss_20b()
    config.training.local_batch_size = 16
    config.training.disable_cuda_graphs = True
    return config


def gptoss20b_workers() -> Trainer.Config:
    """gptoss20b_bs16 plus a fed dataloader. Base config for the batch sweep.

    Stock leaves the dataloader single-process: num_workers=0, so tokenization and
    collation happen inline in the training process and the GPUs sit idle while
    the next batch is built. At bs=16 (1,048,576 tokens/step) that showed up as a
    bimodal per-step MFU in job 822 -- ~19.4% on most steps, ~8.1% on 4 of 30,
    with nvidia-smi recording utilization dropping to 0% while peak memory stayed
    flat. Raw mean 17.78% vs 19.28% over the unstalled steps.

    num_workers=8 (one per GPU, well inside OMP_NUM_THREADS=24) moves that work
    into separate processes; persistent_workers keeps them alive across the
    infinite-stream restarts; prefetch_factor=4 gives each worker a 4-batch
    runway, so 32 batches are in flight; pin_memory makes the H2D copy async.

    Worker sharding is safe here: HuggingFaceTextDataset applies
    split_dataset_by_node(dp_rank, dp_world_size) and HF datasets 4.7.0 then
    shards each node's stream across workers via ex_iterable.shard_data_sources,
    so workers do not duplicate samples. c4 `en` has 1024 train shards, i.e. 128
    per rank at dp_shard=8, so up to 128 workers stay fed. Above that some
    workers would get no shard.

    Batch size is swept from here on the command line rather than by adding a
    config per point:
        TT_CONFIG=gptoss20b_workers TT_EXTRA="--training.local_batch_size 24"

    Memory ceiling: measured peaks fit total = 41.8 + 7.0 + 4.35*bs GiB, where
    41.8GiB is fp32 params + grads + the two AdamW moments (20.91e9 * 16 B / 8
    ranks, fixed). That model predicted 118.4GiB at bs=16 against a measured
    118.62GiB. Extrapolating to the 178.35GiB limit:
        bs=24 -> ~153GiB, bs=28 -> ~171GiB, bs=30 -> ~180GiB, bs=32 -> ~188GiB
    so the ceiling is around bs=28 and bs=32 should OOM.

    Worth knowing before spending the headroom on batch: the reason there IS
    headroom is FullAC, which is on in stock gpt_oss_20b and recomputes the
    entire forward pass during backward. That recompute is real wall-clock that
    the MFU numerator does not count, so trading the free memory for
    SelectiveAC/MemoryBudgetAC is likely a bigger MFU win than more batch --
    batch already flattened to +1 point from bs=3 to bs=16.
    """
    config = gptoss20b_bs16()
    config.dataloader.num_workers = 8
    config.dataloader.persistent_workers = True
    config.dataloader.pin_memory = True
    config.dataloader.prefetch_factor = 4
    return config


def gptoss20b_noac() -> Trainer.Config:
    """FullAC removed: activation_checkpoint=None. Keeps the fed dataloader.

    Why this should be the biggest remaining lever. Stock gpt_oss_20b runs
    FullAC, which recomputes the entire forward pass during backward. Counting
    forward as 1 unit and backward as 2, FullAC makes a step cost 1 + 1 + 2 = 4
    units against 3 without it, but the MFU numerator only ever counts the
    useful 3. So removing AC should cut ~25% of step time and lift MFU by ~33%
    at the same batch -- roughly 19.3% -> ~25% if memory allows. Nothing else on
    the list (EP=8, compile) has that kind of arithmetic behind it.

    MEASURED (8x B200, steps 9-25, 8 dataloader workers):
        bs=1  16.56% MFU, 372.59 TFLOPs/GPU, 104.84GiB (job 836)
        bs=2  21.23% MFU, 477.71 TFLOPs/GPU, 161.55GiB (job 837)  <- default
        bs=3  OOM, 175.19GiB allocated and 2.30GiB short  (job 838)
        bs=4  OOM, 176.19GiB allocated and 1.41GiB short  (job 832)
        bs=8  OOM                                          (job 833)
    bs=2 is the best result on the whole 20b sweep, beating FullAC's best
    (19.29% at bs=24, job 829) by +1.94 points at comparable memory. The bs=1
    pair against job 806's 13.76% is the clean single-variable AC measurement:
    +2.80 points, +20% relative, i.e. less than the ~33% the FLOP arithmetic
    predicts because some recompute overlaps communication.

    The memory cost is severe, which is why the default batch is 2 and not more.
    Measured peaks under FullAC fit total = 41.8 + 7.0 + 4.35*bs GiB, where
    41.8GiB is fp32 params + grads + the two AdamW moments (fixed). FullAC stores
    only layer boundaries, so 4.35GiB/batch-unit is near the floor; without AC
    every intermediate in all 24 layers stays live, measured at ~34.5GiB per
    batch unit -- about 8x. The bs=24 FullAC operating point is unreachable here.

    Sweep batch size on the command line if revisiting:
        TT_CONFIG=gptoss20b_noac TT_EXTRA="--training.local_batch_size 1"

    bs=2 sits at 90.58% of memory. That is under the ~95% thrash threshold from
    job 831 and ran with zero stalls, but there is little headroom. Watch for
    that failure mode when pushing: with
    PYTORCH_ALLOC_CONF=expandable_segments:True the allocator does not
    necessarily raise OutOfMemoryError, it just thrashes near 99% and collapses
    to 2-3% MFU while still reporting steps. Any point whose peak memory sits
    above ~95% should be read as over the wall even if it does not crash.

    Next thing to try is SelectiveAC or MemoryBudgetAC, which should dominate
    both endpoints -- partial recompute costs a fraction of FullAC's tax while
    freeing enough memory for a batch well above 2.

    expert_parallel_degree=1, dtype=float32 and no compile stay stock so AC is
    the only variable against gptoss20b_workers.
    """
    config = gptoss20b_workers()
    config.activation_checkpoint = None
    config.training.local_batch_size = 2
    return config


def gptoss20b_selac() -> Trainer.Config:
    """SelectiveAC instead of FullAC or nothing. Should dominate both endpoints.

    The 20b sweep so far has only measured the two extremes of the AC axis:
        FullAC,  bs=24  ->  19.29%  (155.14GiB, job 829)
        no AC,   bs=2   ->  21.23%  (161.55GiB, job 837)
    Both are constrained by the same thing from opposite directions. FullAC
    recomputes every transformer block, so it pays a recompute tax on every step
    that the MFU numerator never counts, but it is cheap enough in memory
    (~4.35GiB/batch-unit) to run bs=24. No AC pays no tax but costs ~34.5GiB per
    batch unit, which caps the batch at 2 and leaves only 90.58% memory.

    SelectiveAC sits between them: it saves the outputs of ops that are expensive
    to recompute and recomputes every second matmul (see
    activation_checkpoint.py::SelectiveAC and _get_default_save_ops). The default
    force_recompute_mm_shapes_by_fqns=["moe.router.gate"] keeps the router gate
    mm out of the save set, which matters here because gpt_oss routes top-4 of 32
    experts. So the expectation is a fraction of FullAC's recompute tax at a
    fraction of no-AC's memory, i.e. a usable batch AND most of the 21.23%.

    Batch is a guess pending measurement. If SelectiveAC's activation cost per
    batch unit lands midway between 4.35 and 34.5GiB, call it ~15GiB, then
    41.8 + 7.0 + 15*bs projects bs=8 -> ~169GiB, which is already near the wall.
    Default is bs=8 and the sweep below walks down from there if it OOMs.

    On the 120b, SelectiveAC measured 8.22% against FullAC's 8.12% at equal
    batch -- a small win on recompute alone. The real prize here is that it
    should permit a batch far above no-AC's 2.

    Sweep:
        TT_CONFIG=gptoss20b_selac TT_EXTRA="--training.local_batch_size 12"

    ep=1, dtype=float32 and no compile stay stock, so AC policy is the only
    variable against gptoss20b_workers and gptoss20b_noac.
    """
    config = gptoss20b_workers()
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.local_batch_size = 8
    return config


def gptoss20b_selac_compile() -> Trainer.Config:
    """SelectiveAC + torch.compile. Exists to isolate compile from AC policy.

    This config is not interesting on its own -- it is the control that makes
    gptoss20b_membudget interpretable. MemoryBudgetAC *requires* compile
    (trainer.py:171 raises unless compile.enable and "model" in
    compile.components), so a MemoryBudgetAC result moves two variables at once
    against the uncompiled sweep. Running SelectiveAC both with and without
    compile gives the compile delta on its own, which can then be subtracted.

    compile cannot be paired with FullAC for the same purpose: on the 120b that
    combination crashed in AOTAutograd with "AssertionError: Node add_21 was
    invalid, but is output". SelectiveAC + compile is a known-working pair there
    (7.77% vs FullAC's uncompiled 8.12% at 120b).

    Mechanics, from the gpt_oss_120b_compile notes: compile on a token-choice MoE
    works via per-TransformerBlock compile, with dynamo
    capture_scalar_outputs=True for the data-dependent token-dispatch shapes and
    skip_fwd_side_effects_in_bwd_under_checkpoint=True so AC recompute does not
    replay forward side effects. CUDA graphs must stay off, which
    disable_cuda_graphs already handles -- compile does its own capture and the
    varlen cu_seqlens change shape every step regardless.

    ep=1 here, so the dynamo recompile_limit bump that parallelize_gptoss applies
    under EP/TP for gpt_oss's alternating sliding-window / full-attention layers
    is not in play. If compile hits a recompile limit anyway, that is the cause.
    """
    config = gptoss20b_selac()
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    return config


def gptoss20b_selac_compile_bs12() -> Trainer.Config:
    """selac_compile at bs=12: spend the memory compile handed back.

    gptoss20b_selac_compile won the sweep at 33.60% but ran at bs=8, using only
    109.46GiB of 178.35GiB (61.4%). That batch was picked before compile was in
    the picture -- it was sized for the uncompiled SelectiveAC run, which needed
    122.31GiB at the same bs=8. Compile then cut peak memory by ~10% and nobody
    re-spent the ~69GiB of headroom that opened up.

    Reasons to expect this helps rather than plateaus:
      - bs=2 -> bs=3 inside gptoss20b_noac_compile was worth 3.5 points
        (29.43% -> 32.91%), so batch is still on the steep part of the curve at
        these sizes, unlike the flat bs=16-28 region measured under FullAC.
      - ep=1 means every rank runs grouped GEMMs across all 32 experts. bs=8
        leaves ~8192 tokens per expert after top-4 routing; bs=12 makes that
        ~12288, still in the range where M growth helps.

    Memory: uncompiled SelectiveAC measured 122.31GiB at bs=8 and 159.75GiB at
    bs=12, i.e. ~9.4GiB per batch unit. Compile ran ~10% under that, so ~8.4GiB
    per unit, projecting 109.46 + 4*8.4 = ~143GiB here -- comfortable.
    """
    config = gptoss20b_selac_compile()
    config.training.local_batch_size = 12
    return config


def gptoss20b_selac_compile_bs16() -> Trainer.Config:
    """selac_compile at bs=16. Expected to be at or just over the wall.

    Same reasoning as gptoss20b_selac_compile_bs12, pushed one step further to
    find the ceiling. At ~8.4GiB per batch unit this projects
    109.46 + 8*8.4 = ~176.7GiB against a 178.35GiB limit -- essentially exactly
    at the wall, so this run is as much a ceiling probe as a performance point.

    Read the outcome carefully. The failure mode to watch for is NOT an
    OutOfMemoryError: job 831 (FullAC, bs=32) sat at 98.97% memory with
    PYTORCH_ALLOC_CONF=expandable_segments:True and never raised, it just
    thrashed to 2-3% MFU while continuing to report steps. Any result here with
    peak memory above ~95% should be treated as over the wall regardless of
    whether it crashed, and bs=12 taken as the operating point instead.
    """
    config = gptoss20b_selac_compile()
    config.training.local_batch_size = 16
    return config


def gptoss20b_membudget_high() -> Trainer.Config:
    """MemoryBudgetAC(0.9) at bs=8: spend spare memory on LESS recompute.

    This is the config that answers "what do we do with 69GiB of spare memory".
    The mistake in every earlier attempt was trying to spend it on *more work*
    (bigger batch), which does nothing because the expert GEMMs are already
    saturated at bs=8 -- selac_compile measured 33.60/33.67/33.57% at bs=8/12/16.
    Memory should instead buy *less work*, and MemoryBudgetAC's budget is exactly
    that dial: 0.0 recomputes everything (~FullAC), 1.0 recomputes nothing
    (~no AC).

    gptoss20b_membudget ran budget=0.5 at bs=8 and used only 104.46GiB of
    178.35GiB (58.6%), leaving ~74GiB idle while still paying for half the
    recompute it could have skipped. Raising the budget to 0.9 keeps the batch
    where it is and converts that idle memory directly into skipped recompute.

    Why this should beat both endpoints of what we already measured: at bs=8 the
    no-AC comparison is not available (no AC needs ~34.5GiB/batch-unit and OOMs
    above bs=3), so 0.9 is the closest thing to "no recompute at a healthy batch"
    that fits. The uncompiled AC sweep showed removing recompute is worth ~2.8
    points at fixed batch (13.76% -> 16.56% at bs=1), and no-AC + compile at the
    starved bs=3 still reached 32.91%, so a compiled bs=8 run with almost no
    recompute is the most promising untried point.

    Budget sizing: 0.5 cost 104.46GiB. If activation memory scales roughly
    linearly in the budget, 0.9 lands near 104 + 0.8*(161 - 104) ~ 150GiB using
    no-AC's 161.55GiB at bs=2 only as a loose upper anchor -- the honest answer
    is that the mapping from budget to bytes is the partitioner's decision and
    this is a measurement, not a projection. If it OOMs or lands above ~95%
    (job 831's thrash threshold), step down to 0.7.
    """
    config = gptoss20b_membudget()
    config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.9)
    return config


def gptoss20b_noreshard() -> Trainer.Config:
    """selac_compile + fsdp_reshard_after_forward="never": memory for NCCL.

    The second way to spend spare memory on less work rather than more. By
    default FSDP frees each module's all-gathered parameters after forward and
    re-gathers them in backward. "never" keeps them resident, so the backward
    all-gather disappears entirely -- pure communication removed in exchange for
    memory, at completely fixed batch and compute.

    Sizing: parameters are all-gathered in bf16 under mixed precision
    (mixed_precision_param defaults to "bfloat16"), so holding every layer
    gathered costs ~20.91e9 * 2 = ~41.8GiB on top of the 109.46GiB that
    selac_compile already peaks at, for ~151GiB against the 178.35GiB limit.
    That fits, but only just, and the estimate assumes all 24 layers stay
    resident simultaneously -- FSDP's "never" applies per-module, so the true
    figure depends on how parallelize_gptoss wrapped the blocks. Treat >95% peak
    as over the wall per job 831.

    Worth trying because NCCL is a plausible remaining bottleneck that no lever
    so far has touched: every experiment to date changed compute (AC, compile,
    attention backend) or work per step (batch), never communication volume. On
    Qwen3.5-122B the fp32 gradient reduce-scatter alone was 25% of the step, so
    collectives are not a negligible slice on this cluster.

    If this wins, the follow-up is enable_fsdp_symm_mem (configs.py:153) and
    HSDP (data_parallel_replicate_degree=2, shard=4), which trade memory for
    smaller collective groups in a similar spirit.
    """
    config = gptoss20b_selac_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def gptoss20b_membudget_mid() -> Trainer.Config:
    """MemoryBudgetAC(0.7). Step-down after budget=0.9 OOMed.

    gptoss20b_membudget_high (0.9) died at 175.48GiB allocated, 2.81GiB short of
    the 178.35GiB limit, before completing a step (job 1013). Budget 0.5 sat at
    104.46GiB (job 998). So the partitioner's memory response to the budget is
    steep between 0.5 and 0.9 -- roughly +70GiB over that range, or ~17GiB per
    0.1 of budget. Linear interpolation puts 0.7 near ~140GiB, which should fit
    with the same kind of margin gptoss20b_noreshard had at 143.55GiB.

    That interpolation is crude: the budget is a target the inductor partitioner
    solves against, not a byte count, so the mapping need not be linear. This is
    a measurement.
    """
    config = gptoss20b_membudget()
    config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.7)
    return config


def gptoss20b_noreshard_membudget() -> Trainer.Config:
    """Stack the two memory-for-work trades: no-reshard FSDP + MemoryBudgetAC.

    These two spend spare memory on different things and should compose:
      - fsdp_reshard_after_forward="never" removes the backward all-gather,
        buying *communication* with memory. Measured alone: 34.30% at 143.55GiB
        (job 1014), the current best, up from selac_compile's 33.60%.
      - MemoryBudgetAC(0.5) lets the partitioner skip recompute, buying *compute*
        with memory. Measured alone: 33.30% at 104.46GiB (job 998).

    Memory arithmetic from the measured points: no-reshard cost
    143.55 - 109.46 = +34.09GiB on top of SelectiveAC (less than the ~41.8GiB
    bf16-parameter estimate, so FSDP is not holding every layer resident at
    once). Applying that same delta to membudget(0.5)'s 104.46GiB gives
    ~138.6GiB here -- inside the limit, and notably *cheaper* than no-reshard
    alone with SelectiveAC, because budget=0.5 uses less activation memory than
    SelectiveAC does.

    Budget stays at 0.5 rather than 0.7 deliberately: 0.9 already OOMed at
    109GiB-class baselines (job 1013), and no-reshard adds ~34GiB on top, so
    raising the budget here would very likely reproduce that OOM. If this
    combination lands well under the wall, gptoss20b_membudget_mid's result
    indicates whether a 0.6-0.7 variant is worth a follow-up.
    """
    config = gptoss20b_membudget()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def gptoss20b_fused_bs12() -> Trainer.Config:
    """Push the fused-bias best config to bs=12, with more dataloader workers.

    Baseline to beat: gptoss20b_noreshard_membudget on the gptoss20b_fused_bias
    branch measured 35.60% at 131.39GiB (73.67%), job 1024. That leaves ~47GiB
    unused, so bs=12 is reachable: SelectiveAC cost ~8.4GiB per batch unit
    compiled, and MemoryBudgetAC(0.5) should be at or under that, projecting
    131.39 + 4*8.4 = ~165GiB. Tight -- above the ~95% thrash line is 169GiB, so
    this is close to the wall.

    Expectation is honestly LOW. On the pre-fusion branch, selac_compile measured
    33.60 / 33.67 / 33.57% clean at bs=8 / 12 / 16 -- batch was already saturated
    under compile, and the only thing the bigger batches changed was
    reintroducing dataloader stalls that dragged the raw means down. The fused
    bias removed memory traffic rather than adding compute, so there is no
    specific reason to think it moved the saturation point.

    The one reason to run it anyway: the earlier bs=12/16 runs each lost a step to
    input starvation, so their raw means understated them. num_workers goes 8 ->
    16 here (still well inside OMP_NUM_THREADS=24, and c4 has 128 shards per rank
    so there is plenty to shard) to test whether a fed dataloader at bs=12 beats
    bs=8 once stalls are removed. If the clean mean still matches bs=8, batch is
    conclusively finished as a lever on this model.
    """
    config = gptoss20b_noreshard_membudget()
    config.training.local_batch_size = 12
    config.dataloader.num_workers = 16
    return config


def gptoss20b_fused_budget06() -> Trainer.Config:
    """Less activation checkpointing: MemoryBudgetAC 0.5 -> 0.6, with no-reshard.

    "Less AC" on this stack means a higher memory_budget. What is already known
    about that axis, all measured pre-fusion without no-reshard:
        budget 0.5  ->  33.30%, 104.46GiB  (job 998)
        budget 0.7  ->  OOM at 175.48GiB   (job 1016)
        budget 0.9  ->  OOM at 175.48GiB   (job 1013)
    0.7 and 0.9 failed with byte-identical allocation numbers, which means the
    partitioner emitted the *same* plan for both -- the knob is a cliff somewhere
    between 0.5 and 0.7, not a ramp. 0.6 is the untested point on the near side of
    that cliff.

    Risk is real: this stacks 0.6 on top of no-reshard's ~+34GiB, and 0.5 +
    no-reshard already sits at 131.39GiB. If 0.6 lands anywhere near where 0.7
    did, this OOMs. That is worth one run to locate the cliff edge, because if
    0.6 fits it is strictly less recompute than the current best at the same
    batch.

    If it OOMs, the conclusion is that 0.5 is the most aggressive usable budget
    with no-reshard enabled, and the AC axis is closed.
    """
    config = gptoss20b_noreshard_membudget()
    config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.6)
    return config


def gptoss20b_fused_noac_noreshard() -> Trainer.Config:
    """The least-AC configuration that can fit: no AC at all, plus no-reshard.

    Floor of the AC axis -- zero recompute. Both memory-for-work trades stacked on
    top of it, so this is the most memory-hungry configuration in the registry.

    Sizing from measured points: noac_compile was 117.42GiB at bs=2 and 151.00GiB
    at bs=3 (jobs 999, 1000), i.e. ~33.6GiB per batch unit. no-reshard adds
    ~34GiB (109.46 -> 143.55 on SelectiveAC). So bs=2 projects to ~151GiB and
    bs=3 to ~185GiB, over the 178.35GiB limit. Hence bs=2.

    Expectation, again honestly low: no-AC at bs=2 reached only 29.43% compiled
    against 32.91% at bs=3, because ep=1 makes every rank grouped-GEMM over all
    32 experts and bs=2 leaves ~2048 tokens per expert. Removing all recompute
    does not pay for that starvation -- that is exactly what the pre-fusion grid
    showed. no-reshard was worth +0.70 elsewhere, which will not close a
    6-point gap.

    Run it to complete the grid on the fusion branch rather than because it is
    expected to win. The informative outcome is whether the fused bias changed
    the starved-batch regime at all, since that regime is the one where per-token
    epilogue traffic is largest relative to GEMM work.
    """
    config = gptoss20b_noac_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.local_batch_size = 2
    return config


def gptoss20b_mxfp8() -> Trainer.Config:
    """MXFP8 on the dense linears (attention). Experts stay bf16 -- see below.

    IMPORTANT, and the reason this config does not touch the experts: the MXFP8
    grouped-GEMM path is architecturally unusable on gpt_oss. The CuTeDSL
    quantization kernel on sm_100 requires the contraction dim K % 128 == 0, and
    gpt_oss has dim = hidden_dim = 2880, where 2880 % 128 == 64. Both expert
    GEMMs are blocked. MXFP8GroupedExpertsConverter.Config.pad_multiple does not
    help -- it pads per-expert *token groups* (the M dimension), not K. This is
    the same wall gpt_oss_120b_mxfp8 hit ("AssertionError: K must be divisible by
    128"), and the 20b has identical dims, so it transfers exactly.
    gptoss20b_mxfp8_experts exists to verify that rather than assume it.

    What MXFP8 *can* reach is the dense linears. torchao's MXFP8Linear uses 1x32
    block scaling, so it needs only K % 32 == 0: qkv K=2880, wo K=4096 and
    lm_head K=2880 all satisfy that.

    That is worth more than it sounds. Because only 4 of 32 experts fire per
    token, the dense weights are 1.80B of the 4.19B *active* parameters --
    42.9% of the per-token GEMM work, against the experts' 57.1%. So quantizing
    attention alone addresses a large minority of the FLOPs even with the MoE
    left in bf16.

    fqns=["attention"] follows the gpt_oss_120b_mxfp8 precedent and deliberately
    excludes two things: the router gate, because quantizing it perturbs expert
    assignment, and lm_head, whose 201,088-wide output feeds the loss directly.
    gptoss20b_mxfp8_lmhead tests adding lm_head.

    READ THE RESULT ON TOKENS/SEC, NOT MFU. MFU here divides by the bf16 dense
    peak (2.25e15, from tools/utils.py), but MXFP8 GEMMs run against a higher
    hardware peak, so any MXFP8 run reports an inflated MFU that is not
    comparable to the bf16 numbers in RESULTS_GPTOSS20B.md. The control to beat
    is gptoss20b_noreshard_membudget at 25,673 tok/s/GPU (reported 35.60%, job
    1024, with the fused-bias commit).

    Memory expectation is modest, not the halving the datatype name suggests.
    These converters do *dynamic* quantization: master weights stay fp32 and are
    quantized per step, so parameter and optimizer memory are unchanged. What
    shrinks is quantized activations and GEMM operands.
    """
    config = gptoss20b_noreshard_membudget()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "20b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention"],
            ),
        ],
    )
    return config


def gptoss20b_mxfp8_lmhead() -> Trainer.Config:
    """gptoss20b_mxfp8 plus MXFP8 on lm_head.

    lm_head is dim 2880 -> vocab 201,088, which the fused-CE sizing put at
    227.7 TFLOP per step per rank -- 10.5% of step time and the single largest
    dense GEMM in the model. K=2880 satisfies MXFP8Linear's K % 32 == 0, so
    unlike the experts it is reachable.

    Split from gptoss20b_mxfp8 rather than folded in because it carries a
    numerics risk the attention linears do not: lm_head's output goes straight
    into the cross entropy, so quantization error there lands directly on the
    loss and its gradient. gpt_oss_120b_fp8 excluded lm_head for exactly this
    reason. Watch the loss curve against the control (step 25 loss ~8.0 across
    every healthy run in this campaign) and treat a visibly higher loss as
    grounds to reject even if tokens/sec improves.

    Router gate stays bf16 -- attention and lm_head only.
    """
    config = gptoss20b_noreshard_membudget()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "20b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "lm_head"],
            ),
        ],
    )
    return config


def gptoss20b_mxfp8_experts() -> Trainer.Config:
    """MXFP8 grouped experts. EXPECTED TO FAIL -- this verifies the K%128 wall.

    Run to confirm empirically that the 120b's "AssertionError: K must be
    divisible by 128" reproduces on the 20b rather than inferring it from
    matching dims. If it somehow runs, the experts are 57.1% of active FLOPs and
    this becomes the most valuable config in the registry; if it raises inside
    torchao/prototype/moe_training/kernels/mxfp8/cutedsl_quantize_2d_1x32.py as
    predicted, the MoE MXFP8 path is closed for this architecture and the
    remaining low-precision option is Float8, which needs only 16-element
    alignment (2880 % 16 == 0) -- the route gpt_oss_120b_fp8 takes.

    pad_multiple=128 matches the kernel's M-dimension requirement on sm_100, so
    a failure here is specifically about K and not about token-group padding.
    """
    config = gptoss20b_noreshard_membudget()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "20b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def gptoss20b_fp8_experts() -> Trainer.Config:
    """Float8 on the expert grouped GEMMs, bf16 dense. Isolates the expert half.

    MXFP8 cannot reach the experts on gpt_oss -- the CuTeDSL kernel needs
    K % 128 == 0 and dim = hidden_dim = 2880 leaves remainder 64 (verified, job
    1049). Float8GroupedExpertsConverter needs only 16-element alignment
    (PAD_MULTIPLE = 16, "16 byte alignment / 1 byte per elem") and 2880 % 16 == 0,
    so it reaches exactly the GEMMs MXFP8 could not.

    This is the larger half of the model: the 4 active experts are 2.39B of the
    4.19B active parameters, 57.1% of per-token GEMM work, against the dense
    side's 42.9%.

    Notably it does NOT require expert parallelism. Float8GroupedExpertsConverter
    checks only for torchao and SM89+ (B200 is SM100); compile is a warning, not
    a requirement. The gpt_oss_120b_fp8 docstring asserts "EP is required by the
    grouped-experts converter anyway" -- that is not what the converter does, and
    it matters here because ep=8 was an 18-point regression on the 20b (15.28%
    against 33.60%, job 1041). Running Float8 experts at ep=1 avoids paying that.

    Dense linears stay bf16 so this attributes the expert contribution on its
    own. gptoss20b_mxfp8_fp8_full stacks it with the measured MXFP8 dense win.

    Control to beat (all bf16, job 1024): 800.91 TFLOPs/GPU, 25,577 tok/s/GPU,
    loss 8.037 at step 25. Read tokens/sec and TFLOPs, not MFU -- torchtitan
    reports "mfu: N/A" once any GEMM is low-precision.

    Watch the loss. Float8 on the experts perturbs the largest share of the
    compute, and unlike MXFP8-on-attention there is no measurement yet saying it
    is numerically benign on this model. Every healthy run in this campaign sits
    near 8.0 at step 25.
    """
    config = gptoss20b_noreshard_membudget()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "20b",
        converters=[
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def gptoss20b_fp8_experts_nocompile() -> Trainer.Config:
    """Float8 experts WITHOUT compile. Diagnostic for "cutlass cannot run, error 7".

    gptoss20b_fp8_experts converts cleanly ("Converted GroupedExperts to use
    dynamic float8 rowwise") and then dies before step 1. With
    CUDA_LAUNCH_BLOCKING=1 the async "CUDA error: unspecified launch failure"
    resolves to:

        RuntimeError: cutlass cannot run, error 7

    raised from the aten _scaled_grouped_mm dispatch (torch/_ops.py:916) reached
    through AOTAutograd's runtime wrapper -- i.e. CUTLASS itself refuses to
    execute, inside the compiled region. Note this is a different class of
    failure from MXFP8's: MXFP8 failed a clean Python-level shape assertion
    ("K must be divisible by 128"), which is a documented constraint. A CUTLASS
    execution failure is either an unsupported configuration the wrapper does not
    pre-check, or a bug.

    This config removes compile as a variable. If Float8 experts run eagerly, the
    kernel handles these shapes and the problem is in the compiled path
    (inductor's chosen layouts/strides for the grouped mm, or the AOTAutograd
    wrapper). If it fails identically, the kernel cannot do gpt_oss's expert
    shapes at all and the Float8 MoE path is closed here the way MXFP8's is.

    SelectiveAC rather than MemoryBudgetAC because Trainer.Config requires
    compile for MemoryBudgetAC, so the no-compile control cannot use it. That
    makes this config slow -- gptoss20b_selac measured 20.38% uncompiled -- but
    it is a correctness probe, not a performance measurement. Read only whether
    it reaches step 1.
    """
    config = gptoss20b_selac()
    model_compile_enabled = False
    config.model_spec = model_registry(
        "20b",
        converters=[
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def gptoss20b_mxfp8_fp8_full() -> Trainer.Config:
    """Both halves in low precision: MXFP8 dense + Float8 experts.

    The point of the exercise -- every GEMM in the model quantized by whichever
    format can reach it:

        attention qkv/wo, lm_head   42.9% of active FLOPs   MXFP8 (K % 32)
        expert grouped GEMMs        57.1% of active FLOPs   Float8 (K % 16)
        router gate                 0.0006%                 bf16, deliberately

    Two formats rather than one because neither covers everything: MXFP8's
    grouped kernel is blocked by K % 128 on 2880, and while Float8Linear could
    handle the dense side too, MXFP8 there is already measured at +3.50%
    tokens/sec (828.97 TFLOPs/GPU, job 1048) and there is no reason to give that
    up untested.

    Expected to be roughly additive if the two halves compose -- MXFP8 dense
    contributed +3.50% over the bf16 control, so a similar-magnitude expert
    contribution would land in the high single digits. That is a guess, not a
    projection: the expert GEMMs are grouped and token-routed rather than dense,
    so their speedup from halving operand bytes need not track the dense case,
    and the token dispatcher gets swapped to pad groups to 16 which changes the
    dispatch path itself.

    If gptoss20b_fp8_experts shows a loss regression, this config inherits it --
    check that one first.
    """
    config = gptoss20b_noreshard_membudget()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "20b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "lm_head"],
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def gptoss20b_selac_compile_flex() -> Trainer.Config:
    """selac_compile with FlexAttention instead of varlen.

    IMPORTANT: varlen is NOT a new thing to try here -- it is what every run in
    RESULTS_GPTOSS20B.md already used. model_registry defaults to
    attn_backend="varlen" (__init__.py:391) and gpt_oss_20b calls
    model_registry("20b") without overriding it, so the 13.76% baseline through
    the 33.60% best were all varlen. Varlen is also the reason
    disable_cuda_graphs is mandatory throughout: its cu_seqlens metadata changes
    shape every step, which is what broke CUDA graph replay at step 2.

    So the actual open question on the attention axis is flex vs varlen, and
    there is a specific reason to expect flex could win on this model.
    gpt_oss sets sliding_window_size=128 on even-indexed layers
    (__init__.py:209), i.e. 12 of 24 layers attend over 128 positions out of
    seq_len=8192. FlexAttention expresses that as a BlockMask and skips blocks
    entirely outside the window, so those layers should cost close to nothing.
    Whether varlen's kernel exploits its window_size argument as effectively is
    unknown, and it is exactly the kind of gap that would not show up in MFU --
    the FLOP counter bills full attention on all 24 layers regardless
    (get_nparams_and_flops passes no window info), so a real speedup here shows
    up as higher tok/s and a *correspondingly* higher, still-overstated MFU.

    Known risk: FlexAttention under torch.compile has bitten this stack before.
    On qwen3_5, compiling the decoder together with the vision encoder's
    compiled_create_block_mask reproducibly failed with "Node add_21 was invalid,
    but is output". gpt_oss has no vision tower, so the specific collision does
    not apply, but flex + compile is the less-travelled path. If this fails,
    gptoss20b_selac_flex_nocompile isolates flex from compile.

    Everything else matches gptoss20b_selac_compile (bs=8, SelectiveAC, ep=1,
    fp32) so the attention backend is the only variable against 33.60%.
    """
    config = gptoss20b_selac_compile()
    config.model_spec = model_registry("20b", attn_backend="flex")
    return config


def gptoss20b_selac_compile_flexflash() -> Trainer.Config:
    """selac_compile with FlexAttention's FLASH backend. B200-only.

    Same reasoning as gptoss20b_selac_compile_flex, but using the "flex_flash"
    backend, which get_attention_config maps to
    FlexAttention.Config(block_size=(256, 128), kernel_options={"BACKEND":
    "FLASH"}) and which requires CUDA capability >= 9.0 (config_utils.py:86-94).
    B200 is capability 10.0, so it qualifies.

    This is the configuration most likely to beat varlen if the sliding-window
    hypothesis is right: it keeps FlexAttention's BlockMask, so the 12
    window-128 layers can skip out-of-window blocks, while running a flash
    kernel rather than the default flex lowering. The (256, 128) block size is
    coarser than flex's default, which matters for a 128-wide window -- a 256
    query block spans two windows, so verify this actually helps rather than
    forcing extra masked work.

    Only the attention backend differs from gptoss20b_selac_compile (33.60%).
    """
    config = gptoss20b_selac_compile()
    config.model_spec = model_registry("20b", attn_backend="flex_flash")
    return config


def gptoss20b_selac_flex_nocompile() -> Trainer.Config:
    """FlexAttention without compile. Fallback control if flex + compile breaks.

    Isolates the attention backend from compile, comparing against
    gptoss20b_selac's 20.38% (bs=8, SelectiveAC, varlen, no compile) rather than
    against the compiled 33.60%. Only worth running if
    gptoss20b_selac_compile_flex fails to start, or if flex wins and the split
    between backend and compile needs attributing.
    """
    config = gptoss20b_selac()
    config.model_spec = model_registry("20b", attn_backend="flex")
    return config


def gptoss20b_noac_compile() -> Trainer.Config:
    """No AC + compile. The one cell the AC x compile grid was missing.

    The sweep had measured three of four corners:
        FullAC,      no compile, bs=24  ->  19.29%  (job 829)
        no AC,       no compile, bs=2   ->  21.23%  (job 837)
        SelectiveAC, compile,    bs=8   ->  33.60%  (job 997)
    and never ran no-AC with compile. Worth doing because the two effects are
    independent: no AC removes the recompute tax entirely, while compile's 1.65x
    (measured cleanly as selac 20.38% -> selac_compile 33.60% at fixed bs=8)
    comes from fusion and better memory planning. If they compose, this is the
    fastest configuration available.

    FullAC + compile is the corner that cannot be filled: on the 120b it crashed
    in AOTAutograd with "AssertionError: Node add_21 was invalid, but is output".
    compile + AC=None is known-working there (6.42% with EP=4), so this pairing
    is expected to run.

    The catch is the same as gptoss20b_noac -- memory. No AC costs ~34.5GiB per
    batch unit against FullAC's ~4.35GiB, which capped the uncompiled version at
    bs=2 (161.55GiB, 90.58%) with bs=3 OOMing 2.30GiB short. Compile helps here:
    it *reduced* peak memory 122.31 -> 109.46GiB in the SelectiveAC pair, about
    10.5%, presumably from inductor's allocation planning. Applying that to
    no-AC's 161.55GiB projects ~145GiB at bs=2, which should fit comfortably, and
    ~176GiB at bs=3 -- right at the 178.35GiB wall.

    So bs=2 is the default and bs=3 is worth a shot, since compile's memory
    saving may be exactly what makes the batch bs=3 OOM missed by 2.30GiB fit.

    Expectation to check against: no AC at bs=2 was 21.23% uncompiled. Naively
    applying compile's 1.65x gives ~35%, which would beat selac_compile's 33.60%.
    But bs=2 is batch-starved next to bs=8 -- at ep=1 every rank runs grouped
    GEMMs over all 32 experts, and bs=2 leaves only ~2048 tokens per expert -- so
    the honest prediction is somewhere between 30% and 35%, and it may well lose
    to selac_compile despite carrying no recompute tax. That is the measurement.

    ep=1, dtype=float32 stay stock. disable_cuda_graphs is inherited and required:
    compile does its own capture and gpt_oss's varlen cu_seqlens change shape
    every step.
    """
    config = gptoss20b_noac()
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    return config


def gptoss20b_membudget() -> Trainer.Config:
    """MemoryBudgetAC(0.5) + compile. The policy designed to pair with compile.

    Rather than a fixed rule about which ops to save, MemoryBudgetAC hands the
    decision to the inductor partitioner under a memory budget: 0.0 behaves
    roughly like FullAC (save nothing, recompute everything) and 1.0 roughly like
    no AC. 0.5 is the default and what was chosen on the 120b, where it measured
    7.94% at 4.00GiB against SelectiveAC's 7.77% at 5.51GiB under compile.

    Because the budget is a knob rather than a policy, this is the config that
    can actually target the 20b's real constraint. The no-AC result showed the
    recompute tax is worth ~2.8 points at fixed batch, and the FullAC result
    showed batch saturates by 16. So the target is the smallest amount of
    recompute that still permits bs>=8, which is what a budget expresses directly
    and neither endpoint can.

    NOTE this config moves TWO variables against the uncompiled sweep -- AC
    policy and compile -- because Trainer.Config requires compile for
    MemoryBudgetAC (trainer.py:171). Compare against gptoss20b_selac_compile,
    not against gptoss20b_selac, to keep compile out of the comparison.

    Sweep the budget as well as the batch if this is close:
        TT_EXTRA="--activation_checkpoint.memory_budget 0.3"
    Lower budget -> less memory, more recompute. If it OOMs at bs=8, lower the
    budget before lowering the batch; that is the whole point of the knob.
    """
    config = gptoss20b_selac()
    config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.5)
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    return config


def gptoss20b_mxfp8_lmhead_profile() -> Trainer.Config:
    """gptoss20b_mxfp8_lmhead (the take-2 reference) with the torch profiler
    capturing step 20, traces to outputs/profiling/traces_20b_take2/. Run with
    TT_STEPS=24. Profiling adds overhead: take throughput from the unprofiled
    config, not from this.

    Why: the only 20b trace on disk (traces_20b_best, 2026-09-02, 2.88 s
    profiled step) ranks, after the grouped GEMMs (1,345 ms): fp32
    ReduceScatter 587 ms on the comm stream, three fused swiglu-backward /
    bias-grad kernels carrying index_put 333 ms, chunk_cat<float, bf16>
    90 ms, dispatch-gather indexing_backward_kernel 67 ms, router SIMT
    sgemm 37 ms. This re-takes it on the take-2 base.

    Job 6215 (2026-09-30, 2.51 s profiled step, rank 0): grouped GEMM 1,145
    ms (the forward kernel runs 96 times against 48 per backward kernel:
    MemoryBudgetAC recomputes every expert GEMM), fp32 ReduceScatter 444 ms
    on the comm stream, swiglu-backward + bias-grad index_put kernels 312
    ms, FA4 181 ms, chunk_cat<float, bf16> 69 ms, dispatch-gather
    indexing_backward_kernel 69 ms.
    """
    config = gptoss20b_mxfp8_lmhead()
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=20,
        save_traces_folder="profiling/traces_20b_take2",
    )
    config.training.steps = 24
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce() -> Trainer.Config:
    """gptoss20b_mxfp8_lmhead with gradients reduce-scattered in bf16.

    The reference still reduces in fp32 (mixed_precision_reduce defaults to
    "float32"), so every step FSDP widens each block's bf16 gradients into an
    fp32 buffer (chunk_cat_cuda_kernel<float, BFloat16>, 90 ms per profiled
    step on the compute stream in traces_20b_best) and reduce-scatters twice
    the bytes (ReduceScatter_Sum_f32, 587 ms on the comm stream; mostly
    overlapped, but the last blocks' reduce-scatters trail the backward).
    Same lever as gpt_oss_120b_bf16reduce, +5.0% there (jobs 2295 vs 2292,
    confirmed three times), and every take-2 Qwen3 config reduces in bf16.

    NUMERICS: gradients summed across 8 shards in bf16. On the 120b, seeded
    loss diffs stayed within +-0.016 over 600 paired steps.

    MEASURED (job 6212, steps 21-100): 840.97 TF/GPU mean, 841.44 median,
    130.61GiB -- +2.0% over the same-week reference (gptoss20b_mxfp8_lmhead,
    job 6237: 824.26 / 824.17). Seeded (seed 42, 300 steps, bs=5,
    save_grouped_mm; jobs 6377 vs 6376): loss within +-0.015 of fp32 reduce
    per 100-step window.
    """
    config = gptoss20b_mxfp8_lmhead()
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16train() -> Trainer.Config:
    """_bf16reduce with full-bf16 training: params, grads AND AdamW states in bf16
    (the GB300 recipe, split/04; take2-qwen3/bf16-training-recipe).

    What it removes on this model: the fp32 -> bf16 parameter cast FSDP
    does before each all-gather, half of the fused-AdamW traffic (24 ms per
    profiled step in traces_20b_best), and the fp32 sharded-gradient
    buffers. 16 -> 8 bytes per parameter frees ~19.5GiB per rank at 20.9B
    parameters, from 131.53GiB. On Qwen3-30B-A3B it was flat at equal batch
    (job 5470) and paid only once the memory was spent. Here the memory goes
    to saved expert GEMMs (take2-gptoss20b/sac-save-grouped-mm); spending it
    on MemoryBudgetAC 0.6 instead OOMed asking for the same 1.41GiB as job
    1026 (job 6217).

    NUMERICS: no fp32 master weights. AdamW updates below bf16's resolution
    of the weight are lost; needs a seeded loss comparison before a real run.

    MEASURED (job 6216): 845.80 / 846.50 TF/GPU, 110.49GiB (-21.0GiB) --
    +2.6% over the reference (job 6237), +0.6% over _bf16reduce. Seeded
    against fp32 (jobs 6378 vs 6376, bs=5, 300 steps): +0.16 loss over steps
    11-100, then -0.015 (101-200) and -0.048 (201-300). No divergence; not
    cleared for long runs.
    """
    config = gptoss20b_mxfp8_lmhead_bf16reduce()
    config.training.dtype = "bfloat16"
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16optstates() -> Trainer.Config:
    """_bf16reduce with fp32 master weights + bf16 AdamW moments.

    The convergence-safe half of _bf16train: fused_opt_states_bf16 keeps fp32
    params and halves only exp_avg / exp_avg_sq (~-9.7GiB per rank). On
    Qwen3-30B-A3B it trailed _bf16train by ~0.8% (jobs 6174 vs 6173) and is
    the choice for real training.

    MEASURED (job 6220): 840.75 / 841.12 TF/GPU, 120.88GiB -- level with
    _bf16reduce at -9.7GiB.
    """
    config = gptoss20b_mxfp8_lmhead_bf16reduce()
    config.optimizer.implementation = "fused_opt_states_bf16"
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_bs6() -> Trainer.Config:
    """_bf16train with SelectiveAC saving the expert GEMMs, at bs=6, FSDP
    resharding by default.

    MemoryBudgetAC(0.5) recomputes the expert GEMMs: in traces_20b_best the
    forward-layout grouped-mm kernel runs 106 times against 58 for each
    backward kernel, i.e. all 48 per-step expert GEMMs twice, ~300 ms. Stock
    SelectiveAC is no better -- aten._grouped_mm is in neither save list and
    unlisted ops default to PREFER_RECOMPUTE. save_grouped_mm=True
    (perf/sac-save-grouped-mm) fixes that: +13.9% on gpt-oss-20b at equal
    batch on 64x GB300 (job 61 vs sac_compile).

    Memory, from B200 measurements: selac_compile ~7.5GiB per batch unit
    (job 997), saving mlp1 (R, 2F) + mlp2 (R, D) ~11.8GiB per unit (GB300 job
    61), ~29.5GiB fixed under bf16 training -> bs=6 ~146GiB. no-reshard
    (+34GiB) does not fit at bs=6; _noreshard_bs5 trades one batch unit for
    it. Try bs=7 (~165GiB) with TT_EXTRA if this lands low.

    MEASURED (job 6213): 933.93 / 935.10 TF/GPU, 143.51GiB (80.5%) -- +13.3%
    over the reference (job 6237). _noreshard_bs5 beats it.
    """
    config = gptoss20b_mxfp8_lmhead_bf16reduce_bf16train()
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.training.local_batch_size = 6
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5() -> Trainer.Config:
    """_sacgmm with no-reshard kept, at bs=5 (~160GiB estimated).

    no-reshard was +0.70 pts on selac_compile (job 1014). This asks whether
    it is still worth a batch unit once the expert GEMMs are saved; on GB300
    it stopped paying under memory pressure (880.7 at bs=6 vs 917.0 at bs=4).

    MEASURED (job 6218): 959.78 / 961.37 TF/GPU, 160.95GiB (90.2%) -- +16.4%
    over the reference (job 6237). Re-measured 962.37 / 964.31 (job 6371);
    965.64 / 967.05 with TORCHTITAN_FP32_MATMUL_PRECISION=tf32 (job 6372).
    bs=6 with no-reshard does not fit (~+19GiB per batch unit).
    """
    config = gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_bs6()
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.local_batch_size = 5
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_sacgmm_bs5() -> Trainer.Config:
    """_sacgmm without bf16 training: fp32 master weights and AdamW states.

    Separates save_grouped_mm from the bf16 recipe. fp32 costs ~19.5GiB more
    fixed memory (~49GiB), so bs=5 at default reshard (~146GiB).

    MEASURED (job 6219): 910.71 / 912.44 TF/GPU, 143.65GiB -- +10.5% over
    the reference (job 6237): save_grouped_mm without bf16 training.
    """
    config = gptoss20b_mxfp8_lmhead_bf16reduce()
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.training.local_batch_size = 5
    return config


def gptoss20b_mxfp8_lmhead_sacgmm_bs5() -> Trainer.Config:
    """save_grouped_mm at bs=5 with the reference's fp32 numerics: fp32 master
    weights, fp32 AdamW states and fp32 gradient reduce.

    The numerics control for the bf16 recipe. Run seeded (--debug.seed 42)
    beside _bf16reduce_sacgmm_bs5 and _bf16train_sacgmm_noreshard_bs5:
    same batch and same math (reshard policy does not change the result),
    so loss differences are bf16 reduce / bf16 training alone. Default
    reshard because fp32 + no-reshard at bs=5 (~178GiB) does not fit.

    Ran seeded only (job 6376, seed 42, 300 steps, 146.54GiB) as the
    reference for jobs 6377-6379.

    Ran seeded only (job 6376, seed 42, 300 steps, 146.54GiB) as the
    reference for jobs 6377-6379.
    """
    config = gptoss20b_mxfp8_lmhead()
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    config.parallelism.fsdp_reshard_after_forward = "default"
    config.training.local_batch_size = 5
    return config


def _enable_expert_bias_grad_gemm(config: Trainer.Config) -> Trainer.Config:
    for layer in config.model_spec.model.layers:
        layer.moe.routed_experts.inner_experts.bias_grad_gemm = True
    return config


def gptoss20b_mxfp8_lmhead_biasgemm() -> Trainer.Config:
    """gptoss20b_mxfp8_lmhead with the expert bias gradients as one-hot GEMMs
    (GptOssGroupedExperts.Config.bias_grad_gemm, moe.ExpertBiasAdd).

    Target: in traces_20b_best the three kernels that fuse the swiglu
    backward with the mlp1/mlp2 bias-gradient index_put scatter run 135.6,
    100.6 and 97.2 ms per profiled step (3.4-4.7 ms per layer): R = 262,144
    rows land on 33 bias rows, ~8,000 atomic adds per address. The same
    gradient is (P, R) x (R, X) one-hot GEMM; the forward gather-add is
    unchanged, so its fusion into the swiglu is kept.

    NUMERICS: forward and every non-bias gradient are bitwise identical to
    stock (CPU check); the bias gradients accumulate in fp32 inside the GEMM
    and are deterministic. Against an fp64 segment sum at 2,000 rows per
    expert: one-hot GEMM 1.7e-3 relative error, eager bf16 index_put 5.3e-2.

    MEASURED (job 6238): 832.41 / 832.44 TF/GPU, 131.53GiB -- +1.0% over the
    reference (job 6237), at the edge of the ~0.7% noise band.
    """
    return _enable_expert_bias_grad_gemm(gptoss20b_mxfp8_lmhead())


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5_biasgemm() -> Trainer.Config:
    """The wave-1 best (_bf16train_sacgmm_noreshard_bs5, job 6218, 959.78
    TF/GPU) with the one-hot-GEMM expert bias gradient.

    On the reference _biasgemm measured +1.0% (832.41 vs 824.26, jobs 6238
    vs 6237), at the edge of the ~0.7% noise band; this re-measures it on
    the stack, where saving the expert GEMMs changed what the backward does.

    MEASURED (job 6373): 968.89 / 970.72 TF/GPU, +0.7% over _noreshard_bs5
    (job 6371). Best without the swiglu layout change (see _swigluhalves), run with
    TORCHTITAN_FP32_MATMUL_PRECISION=tf32 and
    TT_INDUCTOR="TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
    TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1": 986.95 / 991.61 TF/GPU,
    160.95GiB (job 6374), 987.14 / 988.97 seeded (job 6379) -- +19.7% over
    the reference (job 6237). Reproduce with:

        TORCHTITAN_FP32_MATMUL_PRECISION=tf32 \\
        TT_INDUCTOR="TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1" \\
        sbatch --time=01:30:00 --export=ALL,TT_REPO=<worktree>,TT_CONFIG=<this config> \\
            benchmarks/gpt_oss_20b/run_gptoss20b_take2.sbatch
    """
    return _enable_expert_bias_grad_gemm(
        gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5()
    )


def _set_swiglu_contiguous_halves(config: Trainer.Config) -> Trainer.Config:
    for layer in config.model_spec.model.layers:
        layer.moe.routed_experts.inner_experts.swiglu_interleaved = False
    return config


def gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5_biasgemm_swigluhalves() -> Trainer.Config:
    """The wave-2 stack (_sacgmm_noreshard_bs5_biasgemm; job 6374 with tf32 +
    pointwise autotune: 986.95 TF/GPU mean, 991.61 median) with the mlp1
    output stored as contiguous [gate | linear] halves instead of interleaved.

    In that stack's profile (job 6375, 1.31 s step) the swiglu forward and
    the two swiglu-backward kernels are the largest non-GEMM compute: 27.9 +
    59.6 + 45.5 = 133 ms, ~2.5 ms per layer for the backward against ~0.7 ms
    for its bytes. The interleaved split x[..., ::2] / x[..., 1::2] makes
    every load and store in them stride 2. Same model up to a fixed
    permutation of mlp1's rows (CPU check: outputs and all grads bitwise
    equal after permuting); from-scratch training only.

    MEASURED, both with TORCHTITAN_FP32_MATMUL_PRECISION=tf32 and pointwise
    autotune (TT_INDUCTOR as in _biasgemm above): 1,010.27 TF/GPU mean,
    1,012.19 median, 32,263 tok/s/GPU, 160.95GiB (job 7258) against the
    same-day control _sacgmm_noreshard_bs5_biasgemm at 986.40 / 990.12
    (job 7257) -- +2.4% mean, +2.2% median. THE BEST CONFIG: +22.6% over
    the reference gptoss20b_mxfp8_lmhead (824.26, job 6237). Loss@100 6.576
    vs 6.514 (unseeded; the layouts differ by a row permutation, so the
    same seed is not the same initialization).
    """
    return _set_swiglu_contiguous_halves(
        gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5_biasgemm()
    )

def gpt_oss_120b_compile() -> Trainer.Config:
    """gpt_oss_120b with torch.compile enabled.

    torch.compile on a token-choice MoE works via per-TransformerBlock compile
    (torchtitan/distributed/compile.py::apply_compile), not whole-model compile:
    each block is compiled with fullgraph=True, and the repeated structure means
    one compile is reused across all 36 layers. Two dynamo settings there make the
    MoE traceable -- capture_scalar_outputs=True for the data-dependent shapes in
    token dispatch, and skip_fwd_side_effects_in_bwd_under_checkpoint=True so AC
    recompute does not try to replay forward side effects. With EP or TP enabled,
    parallelize_gptoss() additionally raises dynamo's recompile_limit to 12,
    because GPT-OSS alternates sliding-window and full-attention layers.

    CUDA graphs must stay off: GPT-OSS uses the varlen attention backend, whose
    cu_seqlens change shape every step. This is independent of torch.compile,
    which does its own capture.

    Reference: CI case "gpt_oss_fsdp+tp+ep+compile"
    (tests/integration_tests/models.py) runs exactly --compile.enable with
    --training.disable_cuda_graphs on gpt_oss.

    AC policy is NOT FullAC here. Measured on gpt_oss_debugmodel, 4x B200:
        compile + FullAC              -> crash, "AssertionError: Node add_21 was
                                         invalid, but is output" (AOTAutograd)
        compile + AC=None + EP=4      -> works, mfu 6.42%, 10.87GiB
        compile + SelectiveAC         -> works, mfu 7.77%,  5.51GiB
        compile + MemoryBudgetAC(0.5) -> works, mfu 7.94%,  4.00GiB  <- chosen
    AC=None is not an option at 120b: it peaks at 172.49GiB and then OOMs.
    MemoryBudgetAC is the policy designed to pair with compile -- it lets the
    partitioner pick what to save, and Trainer.Config *requires* compile for it.
    If this OOMs at 120b, lower memory_budget toward 0.0 (0.0 ~= FullAC memory,
    1.0 ~= no AC); SelectiveAC is a known-fitting fallback (158.52GiB at 120b).

    Note: because compile+FullAC crashes, this config cannot isolate compile as a
    single variable against the 8.12% FullAC baseline. The AC contribution is
    small though -- SelectiveAC vs FullAC measured 8.22% vs 8.12% uncompiled.
    """
    model_spec = model_registry("120b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
            # 116.83B params in fp32 needs ~234 GiB/GPU of param+grad+AdamW
            # state vs 178.35 GiB usable on a B200. Full bf16 is ~117 GiB/GPU.
            dtype="bfloat16",
            # Required: varlen attention's cu_seqlens change shape every step.
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        # checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=MemoryBudgetAC.Config(memory_budget=0.5),
        # components defaults to ["model", "loss"]; "model" is the one that
        # triggers the per-TransformerBlock compile described above.
        compile=CompileConfig(enable=True, components=["model", "loss"]),
    )


def gpt_oss_120b_ep8() -> Trainer.Config:
    """gpt_oss_120b with expert parallelism across all 8 GPUs.

    Single-variable change against the measured 8.12% MFU baseline (which ran
    FullAC + bf16 + disable_cuda_graphs at EP=1): only expert_parallel_degree
    moves, from 1 to 8. AC stays FullAC and compile stays off so the delta is
    attributable to EP alone.

    Why EP is the most promising knob here: at EP=1 every rank runs grouped GEMMs
    over all 128 experts for only 8192 tokens, so each per-expert GEMM is tiny and
    latency-bound. At EP=8 each rank owns 16 experts and receives tokens routed
    from all ranks, giving larger, better-shaped GEMMs -- paid for with an
    all-to-all dispatch/combine per MoE layer.

    disable_cuda_graphs=True is mandatory, not optional. With EP > 1 the default
    AllToAllTokenDispatcher synchronizes with the host during dispatch, so
    Trainer.Config.__post_init__ -> _validate_cuda_graphs() raises:
        "CUDA graphs support only expert parallel token dispatcher configurations
         without CPU synchronization. ... Unsupported token dispatcher:
         AllToAllTokenDispatcher.Config"
    That is the error job 174 hit. At EP=1 the guard returns early, which is why
    the baseline never tripped it. The error text suggests MinimalAsyncEP or
    HybridEP's non_blocking_capacity_factor as ways to keep CUDA graphs, but both
    are dead ends for GPT-OSS: varlen attention's cu_seqlens change shape every
    step, so graph replay would still fail. Those need attn_backend="flex".

    dtype="bfloat16" is also mandatory. EP redistributes experts across ranks but
    does not reduce total per-GPU state, so fp32 still needs ~234 GiB/GPU vs the
    178.35 GiB a B200 has.
    """
    model_spec = model_registry("120b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=10000,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


# ---------------------------------------------------------------------------
# MFU experiments. Measured so far on 1 node x 8 B200, 116.83B params:
#   gpt_oss_120b (bf16, FullAC, EP=1)   8.12% mfu   182.6 TFLOPs  151.5GiB
#   gpt_oss_120b_compile (EP=1)        10.17% mfu   228.7 TFLOPs  168.4GiB
#   gpt_oss_120b_ep8 (no compile)      10.18% mfu   229.0 TFLOPs  126.9GiB
# compile and EP=8 each bought ~+25% but plateaued at the same ~10.2%, which
# says they are hitting different ceilings. EP=8 is the cheap one on memory
# (71% vs 94%), leaving ~50GiB to spend. The three configs below spend it.
# ---------------------------------------------------------------------------


def gpt_oss_120b_ep8_compile() -> Trainer.Config:
    """Idea 1: stack the two wins that plateaued separately.

    compile (EP=1) and EP=8 (no compile) both landed at ~10.2%, but they remove
    different overheads: compile fuses pointwise/norm work and cuts kernel-launch
    count inside each block, while EP=8 changes the expert GEMM shapes from
    "128 experts x 8192 tokens" to "16 experts x ~4x the tokens". Neither touches
    the other's bottleneck, so stacking them should not simply re-plateau.

    AC has to thread a needle here:
      FullAC + compile        -> "AssertionError: Node add_21 was invalid, but is
                                 output" (AOTAutograd)
      MemoryBudgetAC + EP + compile -> "RuntimeError: Cannot compute the size of
                                 FakeScriptObject on node primals_17" (job 190).
                                 The budget partitioner has to size every node and
                                 cannot size the EP process-group ScriptObject.
                                 MemoryBudgetAC is fine at EP=1 (that is what the
                                 10.17% gpt_oss_120b_compile run used).
      SelectiveAC + compile   -> works (debugmodel: 7.77%, and no partitioner
                                 budget involved, so EP's ScriptObject is a
                                 non-issue). <- chosen
    """
    config = gpt_oss_120b_ep8()
    config.training.local_batch_size = 1
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    return config


def gpt_oss_120b_ep8_bs3() -> Trainer.Config:
    """Idea 2: spend EP=8's freed memory on batch size, not on saving memory.

    This targets the root cause of the low MFU rather than an overhead. At
    local_batch_size=1 each rank pushes only 8192 tokens per step, so after
    top-4 routing across 16 local experts (EP=8) each per-expert grouped GEMM
    sees ~2048 tokens -- small enough to be launch/latency-bound rather than
    tensor-core-bound. Three tokens' worth of batch triples the M dimension of
    every expert GEMM and amortizes the all-to-all, the optimizer step and the
    FSDP all-gathers over 3x the work, none of which get more expensive.

    Memory: EP=8 measured 126.9GiB, of which ~108.8GiB is bf16 params+grads+Adam
    (8 B/param, fixed) and only ~18GiB is activations. FullAC keeps activations
    ~linear in tokens, so bs=3 projects to ~108.8 + 54 = ~163GiB, inside the
    178.35GiB limit. bs=4 projects to ~181GiB and should OOM -- if bs=3 has room
    to spare, 4 is the next thing to try, and if it OOMs, drop to 2.

    FullAC and no compile are kept so this is a clean single-variable move off
    gpt_oss_120b_ep8 (10.18%).
    """
    config = gpt_oss_120b_ep8()
    config.training.local_batch_size = 3
    return config


def gpt_oss_120b_mxfp8() -> Trainer.Config:
    """Idea 3: stop paying bf16 for the expert GEMMs -- MXFP8 on Blackwell.

    116.83B of the 116.83B params are 114.7B sparse (the experts), so essentially
    all the FLOPs are in the MoE grouped GEMMs. B200 has native MXFP8 grouped-GEMM
    support via torch._scaled_grouped_mm (cuBLAS/CUTLASS), up to 2x bf16 on good
    shapes; torchtitan's own docs report up to 28% end-to-end on B200. MXFP8 uses
    1x32 block scaling rather than tensorwise, which is why it holds accuracy
    better than plain fp8.

    Requirements, all satisfied here:
      - sm_100 (B200) and torchao >= 0.14  (installed: 0.18.0)
      - EP must be enabled for the grouped-experts converter -> EP=8
      - pad_multiple=128, NOT the default 32: the CuTeDSL quantization kernel on
        sm_100 requires 128. deepseek_v3 sets this for the same reason.
      - compile strongly recommended (and required for MemoryBudgetAC)
    The router gate and lm_head are deliberately left in bf16; only `attention`
    linears and the expert grouped GEMMs are quantized, mirroring deepseek_v3.

    IMPORTANT when reading the result: torchtitan computes MFU against the *bf16*
    peak (2.25e15), so a low-precision run inflates the MFU number -- it is no
    longer "fraction of achievable peak". metrics.py flags this directly. Compare
    this config on tokens/sec, which is precision-neutral, not on mfu%.
    """
    config = gpt_oss_120b_ep8()
    config.training.local_batch_size = 1
    # SelectiveAC, not MemoryBudgetAC: the budget partitioner cannot size EP's
    # process-group ScriptObject. See gpt_oss_120b_ep8_compile for the details.
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def gpt_oss_120b_fp8() -> Trainer.Config:
    """Idea 3 (corrected): rowwise Float8 for the expert GEMMs.

    Same goal as gpt_oss_120b_mxfp8 -- stop paying bf16 for the grouped GEMMs that
    hold 114.7B of the 116.83B params -- but via the float8 path, because MXFP8 is
    architecturally unusable on this model. gpt_oss_120b_mxfp8 died with
    "AssertionError: K must be divisible by 128" inside
    torchao/prototype/moe_training/kernels/mxfp8/cutedsl_quantize_2d_1x32.py:
    GPT-OSS has dim = hidden_dim = 2880, and 2880 / 128 = 22.5. No padding knob
    fixes that -- pad_multiple pads the token (M) dimension, not the contraction
    dimension K.

    Float8 needs only 16-element alignment (Float8GroupedExpertsConverter
    PAD_MULTIPLE = 16, "16 byte alignment / 1 byte per elem"), and 2880 % 16 == 0,
    so this path fits GPT-OSS's shapes. Requires SM89+; B200 is SM100.

    Built on gpt_oss_120b_ep8_compile, the best measured config (13.52% at step
    50), so this tests precision on top of EP=8 + compile rather than in isolation.
    EP is required by the grouped-experts converter anyway, and compile is required
    for the fused quantize+GEMM to actually pay off.

    Router gate and lm_head stay bf16 via filter_fqns -- quantizing the router
    perturbs expert assignment, and the 201088-wide lm_head feeds the loss.

    Same reading caveat as MXFP8: MFU is computed against the bf16 peak (2.25e15),
    so a float8 run inflates mfu%. Compare on tokens/sec.
    """
    config = gpt_oss_120b_ep8_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            Float8LinearConverter.Config(
                recipe_name="rowwise",
                filter_fqns=["lm_head", "gate"],
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def gpt_oss_120b_ep8_compile_bs_increase() -> Trainer.Config:
    config = gpt_oss_120b_ep8()
    config.training.local_batch_size = 2
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    return config



def _gpt_oss_dist_muon_optimizer(
    *,
    num_layers: int,
    lr: float,
) -> OptimizersContainer.Config:
    """DistMuon on the MoE expert weight stacks, AdamW on everything else.

    Why only the experts: of 116.83B total params, 114.71B are the sparse expert
    stacks (98.2%). AdamW keeps two states per param (exp_avg + exp_avg_sq);
    Muon keeps one momentum buffer. At training.dtype=bfloat16 that is 2 B/param
    saved, so moving just the experts to Muon frees roughly
        114.71e9 * 2 B / 8 GPUs = 28.7 GB = ~26.7 GiB per GPU.
    Putting attention/gate/norms on Muon too would add well under 1 GiB of
    savings while adding real risk, so they stay on AdamW.

    Specifically NOT on Muon:
      - layers.*.attention.qkv_linear.wqkv.weight -- GPT-OSS fuses q, k and v
        (64 q heads + 8 k + 8 v) into one (5120 x dim) tensor. Kimi hands its
        separate wq/wkv_b to Muon with AttentionPerHeadComputeView(n_heads),
        which does not map onto a fused tensor mixing three different head
        counts. Only ~531M params across 36 layers, so nothing is lost.
      - biases, norms, attention sinks, embeddings, lm_head: not matrices.
        Moonlight (arXiv 2502.16982 Sec 2.2) keeps these on AdamW.

    Compute sharding: the expert stacks are E-major (mlp1 is (E, 2*hidden, dim),
    mlp2 is (E, dim, hidden)), sharded over the EP mesh axis. Note this differs
    from Kimi's per-expert config, which shards over dp_shard + efsdp + ep --
    at EP=8 on a single 8-GPU node the mesh comes up as
    ['batch', 'loss', 'ep', 'fsdp'] with NO efsdp axis, because EP consumes all
    8 ranks. Referencing efsdp here would be wrong for this topology.

    REQUIRES EP > 1. At EP=1 there is no 'ep' mesh axis and this map is invalid.
    Also requires TP=1 and PP=1: DistMuon does not support TP's _StridedShard
    layouts, and PP hands each stage only a subset of the param-group patterns.
    """
    per_expert = MuonComputeShardingConfig(
        compute_layout=ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.EP.value: Shard(0),
            },
        )
    )
    expert_projections = ("mlp1_weight_EGD", "mlp2_weight_EDF")

    def shardings_for_layer(layer_id: int) -> dict[str, MuonComputeShardingConfig]:
        prefix = f"layers.{layer_id}.moe.routed_experts.inner_experts"
        return {
            f"{prefix}.{projection}": per_expert for projection in expert_projections
        }

    per_layer = tuple(shardings_for_layer(i) for i in range(num_layers))
    compute_sharding_by_fqn = {
        fqn: sharding for layer in per_layer for fqn, sharding in layer.items()
    }
    # Bucket two layers at a time so the orthogonalization collectives amortize
    # their launch overhead, mirroring the kimi_k2_7 recipe.
    bucket_layer_ids = tuple(
        tuple(range(first, min(first + 2, num_layers)))
        for first in range(0, num_layers, 2)
    )
    bucket_configs = tuple(
        BucketConfig(
            name="layers." + "-".join(map(str, layer_ids)),
            patterns=tuple(fqn for i in layer_ids for fqn in per_layer[i]),
        )
        for layer_ids in bucket_layer_ids
    )
    muon_pattern = (
        r"routed_experts\.inner_experts\.(?:" + "|".join(expert_projections) + r")$"
    )
    return OptimizersContainer.Config(
        implementation="foreach",
        param_groups=[
            ParamGroupConfig(
                pattern=muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs={
                    "lr": lr,
                    "weight_decay": 0.1,
                    "foreach": False,
                    # Scale updates to AdamW magnitude so the shared lr and the
                    # existing warmup/cosine schedule stay meaningful.
                    "adjust_lr_fn": "match_rms_adamw",
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "bucket_configs": bucket_configs,
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
            }
        },
    )


def gpt_oss_120b_muon() -> Trainer.Config:
    """Best measured config (EP=8 + compile + SelectiveAC + bs=2) with DistMuon.

    Single variable versus gpt_oss_120b_ep8_compile_bs_increase (~27.9% mfu at
    step 600, 172.58GiB / 96.77%): only the optimizer changes. That run was
    memory-bound -- it was emitting expandable_segments mapping-failure warnings
    with 5.5MiB free -- so the ~26.7GiB Muon frees is the point. If the memory
    drop lands as predicted, local_batch_size=3 becomes reachable, which is where
    the actual throughput win would come from.

    Caveats when reading the result:
      - Loss is NOT comparable to the AdamW runs. Muon is a different optimizer
        with different update geometry; lr 8e-4 was tuned for AdamW and is very
        likely wrong here. Judge this run on memory and tok/s, not on loss.
      - DistMuon's orthogonalization needs its own workspace, so the net memory
        saving will be less than the 26.7GiB of optimizer state removed.
      - Read it at 300+ steps. MFU on this model climbs from ~22% at step 50 to
        ~28% by step 600, so short reads understate everything.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    num_layers = len(config.model_spec.model.layers)
    config.optimizer = _gpt_oss_dist_muon_optimizer(num_layers=num_layers, lr=8e-4)
    return config


def gpt_oss_120b_muon_bs3() -> Trainer.Config:
    """Best measured config (EP=8 + compile + SelectiveAC + bs=2) with DistMuon.

    Single variable versus gpt_oss_120b_ep8_compile_bs_increase (~27.9% mfu at
    step 600, 172.58GiB / 96.77%): only the optimizer changes. That run was
    memory-bound -- it was emitting expandable_segments mapping-failure warnings
    with 5.5MiB free -- so the ~26.7GiB Muon frees is the point. If the memory
    drop lands as predicted, local_batch_size=3 becomes reachable, which is where
    the actual throughput win would come from.

    Caveats when reading the result:
      - Loss is NOT comparable to the AdamW runs. Muon is a different optimizer
        with different update geometry; lr 8e-4 was tuned for AdamW and is very
        likely wrong here. Judge this run on memory and tok/s, not on loss.
      - DistMuon's orthogonalization needs its own workspace, so the net memory
        saving will be less than the 26.7GiB of optimizer state removed.
      - Read it at 300+ steps. MFU on this model climbs from ~22% at step 50 to
        ~28% by step 600, so short reads understate everything.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    num_layers = len(config.model_spec.model.layers)
    config.optimizer = _gpt_oss_dist_muon_optimizer(num_layers=num_layers, lr=8e-4)
    return config


def gpt_oss_120b_mixed_precision() -> Trainer.Config:
    """Best config (EP=8 + compile + SelectiveAC + bs=2) with bf16 gradient reduction.

    Sets training.mixed_precision_reduce="bfloat16", which becomes FSDP's
    MixedPrecisionPolicy(reduce_dtype=...) in parallelize_gptoss(). By default
    gradients are upcast to fp32 for the reduce-scatter; bf16 halves those bytes.

    Two things to know before relying on this:

    1. The field is typed Literal["float32"] in TrainingConfig, i.e. upstream
       permits only "float32" (contrast mixed_precision_param, which offers both).
       Python does not enforce Literal at runtime and TORCH_DTYPE_MAP has a
       "bfloat16" entry, so this assignment works -- but it is off the supported
       path, a type checker will flag it, and the equivalent CLI override
       (--training.mixed_precision_reduce bfloat16) is REJECTED by tyro. The
       narrowing is deliberate: bf16 carries 8 mantissa bits and summing shards
       compounds rounding error in the gradient. Watch loss/grad_norm on long runs.

    2. The measured gain is small and possibly noise. Observed 27.5% at step 300,
       versus AdamW fp32-reduce at comparable horizons: 26.93% @ step 340 (job 196)
       and 26.96% @ step 370 (job 204), reaching 27.86% @ step 620 (job 208). So
       this is roughly +0.5pp at matched step count, inside run-to-run spread.
       That is consistent with the topology: at EP=8 on 8 GPUs there is no efsdp
       axis, so the 114.71B expert params are sharded by EP alone and their
       gradients are already complete on the owning rank after the all-to-all
       combine -- they never reduce-scatter. Only the 2.11B dense params do, which
       is ~8.4GB fp32 vs ~4.2GB bf16 against a ~1.18s step. To confirm or reject
       it, compare against job 208/214 at 600+ steps, not at 300.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def _gpt_oss_debugmodel_min_async_ep(*, compile_model: bool) -> Trainer.Config:
    """4-GPU validation of the minimal_async_ep dispatcher (EP=4, 8 experts)."""
    config = _gpt_oss_debugmodel()
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(expert_parallel_degree=4)
    # REQUIRED by MinimalAsyncEP, see gpt_oss_120b_ep8_nocompile_bs2_minasync docstring.
    config.activation_checkpoint = FullAC.Config()
    if compile_model:
        config.compile = CompileConfig(enable=True, components=["model", "loss"])
    config.model_spec = model_registry(
        "debugmodel", moe_comm_backend="minimal_async_ep"
    )
    return config


def gpt_oss_debugmodel_min_async_ep() -> Trainer.Config:
    return _gpt_oss_debugmodel_min_async_ep(compile_model=True)


def gpt_oss_debugmodel_min_async_ep_nocompile() -> Trainer.Config:
    return _gpt_oss_debugmodel_min_async_ep(compile_model=False)


def gpt_oss_120b_ep8_nocompile_bs2_minasync() -> Trainer.Config:
    """EP=8 + bs=2 using the MinimalAsyncEP token dispatcher instead of all-to-all.

    Every other config here uses moe_comm_backend="standard"
    (AllToAllTokenDispatcher), which host-synchronizes during dispatch. That sync
    is the price we paid for the EP=8 win, and nothing has varied it yet.
    minimal_async_ep is built into torchtitan (torchtitan/distributed/
    minimal_async_ep), so unlike deepep (H100/NVLink-switch) and hybridep
    (GB200/NVLink72) it needs no external library.

    IMPORTANT -- this config cannot use SelectiveAC like the rest of the winning
    line. maybe_update_minimal_async_ep_config() hard-requires full recompute:
        "MinimalAsyncEP requires full recompute: set activation-checkpoint:full
         for eager training or --compile.memory_policy full for graph_trainer."
    compile.memory_policy does not exist on CompileConfig (its fields are enable,
    enable_async_tensor_parallel, components, backend) -- that knob belongs to the
    experimental graph_trainer -- so on this path FullAC is the only option.

    That collides with a known failure: compile + FullAC crashed the debugmodel
    with "AssertionError: Node add_21 was invalid, but is output" (AOTAutograd).
    Whether that reproduces alongside this dispatcher is settled empirically by
    gpt_oss_debugmodel_min_async_ep / _nocompile on 4 GPUs before spending a node.

    Other requirements, all satisfied: expert_parallel_degree > 1 (8), num_experts
    divisible by EP (128 / 8 = 16), and spmd_backend != "full_dtensor".
    hidden_dim / num_max_tokens_per_rank / dtype are left None here on purpose --
    the trainer fills them via model_config.update_from_config() (trainer.py:348)
    before the model is built.

    Comparison note: because AC is forced from SelectiveAC to FullAC, this is not
    a single-variable test against the ~27.9% bs=2 result. SelectiveAC vs FullAC
    measured 8.22% vs 8.12% uncompiled, so the AC term is small, but it is not nil.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.activation_checkpoint = FullAC.Config()
    # CONFIRMED on 4x B200 (jobs 281/282, gpt_oss_debugmodel_min_async_ep*):
    #   minimal_async_ep + FullAC + compile -> AssertionError: Node add_21 was
    #                                          invalid, but is output
    #   minimal_async_ep + FullAC, no compile -> trains fine
    # So compile must be OFF. Compare against gpt_oss_120b_ep8_nocompile_bs2_alltoall_control,
    # NOT against the ~27.9% compiled result.
    config.compile = CompileConfig(enable=False)
    config.model_spec = model_registry("120b", moe_comm_backend="minimal_async_ep")
    return config


def gpt_oss_120b_ep8_nocompile_bs2_alltoall_control() -> Trainer.Config:
    """Control for gpt_oss_120b_ep8_nocompile_bs2_minasync: same settings, standard all-to-all.

    Identical to gpt_oss_120b_ep8_nocompile_bs2_minasync (EP=8, bs=2, FullAC, compile off)
    except moe_comm_backend stays "standard". Without this datapoint the
    min_async_ep number is uninterpretable -- we have no EP=8 + FullAC + bs=2 +
    no-compile measurement to compare it against, only compiled ones.
    """
    
    # Saw 14% MFU. Biggest problem is no torch compile. Try to fix
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.activation_checkpoint = FullAC.Config()
    config.compile = CompileConfig(enable=False)
    return config


def gpt_oss_120b_membudget_bs3() -> Trainer.Config:
    """Idea 2: MemoryBudgetAC to buy local_batch_size=3 while KEEPING compile+inductor.

    Batch size is the only lever with a proven large payoff here (bs 1->2 was
    +63%, 13.5% -> 27.9%), and memory is what blocks bs=3. Muon showed that
    freeing memory is not enough on its own if you pay compute for it -- it freed
    23.2GiB but halved throughput orthogonalizing 114.7B params. MemoryBudgetAC
    pays in a little recompute instead, which is far cheaper, and unlike the
    minimal_async_ep route it stays inside the compiled configuration that
    actually delivers 27.9%.

    Memory arithmetic. bf16 params+grads+Adam is a fixed ~108.8GiB/GPU. Measured
    totals for EP=8 + compile: 144.96GiB at bs=1 (job 193) and 168.84-173.56GiB
    at bs=2 (jobs 204/208/214), i.e. activations ~36GiB then ~65GiB -- roughly
    linear. Extrapolating, bs=3 with SelectiveAC needs ~94GiB of activations for
    ~203GiB total, about 25GiB over the 178.35GiB limit. FullAC at bs=3 measured
    158.03GiB total (job 191), so FullAC-level activation memory (~49GiB) fits
    comfortably. memory_budget is therefore set low (0.3), near the FullAC end of
    the dial: 0.0 == activation memory of full recompute, 1.0 == no recompute.
    If it still OOMs, go lower; if it fits with room to spare, raise it to trade
    memory back for speed.

    Why this needs the unsafe flag. MemoryBudgetAC + EP previously died with
    "RuntimeError: Cannot compute the size of FakeScriptObject on node primals_17"
    (job 190) -- the budget partitioner must size every node and cannot size EP's
    process-group ScriptObject. The error names the escape hatch, and we set it
    below. It is labelled unsound in general, but the object here is a process
    group, which holds no tensor storage, so zero-size is accurate for this case.

    MemoryBudgetAC also *requires* compile ("model" in compile.components), which
    Trainer.Config validates -- satisfied via the base config.

    NOT yet validated on hardware. The debugmodel has repeatedly failed to predict
    120b behaviour (MemoryBudgetAC+EP, Float8, and compile+FullAC under
    minimal_async_ep all passed small and broke at scale), so this one goes
    straight to a 400-step 8-GPU probe. Watch for: an OOM (lower the budget), the
    FakeScriptObject error resurfacing (flag not taking effect), or a NCCL
    ALLTOALL_BASE timeout like job 291 (recompute desynchronizing the EP
    collectives -- which would mean partial recompute under compile is unsafe with
    EP generally, not just for FullAC).
    """
    import torch._functorch.config as _functorch_config

    # Must be set before torch.compile traces anything; config functions run at
    # startup, well before parallelize_gptoss() applies compile.
    _functorch_config.unsafe_treat_script_objects_as_zero_size = True

    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.training.local_batch_size = 3
    config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.3)
    return config


def _gpt_oss_120b_mxfp8(fqns: list[str]) -> Trainer.Config:
    """MXFP8 on the dense linears of the best 120b config. Experts stay bf16.

    Base is gpt_oss_120b_ep8_compile_bs_increase (EP=8, compile+inductor,
    SelectiveAC, bs=2), the best measured 120b config: 27.58% MFU / 620.54
    TFLOPs/GPU / 13,701 tok/s/GPU sustained over steps 481-620 (job 208).

    Why the experts are untouched. The MXFP8 grouped-GEMM path is architecturally
    unusable on gpt_oss: the CuTeDSL quantize kernel asserts K % 128 == 0
    (torchao/prototype/moe_training/kernels/mxfp8/cutedsl_quantize_2d_1x32.py:998)
    and gpt_oss has dim = hidden_dim = 2880, remainder 64. Verified twice --
    job 192 on the 120b, job 1049 on the 20b. pad_multiple pads per-expert token
    groups (M), not K, so it cannot fix this.

    torchao also ships a parallel flydsl_* MXFP8 kernel family with no such
    assertion, which would have been the way out. It is unreachable: it needs a
    runtime package named "flydsl" (_missing_flydsl_runtime_packages() reports
    it missing), and the "flydsl" on PyPI is "FlyDSL - ROCm Domain Specific
    Language", an AMD project, not the NVIDIA one torchao expects. Do not install
    it on this box.

    MXFP8Linear needs only K % 32 == 0 (1x32 block scaling), which every dense
    GEMM here satisfies: qkv K=2880, wo K=4096, lm_head K=2880.

    Expected upside, sized honestly. Of the 5.71B active params, attention is
    ~955M (36 layers x (wqkv 5120x2880 + wo 4096x2880)) and the 4 active experts
    are ~3.59B, so fqns=["attention"] reaches only ~21% of per-token GEMM work.
    Adding lm_head (2880 x 201,088) reaches roughly another ~11%. Note this is
    NOT the "42.9% dense" figure from the 20b docstring -- that counts embeddings
    and lm_head, and embeddings are a lookup, not a GEMM. The 20b measured +3.50%
    tok/s from the attention-only version; expect the same order here.

    The router gate is deliberately never quantized: perturbing it changes expert
    assignment, which is a correctness risk rather than a speed tradeoff.

    READ TOKENS/SEC, NOT MFU. torchtitan computes MFU against the bf16 dense peak
    (2.25e15, tools/utils.py), so any low-precision run reports an inflated or
    N/A MFU that is not comparable to the bf16 line of results. The number to
    beat is 13,701 tok/s/GPU (620.54 TFLOPs/GPU), job 208.

    Memory goes UP, not down -- do not expect the datatype name to save memory.
    These converters do *dynamic* quantization: master weights stay bf16 and are
    re-quantized every step, so the fp8 data (1 B/elem) plus its e8m0 scales (one
    per 32 elems) are allocated *in addition to* the bf16 tensors, which must
    persist for the optimizer and for autograd. Nothing is replaced. Measured at
    bs=1 (job 1304 vs 1305): model memory at init is byte-identical at 30.07GiB
    and step 1 matches within 0.02GiB, but steady state is 146.52 vs 145.02GiB,
    +1.50GiB -- entirely per-step transients. That is the expected size: the
    ~955M attention params MXFP8 touches are ~0.96GiB of fp8 copies plus scales,
    and quantized activations account for the rest. A memory win would require
    persistently storing weights in fp8, which is a different technique.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns,
            ),
        ],
    )
    return config


def gpt_oss_120b_mxfp8_linears() -> Trainer.Config:
    """Direct port of the proven gptoss20b_mxfp8 recipe (+3.50% tok/s) to 120b."""
    return _gpt_oss_120b_mxfp8(["attention"])


def gpt_oss_120b_mxfp8_linears_lmhead() -> Trainer.Config:
    """gpt_oss_120b_mxfp8_linears plus lm_head, mirroring gptoss20b_mxfp8_lmhead.

    lm_head is 2880 -> 201,088 applied to every token, so it is worth roughly as
    much again as attention. It feeds the loss directly, which is why it is split
    out from the attention-only config rather than bundled in.
    """
    return _gpt_oss_120b_mxfp8(["attention", "lm_head"])


def gpt_oss_120b_mxfp8_linears_bs1() -> Trainer.Config:
    """MXFP8 dense linears at bs=1, to test the memory-pressure hypothesis.

    gpt_oss_120b_mxfp8_linears (bs=2) was a 22.5% throughput regression: 9,728
    vs 12,551 tok/s/GPU mean over steps 200-340 (job 1300 vs job 208), with the
    MXFP8 run oscillating 8.4k-10.6k while bf16 held a smooth 12.9-13.3k. It ran
    at 95.85% memory and logged 7 expandable_segments mapping failures against
    bf16's 3, with zero recompiles -- so the instability is allocator contention,
    not Dynamo thrash.

    The suspicion is that dynamic quantization needs scratch space the bs=2 config
    does not have. Supporting evidence: the 20b recipe this was ported from
    (+3.50% tok/s) ran on gptoss20b_noreshard_membudget at 131.39GiB peak, with
    real headroom, whereas the 120b bs=2 best sits at 172.58GiB / 96.77%.

    So this drops to bs=1, where gpt_oss_120b_ep8_compile measured 144.96GiB
    (81.28%) -- roughly 33GiB of headroom. If MXFP8 turns positive here while
    negative at bs=2, memory pressure is the cause and MXFP8 is only usable on
    120b in configurations with slack. If it is still negative, the quantize
    overhead simply exceeds the GEMM saving on this model's ~21% dense GEMM share,
    and MXFP8 is not worth pursuing on the 120b at all.

    Must be compared against gpt_oss_120b_ep8_compile run to the SAME step count.
    The existing bs=1 datapoint (job 193) stopped at step 60 -- 13.52% MFU at step
    50, still climbing to 14.34% at step 60 -- which is far too early to compare
    against a 400-step run on this model.

    Read tokens/sec, not MFU: torchtitan reports mfu N/A once any GEMM is
    low-precision.
    """
    config = gpt_oss_120b_ep8_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention"],
            ),
        ],
    )
    return config


def gpt_oss_120b_noreshard() -> Trainer.Config:
    """Best 120b config + fsdp_reshard_after_forward="never".

    Base is gpt_oss_120b_ep8_compile_bs_increase (EP=8, compile+inductor,
    SelectiveAC, bs=2): 27.58% MFU / 620.54 TFLOPs/GPU / 13,701 tok/s/GPU
    sustained over steps 481-620 (job 208). Only the reshard policy changes.

    "never" keeps FSDP parameters unsharded after forward instead of freeing them
    and re-all-gathering for backward, spending memory to remove one all-gather
    per FSDP module per step. Communication is the untouched axis on this model:
    every lever tried so far was compute (compile, MXFP8), parallelism (EP), batch
    size, or memory (Muon, MemoryBudgetAC).

    On the "+34GiB so it needs bs=1" warning: that number looks like it came from
    a 20b run at ep=1, where FSDP shards all ~20.9B params -- unsharded 41.8GB vs
    sharded 5.2GB is ~+34GiB. It should not transfer. At EP=8 on 8 GPUs the mesh
    is ['batch', 'loss', 'ep', 'fsdp'] with no efsdp axis, so the 114.71B expert
    params are EP-sharded and reshard_after_forward never applies to them. Only
    the 2.11B dense params are FSDP-managed: ~3.93GiB unsharded in bf16 against
    ~0.49GiB sharded, so roughly +3.4GiB, inside the ~5.8GiB of headroom at bs=2.

    That is an estimate, not a measurement, which is why this sits at bs=2 and
    gpt_oss_120b_noreshard_bs1 exists as the fallback. bs=1 has a matched 400-step
    bf16 control already measured (job 1305: 403.35 TFLOPs/GPU, 8,906 tok/s/GPU,
    17.93% MFU), so a bs=1 result is interpretable immediately.

    MFU is valid here -- nothing is quantized -- so TFLOPs/GPU, tok/s and MFU are
    all directly comparable to job 208.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def gpt_oss_120b_noreshard_bs1() -> Trainer.Config:
    """gpt_oss_120b_noreshard at bs=1, the fallback if the bs=2 version OOMs.

    bs=1 base (gpt_oss_120b_ep8_compile) measured 145.02GiB / 81.31%, so ~33GiB
    of headroom absorbs the unsharded parameters even if the +3.4GiB estimate in
    gpt_oss_120b_noreshard is badly wrong. Control: job 1305, 403.35 TFLOPs/GPU.
    """
    config = gpt_oss_120b_ep8_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


# ---------------------------------------------------------------------------
# Communication-efficiency wave (2026-09-16).
#
# Motivation and the correction it rests on. The trace that started this work
# (job 1399, outputs/profiling/traces_120b_best/iteration_400) shows 41.0% of
# kernel time in NCCL and only 9.6% of that comm hidden behind compute. Both
# numbers are real, but they do NOT describe the 660 TFLOPs configuration:
#
#   job 237  gpt_oss_120b_ep8_compile_bs_increase  ~660 TFLOPs  step ~1,122 ms
#   job 1399 same config, profiled                 ~460 TFLOPs  step ~1,674 ms
#
# Job 1399 was missing three environment variables that job 237 set:
#   TORCH_NCCL_AVOID_RECORD_STREAMS=1
#   NCCL_NVLS_ENABLE=0
#   PYTORCH_ALLOC_CONF=...,garbage_collection_threshold:0.8
# and it ran at 173.52GiB (97.29%) against job 237's 169.50GiB (95.04%).
#
# What the trace actually shows, measured per rank: all 7 non-rank-4 ranks each
# sit ~185 ms inside ONE ncclDevKernel_SendRecv waiting for rank 4, whose GPU is
# 29 ms busy out of that 185 ms window while its CPU is stuck in
# CompiledFunctionBackward. So the dominant "communication" cost in that trace
# is rank desynchronization under memory pressure, not bytes on the wire.
# Correlating every NCCL kernel to its collective through the profiler's
# "External id" (analyze_trace_comm.py) prices this exactly:
#
#   collective              bytes/step   kernel time   best rate   transfer   wait
#   all_to_allv  (bf16)       55.87 GB     564.86 ms   713 GB/s    78.33 ms  486.53 ms
#   reduce_scatter (fp32)      8.51 GB     122.27 ms   638 GB/s    13.34 ms  108.93 ms
#   all_gather   (bf16)        1.21 GB      29.21 ms    82 GB/s    14.79 ms   14.42 ms
#   ---------------------------------------------------------------------------------
#   total                                   716.34 ms              106.46 ms  609.88 ms
#
# "transfer" is the time those same bytes would take at the best rate the run
# itself achieved. So of 716 ms of NCCL kernel time, ~107 ms moves data and
# ~610 ms is ranks waiting for each other. The largest dispatches (804-819 MB)
# hit 693-713 GB/s, close to NVLink peak, while 52-135 MB calls land at
# 0.3-14 GB/s -- those are not slow transfers, they are stalls.
# Bandwidth is not the problem; arrival skew is.
#
# Trace compute union is 1,050 ms. Against job 237's 1,122 ms step that leaves
# only ~70 ms of exposed comm at 660 TFLOPs, so the 41% headline overstates the
# available win by roughly an order of magnitude. run_profile_v2.sbatch re-takes
# the trace with job 237's environment to replace these estimates with a
# measurement.
#
# The one comm cost the trace shows that is NOT overlap-dependent, and therefore
# survives the correction, is the fp32 gradient path (see gpt_oss_120b_bf16reduce).
# ---------------------------------------------------------------------------


def gpt_oss_120b_bf16reduce() -> Trainer.Config:
    """Best 120b config with the FSDP gradient reduce-scatter in bf16.

    Single variable against gpt_oss_120b_ep8_compile_bs_increase (job 237,
    ~660 TFLOPs/GPU, 14,600 tok/s/GPU, 169.50GiB): only
    training.mixed_precision_reduce moves from "float32" to "bfloat16".

    Why this one is not overlap-dependent. FSDP reduces gradients in fp32 while
    the model trains in bf16, so every step pays twice:

      ncclDevKernel_ReduceScatter_Sum_f32   122.27 ms   fp32 bytes on the wire
      chunk_cat_cuda_kernel<float, BFloat16> 75.94 ms   bf16 -> fp32 copy-in

    The 75.94 ms is ordinary compute on the critical path -- it is not comm and
    cannot be hidden by better overlap -- and halving the reduce-scatter payload
    is a byte reduction that does not depend on arrival skew. Verified against
    the trace's record_param_comms: reduce-scatter input totals 2.114e9 elements
    per step (2 x 579,133,440 for tok_embeddings and lm_head at vocab 201,088 x
    dim 2,880, plus 36 x 26,924,672 for the per-layer dense params), which is
    exactly the 2.114B dense parameter count. The 114.71B expert params are
    EP-sharded and never enter this collective, so bf16 reduction here touches
    only 1.8% of the model's parameters.

    Expected size, stated before measuring: the Qwen3.5-122B precedent on this
    same cluster measured +4.5% (RESULTS_QWEN35.md §17, jobs 647/648), and the
    lesson recorded there applies -- a profile bucket's size is an upper bound on
    what removing it can save, not an estimate, because FSDP already overlaps
    gradient reduction with backward compute. Anything from +3% to +8% would be
    consistent.

    THIS CHANGES TRAINING NUMERICS and is not cleared for a real run. Gradients
    reduced across 8 shards in bf16 accumulate rounding error that fp32
    reduction exists to prevent. It must be read against a float32 control at
    the same --debug.seed; gpt_oss_120b_ep8_compile_bs_increase run with the same
    seed is that control, which is why no separate control config is added here.
    On Qwen3.5 the paired curves converged (gap narrowing ~30x from step 15 to
    150) with no instability, but 400 steps can reveal a problem, not prove its
    absence at 10k.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def gpt_oss_120b_fsdp_symm_mem() -> Trainer.Config:
    """Best 120b config with FSDP2 symmetric-memory collectives.

    Single variable against gpt_oss_120b_ep8_compile_bs_increase: only
    parallelism.enable_fsdp_symm_mem moves from False to True. This routes the
    FSDP all-gather and reduce-scatter through symmetric memory (direct NVLink
    peer copies) instead of NCCL ring kernels.

    Prior expectation is LOW. The same flag measured exactly flat on
    Qwen3.5-122B on this cluster -- 11.82% MFU and 3,650 tok/s against a 3,648
    baseline, identical peak memory (RESULTS_QWEN35.md §18, job 785). It is
    included because the FSDP side here is a different shape than Qwen3.5's: at
    EP=8 only the 2.114B dense params are FSDP-managed, and 1.16B of those are
    two 579M-element embedding matrices, so this run's collectives are a few
    very large buckets (2 x 144.8 MiB all-gathers) rather than many medium ones.
    Symmetric memory helps large transfers more than small, so the Qwen3.5 null
    result does not transfer cleanly.

    Numerics are unchanged, so loss is directly comparable to job 237 and MFU is
    valid. Cheap to run and independent of gpt_oss_120b_bf16reduce; if both win
    they should be stacked and re-measured, not assumed additive.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.parallelism.enable_fsdp_symm_mem = True
    return config


# ---------------------------------------------------------------------------
# Wave 2: the non-GEMM compute the corrected trace reading exposes.
#
# Once the 41% NCCL figure is understood as rank skew rather than bytes (see the
# wave-1 block above), the trace's 570.54 ms of non-GEMM, non-NCCL kernel time
# becomes the thing worth attacking. Attributed to call sites by External id:
#
#   ms      n    aten call site              what it is
#   ------  ---  --------------------------  --------------------------------
#   83.70   200  _fused_adamw_               optimizer step
#   75.94    74  _chunk_cat                  FSDP bf16->fp32 grad copy-in
#   38.71   439  copy_ (Memcpy DtoD)         staging copies
#   26.43    36  <inductor fused pointwise>  router math + token permute
#   23.89    36  <inductor fused pointwise>  index_put / index_select permute
#   22.03    36  <inductor fused pointwise>  permute backward
#   21.76    36  _index_put_impl_            MoE combine backward scatter
#   21.56    36  <inductor fused pointwise>  index_put / index_select permute
#   21.47    36  <inductor fused pointwise>  router + permute (fwd)
#   16.55    37  div (fp32)                  FSDP gradient divide
#   18.17   200  _foreach_norm/_foreach_mul_ grad-norm clipping
#
# Grouping those: ~178 ms is the MoE token permute/unpermute cluster (14
# inductor pointwise kernels, all n=36, i.e. once per layer), ~228 ms is the
# fp32 gradient path (_chunk_cat + fp32 div + the fp32 reduce-scatter), and
# ~84 ms is the optimizer. Only 517 ms of the step is in GEMMs.
#
# The expert grouped GEMMs are the healthy part: ~352 TFLOP of expert work in
# 341.80 ms is ~1,030 TFLOPs, 46% of the 2.25e15 bf16 peak. The dense GEMMs
# (attention projections plus the 201,088-wide lm_head) manage ~530 TFLOPs.
#
# A second pattern from the logs drives the ordering below. Every configuration
# that pushes HBM to >=97% collapses, and not by a little:
#
#   job 237  gpt_oss_120b_ep8_compile_bs_increase   95.04%   ~660 TFLOPs
#   job 1399 same config, thinner env               97.29%   ~460 TFLOPs
#   job 1300 gpt_oss_120b_mxfp8_linears             96.23%   ~460 TFLOPs
#   job  661 gpt_oss_120b_membudget_bs3             97.55%     ~32 TFLOPs
#   job 1518 gpt_oss_120b_noreshard                 97.56%     ~20 TFLOPs, then OOM
#
# That is the same mechanism as the trace's rank-4 straggler: near the limit the
# allocator stalls, one rank falls behind, and every other rank bills the wait
# to whatever collective it is parked in. So memory headroom is worth more here
# than any single kernel on the list above, and a wave-2 candidate that spends
# memory has to be judged against that, not just on its own kernel savings.
#
# Nothing in this block changes the permute implementation, because none of the
# routes to one are open: TorchAOTokenDispatcher is reachable only through the
# quantization converters, MXFP8GroupedExpertsConverter is architecturally
# blocked (the CuTeDSL 1x32 quantize kernel needs K % 128 == 0 and GPT-OSS has
# dim 2880), deepep and hybridep are not installed in the venv, and
# minimal_async_ep hard-requires FullAC, which crashes under compile
# ("AssertionError: Node add_21 was invalid, but is output"). What is left is
# making the generated kernels better (inductor autotuning, driven by env vars
# in run_gptoss120b_v2.sbatch since CompileConfig exposes no inductor knobs),
# bounding how much of the cost is routing imbalance, and one retry of the
# float8 path.
# ---------------------------------------------------------------------------


def gpt_oss_120b_balanced_diag() -> Trainer.Config:
    """DIAGNOSTIC CEILING, not a shippable config: forced round-robin routing.

    Sets debug.moe_force_load_balance on the best config. The router's own
    top-k choice is replaced by round-robin assignment, so every expert receives
    exactly the same number of tokens.

    This is NOT a throughput candidate -- it changes which expert sees which
    token, so the model being trained is not GPT-OSS-120B and the loss is
    meaningless. It exists to put a number on how much of the MoE cost is
    imbalance rather than intrinsic work, which three separate measurements
    currently blame imbalance for:

      - the 144 bf16 all-to-all payloads in the trace span 200.9-819.4 MB,
        a 4x spread across calls that should be identical in a balanced model
      - the permute kernels' cost scales with the largest expert group, not the
        average, because grouped-mm pads to the biggest group
      - rank 4 being ~185 ms late is consistent with it having drawn the heavy
        share of routed tokens that step

    Under forced balance all three effects vanish at once. Read the result as an
    upper bound on what a real fix (auxiliary load-balancing loss, router
    temperature, expert-capacity tuning) could recover, and read it on tok/s
    only. If it is flat, imbalance is not the problem and this whole line of
    attack closes -- which is worth one node-hour to know.

    Numerics note: GptOssMoE subclasses the common MoE and does not override the
    router, and GptOssModel subclasses the common Decoder, whose
    update_from_config propagates debug.moe_force_load_balance into
    router._debug_force_load_balance. So the flag does take effect on this model.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    config.debug = DebugConfig(moe_force_load_balance=True)
    return config


def gpt_oss_120b_fp8_bs2() -> Trainer.Config:
    """Float8 rowwise linears + float8 grouped experts on the best config.

    gpt_oss_120b_fp8 already exists but was built on gpt_oss_120b_ep8_compile
    (local_batch_size=1) and crashed when it ran as job 194:

        CUDA error: unspecified launch failure  (cudaErrorLaunchFailure)
        raised from currentStreamCaptureStatusMayInitCtx

    This is the retry at bs=2 on the current best base rather than a re-run of
    the same thing: the base config, the environment (job 194 predates
    TORCH_NCCL_AVOID_RECORD_STREAMS / garbage_collection_threshold:0.8) and the
    memory profile are all different now. It is the speculative entry of wave 2
    and should be queued last.

    Why it is worth one attempt despite the crash. Float8GroupedExpertsConverter
    is the only open route to replacing the MoE permute: it swaps in
    TorchAOTokenDispatcher, which uses torchao's permute_and_pad kernel instead
    of the inductor-generated permute that costs ~178 ms per step. So this
    single config attacks the largest non-GEMM pool AND the 341.80 ms of expert
    grouped GEMM at once. MXFP8 cannot do this -- it needs K % 128 == 0 and
    GPT-OSS's dim is 2880 -- but float8 needs only 16-element alignment and
    2880 % 16 == 0.

    Reading the result: MFU is computed against the bf16 peak (2.25e15), so a
    float8 run inflates mfu% and metrics.py reports N/A. Compare on tok/s, which
    is precision-neutral. Loss is not comparable to the bf16 runs.

    Router gate and lm_head stay bf16 via filter_fqns: quantizing the router
    perturbs expert assignment, and the 201,088-wide lm_head feeds the loss.

    Memory is the risk as much as the crash. Every config in this registry that
    reached >=97% of HBM collapsed, and bs=2 already sits at 95.04%. Float8
    keeps master weights in bf16 and adds quantized copies plus padding, so this
    could land the wrong side of that line even if the launch failure is gone.
    If it OOMs or degrades, gpt_oss_120b_fp8 at bs=1 is the fallback shape.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            Float8LinearConverter.Config(
                recipe_name="rowwise",
                filter_fqns=["lm_head", "gate"],
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


# ---------------------------------------------------------------------------
# Wave 3: expert load balance, which turns out to be the throughput limiter.
#
# gpt_oss_120b_balanced_diag (job 2278) was queued as a diagnostic ceiling and
# came back as the largest result in this whole registry:
#
#   step            100   200   300   400   500   600   peak HBM
#   job 2278 forced 686   689   681   684   682   683   163.68GiB (91.77%)
#   job 2268 control 404   494   523   532   538   517   171.28GiB (96.03%)
#   job  237 (*)     464   550   616   631   646   656   169.50GiB (95.04%)
#   (*) different LR trajectory, see the TT_STEPS note below
#
# Forced round-robin routing is FLAT AT THE CEILING FROM STEP 100. Every other
# run climbs toward it for hundreds of steps and never arrives. So the ramp that
# looked like ordinary warmup is the auxiliary-loss-free load balancer slowly
# converging, and the gap between the ramp and 683 is the cost of imbalance:
# +27% throughput and 7.6GiB of peak memory, the latter mattering twice over
# because every config here that touched 97% of HBM collapsed.
#
# The mechanism is in optimizer.py::_update_expert_bias:
#
#   expert_bias_delta_E = load_balance_coeff * sign(mean(tokens_per_expert) -
#                                                   tokens_per_expert)
#
# A fixed-size sign step, DeepSeek-V3 style (arXiv 2408.15664), applied once per
# optimizer step. GPT-OSS's 120b flavor hardcodes load_balance_coeff=1e-3
# (gpt_oss/__init__.py:363), so a routing bias can travel at most 0.001 per
# step: order 1000 steps to move one bias by 1.0 against softmax logits of order
# 1. That is why balance is still converging at step 600, and it is the knob.
#
# The configs below raise it. This is a legitimate training knob, not a
# benchmark cheat -- it is the published bias update rate, and the balancer adds
# no gradient term to the objective (that is what "auxiliary-loss-free" means).
# Too large should overshoot and oscillate, so 10x and 30x bracket it rather
# than guessing one value.
#
# A METHODOLOGY NOTE that applies to every comparison here. Wave 1 and 2 ran
# TT_STEPS=600, and LRSchedulersContainer clamps warmup to the total step count
# (lr_scheduler.py:107-112), so those runs rewrote warmup_steps 2000 -> 600 and
# reached the full 8e-4 LR by step 600 instead of ~2.4e-4. They are internally
# comparable but are on a different, more aggressive LR trajectory than job 237,
# which plausibly explains why the control plateaued at ~535 where job 237
# reached 656: a faster-moving router gate balances worse. Wave 3 leaves
# training.steps at the config's 10000 and caps the run with slurm --time
# instead, so the LR trajectory is the real one.
# ---------------------------------------------------------------------------


def _gpt_oss_120b_with_load_balance_coeff(coeff: float) -> Trainer.Config:
    """Best 120b config with the expert-bias update rate set to `coeff`.

    Reaches into the per-layer MoE configs because load_balance_coeff is a model
    flavor argument, hardcoded at 1e-3 by gpt_oss/__init__.py::_120b, with no
    Trainer.Config field and no CLI override. All 36 layers must agree:
    register_moe_load_balancing_hook raises if load_balance_coeff is set on some
    MoE layers and not others, and nothing enforces a consistent VALUE, so an
    inconsistent sweep would silently train 36 different balancers.
    """
    config = gpt_oss_120b_ep8_compile_bs_increase()
    for layer in config.model_spec.model.layers:
        layer.moe.load_balance_coeff = coeff
    return config


def gpt_oss_120b_lbc1e2() -> Trainer.Config:
    """load_balance_coeff 1e-3 -> 1e-2 (10x). Single variable off job 237's config.

    The conservative end of the bracket. At 10x, a routing bias can move 0.01
    per step, so the ~1.0 of bias travel that balance appears to need arrives in
    ~100 steps rather than ~1000. If the +27% ceiling from job 2278 is really
    just convergence speed, this should track much closer to 683 early and
    plateau higher than the ~535-656 band.

    Numerics: this changes routing, so loss is not bit-comparable to the control,
    but unlike gpt_oss_120b_balanced_diag it is a real model -- the router still
    chooses experts, it is just nudged toward balance faster. MFU stays valid
    (nothing is quantized). Watch peak HBM: job 2278 freed 7.6GiB by balancing,
    and any of that which materializes here is worth as much as the throughput.
    """
    return _gpt_oss_120b_with_load_balance_coeff(1e-2)


def gpt_oss_120b_lbc3e2() -> Trainer.Config:
    """load_balance_coeff 1e-3 -> 3e-2 (30x). The aggressive end of the bracket.

    Exists to find the overshoot. A sign-step update has no damping, so the bias
    hunts around the balanced point with an amplitude proportional to the step
    size; at some coeff the hunting itself costs more than the imbalance it
    fixes, and expert assignment starts churning between steps, which also
    defeats any expert specialization the model is trying to learn.

    If this beats gpt_oss_120b_lbc1e2, the knob is not yet saturated and the
    sweep should continue upward. If it is worse, the answer is between 1e-3 and
    3e-2 and gpt_oss_120b_lbc1e2 is the better starting point. Either result
    bounds the useful range, which is the reason to run both rather than pick
    one. Judge on throughput AND on whether the loss curve stays smooth --
    routing churn shows up as loss noise before it shows up as divergence.
    """
    return _gpt_oss_120b_with_load_balance_coeff(3e-2)


# ---------------------------------------------------------------------------
# Wave 4 (overnight 2026-09-16/17): memory headroom, from the AC save set.
#
# What waves 1-3 settled:
#
#   lever                          result
#   -----------------------------  ------------------------------------------
#   bf16 grad reduce               +5% twice (job 2269 vs 2268 at clamped LR,
#                                  job 2295 vs 2292 at the real LR). KEEP.
#   TT_STEPS fix                   job 2292 reproduced job 237 -- 628 TFLOPs
#                                  at steps 600-1200 vs 237's 646-671. The LR
#                                  clamp WAS the reason waves 1-2 sat at ~535.
#   load_balance_coeff up          BACKWARDS for throughput. At step 1000:
#                                  1e-3 -> 629, 1e-2 -> 554, 3e-2 -> 503.
#                                  Monotonic. But loss and stability improve
#                                  monotonically the other way (grad_norm at
#                                  step ~2200: 1e-3 -> 4.2, 3e-2 -> 1.8).
#   pointwise autotune             +1-3%, inside the control's own variance,
#                                  and costs 2.7GiB. Marginal.
#   fsdp symm mem                  flat. MAX_AUTOTUNE=1 Triton codegen bug.
#                                  float8 crashes reproducibly. All closed.
#   NCCL bandwidth                 closed. 586-670 GB/s in isolation vs 98.9
#                                  aggregate in-run; NVLS/buffers/channels all
#                                  flat; only 106ms of 468ms NCCL is transfer.
#
# And the thing that reframes what is left. Every run diverges once warmup ends
# at step 2000 and the LR reaches its full 8e-4 (job 2292):
#
#   step    600   1000   1400   2000   2400   2800   3000
#   tflops  627    629    532    596    562    528    503
#   g_norm 0.54   0.36   0.36   1.42   6.50  38.75  24.75
#   loss   5.10   4.38   4.34   4.48   4.74   4.93   4.84   <- rising
#
# Loss turns upward at ~step 1400 and grad_norm goes to 39. The throughput decay
# is downstream of that: as the router destabilizes, expert assignment collapses
# toward fewer experts, balance degrades, and throughput follows. Job 237 never
# saw this because it stopped at step ~2000.
#
# So the honest figure for this configuration is ~628 TFLOPs (bf16reduce ~660),
# measured over steps 600-1200, and anything past ~1400 is measuring a diverging
# run. gpt_oss_120b_bf16reduce_lr3e4 below exists to fix that.
#
# THE REMAINING LEVER, and it comes from reading the AC policy rather than the
# trace. activation_checkpoint.py::_get_default_save_ops puts the collectives in
# the MUST_SAVE set:
#
#   comm_ops = [reduce_scatter_tensor, all_to_all_single, deepep.*, hybridep.*]
#
# with the stated rationale "to avoid re-communication". For GPT-OSS-120B at
# EP=8 that is an expensive default. Each layer's dispatch all-to-all output is
# ~65,536 rows x 2,880 x 2B = 377MB and the combine's is the same, so keeping
# both across 36 layers is order 25GiB of the ~49GiB of activation memory at
# bs=2 (bs=1 measured 144.96GiB and bs=2 169.50GiB, so ~24.5GiB per unit of
# batch on top of ~120.5GiB of fixed param/grad/optimizer state).
#
# Spending 25GiB to avoid re-communication is the wrong trade on this model, for
# a reason specific to what waves 1-3 measured: the all-to-all only moves ~78ms
# of actual bytes per step, so recomputing the dispatch in backward should cost
# order 20ms, while the memory it frees is the variable that has correlated with
# throughput all along -- 91.77% HBM went with 683 TFLOPs, 95.92% with 628, and
# everything at 97%+ collapsed. It should also put local_batch_size=3 in reach,
# and bs 1->2 was +63%.
#
# The risk is a hang, not a slowdown, and it is the likely reason for the
# MUST_SAVE default: re-issuing a collective during backward requires every rank
# to do it in the same order. The wave-4 jobs therefore run with a short
# walltime so a deadlock cannot consume the night.
# ---------------------------------------------------------------------------


class GptOssSelectiveACRecomputeA2A(SelectiveAC):
    """SelectiveAC that recomputes the EP all-to-all instead of saving it.

    Drops only ``_c10d_functional.all_to_all_single`` from the MUST_SAVE set,
    leaving ``reduce_scatter_tensor`` saved. The reduce-scatter is FSDP's
    gradient reduction, which happens in backward already and has no forward
    output worth saving or recomputing; the all-to-all is the MoE dispatch and
    combine, whose saved outputs are the large tensors.

    Everything else about the policy is inherited, including the
    "recompute every second matmul" balance and the force-recompute of
    moe.router.gate. So this is a single-op change to the save set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        pass

    def get_save_ops(self) -> set:
        ops = super().get_save_ops()
        # _get_default_save_ops returns a set (it seeds from a set comprehension
        # over compute_intensive_ops, then set.update(dict) folds in the
        # resolved comm/compute op dicts by key), so discard is the right call
        # and is a no-op if a future torch renames the op.
        ops.discard(torch.ops._c10d_functional.all_to_all_single.default)
        return ops


def gpt_oss_120b_bf16reduce_a2arecompute() -> Trainer.Config:
    """bf16 grad reduce + recompute the EP all-to-all instead of saving it.

    Single variable against gpt_oss_120b_bf16reduce (job 2295, ~660 TFLOPs over
    steps 600-1200 at 171.86GiB / 96.36%): only the AC save set changes.

    Read the MEMORY first, not the throughput. The hypothesis is ~25GiB freed,
    which would land this near 82-85% of HBM. If peak memory barely moves, the
    estimate of what those saved buffers cost was wrong and
    gpt_oss_120b_bf16reduce_a2arecompute_bs3 should be cancelled rather than
    left in the queue. If memory drops a lot but throughput is flat or slightly
    down, that is still the desired outcome -- the freed memory is the product,
    and bs=3 is where it gets spent.

    Failure mode to expect: a hang, not a slowdown. Re-issuing a collective in
    backward needs all 8 ranks to agree on ordering, which is the likely reason
    comm ops are MUST_SAVE by default. comm.train_timeout_seconds is 100, so a
    deadlock should abort rather than sit, but these jobs are queued with a short
    walltime anyway.
    """
    config = gpt_oss_120b_bf16reduce()
    config.activation_checkpoint = GptOssSelectiveACRecomputeA2A.Config()
    return config


def gpt_oss_120b_bf16reduce_a2arecompute_bs3() -> Trainer.Config:
    """The payoff run: local_batch_size 2 -> 3 on the memory freed above.

    Batch size is the only lever on this model with a proven large payoff --
    bs 1->2 was +63% (13.5% -> 27.9% MFU) -- and memory is the only thing that
    has ever blocked bs=3. Two earlier attempts died of exactly that:
    gpt_oss_120b_membudget_bs3 (job 661) fell to ~32 TFLOPs at 97.55% and
    gpt_oss_120b_ep8_bs3 needs FullAC, which crashes under compile.

    Arithmetic, to be checked against the run rather than trusted: fixed state
    is ~120.5GiB, non-a2a activations are ~12GiB per unit of batch if the a2a
    buffers really are ~25GiB of the ~49GiB at bs=2, so bs=3 projects to
    ~120.5 + 36 = ~156GiB (~88%). That is inside the envelope but not
    comfortably, and 88% is already past where job 2292 sat.

    DEPENDS on gpt_oss_120b_bf16reduce_a2arecompute freeing what it is supposed
    to. Check that job's peak memory before reading anything into this one; if
    the bs=2 version did not drop well below 96%, this OOMs and the result means
    nothing about batch size.
    """
    config = gpt_oss_120b_bf16reduce_a2arecompute()
    config.training.local_batch_size = 3
    return config


def gpt_oss_120b_bf16reduce_lr3e4() -> Trainer.Config:
    """bf16 grad reduce at lr 3e-4 instead of 8e-4, to stop the divergence.

    Not a throughput lever -- a measurement-validity one, and a prerequisite for
    this config being usable for real training at all. Job 2292 shows loss
    turning upward at ~step 1400 and grad_norm reaching 38.75 by step 2800, with
    throughput decaying 629 -> 503 as expert routing collapses. Every number
    past ~step 1400 in waves 1-3 is measured on a diverging run, and the only
    reason job 237 looked clean is that it stopped at step ~2000.

    8e-4 is the stock gpt_oss_120b value and it is simply too high for 116.83B
    params at global batch 16 sequences; grad_norm sitting at 39 against
    max_norm 1.0 means the clip is scaling gradients down ~39x, which is not
    training. 3e-4 is a conservative step down rather than a tuned value.

    What to look for: does the 628/660 TFLOPs plateau HOLD past step 2000
    instead of decaying? If yes, the sustainable figure for this configuration
    is the plateau, the decay was a symptom of divergence, and long runs become
    interpretable. Loss should also keep falling rather than turning up, which
    is the actual point.
    """
    config = gpt_oss_120b_bf16reduce()
    config.optimizer = default_adamw(lr=3e-4)
    return config


def gpt_oss_120b_bf16reduce_norng() -> Trainer.Config:
    """bf16 grad reduce with preserve_rng_state=False in the AC policy.

    Cheap rider, small expected effect. ActivationCheckpointing.Config defaults
    preserve_rng_state=True, which stashes and restores RNG state around every
    checkpointed region so recompute reproduces forward bit-exactly. GPT-OSS has
    no dropout and nothing else stochastic inside a TransformerBlock, so there is
    no RNG for the recompute to diverge on and the stash/restore is pure
    overhead: 36 blocks x 2 (save + restore) per step, each touching CUDA RNG
    state.

    Expect low single digit percent at best. It is queued because it is one line,
    numerics-neutral in the absence of in-block randomness, and independent of
    everything else in this wave, so it costs only queue time. If the model ever
    gains dropout this must be reverted.
    """
    config = gpt_oss_120b_bf16reduce()
    config.activation_checkpoint = SelectiveAC.Config(preserve_rng_state=False)
    return config


def gpt_oss_120b_lbc3e4() -> Trainer.Config:
    """load_balance_coeff 1e-3 -> 3e-4, completing the sweep downward.

    The wave-3 bracket came back monotonic in the wrong direction: at step 1000,
    1e-3 gave 629 TFLOPs, 1e-2 gave 554 and 3e-2 gave 503. Nothing in that
    establishes that 1e-3 is the optimum -- it was only the smallest value
    tested. A sign-step update with no damping oscillates with amplitude
    proportional to the step size, so a smaller step should balance more slowly
    but sit more quietly once there, and the trend says quieter is faster.

    The expected cost is training quality, which moved the other way across the
    same sweep: grad_norm at step ~2200 was 4.16 at 1e-3, 1.84 at 3e-2, and loss
    at matched steps was better at higher coefficients. So this may buy
    throughput and pay for it in stability, which would make it a real trade
    rather than a free win, and worth knowing either way before anyone picks a
    coefficient for a long run.

    Run with gpt_oss_120b_bf16reduce's own control (job 2295) in mind: this one
    is built on the plain best config, NOT on bf16reduce, so its comparison is
    job 2292 at 628 TFLOPs. Keeping it off bf16reduce keeps it single-variable
    against the wave-3 sweep it extends.
    """
    return _gpt_oss_120b_with_load_balance_coeff(3e-4)


def gpt_oss_120b_bf16reduce_lr3e4_biasgemm() -> Trainer.Config:
    """gpt_oss_120b_bf16reduce_lr3e4 with the expert bias gradients as one-hot
    GEMMs (GptOssGroupedExperts.Config.bias_grad_gemm, moe.ExpertBiasAdd).

    The gather-add bias[expert_idx_R] differentiates to an
    index_put(accumulate=True) that inductor fuses into the swiglu-backward
    kernels as atomic adds. At EP=8, bs=2 each rank receives ~65,536 routed
    rows onto 17 bias rows (16 local experts + the padding row): ~3,900
    atomics per address, per layer, for both mlp1 and mlp2. The one-hot GEMM
    computes the same gradient with fp32 accumulation, deterministically.

    On GPT-OSS-20B: +1.0% on its reference (job 6238), +0.7% on the stack
    (job 6373). Forward and all non-bias gradients are bitwise identical.

    MEASURED (200 steps, TF/GPU mean / median over steps 101-200, same day):
        control _lr3e4        (job 7597)  499.23 / 511.53  173.76GiB
        this config           (job 7602)  516.73 / 522.26  172.90GiB   +3.5%
    Replicate controls landed within +-0.5% of each other over this window.
    """
    return _enable_expert_bias_grad_gemm(gpt_oss_120b_bf16reduce_lr3e4())


def gpt_oss_120b_bf16reduce_lr3e4_swigluhalves() -> Trainer.Config:
    """gpt_oss_120b_bf16reduce_lr3e4 with mlp1's output stored as contiguous
    [gate | linear] halves instead of interleaved gate / linear columns.

    The interleaved split x[..., ::2] / x[..., 1::2] makes every load and
    store in the swiglu forward and backward stride 2; on GPT-OSS-20B those
    kernels were the largest non-GEMM compute (133 ms of a 1.31 s step, job
    6375), the backward ~3.5x its byte cost, and the halves layout measured
    +2.4% (1,010.27 vs 986.40 TF/GPU, jobs 7258 vs 7257). The 120b has the
    same expert shapes (dim = hidden = 2880) and 36 layers instead of 24.

    Same model up to a fixed permutation of mlp1's output rows; from-scratch
    training only (GptOssStateDictAdapter refuses HF conversion with it).

    MEASURED (200 steps, TF/GPU mean / median over steps 101-200, same day):
        control _lr3e4        (job 7597)  499.23 / 511.53  173.76GiB
        this config           (job 7599)  525.33 / 531.78  169.30GiB   +5.2%
    -4.5GiB peak and 6 slow steps (<350 TF/GPU) against the control's 23.
    """
    return _set_swiglu_contiguous_halves(gpt_oss_120b_bf16reduce_lr3e4())


def _set_loader_workers(config: Trainer.Config) -> Trainer.Config:
    config.dataloader.num_workers = 4
    config.dataloader.persistent_workers = True
    config.dataloader.prefetch_factor = 4
    config.dataloader.pin_memory = True
    return config


def gpt_oss_120b_bf16reduce_lr3e4_loaderworkers() -> Trainer.Config:
    """gpt_oss_120b_bf16reduce_lr3e4 with the c4 dataloader moved off the
    training process: 4 persistent, pinned, prefetching workers per rank.

    Every 120b config so far leaves num_workers at 0, so the streaming c4
    reads and tokenization run inline every step, and with CUDA graphs off
    (varlen) that CPU time sits on the kernel-launch path; a shard fetch
    stalls a step outright every ~2.13M tokens/rank (~130 steps at bs=2).
    The 120b's per-rank CPU stalls already show up as rank skew in the EP
    all-to-all (traces_120b_best: seven ranks parked ~185 ms in one SendRecv
    waiting for a rank whose CPU was busy). Same change as
    take2-qwen3/dataloader-workers, +4.4% tok/s on Qwen3-30B-A3B (job 6180
    vs 6177). Worker interleaving changes the sample order.

    MEASURED (200 steps, TF/GPU mean / median over steps 101-200, same day):
        control _lr3e4        (job 7597)  499.23 / 511.53  173.76GiB
        this config           (job 7600)  530.20 / 534.63  170.54GiB   +6.2%
    Steps 21-100 are slower while the workers spin up (21-200 mean -0.2%).
    """
    return _set_loader_workers(gpt_oss_120b_bf16reduce_lr3e4())


def gpt_oss_120b_bf16reduce_lr3e4_norng() -> Trainer.Config:
    """gpt_oss_120b_bf16reduce_lr3e4 with preserve_rng_state=False in the
    SelectiveAC policy.

    GPT-OSS has no dropout or other in-block randomness, so stashing and
    restoring CUDA RNG state around each of the 36 checkpointed blocks buys
    nothing. On the diverging lr=8e-4 base it read 651.42 mean / 664.30
    median TF/GPU over steps 200-1500 at 169.29GiB (job 2310) against
    654.69 / 666.06 at 172.54GiB for its control (job 2311) -- level on
    throughput, -3.25GiB peak -- but both runs were past the divergence that
    gpt_oss_120b_bf16reduce_lr3e4 fixes, so this re-measures it on the
    stable trajectory. Numerics-neutral while the block stays deterministic.

    MEASURED (200 steps, TF/GPU mean / median over steps 101-200, same day):
        control _lr3e4        (job 7597)  499.23 / 511.53  173.76GiB
        this config           (job 7603)  508.47 / 518.15  173.27GiB   +1.9%
    """
    config = gpt_oss_120b_bf16reduce_lr3e4()
    config.activation_checkpoint = SelectiveAC.Config(preserve_rng_state=False)
    return config


def gpt_oss_120b_bf16reduce_lr3e4_mxfp8_attn_lmhead() -> Trainer.Config:
    """gpt_oss_120b_bf16reduce_lr3e4 with MXFP8 on the attention linears and
    lm_head, as in every GPT-OSS-20B and Qwen3-30B-A3B take-2 config.

    A retry, not a first try. gpt_oss_120b_mxfp8_linears_lmhead (job 1301)
    read 468.26 TF/GPU on the old lr=8e-4 base at 170.77GiB / 96%, against
    ~566 for the bf16 base at the time (job 208) -- and its attention-only
    sibling oscillated with 7 expandable_segments mapping failures (job
    1300). That was diagnosed as allocator pressure, not quantize cost: the
    MXFP8 transients (+1.50GiB, jobs 1304 vs 1305) landed a 96% config in
    the >=97% region where every 120b config has collapsed. The lr=3e-4 base
    peaks at 169.13GiB (94.83%, job 2307), ~3GiB lower, which is the room
    this needs. Attention + lm_head are ~32% of the per-token GEMM work.

    Read TF/GPU, not MFU (N/A for low-precision runs). lm_head output feeds
    the loss directly: reject on a visibly higher loss curve even if faster.

    MEASURED -- NEGATIVE, not in any stack (200 steps, TF/GPU mean / median over
    steps 101-200, same day):
        control _lr3e4        (job 7597)  499.23 / 511.53  173.76GiB
        this config           (job 7606)  489.43 / 496.49  170.75GiB   -2.0%
    """
    config = gpt_oss_120b_bf16reduce_lr3e4()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "120b",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "lm_head"],
            ),
        ],
    )
    return config


def gpt_oss_120b_bf16reduce_lr3e4_take2stack() -> Trainer.Config:
    """Every bf16 take-2 lever above at once, so one night yields both the
    per-lever A/B and the stacked number: bias grad GEMM, contiguous-halves
    swiglu, dataloader workers and SelectiveAC without RNG preservation.

    Run with TORCHTITAN_FP32_MATMUL_PRECISION=tf32 and pointwise autotune
    (TT_INDUCTOR="TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
    TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1"), the GPT-OSS-20B best's
    environment. MXFP8 is left out: it is the one lever here that adds memory
    on a config at 94.8% of HBM. Drop any lever that measures negative on its
    own and re-run before quoting this.

    MEASURED with tf32 + pointwise autotune:
        200 steps (job 7605)    571.35 / 574.00 over steps 101-200, 164.27GiB,
                                +14.4% over _lr3e4 (job 7597, 499.23 / 511.53)
        normal warmup (job 7627) 772.73 / 776.29 over steps 1000-1500,
                                737.17 / 755.77 over 200-1500, 165.88GiB (93.01%),
                                vs job 2307 683.39 / 693.04 and 646.41 / 676.34:
                                +13.1% / +12.0% at the plateau.
    tf32, pointwise autotune and cuBLAS 13.8 were each neutral alone (jobs 7601,
    7604, 7607: within +-0.5% of the control).
    """
    config = gpt_oss_120b_bf16reduce_lr3e4_norng()
    _enable_expert_bias_grad_gemm(config)
    _set_swiglu_contiguous_halves(config)
    _set_loader_workers(config)
    return config


# ---------------------------------------------------------------------------
# Take 2, wave 2 (2026-10-03): three more levers on the take-2 stack.
#
# gpt_oss_120b_bf16reduce_lr3e4_take2stack + tf32 + pointwise autotune (job
# 7627, normal warmup): 772.73 mean / 776.29 median TF/GPU over steps
# 1000-1500 (+13.1% / +12.0% over job 2307), peak 165.88GiB (93.01%); over
# 200 steps (job 7605) 571.35 / 574.00 over steps 101-200, 164.27GiB.
# ---------------------------------------------------------------------------


def gpt_oss_120b_bf16reduce_lr3e4_take2stack_sacgmm_tail6() -> Trainer.Config:
    """The take-2 stack, saving the expert GEMM outputs in the last 6 of 36
    blocks instead of recomputing them (SelectiveAC.Config.save_grouped_mm
    with save_grouped_mm_from_layer=30).

    SelectiveAC recomputes every aten._grouped_mm in backward because torch's
    compute_intensive_ops predates grouped mm. Saving them all was
    GPT-OSS-20B's largest take-2 lever (+16.4%, job 6218), but here it costs
    ~1.13GiB per block at bs=2 (mlp1 out ~65,536 x 5,760 bf16 + mlp2 out
    ~65,536 x 2,880), ~41GiB for all 36 -- far past the 178GiB card. The
    stack peaks ~12GiB under the card (93%), and every 120b config at >=97%
    has stalled, so this spends ~6.8GiB of it on 6 blocks (~172GiB, 96.5%).
    Sweep the count on the command line:
        TT_EXTRA="--activation_checkpoint.save_grouped_mm_from_layer 33"

    If peak memory does not rise by ~1.1GiB per saved block, compile reused
    one block's graph and policy across layers and the per-layer cutoff did
    nothing -- read the memory before the throughput.

    MEASURED (200 steps, tf32 + pointwise autotune, TF/GPU mean / median over
    steps 101-200): 569.59 / 570.54 at 168.95GiB (job 7629) vs the stack's
    560.59 / 567.34 at 167.00GiB (job 7628), +1.6% -- but only +1.95GiB, not the
    ~6.8GiB predicted. save_grouped_mm_from_layer does not take effect under
    compile (see its docstring), so this is not a measurement of saving 6 blocks.
    Saving all 36 is _take2stack_skipexperts_sacgmmall.
    """
    config = gpt_oss_120b_bf16reduce_lr3e4_take2stack()
    config.activation_checkpoint = SelectiveAC.Config(
        preserve_rng_state=False,
        save_grouped_mm=True,
        save_grouped_mm_from_layer=30,
    )
    return config
