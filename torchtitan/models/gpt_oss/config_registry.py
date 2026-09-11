# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import (
    FullAC,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
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
