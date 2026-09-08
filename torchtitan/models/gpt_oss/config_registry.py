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
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
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


# ---------------------------------------------------------------------------
# GB300 tuning sweep (64x GB300 = 16 nodes x 4).
#
# All of these descend from gpt_oss_20b, so anything not named here stays at the
# baseline: seq_len 8192, FSDP over all ranks, expert_parallel_degree 1, AdamW
# lr 8e-4, C4.
#
# Measured numbers for each are in gb300/TUNING_RESULTS.md. Baseline for
# comparison is 287.8 TFLOP/s/GPU.
#
# Note every one of these still needs --training.disable_cuda_graphs at launch:
# varlen attention's cu_seqlens changes length per step, so CUDA graph capture
# fails on step 2. That is a property of the stock model config, not of these.
# ---------------------------------------------------------------------------


def _20b_no_ac(local_batch_size: int) -> Trainer.Config:
    """Baseline with activation checkpointing removed and batch size raised.

    The baseline holds only 18.46 GiB of 284 GB (6.68%), so it is nowhere near
    memory bound - it recomputes activations it had ample room to keep. Dropping
    FullAC trades that headroom for the recompute, and the freed step time only
    pays off if the batch grows to fill the GPU.
    """
    config = gpt_oss_20b()
    config.activation_checkpoint = None
    config.training.local_batch_size = local_batch_size
    return config


def gpt_oss_20b_gb300_noac() -> Trainer.Config:
    """No AC at the largest batch that fits: 4.

    Measured on 64x GB300, 6 steps each:
        bs=2  46.0% memory  346.0 TFLOP/s
        bs=4  86.5% memory  482.4 TFLOP/s   <- this
        bs=6  OOM
        bs=8  OOM
    FSDP shards parameters, gradients and optimizer state but NOT activations,
    and activations are what scale with local_batch_size - so with AC off the
    batch is what pushes the GPU into OOM, not the model.

    86.5% is close to the edge. If a longer run OOMs on fragmentation, drop to 3
    rather than re-enabling AC.
    """
    return _20b_no_ac(local_batch_size=4)


def gpt_oss_20b_gb300_compile() -> Trainer.Config:
    """torch.compile on top of the stock baseline. KNOWN TO FAIL - kept as the
    record of what "just enable compile" does on this stack.

        torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
        AssertionError: Node add_21 was invalid, but is output
        ... in min_cut_rematerialization_partition -> _extract_fwd_bwd_modules

    It is FullAC + compiling the *model* that breaks, not compile itself:
    activation checkpointing and inductor's min-cut partitioner are both deciding
    what to recompute. Both escapes work - drop AC (gpt_oss_20b_gb300_noac_compile)
    or leave the model eager (gpt_oss_20b_gb300_compile_loss_only).
    """
    config = gpt_oss_20b()
    config.compile.enable = True
    return config


def gpt_oss_20b_gb300_noreshard() -> Trainer.Config:
    """FSDP keeps parameters gathered after forward instead of resharding.

    Spends memory to avoid the re-all-gather in backward. Cheap here precisely
    because the baseline is using so little memory.
    """
    config = gpt_oss_20b()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def _20b_mxfp8(config: Trainer.Config, flavor: str = "20b") -> Trainer.Config:
    """Apply MXFP8 to the MoE grouped GEMMs and the dense Linears in attention.

    fqns is an include-list (substring match). gpt-oss has no shared experts and
    no per-layer feed_forward - every layer's FFN is the MoE - so 'attention' is
    the whole dense-Linear surface. The router gate and lm_head stay in bf16.

    pad_multiple=128 is required by the CuTeDSL quantization kernel on sm_100+.
    """
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        flavor,
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


def gpt_oss_20b_gb300_mxfp8() -> Trainer.Config:
    """MXFP8 quantization, everything else baseline."""
    return _20b_mxfp8(gpt_oss_20b())


# --- compounded ------------------------------------------------------------


def gpt_oss_20b_gb300_noac_compile() -> Trainer.Config:
    """No AC + compile. Compile frees enough activation memory to raise the batch
    past the no-compile ceiling.

    Measured at 6 steps, no-AC + compile:
        bs=4   54.2% memory
        bs=6   77.0% memory   <- this, the largest with real headroom
        bs=8   98.8% memory   (fits, but see _maxbs below)
        bs=10  OOM
    Without compile the same config OOMs at bs=6, so compile is worth ~2 steps of
    batch here on top of whatever it does for speed.
    """
    config = gpt_oss_20b_gb300_noac()
    config.compile.enable = True
    config.training.local_batch_size = 6
    return config


def gpt_oss_20b_gb300_noac_compile_maxbs() -> Trainer.Config:
    """The largest batch that fit at all: 8, at 98.8% of GPU memory.

    Separate from the config above because 98.8% is not a setting to hand someone
    as a default - it survived a 6-step probe, and fragmentation over a long run
    is a real risk. Run to see what the last two steps of batch are worth; use
    _noac_compile if you want something that reliably finishes.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.training.local_batch_size = 8
    return config


def gpt_oss_20b_gb300_noac_compile_noreshard() -> Trainer.Config:
    """No AC + compile + never reshard.

    Batch drops back to 4: never-reshard keeps parameters gathered after forward,
    which on its own took the baseline from 18.5 GiB to 56.5 GiB. Stacking that on
    top of bs=6 (77%) would not fit.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.local_batch_size = 4
    return config


def gpt_oss_20b_gb300_noac_compile_noreshard_mxfp8() -> Trainer.Config:
    """All four tunings together."""
    return _20b_mxfp8(gpt_oss_20b_gb300_noac_compile_noreshard())


def gpt_oss_20b_gb300_compile_loss_only() -> Trainer.Config:
    """torch.compile restricted to the loss, leaving the model graph eager.

    Compiling the model fails in AOTAutograd's min-cut rematerialization
    partitioner ("Node add_N was invalid, but is output"). This narrows compile to
    the component that does not go through that partitioner, so there is still a
    compile data point if the model graph cannot be compiled on this stack.
    """
    config = gpt_oss_20b()
    config.compile.enable = True
    config.compile.components = ["loss"]
    return config


# ---------------------------------------------------------------------------
# Round 2. Chosen from what round 1 measured (250 steps, 64x GB300):
#
#   baseline                    281.4 TFLOP/s   6.7% mem
#   noreshard                   322.2           20.4%
#   compile_loss_only           282.1           6.7%     (no effect)
#   noac            bs=4        569.2           86.5%
#   noac_compile    bs=6        942.3           77.0%    <- best
#   noac_compile_maxbs bs=8     ~205            98.7%    <- fell off a cliff
#
# Two things drive everything here. Memory utilisation is the lever - the
# baseline recomputes activations it had 93% of the GPU spare to keep. And there
# is a sharp cliff near the top: bs=8 at 98.7% is 4.5x SLOWER than bs=6 at 77%,
# because the allocator thrashes rather than because the work changed.
#
# So the interesting question is no longer "how much memory can we use" but
# "how close to the cliff can we get, and can we get more useful work into the
# memory we can safely use". These four attack that from different sides.
# ---------------------------------------------------------------------------


def gpt_oss_20b_gb300_noac_compile_bs7() -> Trainer.Config:
    """Bisect the cliff: bs=6 gives 942 TFLOP/s at 77%, bs=8 gives ~205 at 98.7%.

    bs=7 should land near 88%. This says whether 942 is the peak or whether
    there is another step of batch to take before the allocator gives out - and
    where the edge actually is, which nothing else in the sweep pins down.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.training.local_batch_size = 7
    return config


def gpt_oss_20b_gb300_sac_compile() -> Trainer.Config:
    """Selective (per-op) AC instead of none, + compile, at a larger batch.

    No-AC and full-AC are the two ends of a spectrum and round 1 only measured
    the ends. SelectiveAC saves the ops that are expensive to recompute and
    recomputes every second matmul, so it should sit well below no-AC's memory at
    the same batch - buying batch back without paying full-AC's recompute bill.

    bs=8 is an estimate, not a measurement: no-AC fit bs=8 at 98.8%, and SAC
    stores strictly less, so this should land with headroom. If it OOMs, drop to
    7 - do not conclude SAC is unviable.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.local_batch_size = 8
    return config


def gpt_oss_20b_gb300_noac_compile_noreshard_bs6() -> Trainer.Config:
    """The two strongest levers together, at the batch that worked best.

    Round 1's compounded config held never-reshard at bs=4 and reached 917.0
    TFLOP/s in 66.3% of memory - within 3% of the best result (942.3 at bs=6)
    while using 11 points less memory. That spare memory is the whole point:
    never-reshard was carrying bs=4 to nearly the same throughput as bs=6
    without it, and nothing has yet tried it at bs=6.

    Should land near 85-90% memory, so it is deliberately below the bs=7 probe -
    if bs7 finds the cliff lower than expected, this is the config that has to
    move.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.local_batch_size = 6
    return config


def gpt_oss_20b_gb300_noac_compile_hsdp() -> Trainer.Config:
    """HSDP: shard within a node (4), replicate across nodes (16).

    Every config so far shards all 64 ranks, so each all-gather crosses the whole
    rack. This confines the parameter all-gather to the 4 GPUs inside a node and
    reduces the cross-node traffic to gradient all-reduce.

    Worth testing precisely because 20B is small for 64-way sharding: the shards
    are tiny and the collective is latency-bound, which is the regime where
    narrowing the shard group usually wins. Costs memory (each node holds a full
    replica's shard), so the batch backs off to 4.
    """
    config = gpt_oss_20b_gb300_noac_compile()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.data_parallel_replicate_degree = 16
    config.training.local_batch_size = 4
    return config



# ---------------------------------------------------------------------------
# Shared helpers for the large-batch runs
# ---------------------------------------------------------------------------


def _stage_local_data(config: Trainer.Config) -> Trainer.Config:
    """Point the dataloader at locally staged C4, and loosen the watchdog.

    Hub-streamed C4 cannot feed 64 GB300s above local_batch_size ~8. Past that
    the Hub throttles (142 retries / 87 connection errors at bs=24), ranks
    starve, one misses the 100 s collective window, and the NCCL watchdog aborts
    - which additionally strands nodes in DRAIN with wedged GPUs needing a driver
    reload. See TUNING_RESULTS.md round 3.

    train_timeout_seconds is raised from 100 to 600 as a safety net, not a fix:
    with local shards there should be no stall to absorb, but a watchdog abort at
    this scale damages the cluster, so it is worth not tripping on a hiccup.
    """
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_local")
    config.comm.train_timeout_seconds = 600
    return config


# --- SelectiveAC at batches that actually fill the GPU ----------------------
#
# Round 2 ran SelectiveAC at bs=8 and it reached 859.6 TFLOP/s in 27.2% of
# memory, while every other config near that throughput needed 80-90%. Per-op
# SAC costs ~7.7 GiB per batch unit against no-AC's ~33, so bs=8 was nowhere
# near its ceiling. Measured at 8 steps:
#     bs=16  48.4%   bs=24  70.8%   bs=28  82.8%   bs=32  93.9%
# Nothing OOMed up to 32. These three bracket the 88.9% where no-AC peaked.


def gpt_oss_20b_gb300_sac_compile_bs24() -> Trainer.Config:
    """SelectiveAC at 70.8% memory - the conservative point."""
    config = gpt_oss_20b_gb300_sac_compile()
    config.training.local_batch_size = 24
    return _stage_local_data(config)


def gpt_oss_20b_gb300_sac_compile_bs28() -> Trainer.Config:
    """SelectiveAC at 82.8% memory - just under where no-AC peaked."""
    config = gpt_oss_20b_gb300_sac_compile()
    config.training.local_batch_size = 28
    return _stage_local_data(config)


def gpt_oss_20b_gb300_sac_compile_bs32() -> Trainer.Config:
    """SelectiveAC at 93.9% memory - past the no-AC peak, short of the 98.7%
    that cratered. Whether this is the best config or another cliff casualty is
    exactly the open question."""
    config = gpt_oss_20b_gb300_sac_compile()
    config.training.local_batch_size = 32
    return _stage_local_data(config)


def gpt_oss_20b_gb300_noac_compile_bs7_local() -> Trainer.Config:
    """Control for the dataset change: bs=7 measured 963.5 TFLOP/s on Hub-streamed
    C4 with essentially no throttling (0 retries, 1 connection error).

    Re-running it on local shards should reproduce that. If it does, local vs
    streamed is neutral at this batch and the large-batch SAC numbers are directly
    comparable to the rest of the table. If it does not, every cross-dataset
    comparison in this sweep needs re-examining - which is why this runs first.
    """
    return _stage_local_data(gpt_oss_20b_gb300_noac_compile_bs7())


# ---------------------------------------------------------------------------
# GB300 tuning sweep, 120B. Reasoned from the 20B results rather than repeating
# them, because 120B sits in a different regime.
#
# What the 20B sweep established:
#   - memory utilisation is the lever; the peak was ~89% and above ~94% it falls
#     off a cliff (bs=8/98.7% ran 43% SLOWER than the baseline)
#   - compile alone is worth +0.2%; its value is that it frees activation memory
#     and so buys batch
#   - never-reshard helps while memory is slack (+14.5% at bs=1) and HURTS near
#     the ceiling (880.7 at 90% vs 917.0 at 66%)
#   - per-op SelectiveAC costs ~7.7 GiB/batch-unit vs no-AC's ~33 - 4.4x cheaper
#   - HSDP does not help: all 64 GPUs are one NVLink fabric, so 64-way sharding
#     pays no cross-node penalty. Not repeated here.
#
# Why 120B differs:
#   - the baseline already sits at 68.33 GiB (24.7%) against 20B's 18.46 (6.7%),
#     so there is far less headroom to spend
#   - 36 layers against 24 means ~1.5x the activation cost per batch unit, and
#     activations are the part that does NOT shard
#   - per-GPU throughput is 56% of 20B's at the same batch, i.e. 120B is more
#     COMMUNICATION-bound: more parameters to all-gather per token of useful work
#
# Two predictions follow, and the configs below are built to test them:
#   1. no-AC will barely raise the batch here (maybe to 2-3) because 120B's
#      activations are already large - so the 20B winner may not transfer.
#   2. SelectiveAC should matter MORE than on 20B, and never-reshard should help
#      MORE, because the binding constraint is communication rather than compute.
# ---------------------------------------------------------------------------


def gpt_oss_120b_gb300_noreshard() -> Trainer.Config:
    """never-reshard at the stock batch. Cheapest possible test of prediction 2.

    On 20B this was +14.5% at bs=1 with memory slack to spare. 120B is more
    communication-bound, and this is the one knob that directly removes a
    collective (the backward re-all-gather), so it should do better here. If it
    does not, the "more communication-bound" reading of the baseline is wrong.
    """
    config = gpt_oss_120b()
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


def gpt_oss_120b_gb300_noac_compile() -> Trainer.Config:
    """No AC + compile, batch from the probe. The 20B winner, ported.

    local_batch_size is set from probe measurements; see TUNING_RESULTS.md. The
    expectation is that this does far less here than the 3.35x it gave on 20B,
    because 120B has ~1.5x the activation cost per batch unit and started with a
    quarter of the memory free rather than 93%.
    """
    config = gpt_oss_120b()
    config.activation_checkpoint = None
    config.compile.enable = True
    # Measured (8-step probe): bs=2 53.9%, bs=3 76.6%. bs=3 it is. Compare with
    # 20B, where no-AC + compile reached bs=6-7 at similar occupancy - prediction
    # 1 holds, 120B's activations leave far less room to buy batch with.
    config.training.local_batch_size = 3
    return _stage_local_data(config)


def gpt_oss_120b_gb300_sac_compile() -> Trainer.Config:
    """SelectiveAC + compile at a batch that fills the GPU.

    The config prediction 2 is really about. Per-op SAC was 4.4x cheaper than
    no-AC on 20B; scaled by 120B's 1.5x layer count that is ~11.6 GiB per batch
    unit over ~60 GiB fixed, so ~85-90% memory should land near bs=16. That is
    roughly 8x the batch no-AC can afford here, and batch is what pays.
    """
    config = gpt_oss_120b()
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile.enable = True
    # Measured (8-step probe): bs=12 63.1%, bs=16 79.4%, bs=20 95.0%.
    # bs=16 is the pick: 20B peaked at 89% and fell off a cliff by 98.7%, so 95%
    # is not somewhere to sit for a 250-step run. See _bs20 for that test.
    # Prediction 2 holds on memory - SAC affords ~5x the batch no-AC can here.
    config.training.local_batch_size = 16
    return _stage_local_data(config)


def gpt_oss_120b_gb300_sac_compile_noreshard() -> Trainer.Config:
    """Both predictions together: SelectiveAC for batch, never-reshard for the
    collective.

    Batch backs off from the sac_compile setting because never-reshard holds
    parameters gathered after forward, and on 20B stacking it near the ceiling
    was actively worse than not.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.parallelism.fsdp_reshard_after_forward = "never"
    # 12 measured at 63.1% without never-reshard; that leaves room for the
    # gathered parameters this adds.
    config.training.local_batch_size = 12
    return config


def gpt_oss_120b_gb300_sac_compile_bs20() -> Trainer.Config:
    """SelectiveAC at 95.0% memory - deliberately past where 20B was safe.

    Separate from sac_compile so the main config stays somewhere that finishes.
    20B peaked at 88.9% and collapsed to 160.6 TFLOP/s by 98.7%; whether 95% is
    over that edge on 120B is untested, and this is the test.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.training.local_batch_size = 20
    return config


def gpt_oss_120b_gb300_sac_compile_maxbs() -> Trainer.Config:
    """SelectiveAC at bs=20, the largest that fit the probe: 95.0% of memory.

    Kept separate from sac_compile rather than made the default. On 20B the cliff
    sat between 88.9% (fine) and 98.7% (43% slower than baseline), and 95% is
    inside that unmeasured gap. Worth one run to find out; not worth being the
    setting someone inherits.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.training.local_batch_size = 20
    return config


# ---------------------------------------------------------------------------
# EXPERIMENT: save the MoE expert GEMM under SelectiveAC.
#
# SelectiveAC's save set comes from torch's compute_intensive_ops (mm, bmm,
# addmm, convolution, sdpa*, _scaled_mm) plus torchtitan's additions. Verified
# against this venv:
#
#     torch_attn._varlen_attn.default        saved=True
#     aten.linear.default                    saved=True
#     aten._grouped_mm.default               saved=False   <-- the expert GEMM
#
# The policy defaults unlisted ops to PREFER_RECOMPUTE, so today per-op SAC
# recomputes every expert GEMM in backward while faithfully saving the far
# cheaper attention projections. Since aten.index and the pointwise ops are
# recomputable too, SelectiveAC degenerates to full recompute inside the MoE --
# which is where GPT-OSS spends 54% (20B) / 56% (120B) of its real FLOPs.
#
# That predicts both SAC numbers already in the table: the low memory (27.2% at
# 20B bs=8) is activations being thrown away, and the throughput gap is the
# recompute bill (+18% FLOPs in theory, 11% observed).
# ---------------------------------------------------------------------------


def gpt_oss_20b_gb300_sac_gmm_bs8() -> Trainer.Config:
    """Controlled A/B against `sac_compile` (bs=8, measured 859.6 at 27.2%).

    Same batch, same everything, one variable: aten._grouped_mm joins the save
    set. Isolates the mechanism rather than chasing a headline number.

    Memory estimate: saving mlp1 (R, 2F) + mlp2 (R, D) outputs costs ~566 MiB per
    layer per batch unit, x24 layers = ~13.6 GiB/bs-unit on top of SAC's measured
    ~7.7. At bs=8 that is ~65% of memory, well clear of the ~89% cliff.

    Read it per-FLOP, not just as TFLOP/s: if this recovers most of the 11% gap
    to no-AC while staying far below no-AC's 88.9% memory, the batch sweep that
    follows is where the actual win is.
    """
    config = gpt_oss_20b_gb300_sac_compile()  # bs=8
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    return _stage_local_data(config)


def gpt_oss_120b_gb300_sac_gmm_bs5() -> Trainer.Config:
    """The same change where it matters most: SelectiveAC is the *winning* 120B
    config (job 56, 250 steps: 810 TFLOP/s at 84.25% memory), so on the larger
    model the best recipe is the one paying the recompute bill.

    Not a controlled A/B -- saving the expert GEMMs at bs=16 cannot fit, so this
    compares configs at each one's own batch, which is what the rest of this
    sweep does anyway.

    Sizing from job 56's MEASURED steady-state 232.96 GiB at bs=16, not from the
    8-step probe: with the probe's ~11.3 GiB/bs-unit slope that implies ~52 GiB
    fixed. Saving mlp1 (R, 2F) + mlp2 (R, D) adds 36 layers x ~566 MiB =
    ~20.4 GiB/bs-unit, so the slope nearly triples to ~31.7 GiB/bs-unit:

        bs=5  ~76%      bs=6  ~88%      bs=7  ~99%

    bs=5 is the pick. bs=6 lands where 20B peaked, but the estimate carries real
    uncertainty and round 1 showed the failure mode here is soft: bs=8 at 98.7%
    ran 43% SLOWER than the baseline rather than OOMing, wasting the whole run.
    A completed run at 76% is worth more than a coin flip at 88%; bs=6 is the
    follow-up once this reports its real memory.

    NOTE: TUNING_RESULTS_120B.md records job 56 as bs=18. The run log says
    "local batch size 16" and the config says 16 -- the table is a typo.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    config.training.local_batch_size = 5
    return _stage_local_data(config)


def gpt_oss_20b_gb300_sac_gmm_bs11() -> Trainer.Config:
    """Conservative follow-up to bs=8: ~82.5% memory, clear of the cliff.

    Sizing is now MEASURED, not estimated. bs=8 came in at 169.52 GiB (61.31%)
    against the SAC baseline's 75.12 GiB (27.17%), so saving mlp1+mlp2 outputs
    costs (169.52 - 75.12) / 8 = 11.80 GiB per batch unit. On top of SAC's own
    ~7.69 GiB/bs-unit that is ~19.49 GiB/bs-unit over ~13.6 GiB fixed -- and that
    fixed term reproduces from both configs independently, so the model holds:

        bs=8   61.3% (measured)    bs=11  ~82.5%    bs=12  ~89.5%    bs=13  ~96.6%

    Use this one if bs=12 falls off the cliff.
    """
    config = gpt_oss_20b_gb300_sac_gmm_bs8()
    config.training.local_batch_size = 11
    return config


def gpt_oss_20b_gb300_sac_gmm_bs12() -> Trainer.Config:
    """The real test: same memory footprint as the current best config, 71% more
    batch.

    bs=8 already reached 979.5 TFLOP/s at 61.31% memory -- within 0.8% of
    noac_compile_bs7_local's 987.8, on 27.6 points less memory. That leaves the
    interesting question: at EQUAL memory, which recipe wins? bs=12 lands at
    ~89.5% by the measured slope above, against no-AC's 88.9%, so this is the
    apples-to-apples comparison.

    89.5% is marginally past the 88.9% that peaked in round 1, and the failure
    mode there was soft rather than an OOM (bs=8 at 98.7% ran 43% SLOWER than the
    baseline). If this degrades rather than OOMs, that is the cliff and bs=11 is
    the answer.
    """
    config = gpt_oss_20b_gb300_sac_gmm_bs8()
    config.training.local_batch_size = 12
    return config
