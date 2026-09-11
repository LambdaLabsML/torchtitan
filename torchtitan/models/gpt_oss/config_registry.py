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
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.logging import logger
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
# The push to 1000 TFLOP/s/GPU on 64x GB300 (branch lambda_64xgb300_gptoss120b_1k)
#
# Reference is job 56, `gpt_oss_120b_gb300_sac_compile` at bs=18: 786.9
# TFLOP/s/GPU steady-state (810.2 at step 250), 232.96 GiB (84.25%). Reaching
# 1000 needs +27% on that, which is more than any remaining memory/batch knob
# can give -- the 120B sweep already spent those, and bs=20 bought -0.1%.
#
# So these configs go after the two costs the 120B sweep never touched, both of
# which are large *because* 120B runs at EP=1 with fp32 gradient reduction:
#
#   1. MXFP8 on the expert grouped GEMMs and the attention Linears. The MoE FFN
#      is ~57% of this model's per-token FLOPs (36 layers x 4 active experts x
#      3 x 2880^2) and the attention projections another ~15%, so this is the
#      only lever aimed at the majority of the arithmetic. It was recorded as
#      "blocked: SM100 kernels absent" in TUNING_RESULTS.md -- that was a
#      packaging gap, not a hardware one, and it is now fixed; see
#      gb300/MXFP8_120B.md.
#   2. The fp32 gradient reduce-scatter. At EP=1 every one of the ~114 B expert
#      parameters is reduce-scattered in fp32 on every step. On Qwen3.5-122B,
#      a comparable model, that collective measured ~25% of the step and 65% of
#      all NCCL time. bf16 halves those bytes.
#
# and at the interaction the sweep left open: expert parallelism. 120B has only
# ever run EP=1, where FSDP must all-gather every layer's full 128-expert weight
# stack (3.19 B params = 6.4 GB bf16 per layer, ~229 GB per step) on every rank.
# EP shards the experts instead, trading that all-gather for a token all-to-all.
# Job 105 showed EP=8 runs and costs 48.36 GiB against the baseline's 68.33 at
# bs=1, so it does cut the gathered-weight buffers substantially.
#
# The sign of EP at bs=18 is genuinely unknown rather than predicted: the
# weight all-gather EP removes is fixed, but the all-to-all it adds scales with
# tokens, so EP's advantage shrinks as batch grows and these run at 18x the
# batch job 105 used. That is what makes it worth a measurement.
#
# IMPORTANT -- all TFLOP/s here are on torchtitan's STOCK FLOP accounting, the
# same as job 56, so they are directly comparable to it. That accounting charges
# sliding-window layers the full O(L^2) attention cost; branch
# perf/flops-sliding-window corrects it and lowers every 120B number by ~18%
# (786.9 -> 665.0). Deliberately NOT merged here: changing the denominator
# mid-comparison would make these runs incomparable to the reference.
# ---------------------------------------------------------------------------


def _120b_ref() -> Trainer.Config:
    """Job 56's actual configuration: `sac_compile` at **bs=16**.

    `TUNING_RESULTS_120B.md` records job 56 as bs=18. That is an error in the
    table, and three independent readings say so:

      - job 56's step-1 memory is 219.44 GiB (79.36%), and the registry's own
        8-step probe comment records "bs=16 79.4%". bs=18 does not land there.
      - job 56 steady-states at 232.96 GiB (84.25%); job 60 at bs=20 reaches
        273.00 GiB (98.73%). That is 3.62 percentage points per batch unit, so
        bs=18 predicts ~91.5%.
      - measured: bs=18 runs at 254.56 GiB (92.07%). Exactly the prediction,
        and 21.6 GiB -- two batch units -- above job 56.

    So job 56 ran the registry default with no override, and bs=16 is the
    configuration the 786.9 TFLOP/s reference belongs to. Pinned explicitly
    rather than inherited, so this cannot drift again.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.training.local_batch_size = 16
    return config


def gpt_oss_120b_1k_ref_bs18() -> Trainer.Config:
    """bs=18, measured in job 254 and kept because the number is interesting.

    It sits between two configurations that both run well -- bs=16 at 84.25%
    gives 786.9 and bs=20 at 98.73% gives 786.1 -- and is worse than either:
    16,050 tokens/s/GPU against job 56's 17,626, i.e. 0.510 s per batch unit
    against 0.465. Non-monotonic in batch, which is not what a memory-pressure
    story would predict.

    That reading is NOT clean, and the run is recorded as contaminated: the
    login node is one of the 16 allocated nodes, and NFS- and /tmp-heavy
    commands were run on it during steps 40-100 (a `du` over a 17 GB Inductor
    cache, `find` walks over NFS). Job 254 logged 5 dataloader retries where
    job 56 logged 0, and its throughput dips line up in time with those
    commands. Whether bs=18 is genuinely a worse operating point is therefore
    unresolved, and this config exists to settle it on a quiet cluster rather
    than to assert it.
    """
    config = gpt_oss_120b_gb300_sac_compile()
    config.training.local_batch_size = 18
    return config


def gpt_oss_120b_1k_ref() -> Trainer.Config:
    """Control. Re-measures job 56 under this branch's launcher.

    Not redundant with job 56: this branch runs with a per-job TRITON_CACHE_DIR,
    and every other row here is a 100-step run whose steady-state window is
    steps 50-100 rather than 100-250. Comparing a 100-step arm against a
    250-step number would fold the difference in window into the result.
    """
    return _120b_ref()


def _mxfp8_120b(config: Trainer.Config) -> Trainer.Config:
    """MXFP8 on the expert grouped GEMMs + the attention Linears, at 120B.

    Shares `_20b_mxfp8`'s body -- it is already parameterised by flavor, and the
    reasoning carries over unchanged: gpt-oss has no shared experts and no dense
    per-layer feed_forward, so every layer's FFN is the MoE and "attention" is
    the whole dense-Linear surface. Router gate and lm_head stay bf16.
    """
    return _20b_mxfp8(config, flavor="120b")


def gpt_oss_120b_1k_mxfp8() -> Trainer.Config:
    """MXFP8 at job 56's batch. The single largest untried lever.

    Batch is held at 18 on purpose. MXFP8 on dsv4 cost no memory at all (peak
    243.23 GiB against bf16's 243.21), so there should be nothing to re-tune,
    and holding the batch keeps this a clean one-variable comparison against
    `_1k_ref`. If it does shift memory, the follow-up tunes batch against the
    measurement instead of a guess.
    """
    return _mxfp8_120b(_120b_ref())


def gpt_oss_120b_1k_bf16reduce() -> Trainer.Config:
    """bf16 gradient reduce-scatter at job 56's batch.

    CHANGES NUMERICS -- bf16 reduction across 64 shards accumulates rounding
    error that the fp32 default exists to avoid. Paired with a fixed
    --debug.seed control run rather than judged on its loss curve alone, because
    on this cluster the init/data seed is not fixed by default and step-1 loss
    spans 11.80-12.60 across runs, so an unseeded loss comparison proves nothing.

    Expected to matter far more here than the +0.6% it gave on dsv4: there, EP=64
    made every expert rank-local so the flag only touched ~21.6 B non-expert
    parameters. At EP=1 it touches all ~117 B.
    """
    config = _120b_ref()
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def _120b_ep(degree: int, local_batch_size: int = 16) -> Trainer.Config:
    config = _120b_ref()
    config.parallelism.expert_parallel_degree = degree
    config.training.local_batch_size = local_batch_size
    return config


def gpt_oss_120b_1k_ep8() -> Trainer.Config:
    """EP=8 at job 56's batch, isolating the parallelism change.

    EP=8 was dsv4's joint optimum (with 16) and is the middle of the range that
    divides 64. Batch held at 18 so this is one variable against `_1k_ref`; the
    memory it frees is spent in the `_maxbs` variants once it has been measured
    rather than predicted.
    """
    return _120b_ep(8)


def gpt_oss_120b_1k_ep16() -> Trainer.Config:
    """EP=16. dsv4's operating point -- same speed as EP=8 and 14 GiB cheaper.

    On dsv4 the EP optimum was interior: 4 collapsed (each rank all-gathers its
    whole group's experts), 64 fanned the all-to-all across all 16 nodes over IB.
    8 and 16 tied. Both are measured here because that optimum is a property of
    the model's expert count and token volume, not only of the cluster, and
    gpt-oss-120B has 128 experts against dsv4's 384.
    """
    return _120b_ep(16)


def gpt_oss_120b_1k_ep4() -> Trainer.Config:
    """EP=4, one node's worth -- keeps the all-to-all inside a node.

    The case for it on this cluster is that a 4-GPU EP group is exactly one node,
    so the dispatch never crosses IB. The case against is dsv4's EP=4 result
    (-66%), where 96 experts per rank drove the gathered weights to 96% of HBM.
    gpt-oss at EP=4 holds 32 experts per rank, a quarter of that per-layer
    weight, so the failure mode may simply not apply.
    """
    return _120b_ep(4)


# --- combinations, run once the isolated arms above are measured -----------


def gpt_oss_120b_1k_mxfp8_bf16reduce() -> Trainer.Config:
    """The two compute/comms levers together: MXFP8 arithmetic + bf16 reduction.

    They target disjoint costs -- GEMM throughput and gradient-reduction bytes --
    so if both are positive in isolation this should be close to additive. That
    is a prediction worth recording: if it lands well under the sum, the step is
    bound by something neither of them touches.
    """
    return _mxfp8_120b(gpt_oss_120b_1k_bf16reduce())


def gpt_oss_120b_1k_mxfp8_ep8() -> Trainer.Config:
    """MXFP8 + EP=8."""
    return _mxfp8_120b(_120b_ep(8))


def gpt_oss_120b_1k_all() -> Trainer.Config:
    """Everything that measured positive: MXFP8 + bf16 reduce + EP=8."""
    config = _120b_ep(8)
    config.training.mixed_precision_reduce = "bfloat16"
    return _mxfp8_120b(config)


# --- batch variants, for spending memory a lever frees ---------------------
#
# Sized from measurement, not prediction. SelectiveAC on 120B costs ~10.8 GiB
# per batch unit over a ~60 GiB fixed term (job 53), so each 4 units of batch is
# ~43 GiB. Job 56 sits at 232.96 GiB of 276.50 (84.25%), leaving ~15 GiB -- about
# 1.4 units. Anything more has to come from a lever that frees memory.


def gpt_oss_120b_1k_ep8_bs24() -> Trainer.Config:
    """EP=8 spending its freed memory on batch. bs 18 -> 24.

    Only meaningful if `_1k_ep8` measures well under 84% memory. At bs=1, EP=8
    saved 19.97 GiB over EP=1 (48.36 vs 68.33), and if that saving holds at
    scale it covers ~1.8 batch units; 24 assumes the saving grows with batch
    because what EP removes is the per-layer gathered weight buffer that the
    activations then compete with. Tested, not assumed.
    """
    return _120b_ep(8, local_batch_size=24)


def gpt_oss_120b_1k_ep8_bs32() -> Trainer.Config:
    """EP=8 at bs=32. Deliberately past where the memory should sit.

    Kept separate so the config anyone inherits is not the one parked on the
    edge. On 120B SelectiveAC ran at 98.7% with no penalty, so the 20B cliff
    rule does not bind here -- but a clean OOM still ends a run.
    """
    return _120b_ep(8, local_batch_size=32)


def gpt_oss_120b_1k_mxfp8_bs24() -> Trainer.Config:
    """MXFP8 at bs=24, in case MXFP8 frees memory here unlike on dsv4."""
    return _mxfp8_120b(_120b_ep(1, local_batch_size=24))


# --- debug gate ------------------------------------------------------------


def gpt_oss_debugmodel_1k_mxfp8() -> Trainer.Config:
    """4-GPU gate for the MXFP8 build: does it convert and step at all.

    Exists because MXFP8 needed six separate build/kernel fixes to run on
    sm_103 at all, and each of them failed at model-build or first-forward time.
    Finding that out on one node in three minutes is worth far more than finding
    it out after a 16-node job has taken the whole cluster.

    Converts the same fqn surface as the 120B configs ("attention" + every
    GroupedExperts), which the dsv4 work found to matter: a gate whose fqn list
    is narrower than the real config's is not a gate -- theirs converted 12
    Linears where the real model converted 730, and missed the one that broke.
    """
    config = _gpt_oss_debugmodel()
    config.compile.enable = True
    config.activation_checkpoint = SelectiveAC.Config()
    return _20b_mxfp8(config, flavor="debugmodel")


def gpt_oss_120b_1k_ref_profile() -> Trainer.Config:
    """The reference config with the torch profiler on, active at step 12.

    This runs FIRST, before any of the arms above, because the dsv4 sweep on this
    cluster spent an entire overnight batch of 13 experiments tuning parts of the
    step that turned out to cost nothing -- one kernel was 58% of GPU time and
    the expert GEMMs it had been optimising were 1.6%. One profiling run would
    have redirected all of them. The 120B step here is 31.5% MFU, so roughly
    two-thirds of the machine is going somewhere, and which lever matters depends
    entirely on where.

    Step 12 is after compile has settled (first-step compile dominates) but
    before the steady state this config reaches around step 60-80. It profiles a
    step whose *shape* is representative even though its throughput is not yet;
    the kernel mix is what is being read, not the rate.

    Set as a config rather than via --profiler.* flags so there is no question
    about nested-key CLI spelling, and so the profiled configuration is the same
    object the reference arm runs.
    """
    config = _120b_ref()
    config.profiler.enable_profiling = True
    config.profiler.profile_freq = 12
    config.profiler.save_traces_folder = "/mnt/dgxc/traces/ref_120b"
    return config


# ---------------------------------------------------------------------------
# FP8 rowwise, after MXFP8 turned out to be unavailable at this model's shape
#
# The MXFP8 debug gate (job 246) failed in the first backward:
#
#     cutedsl_quantize_2d_1x32.py:907
#     AssertionError: K must be divisible by 128
#
# K there is the GEMM contraction dim, which for every expert GEMM in gpt-oss is
# `dim` = 2880, and 2880 % 128 = 64. `pad_multiple` does not help: it pads the
# per-expert TOKEN groups (the M dim), not K. torchao's grouped-MXFP8 path calls
# `mxfp8_quantize_2d_1x32_cutedsl` unconditionally -- there is no kernel
# preference to fall back to a Triton cast, and the flydsl sibling is the ROCm
# one. So MXFP8 on gpt-oss needs either a padded-K kernel or a model dimension
# it does not have. This is a different blocker from the one TUNING_RESULTS.md
# recorded (that one was "no aarch64 build", which IS fixed -- see
# MXFP8_120B.md, the kernels load and register fine on sm_103/torch 2.14).
#
# FP8 rowwise reaches the same tensor cores with an alignment this model meets:
# `Float8GroupedExpertsConverter.PAD_MULTIPLE` is 16, and 2880 % 16 == 0. It
# also needs nothing built: the GEMM is torch's own `_scaled_grouped_mm`, which
# ships compiled in the torch 2.14 wheel, and both `Float8TrainingOpConfig` and
# `Float8LinearConfig.from_recipe_name` are present in the venv's stock torchao
# 0.18.0. No overlay, no sm_103 build, nothing to go silently missing.
# ---------------------------------------------------------------------------


def _maybe_disable_pointwise_autotune() -> None:
    """Opt-in escape from an Inductor pointwise-autotune crash in the fp8 path.

    The 4-GPU fp8 gate (job 252) died while Inductor was BENCHMARKING candidate
    configs for a fused fp8-cast kernel -- not while running it:

        triton_poi_fused_..._triton_fp8_rowwise_2d_scale_and_cast_25.run(...)
          -> autotune_to_one_config -> benchmark_all_configs -> synchronize
        torch.AcceleratorError: CUDA error: unspecified launch failure
        Sticky error detected

    So the failing thing is one of the autotuner's trial launches, and
    `triton.autotune_pointwise` turns that trial loop off ("this should only be
    disabled for debugging/testing", per Inductor's own comment). It has no
    environment variable of its own, hence this hook.

    Gated on GPTOSS_1K_NO_POINTWISE_AUTOTUNE so it changes nothing unless a job
    asks for it -- the fp8 arms are queued against this worktree already, and a
    change that altered their compilation silently would make them incomparable.
    Disabling it is not free: the autotuner exists to pick pointwise tilings, so
    any run using this needs saying so next to its number.
    """
    import os

    if os.environ.get("GPTOSS_1K_NO_POINTWISE_AUTOTUNE") != "1":
        return
    import torch._inductor.config as inductor_config

    inductor_config.triton.autotune_pointwise = False
    logger.warning(
        "GPTOSS_1K_NO_POINTWISE_AUTOTUNE=1: disabled "
        "torch._inductor.config.triton.autotune_pointwise"
    )


def _fp8_120b(config: Trainer.Config, experts_only: bool = False) -> Trainer.Config:
    """FP8 rowwise on the expert grouped GEMMs, optionally also the dense Linears.

    `filter_fqns` on the Linear converter is an EXCLUDE list (unlike the MXFP8
    converter's include-list `fqns`), so the two exclusions carry the same
    intent the MXFP8 helper expressed by including only "attention": keep the
    router gate and the lm_head in bf16. The router decides expert assignment
    from 128 logits, and quantising lm_head puts fp8 error straight into the
    loss. `module_filter_fn` independently skips any Linear whose in/out
    features are not multiples of 16, which is what excludes `attn_sink`
    (in_features=1).
    """
    _maybe_disable_pointwise_autotune()
    converters: list = [
        Float8GroupedExpertsConverter.Config(
            model_compile_enabled=config.compile.enable
            and "model" in config.compile.components,
        ),
    ]
    if not experts_only:
        converters.append(
            Float8LinearConverter.Config(
                model_compile_enabled=config.compile.enable
                and "model" in config.compile.components,
                recipe_name="rowwise",
                filter_fqns=["lm_head", "router"],
            )
        )
    config.model_spec = model_registry("120b", converters=converters)
    return config


def gpt_oss_120b_1k_fp8_experts() -> Trainer.Config:
    """FP8 rowwise on the expert grouped GEMMs only. The primary lever.

    Experts alone are ~57% of this model's per-token FLOPs -- 36 layers x 4
    active experts x 3 GEMMs of 2880x2880 -- so this is where fp8 pays. Run
    before the combined arm on purpose: the grouped-GEMM swap also replaces the
    token dispatcher (`swap_token_dispatcher`), which is the part most likely to
    interact with SelectiveAC and compile, and isolating it means a failure
    points at one change.
    """
    return _fp8_120b(_120b_ref(), experts_only=True)


def gpt_oss_120b_1k_fp8() -> Trainer.Config:
    """FP8 rowwise on the expert grouped GEMMs and the attention Linears.

    Adds the attention projections (~15% of per-token FLOPs) on top of the
    experts' ~57%, for ~72% of the arithmetic in fp8.
    """
    return _fp8_120b(_120b_ref())


def gpt_oss_120b_1k_fp8_bf16reduce() -> Trainer.Config:
    """FP8 rowwise + bf16 gradient reduce-scatter: the arithmetic lever and the
    bytes-on-the-wire lever together. Prediction is near-additive."""
    return _fp8_120b(gpt_oss_120b_1k_bf16reduce())


def gpt_oss_120b_1k_fp8_ep8() -> Trainer.Config:
    """FP8 rowwise + EP=8."""
    return _fp8_120b(_120b_ep(8))


def gpt_oss_120b_1k_fp8_all() -> Trainer.Config:
    """FP8 rowwise + bf16 reduce + EP=8."""
    config = _120b_ep(8)
    config.training.mixed_precision_reduce = "bfloat16"
    return _fp8_120b(config)


def gpt_oss_120b_1k_fp8_bs24() -> Trainer.Config:
    """FP8 rowwise at bs=24, for spending memory fp8 frees.

    fp8 expert weights are half the bytes of bf16 in the GEMM operands, and the
    saved activations for the expert GEMMs are fp8 too, so unlike MXFP8 on dsv4
    (which cost no memory and freed none) this may actually open batch headroom.
    Sized against the measured footprint of the bs=18 arm rather than assumed.
    """
    return _fp8_120b(_120b_ep(1, local_batch_size=24))


def gpt_oss_debugmodel_1k_fp8() -> Trainer.Config:
    """4-GPU gate for the fp8 path, same role as the MXFP8 gate that caught the
    K%128 assert in three minutes.

    Uses the 120B converter surface rather than a narrower one, for the reason
    the dsv4 work learned the hard way: a gate whose fqn set is smaller than the
    real config's can pass while the real run dies on a module the gate never
    converted.
    """
    config = _gpt_oss_debugmodel()
    config.compile.enable = True
    config.activation_checkpoint = SelectiveAC.Config()
    converters: list = [
        Float8GroupedExpertsConverter.Config(model_compile_enabled=True),
        Float8LinearConverter.Config(
            model_compile_enabled=True,
            recipe_name="rowwise",
            filter_fqns=["lm_head", "router"],
        ),
    ]
    config.model_spec = model_registry("debugmodel", converters=converters)
    return config


# ---------------------------------------------------------------------------
# Wave 2: the remaining comms knob, and batch retunes for whatever frees memory
# ---------------------------------------------------------------------------


def gpt_oss_120b_1k_symmmem() -> Trainer.Config:
    """FSDP all-gather over NVLink symmetric memory.

    Worth a slot here for a reason it was not on dsv4, where it measured flat and
    cost +8 GiB: at EP=1 this model all-gathers every layer's full 128-expert
    weight stack, ~6.4 GB of bf16 per layer and ~229 GB per step, which is the
    largest single thing FSDP does. dsv4 ran at EP=16 with experts already
    rank-local, so there was far less all-gather for symmetric memory to
    accelerate.

    Requires compute capability >= 9.0; GB300 reports (10, 3).
    """
    config = _120b_ref()
    config.parallelism.enable_fsdp_symm_mem = True
    return config


def gpt_oss_120b_1k_fp8_symmmem() -> Trainer.Config:
    """fp8 arithmetic + symmetric-memory all-gather."""
    config = _120b_ref()
    config.parallelism.enable_fsdp_symm_mem = True
    return _fp8_120b(config)


def gpt_oss_120b_1k_fp8_bs20() -> Trainer.Config:
    """fp8 at bs=20. One step of batch, for the case where fp8 frees a little.

    Job 56 sits at 232.96 GiB of 276.50 (84.25%) with ~15 GiB spare, and
    SelectiveAC costs ~10.8 GiB per batch unit -- so bs=20 is roughly the most
    the reference footprint can absorb without a lever that frees memory.
    On 120B, unlike 20B, SelectiveAC at 98.7% measured no penalty at all
    (786.1 against 786.9 at 84.2%), so sitting high here is not the cliff risk
    it was on the smaller model.
    """
    return _fp8_120b(_120b_ep(1, local_batch_size=20))


def gpt_oss_120b_1k_fp8_bf16reduce_bs20() -> Trainer.Config:
    """Everything positive, at the largest batch the reference footprint allows."""
    config = _120b_ep(1, local_batch_size=20)
    config.training.mixed_precision_reduce = "bfloat16"
    return _fp8_120b(config)


def gpt_oss_120b_1k_sac_gmm_bs6() -> Trainer.Config:
    """SelectiveAC that also saves the expert GEMM, at the batch its memory allows.

    Recorded and left UNQUEUED unless the primary levers fall short, because the
    120B numbers already price this trade and it looks like a loss.

    On 20B it was worth +13.9% at an identical batch (859.6 -> 979.5), by
    removing the expert-GEMM recompute that per-op SelectiveAC does by default.
    But it cost 11.80 GiB per batch unit there, which scales to ~17.7 on 120B's
    36 layers, and with SelectiveAC's own ~10.8 that is ~28.5 GiB per batch unit
    against a ~60 GiB fixed term -- about bs=6 at 90% of HBM, against the
    reference's 18.

    The 120B sweep already measured the two ends of exactly this trade-off:
    saving everything (no-AC) affords bs=3 and gives 455.3, saving little
    (SelectiveAC) affords bs=18 and gives 786.9. This config sits between them
    at bs=6, and nothing in those two points suggests the middle wins. Two
    earlier attempts at bs=5 (jobs 62 and 67) died at init with exit 137 before
    producing a step, so it is also the riskiest use of a slot.

    Needs `save_grouped_mm` on SelectiveAC.Config, which is on branch
    perf/sac-save-grouped-mm and is NOT cherry-picked here -- so this function
    will raise until it is. Left that way on purpose: the config records the
    reasoning without implying the code is in place.
    """
    config = _120b_ep(1, local_batch_size=6)
    config.activation_checkpoint = SelectiveAC.Config(save_grouped_mm=True)
    return config


# ---------------------------------------------------------------------------
# Numerics check for fp8, seeded so the comparison is actually a comparison
# ---------------------------------------------------------------------------


def gpt_oss_120b_1k_ref_seeded() -> Trainer.Config:
    """bf16 control for the fp8 loss comparison, with a fixed seed.

    The seed is the whole point. On this cluster the init/data seed is not fixed
    by default and step-1 loss has spanned 11.80-12.60 across runs with
    bit-identical effective configurations, so an unseeded loss curve cannot
    show whether fp8 changed anything. With `--debug.seed` pinned, the bf16 and
    fp8 curves start from the same weights on the same data and any divergence
    is attributable.

    Short on purpose: fp8's effect on loss, if it has one, shows in the first
    tens of steps, and this is a check on numerics rather than a throughput
    measurement.
    """
    config = _120b_ref()
    config.debug.seed = 42
    return config


def gpt_oss_120b_1k_fp8_seeded() -> Trainer.Config:
    """fp8 arm of the seeded loss comparison. Pairs with the config above."""
    config = _120b_ref()
    config.debug.seed = 42
    return _fp8_120b(config)


def gpt_oss_120b_1k_ref_noautotune() -> Trainer.Config:
    """The bf16 reference with Inductor's pointwise autotuner disabled.

    Exists so the fp8 arms have a control that compiles the same way they do.
    fp8 cannot run on this cluster without `GPTOSS_1K_NO_POINTWISE_AUTOTUNE=1`
    (jobs 252 and 257 both died in `benchmark_all_configs`), and pointwise
    kernels are 11.8% of the compute stream, so a matched control is the only
    way to say how much of an fp8 gain is fp8 rather than a compilation
    difference. Expected to come in at or slightly below the 771.9 of job 263 --
    the autotuner exists to pick tilings -- which would mean fp8's gain measured
    against job 263 is understated rather than flattered.
    """
    _maybe_disable_pointwise_autotune()
    return _120b_ref()


# ---------------------------------------------------------------------------
# After fp8 grouped mm turned out to be unavailable on sm_103
#
# Probes 270 and 271 settled it. On GB300 (capability 10,3):
#
#   torch._scaled_mm      2D rowwise   OK     <- the Float8Linear path
#   torch._grouped_mm     bf16         OK     <- 49.7% of the reference step
#   torch._scaled_grouped_mm fp8       ABORTS <- "Arch conditional MMA
#   torchao fp8 grouped wrapper        ABORTS    instruction used without
#                                                targeting appropriate
#                                                compute capability"
#
# and it aborts identically under torch 2.14 and the 2.15.0.dev nightly. CUTLASS
# calls abort() rather than raising, which is why the failure killed the process
# instead of surfacing as an exception -- and why the first 120B fp8 job blamed
# the Inductor autotuner: the autotuner was simply the first thing to touch the
# kernel.
#
# So the expert grouped GEMM -- half the critical path -- cannot be quantized on
# this hardware with this software. MXFP8 needs K % 128 == 0 and K is 2880;
# fp8 needs an sm_103 kernel that neither torch build ships. What remains
# reachable is the 2D path, which is the attention projections and lm_head:
# 9.9% of the compute stream.
#
# The levers below are therefore aimed at making the bf16 expert GEMM itself
# cheaper rather than narrower. The profile says it runs at ~898 TFLOP/s
# against a 2503 bf16 peak -- about 36% -- which is low for GEMMs of this size,
# and the most likely reason is shape: at EP=1 each rank runs 128 expert groups
# of ~4.1k rows each (bs=16: 16*8192*4/128). Expert parallelism makes them
# fewer and larger without changing the arithmetic.
# ---------------------------------------------------------------------------


def gpt_oss_120b_1k_fp8_linear() -> Trainer.Config:
    """fp8 rowwise on the dense Linears only. The one quantization that runs.

    Uses `Float8LinearConverter` alone, with no grouped-experts converter, so it
    never touches `_scaled_grouped_mm` and never calls `swap_token_dispatcher`
    -- which also means, unlike the other fp8 arms, this one changes exactly one
    thing.

    Ceiling is small and known in advance: dense GEMMs are 9.9% of the compute
    stream, so even at 2x on those the step only shortens ~5%. Worth a slot
    because it is the only arithmetic lever left and it composes with the EP
    arms, not because it can reach the target on its own.

    lm_head is excluded along with the router. It is the largest single Linear
    (2880 x 201088) so including it would raise the ceiling, but fp8 error there
    lands directly in the loss, and `ChunkedLossWrapper` already splits it.
    """
    config = _120b_ref()
    config.model_spec = model_registry(
        "120b",
        converters=[
            Float8LinearConverter.Config(
                model_compile_enabled=True,
                recipe_name="rowwise",
                filter_fqns=["lm_head", "router"],
            )
        ],
    )
    return config


def gpt_oss_120b_1k_ep8_fp8_linear() -> Trainer.Config:
    """EP=8 for expert-GEMM shape, plus fp8 on the dense Linears.

    The two surviving levers together. They are disjoint -- one changes the
    shape of the MoE GEMMs, the other the precision of the attention ones -- so
    if both are positive this should be close to additive.
    """
    config = _120b_ep(8)
    config.model_spec = model_registry(
        "120b",
        converters=[
            Float8LinearConverter.Config(
                model_compile_enabled=True,
                recipe_name="rowwise",
                filter_fqns=["lm_head", "router"],
            )
        ],
    )
    return config


def gpt_oss_120b_1k_mxfp8_experts() -> Trainer.Config:
    """MXFP8 on the expert grouped GEMMs only. The primary quantization arm.

    Experts-only rather than experts+Linears, because the fp8 measurement says
    the dense half is where the risk is: `gpt_oss_120b_1k_fp8_linear` (job 281)
    quantized *only* the attention projections and measured 618.8 TFLOP/s
    against the bf16 control's 771.9 -- **-19.8%** -- while freeing 37 GiB.

    A 9.9% share of the compute stream cannot lose 20% by itself, so the cost is
    almost certainly not the GEMMs: `Float8LinearConverter` sets
    `torch._inductor.config.emulate_precision_casts = True` globally
    (float8.py:122), which changes codegen for the *whole* model. The MX
    converters do not set it, so MXFP8 should not inherit that penalty -- but
    the cheap way to find out is to not convert the Linears at all, and put the
    arm on the 49.7% that actually matters.
    """
    config = _120b_ref()
    config.model_spec = model_registry(
        "120b",
        converters=[
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=True,
                pad_multiple=128,
            )
        ],
    )
    return config
