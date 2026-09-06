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
