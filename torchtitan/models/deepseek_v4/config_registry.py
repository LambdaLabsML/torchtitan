# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .mtp import MTPLoss


def deepseek_v4_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_mtp_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len, n_mtp_layers=1)
    return Trainer.Config(
        loss=MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_flash(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_flash", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_pro(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_pro", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


# GB300 (sm_103) requires these FlexAttention tiles at head_dim=512. Without
# them the first forward dies with "No valid triton configs ... Required:
# 294912, Hardware limit: 232448", and without the backward tiles specifically
# the backward hits "CUDA error: unspecified launch failure". Measured on one
# GB300: every larger forward tile fails, and 32x32 only works with num_stages
# and num_warps pinned too. This is a correctness requirement on this hardware,
# not a tuning choice.
_GB300_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_stages": 1,
    "num_warps": 4,
    "BLOCK_M1": 16,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 16,
}


def _pin_gb300_flex_tiles(config: Trainer.Config, block_size: int = 32) -> None:
    """Pin the flex tiles and sparse block size on every flex layer."""
    from torchtitan.models.common.attention import FlexInnerAttention

    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexInnerAttention.Config):
            inner.kernel_options = dict(_GB300_FLEX_KERNEL_OPTIONS)
            inner.block_size = block_size
            # With the tiles pinned, autotune can only re-benchmark the pin
            # against variants that do not fit: throughput-neutral and ~228 s
            # of startup per distinct shape.
            inner.max_autotune = False


def deepseek_v4_flash_8k_gb300(seq_len: int | None = 8192) -> Trainer.Config:
    """DeepSeek-V4 flash at seq_len 8192 on 64x GB300 (16 nodes x 4 GPUs).

    The tuned recipe behind the leaf-compile measurements in this branch.
    Levers, each measured against the one before it:

    - ``expert_parallel_degree=4`` -- one node's NVLink group per expert
      group; the stock 64 scored 18.29 TFLOP/s against 21.2 here.
    - ``block_size=32`` on the DSA block mask (+15.8%), plus the GB300 tile
      pin above, which is mandatory rather than optional.
    - ``FullAC`` -- at 8192 the model OOMs without it.
    - ``mixed_precision_reduce = bfloat16`` (+3.7%).
    - the leaf compiles in this branch, worth +34% on top.

    MoE dispatch: ``minimal_async_ep``, worth +4.9% over the standard
    all-to-all on the pre-compile recipe and ~+15% once the leaf compiles made
    compute cheaper. Upstream deprecated that dispatcher in #4627 as
    unmaintained; this fork restores it because nothing available replaces it
    on this hardware (DeepEP measured 7.4% above standard and needs a source
    build; HybridEP 2.9% below MinimalAsyncEP). It also forces CUDA graphs
    off, since its dispatch has a host sync.

    Run with ``TORCHTITAN_LEAF_COMPILE=all`` (the default) and, for a numerics
    reference, ``TORCHTITAN_LEAF_COMPILE=0``.
    """
    from torchtitan.distributed.activation_checkpoint import FullAC

    config = deepseek_v4_flash(seq_len)
    config.model_spec = model_registry(
        "deepseek_v4_flash", seq_len=seq_len, moe_comm_backend="minimal_async_ep"
    )
    config.training.disable_cuda_graphs = True
    _pin_gb300_flex_tiles(config)
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        expert_parallel_degree=4,
    )
    config.activation_checkpoint = FullAC.Config()
    config.training.mixed_precision_reduce = "bfloat16"
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len or 8192
    config.training.max_context_length = seq_len or 8192
    config.training.steps = 30
    return config


def deepseek_v4_flash_8k_gb300_batched(
    microbatch: int = 4, seq_len: int | None = 8192
) -> Trainer.Config:
    """The GB300 recipe with ``microbatch`` packed sequences per rank.

    Only useful with the batched DSA path: folded, the indexer scores every
    query against every compressed key of the whole stream, so a 2x microbatch
    cost -19% per token and 4x never reached step 1. Batched, the attention
    cost is linear in tokens and the curve rises: measured on 32x GB300,
    1x 137.28 -> 2x 150.57 -> 4x 158.74 TFLOP/s (and 170.86 at EP=2).

    4x is the operating point: 6x adds ~1% for 47 GiB, and 8x exhausts the
    card. Note the trainer's reported memory excludes the MoE dispatcher's
    symmetric buffers, which grow with the microbatch.
    """
    config = deepseek_v4_flash_8k_gb300(seq_len)
    config.training.num_tokens_per_microbatch_per_dp_rank = microbatch * (seq_len or 8192)
    return config


def deepseek_v4_flash_8k_gb300_batched_stages2(
    microbatch: int = 6, seq_len: int | None = 8192
) -> Trainer.Config:
    """The batched GB300 recipe at EP=2 with ``num_stages=2``.

    ``_GB300_FLEX_KERNEL_OPTIONS`` pins ``num_stages=1``, which is part of the
    correctness pin for head_dim=512 and was never tuned for speed. Raising it
    to 2 gives the Triton loop the software pipelining it had none of -- the
    profile showed ``num_stages=1``, ``limitingFactors=SMEM`` and no overlap
    between a tile's global-to-shared load and the previous tile's compute.

    Measured on 32x GB300 at 6x/EP=2: 214.11 vs 172.45 TFLOP/s, +24.2 %, with
    slightly lower memory (220.16 vs 223.36 GiB) and the loss curve tracking
    the control. It applies to the forward template as well as the backward.

    Worth knowing this was neutral at 1x microbatch (25.44 vs 25.45 on an
    earlier attention): pipelining pays in proportion to the independent work
    a launch has, and the batch axis is what supplies it. So it belongs with
    ``_batched``, not with the 1x recipe.
    """
    from torchtitan.models.common.attention import FlexInnerAttention

    config = deepseek_v4_flash_8k_gb300_batched(microbatch, seq_len)
    config.parallelism.expert_parallel_degree = 2
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexInnerAttention.Config):
            opts = dict(inner.kernel_options or {})
            opts["num_stages"] = 2
            inner.kernel_options = opts
    return config


def deepseek_v4_flash_8k_gb300_batched_stages2_profile(
    microbatch: int = 6, seq_len: int | None = 8192
) -> Trainer.Config:
    """``_stages2`` with the torch profiler on for two steps.

    ``warmup=3, active=2`` matches the other GB300 traces, so the kernel
    breakdown is comparable. Profiling distorts step time -- the numbers inside
    a profiled run are not throughput datapoints.
    """
    config = deepseek_v4_flash_8k_gb300_batched_stages2(microbatch, seq_len)
    config.profiler.enable_profiling = True
    config.profiler.profile_freq = 10
    config.profiler.profiler_warmup = 3
    config.profiler.profiler_active = 2
    return config
