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
    # All 16, not the 16/32/32/16 pinned when this workaround was first
    # written. Letting Inductor autotune the backward freely (job 545) picked
    # 16 across the board and ran 11.6% faster end to end -- 181.53 vs 162.62
    # TFLOP/s -- at identical memory. Smaller tiles mean less shared memory
    # per block, so more blocks stay resident; at head_dim=512 occupancy
    # matters more than tile size. The original 16/32/32/16 was chosen to
    # stop a launch failure, never benchmarked against alternatives.
    "BLOCK_M1": 16,
    "BLOCK_N1": 16,
    "BLOCK_M2": 16,
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

    - ``optimizer.implementation = "fused_opt_states_bf16"`` -- fused AdamW
      with bf16 moment buffers, halving optimizer state again on top of the
      bf16 training dtype.
    - ``training.dtype = "bfloat16"`` -- full bf16 training: parameters,
      gradients and optimizer states, with no fp32 master copy. This is the
      lever that makes the model fit at all; torchtitan's ``float32`` default
      costs ~80 GiB per rank here (measured: 178 GiB peak with it, 260 GiB
      without, which on a 277 GiB card means allocator pressure and ~4% lost
      throughput).
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
    optimizer = default_adamw(lr=8e-4)
    optimizer.implementation = "fused_opt_states_bf16"
    config.optimizer = optimizer
    config.training.dtype = "bfloat16"
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


def deepseek_v4_flash_8k_gb300_free_bwd_tiles(
    microbatch: int = 4, seq_len: int | None = 8192, autotune: bool = True
) -> Trainer.Config:
    """The GB300 recipe with the BACKWARD flex tiles unpinned.

    ``BLOCK_M1/N1/M2/N2`` were pinned because at head_dim=512 the backward
    otherwise died with "CUDA error: unspecified launch failure", and the pin
    also caps those tiles at ``block_size`` (32), which is why the backward
    kernel has been the single largest cost in every profile (40%+ of kernel
    time). If a newer Inductor can pick valid backward tiles on its own, they
    may be larger than 32 and the backward may get cheaper. Autotune defaults
    back ON here, since with nothing pinned Inductor has to search for a
    configuration that fits in 232,448 B of shared memory.

    The forward tiles stay pinned: those were needed for the forward to
    compile at all.
    """
    from torchtitan.models.common.attention import FlexInnerAttention

    config = deepseek_v4_flash_8k_gb300_batched(microbatch, seq_len)
    for layer in config.model_spec.model.layers:
        inner = getattr(getattr(layer, "attention", None), "inner_attention", None)
        if isinstance(inner, FlexInnerAttention.Config):
            inner.kernel_options = {
                k: v
                for k, v in _GB300_FLEX_KERNEL_OPTIONS.items()
                if not k.startswith(("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"))
            }
            inner.max_autotune = autotune
    return config


def deepseek_v4_flash_8k_gb300_fp32_params(
    microbatch: int = 4, seq_len: int | None = 8192
) -> Trainer.Config:
    """fp32 parameters with bf16 optimizer states -- the configuration
    ``fused_opt_states_bf16`` is actually for.

    On top of ``training.dtype="bfloat16"`` the flag is a no-op: full bf16
    training already puts parameters, gradients AND optimizer states in bf16,
    so there is nothing left for it to halve (measured: 192.71 GiB either
    way, 162.62 vs 162.68 TFLOP/s). Its real use is the other trade -- keep
    fp32 master weights for convergence safety and pay for them with bf16
    moments instead of fp32 ones. This config measures what that costs.
    """
    config = deepseek_v4_flash_8k_gb300_batched(microbatch, seq_len)
    config.training.dtype = "float32"
    return config


def deepseek_v4_flash_8k_gb300_fastdata(
    microbatch: int = 4, seq_len: int | None = 8192
) -> Trainer.Config:
    """The GB300 recipe with the input pipeline widened.

    This tree uses Grain, not the PyTorch DataLoader, so the familiar
    ``num_workers`` / ``prefetch_factor`` / ``persistent_workers`` /
    ``pin_memory`` knobs do not exist. The equivalents are:

    - ``read_options.num_threads`` (default 16) -- Grain's reader threads,
      the analogue of ``num_workers``;
    - ``read_options.prefetch_buffer_size`` (default 500) -- records read
      ahead, the analogue of ``prefetch_factor``;
    - ``num_prefetch_batches`` (default 2) -- assembled batches held ready.

    Grain's workers are persistent and its output is already pinned, so those
    two flags have no counterpart to set. ``resize_fn`` / ``max_patches`` are
    vision knobs and do not apply to a text model.

    Expectation: little or nothing. The 4x profile has the GPU idle 1.2% of
    the step, which bounds anything the input pipeline can win -- a starved
    loader would show up as idle gaps. Measured here so the question is
    settled rather than assumed.
    """
    config = deepseek_v4_flash_8k_gb300_batched(microbatch, seq_len)
    import grain

    config.dataloader.read_options = grain.ReadOptions(
        num_threads=32, prefetch_buffer_size=2000
    )
    config.dataloader.num_prefetch_batches = 8
    return config


def deepseek_v4_flash_8k_gb300_sac(
    microbatch: int = 1, ep: int = 2, seq_len: int | None = 8192
) -> Trainer.Config:
    """The GB300 recipe with per-op selective AC instead of FullAC.

    FullAC recomputes the whole block, so every kernel in the forward runs
    twice. SelectiveAC keeps the expensive outputs (matmuls, attention,
    comms) and recomputes only the cheap elementwise work, trading memory for
    that second forward. The trade has to be paid for out of the microbatch
    or the parallelism, which is what this sweep varies.

    Prior evidence, all superseded but worth stating: on the pre-batched
    recipe at 1x, stock SelectiveAC measured 96.96 TFLOP/s against FullAC's
    99.50 and cost +118 GiB, i.e. it lost on both counts. Two things have
    changed since -- the backward is ~12% cheaper (retuned flex tiles), which
    raises the relative cost of the recomputed forward, and full bf16
    training freed ~67 GiB of headroom to spend.

    Requires the out-of-place relu in ``Indexer.select``; with the in-place
    version every SAC run died on "Tensor cached during selective activation
    checkpoint has been mutated".
    """
    from torchtitan.distributed.activation_checkpoint import SelectiveAC

    config = deepseek_v4_flash_8k_gb300_batched(microbatch, seq_len)
    config.parallelism.expert_parallel_degree = ep
    # The default FQN list expects an nn.Linear named moe.router.gate; this
    # model routes differently, so the list is emptied rather than matched.
    config.activation_checkpoint = SelectiveAC.Config(
        force_recompute_mm_shapes_by_fqns=[]
    )
    return config
