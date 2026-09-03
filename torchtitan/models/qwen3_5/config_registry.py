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
from torchtitan.components.tokenizer import MultiModalTokenizer

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry, QWEN3_5_SPECIAL_TOKENS


def _dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=128,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        build_mrope_positions=True,
        **kwargs,
    )


def qwen35_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m-test"),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_debugmodel_varlen_attn() -> Trainer.Config:
    config = qwen35_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe() -> Trainer.Config:
    model_spec = model_registry("debugmodel_moe", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m-test"),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=2,
            expert_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_0_8b() -> Trainer.Config:
    model_spec = model_registry("0.8B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-0.8B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_2b() -> Trainer.Config:
    model_spec = model_registry("2B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-2B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-3),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_4b() -> Trainer.Config:
    model_spec = model_registry("4B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-4B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_9b() -> Trainer.Config:
    model_spec = model_registry("9B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_27b() -> Trainer.Config:
    model_spec = model_registry("27B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-27B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_35b_a3b() -> Trainer.Config:
    model_spec = model_registry("35B-A3B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-35B-A3B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_122b_a10b() -> Trainer.Config:
    model_spec = model_registry("122B-A10B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-122B-A10B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=4,
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_397b_a17b() -> Trainer.Config:
    model_spec = model_registry("397B-A17B", moe_comm_backend="standard")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-397B-A17B",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=_dataloader("cc12m"),
        optimizer=default_adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=8,
            expert_parallel_degree=16,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


# ---------------------------------------------------------------------------
# MFU tuning variants for 35B-A3B on 1x8 B200 (dj, 2026-08-19).
#
# Stock qwen35_35b_a3b measures MFU ~0.95% at 86 GiB/GPU (48% of 178 GiB), i.e.
# there is a lot of unused memory. A rank-0 profiler trace of the stock config
# (job 289, step 10) attributes the step as:
#
#   GatedDeltaNet FLA chunk kernels  2640 ms  39.9%   <- 510 kernels
#   NCCL collectives                 1979 ms  29.9%   <- 731 ms EP all_to_all
#   GatedDeltaNet causal_conv1d       999 ms  15.1%   <- 360 kernels
#   everything else (18.4k kernels)   546 ms   8.2%
#   copies / memcpy                   237 ms   3.6%
#   GEMM / matmul                     187 ms   2.8%   <- all MFU counts
#   FlexAttention (ViT)                26 ms   0.4%
#   plus 2223 ms (25.9% of the step) with no kernel resident at all.
#
# So MFU is low because the GEMMs are 3% of the step, not because the FLOP
# model is wrong. These variants target the three biggest non-GEMM costs.
# ---------------------------------------------------------------------------


def _a3b_variant(
    *,
    local_batch_size: int | None = None,
    activation_checkpoint=...,
    expert_parallel_degree: int | None = None,
    tensor_parallel_degree: int | None = None,
    compile_enable: bool = False,
    seq_len: int | None = None,
) -> Trainer.Config:
    """qwen35_35b_a3b with selected knobs overridden. Everything else is stock."""
    cfg = qwen35_35b_a3b()
    if local_batch_size is not None:
        cfg.training.local_batch_size = local_batch_size
    if seq_len is not None:
        cfg.training.seq_len = seq_len
    if activation_checkpoint is not ...:
        cfg.activation_checkpoint = activation_checkpoint
    if expert_parallel_degree is not None:
        cfg.parallelism.expert_parallel_degree = expert_parallel_degree
    if tensor_parallel_degree is not None:
        cfg.parallelism.tensor_parallel_degree = tensor_parallel_degree
    cfg.compile.enable = compile_enable
    return cfg


def qwen35_35b_a3b_noac() -> Trainer.Config:
    """Drop activation checkpointing. FullAC reruns the whole block in backward,
    which is why causal_conv1d_fwd appears 270x for 90 real calls. Memory is
    only half used, so buy the memory back."""
    return _a3b_variant(activation_checkpoint=None)


def qwen35_35b_a3b_selac() -> Trainer.Config:
    """Per-op selective AC instead of full-block recompute."""
    return _a3b_variant(activation_checkpoint=SelectiveAC.Config())


def qwen35_35b_a3b_bs16() -> Trainer.Config:
    """4x the tokens per step to amortize the 26% idle and the 18k tiny kernels."""
    return _a3b_variant(local_batch_size=16)


def qwen35_35b_a3b_noac_bs16() -> Trainer.Config:
    """No recompute + 4x batch."""
    return _a3b_variant(activation_checkpoint=None, local_batch_size=16)


def qwen35_35b_a3b_selac_bs16() -> Trainer.Config:
    return _a3b_variant(
        activation_checkpoint=SelectiveAC.Config(), local_batch_size=16
    )


def qwen35_35b_a3b_ep2() -> Trainer.Config:
    """EP=8 over only 16k tokens makes the per-expert GEMMs tiny and costs
    731 ms/step of all_to_all. Spread experts less."""
    return _a3b_variant(expert_parallel_degree=2)


def qwen35_35b_a3b_notp() -> Trainer.Config:
    """TP=2 on a 2048-dim model shards the GEMMs down to 1024 columns and adds
    sequence-parallel collectives. Try pure FSDP8 + EP8."""
    return _a3b_variant(tensor_parallel_degree=1)


def qwen35_35b_a3b_compile() -> Trainer.Config:
    return _a3b_variant(compile_enable=True)


def qwen35_35b_a3b_tuned() -> Trainer.Config:
    """Best-guess combination of the above."""
    return _a3b_variant(
        activation_checkpoint=SelectiveAC.Config(),
        local_batch_size=16,
        tensor_parallel_degree=1,
        compile_enable=True,
    )


# ---------------------------------------------------------------------------
# THE root-cause variant: sample packing.
#
# Stock cc12m gives one short image+caption per 4096-token sequence. Measured
# with probe_dataloader.py / doc_probe.py on the stock qwen35_35b_a3b config:
#
#   labels.numel()          16,384 positions   <- this is what tps counts
#   real content               ~1,360 positions (one doc of 270-425 per row)
#   supervised (labels!=-100)   42-136 positions  (0.3-0.8%!)
#   positions == 0          15,025 of 16,384
#
# That last line is the killer. Padding slots all carry position_id 0, and
# GatedDeltaNet builds its varlen metadata from document starts, so the FLA
# kernels are launched over ~15,000 length-1 "documents" instead of 4 real
# ones. (It is also why fla's grid hit the CUDA 65535 limit on axis 1:
# N*HV = 15000*32 = 480,000. Flattening the grid made it run, not be sane.)
# That is where 55% of the step goes.
#
# Packing bin-packs short samples into full sequences, so positions are
# contiguous and the document count drops to ~15 real documents per sequence.
# ---------------------------------------------------------------------------


def qwen35_35b_a3b_pack() -> Trainer.Config:
    """Stock 35B-A3B + sample packing. Only the dataloader changes."""
    cfg = qwen35_35b_a3b()
    cfg.dataloader.packing_buffer_size = 64
    return cfg


def qwen35_35b_a3b_pack_selac() -> Trainer.Config:
    cfg = qwen35_35b_a3b_pack()
    cfg.activation_checkpoint = SelectiveAC.Config()
    return cfg


def qwen35_35b_a3b_pack_bs8() -> Trainer.Config:
    """Packing + 2x batch. bs=16 OOMs (job 299), so 8."""
    cfg = qwen35_35b_a3b_pack()
    cfg.training.local_batch_size = 8
    return cfg


def qwen35_35b_a3b_pack_notp() -> Trainer.Config:
    cfg = qwen35_35b_a3b_pack()
    cfg.parallelism.tensor_parallel_degree = 1
    return cfg


def qwen35_35b_a3b_pack_notp_bs8() -> Trainer.Config:
    """Best so far (pack + TP=1, MFU 3.12% @ 51% memory) with 2x the batch."""
    cfg = qwen35_35b_a3b_pack_notp()
    cfg.training.local_batch_size = 8
    return cfg


def qwen35_35b_a3b_pack_notp_selac() -> Trainer.Config:
    """pack + TP=1 with per-op selective AC instead of full-block recompute.
    Full no-AC blew memory to 96% and collapsed throughput (job 298), so this is
    the middle option."""
    cfg = qwen35_35b_a3b_pack_notp()
    cfg.activation_checkpoint = SelectiveAC.Config()
    return cfg


# ---------------------------------------------------------------------------
# 122B-A10B. Structurally a much better MFU target than 35B-A3B: 10B active
# params give 58.88 GFLOP/token vs 19.69, so the same tokens/s is ~3x the MFU.
# Needs dtype=bfloat16 -- stock dtype is float32, and 122.6e9 * 16 B / 8 GPUs
# is ~245 GiB/GPU against 178 GiB usable. bf16 puts it at ~123 GiB/GPU, the
# same trick the gpt_oss_120b runs needed.
# ---------------------------------------------------------------------------


def qwen35_122b_a10b_pack_bf16() -> Trainer.Config:
    cfg = qwen35_122b_a10b()
    cfg.training.dtype = "bfloat16"
    cfg.dataloader.packing_buffer_size = 64
    # Stock qwen35_122b_a10b sets tensor_parallel_degree=4 but the model has
    # n_kv_heads=2, so it dies at startup with
    #   ValueError: tensor_parallel_degree (4) must divide n_kv_heads (2).
    # The stock config cannot run at all (job 319). TP must be 1 or 2.
    cfg.parallelism.tensor_parallel_degree = 2
    return cfg


def qwen35_122b_a10b_pack_bf16_tp1() -> Trainer.Config:
    """TP=1 was the winning setting on 35B-A3B."""
    cfg = qwen35_122b_a10b_pack_bf16()
    cfg.parallelism.tensor_parallel_degree = 1
    return cfg


def qwen35_122b_a10b_pack_bf16_tp2() -> Trainer.Config:
    cfg = qwen35_122b_a10b_pack_bf16()
    cfg.parallelism.tensor_parallel_degree = 2
    return cfg


def qwen35_35b_a3b_pack_notp_compile() -> Trainer.Config:
    """Best config + torch.compile. After the packing fix the largest remaining
    bucket is 1367 ms across 18,372 tiny elementwise kernels (job 318 trace),
    which is what inductor fusion is for."""
    cfg = qwen35_35b_a3b_pack_notp()
    cfg.compile.enable = True
    return cfg


# ---------------------------------------------------------------------------
# Techniques borrowed from the gpt_oss_120b tuning that reached ~29% MFU:
# torch.compile and a bigger batch. Neither transfers directly here --
# compile.enable=True crashes inductor on the GatedDeltaNet path (job 322,
# "Node ... was invalid, but is output") and bs=8 is WORSE than bs=4 because the
# extra batch is nearly all uncounted ViT work (job 316). These try the parts
# that can still apply.
# ---------------------------------------------------------------------------


def qwen35_122b_a10b_pack_bf16_tp1_losscompile() -> Trainer.Config:
    """Compile only the loss, not the model. compile.components accepts just
    "model" and "loss"; dropping "model" avoids the inductor crash while still
    compiling the chunked cross-entropy over a 248,320-token vocab."""
    cfg = qwen35_122b_a10b_pack_bf16_tp1()
    cfg.compile.enable = True
    cfg.compile.components = ["loss"]
    return cfg


def qwen35_122b_a10b_pack_bf16_tp1_ep4() -> Trainer.Config:
    """NCCL is 26% of the step. EP=8 over this token count may be over-spread."""
    cfg = qwen35_122b_a10b_pack_bf16_tp1()
    cfg.parallelism.expert_parallel_degree = 4
    return cfg


def qwen35_122b_a10b_pack_bf16_tp1_ep2() -> Trainer.Config:
    """EP=4 beat EP=8 by 9.3% (job 355 vs 352). Keep walking it down."""
    cfg = qwen35_122b_a10b_pack_bf16_tp1()
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def qwen35_122b_a10b_pack_bf16_tp1_ep1() -> Trainer.Config:
    """No expert parallelism at all; FSDP still shards the expert weights.
    May OOM -- EP=4 already sits at 85.5% memory."""
    cfg = qwen35_122b_a10b_pack_bf16_tp1()
    cfg.parallelism.expert_parallel_degree = 1
    return cfg



# ---------------------------------------------------------------------------
# Activation-checkpointing sweep.
#
# FullAC reruns the whole block in backward. Those FLOPs cost wall clock but are
# NOT in num_flops_per_token, so AC is a direct MFU tax -- roughly a third of the
# step, since forward is ~1/3 of fwd+bwd. Dropping it is one of the largest
# remaining levers.
#
# Earlier attempts all died on memory (jobs 298/299/317) because they removed AC
# without buying memory back anywhere. The levers that free activation memory:
#   - TP: sequence-parallel shards activations. Capped at 2 here -- both MoE
#     variants have n_kv_heads=2 and TP must divide it.
#   - smaller local_batch_size.
# So: TP=2 + bs=2 is ~1/4 the activation memory of TP=1 + bs=4, which is roughly
# what no-AC needs.
# ---------------------------------------------------------------------------


def _a3b_122b_ac_variant(
    *, ac, local_batch_size: int, tp: int = 2, ep: int = 2
) -> Trainer.Config:
    cfg = qwen35_122b_a10b_pack_bf16_tp1()
    cfg.activation_checkpoint = ac
    cfg.training.local_batch_size = local_batch_size
    cfg.parallelism.tensor_parallel_degree = tp
    cfg.parallelism.expert_parallel_degree = ep
    return cfg


def qwen35_122b_pack_tp2_ep2_noac_bs2() -> Trainer.Config:
    return _a3b_122b_ac_variant(ac=None, local_batch_size=2)


def qwen35_122b_pack_tp2_ep2_noac_bs1() -> Trainer.Config:
    return _a3b_122b_ac_variant(ac=None, local_batch_size=1)


def qwen35_122b_pack_tp2_ep2_selac_bs2() -> Trainer.Config:
    return _a3b_122b_ac_variant(ac=SelectiveAC.Config(), local_batch_size=2)


def qwen35_122b_pack_tp2_ep2_selac_bs4() -> Trainer.Config:
    return _a3b_122b_ac_variant(ac=SelectiveAC.Config(), local_batch_size=4)


def qwen35_122b_pack_tp2_ep2_fullac_bs2() -> Trainer.Config:
    """Control: same TP/EP/batch as the no-AC runs, but AC still on. Without this
    the no-AC numbers conflate 'AC removed' with 'batch halved'."""
    return _a3b_122b_ac_variant(ac=FullAC.Config(), local_batch_size=2)


def qwen35_35b_pack_tp2_ep2_noac_bs2() -> Trainer.Config:
    cfg = qwen35_35b_a3b_pack_notp()
    cfg.activation_checkpoint = None
    cfg.training.local_batch_size = 2
    cfg.parallelism.tensor_parallel_degree = 2
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def qwen35_35b_a3b_pack_notp_ep2() -> Trainer.Config:
    """EP reduction was worth +14.8% on 122B-A10B. 35B-A3B ships EP=8 too.
    Measured 5.22% MFU / 4,065 tps (job 390)."""
    cfg = qwen35_35b_a3b_pack_notp()
    cfg.parallelism.expert_parallel_degree = 2
    return cfg


def qwen35_35b_pack_ep2_selac_bs4() -> Trainer.Config:
    """35B has the most headroom (51% at TP=1/bs=4), so selective AC may fit
    without touching TP or batch."""
    cfg = qwen35_35b_a3b_pack_notp_ep2()
    cfg.activation_checkpoint = SelectiveAC.Config()
    return cfg

## DM: None of the compile functions really worked. I think its broken
##
##
##
def qwen35_122b_a10b_pack_bf16_tp1_ep2_compile() -> Trainer.Config:
    """Best config + torch.compile, with GatedDeltaNet blocks excluded.

    Plain compile.enable=True crashes inductor (job 322): apply_compile uses
    fullgraph=True per block, and the GDN blocks wrap FLA custom autograd
    functions dynamo cannot trace. parallelize_qwen3_5 now passes a block_filter
    that skips those, so this compiles 12 full-attention blocks + all 27 vision
    encoder blocks and leaves the 36 GDN blocks in eager. Target: the 1367 ms
    across 18,372 tiny elementwise kernels in the job 318 trace.
    """
    cfg = qwen35_122b_a10b_pack_bf16_tp1_ep2()
    cfg.compile.enable = True
    return cfg


def qwen35_35b_a3b_pack_notp_ep2_compile() -> Trainer.Config:
    cfg = qwen35_35b_a3b_pack_notp_ep2()
    cfg.compile.enable = True
    return cfg


def qwen35_35b_a3b_pack_notp_ep2_compile_decoder() -> Trainer.Config:
    """Isolation A: compile only the 10 full-attention decoder blocks."""
    cfg = qwen35_35b_a3b_pack_notp_ep2()
    cfg.compile.enable = True
    cfg.compile.components = ["decoder", "loss"]
    return cfg


def qwen35_35b_a3b_pack_notp_ep2_compile_vision() -> Trainer.Config:
    """Isolation B: compile only the 27 vision-encoder blocks."""
    cfg = qwen35_35b_a3b_pack_notp_ep2()
    cfg.compile.enable = True
    cfg.compile.components = ["vision", "loss"]
    return cfg
## DM
##
##
##

def qwen35_122b_a10b_pack_bf16_tp1_ep2_workers() -> Trainer.Config:
    """Best config + background dataloader workers.

    Job 567 showed contiguous multi-step MFU dips (7.4% -> 0.9% for ~9 steps)
    with ZERO allocator retries and an UNCHANGED 1965 MHz SM clock, while power
    fell 731W -> 437W at ~90% "utilization". Occupied-but-not-computing at
    constant clock is collectives waiting on a straggler rank, not throttling and
    not memory pressure.
    torchtitan's dataloader defaults to num_workers=0 and the qwen3_5
    _dataloader() helper never overrides it, so each rank's cc12m stream (HTTPS
    shard fetch + JPEG decode + resize) runs inline in the training loop. A stall
    on any rank blocks every other rank in the collective.

    CAVEAT: with a streaming IterableDataset, num_workers>1 can duplicate samples
    across workers unless the dataset shards by worker id. Fine for an MFU probe;
    verify data ordering before using this for a real training run.
    """
    cfg = qwen35_122b_a10b_pack_bf16_tp1_ep2()
    # DM: was 2
    cfg.dataloader.num_workers = 8
    cfg.dataloader.persistent_workers = True
    cfg.dataloader.prefetch_factor = 4
    cfg.dataloader.pin_memory = True
    return cfg


def qwen3_5_122b_swiglu_bias_fusion() -> Trainer.Config:
    """Best 122B config + fused SwiGLU (gate+up GEMM and silu*mul activation).

    Built on qwen35_122b_a10b_pack_bf16_tp1_ep2_workers (sustained 7.95% MFU /
    2,626 tps over 181 steps), so any delta is attributable to the fusion.

    What torchtitan.overrides.fused_swiglu actually does:
      - fuses w1 (gate) and w3 (up) into one w13 GEMM, laid out
        (hidden_dim, 2, dim) so Shard(0) TP keeps a matching slice of both
      - fuses the SiLU-and-mul into one Triton kernel
        (the torchtitan::silu_and_mul custom op)
      - checkpoint-compatible: w13 is split back to w1.weight/w3.weight on save

    NOTE ON "BIAS": there is no bias fusion here, and nothing to fuse.
    fused_swiglu.py never mentions bias, and Qwen3.5's SwiGLU path is bias-free --
    routed experts (w1_EFD/w2_EDF/w3_EFD) and shared experts (w1/w2/w3.weight)
    carry no bias. The only 36 decoder biases are GatedDeltaNet `dt_bias`, which
    is not on this path. The vision encoder does have 220 biases but uses
    VisionMLP, which this override does not target. The name follows the request;
    the mechanism is gate+up + activation fusion only.

    Two overrides are registered by that module and both must be named:
    fused_swiglu targets FeedForward (Qwen3.5's shared experts are
    SigmoidGatedFeedForward, a FeedForward subclass) and fused_grouped_experts
    targets GroupedExperts (the routed experts). The MoE is where the FLOPs are
    -- ~6.8 of 19.7 GFLOP/token on 35B -- and this reaches the elementwise
    traffic torch.compile could not (see notes on the compile sweep).
    """
    cfg = qwen35_122b_a10b_pack_bf16_tp1_ep2_workers()
    for target in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert target not in cfg.override.imports
        cfg.override.imports.append(target)
    return cfg


# ---------------------------------------------------------------------------
# seq_len 8192 sweep (requested for the study).
#
# Two things change when seq_len doubles:
#
# 1. MEMORY. Activation memory scales with seq_len, and the best 122B config
#    already sits at 158.9 GiB (89.1%) of 178.35 GiB. So local_batch_size has to
#    come down to keep tokens/step -- and memory -- roughly constant. bs=2 at
#    8192 gives the same 16,384 tokens/rank/step as bs=4 at 4096, which keeps the
#    comparison against the 7.92-7.95% baseline meaningful.
#
# 2. num_flops_per_token GOES UP, so MFU rises mechanically. The quadratic
#    attention term is 6 * n_full * n_heads * (2*head_dim) * seq_len, which for
#    122B is 4.83 GFLOP/token at 4096 and 9.66 at 8192: total 58.88 -> 63.71
#    GFLOP/token, i.e. **+8.2% MFU before any efficiency change at all**.
#    It is only +8% rather than double because just 12 of 48 layers carry
#    quadratic attention; the other 36 are linear GatedDeltaNet.
#    Compare tokens/s as well as MFU when reading these runs.
#
# The packer bins to seq_len, so at 8192 each bin holds ~23 cc12m samples
# (~350 tokens each) instead of ~12. Images per batch stays under the
# max_images_per_batch=128 cap at bs<=4.
# ---------------------------------------------------------------------------


def qwen35_122b_seq8192_bs2() -> Trainer.Config:
    """Primary: same tokens/step as the 4096/bs4 baseline, so memory should land
    close to the 89% the baseline used."""
    cfg = qwen35_122b_a10b_pack_bf16_tp1_ep2_workers()
    cfg.training.seq_len = 8192
    # DM: real quick test from 2 to 3
    cfg.training.local_batch_size = 3
    return cfg


def qwen35_122b_seq8192_bs1() -> Trainer.Config:
    """Fallback if bs=2 OOMs -- halves tokens/step, so expect lower tps."""
    cfg = qwen35_122b_seq8192_bs2()
    cfg.training.local_batch_size = 1
    return cfg


def qwen35_122b_seq8192_bs4() -> Trainer.Config:
    """Doubles tokens/step vs the baseline. Almost certainly OOMs at 89% -- queued
    to establish the ceiling rather than in expectation of success."""
    cfg = qwen35_122b_seq8192_bs2()
    cfg.training.local_batch_size = 4
    return cfg


def qwen35_35b_seq8192_bs4() -> Trainer.Config:
    """35B has the headroom for this: its best config uses only 51% of memory at
    4096/bs4, so doubling seq_len at the same batch should fit."""
    cfg = qwen35_35b_a3b_pack_notp_ep2()
    cfg.training.seq_len = 8192
    cfg.dataloader.num_workers = 2
    cfg.dataloader.persistent_workers = True
    cfg.dataloader.prefetch_factor = 4
    cfg.dataloader.pin_memory = True
    return cfg


# ---------------------------------------------------------------------------
# Two config-level MFU levers, on top of qwen35_122b_seq8192_bs2 (8.72% MFU /
# 2,686 tps, the best sustained config). 122B at seq_len 8192 only.
# ---------------------------------------------------------------------------


def qwen35_122b_seq8192_bs2_minasync_ep() -> Trainer.Config:
    """Lever 1: MinimalAsyncEP token dispatcher instead of standard all-to-all.

    NCCL is ~26% of the step, so the EP comm path is worth probing. Of the four
    backends only two are usable here: "standard" (current) and
    "minimal_async_ep". "deepep" and "hybridep" both need external libraries
    that are NOT installed (deep_ep / hybridep both fail to import), and are
    targeted at H100/NVLink-Switch and GB200/NVLink72 respectively.

    Managing expectations, honestly:
      - the same dispatcher measured **1.79x WORSE** than standard all-to-all on
        gpt_oss_120b (jobs 305 vs 306: 6.47% vs 11.57% MFU). That was EP=8,
        no-compile, a different model -- so not settled for this config, but the
        prior is bad.
      - we run EP=2, so the expert all-to-all is between just 2 ranks. Most of
        the 26% NCCL is FSDP all-gather / reduce-scatter, not EP, which caps how
        much any EP backend can win.
      - MinimalAsyncEP is documented "for constrained DP>=EP"; we satisfy that
        (dp_shard=8 >= EP=2).
    """
    cfg = qwen35_122b_seq8192_bs2()
    cfg.model_spec = model_registry(
        "122B-A10B", moe_comm_backend="minimal_async_ep"
    )
    return cfg


def qwen35_122b_seq8192_bs2_patchbudget() -> Trainer.Config:
    """Lever 2: uniform patch budget, to cut the pixel_values padding waste.

    pixel_values is padded to (num_images, max_num_patch, patch_dim), and
    max_num_patch swings 1,344-2,560 across batches -- measured 12-25% of patch
    slots wasted (0.375 padded vs 0.306 real patches per position). Narrowing the
    per-image patch distribution shrinks that padding.

    resize_to_patch_budget caps total and per-side patches, then pads to a
    patch_size*merge_size multiple. Note it only ever scales DOWN
    (scale = min(1.0, ...)), so with the default max_patches=4096 against cc12m's
    ~1,100-2,500 real patches per image the cap would never bind and the swap
    would be a pure no-op. So max_patches is lowered to 1024 (= 256 vision tokens
    per image) to actually bind on the larger images and compress the spread.

    HOW TO READ THIS RUN: lowering max_patches also reduces real ViT work, so
    tokens/s will rise partly because there is simply less to do. That is exactly
    what the per-batch vision FLOP term is for -- it counts REAL patches from
    grid_thw, so corrected MFU only improves if efficiency improves, not merely
    because the workload shrank. Compare MFU, not tps, for this one.
    """
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget

    cfg = qwen35_122b_seq8192_bs2()
    cfg.dataloader.resize_fn = resize_to_patch_budget
    cfg.dataloader.max_patches = 1024
    return cfg


def qwen35_122b_seq8192_bs2_combined() -> Trainer.Config:
    """Both winning levers together: uniform patch budget + MinimalAsyncEP.

    Measured separately against the 8.72% / 2,686 tps baseline (job 613):
      - patch budget   (job 629): 9.34%  / 2,883 tps  (+7.1%)
      - MinimalAsyncEP (job 630): 10.33% / 3,182 tps  (+18.5%)

    They touch disjoint subsystems -- the dataloader/vision path and the MoE
    expert-comm path -- so they should be roughly additive. Not guaranteed: if
    either was partly hiding the other's stall (e.g. less padded ViT work leaves
    less compute to overlap the expert all-to-all against), the combination will
    undershoot the sum.
    """
    from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget

    cfg = qwen35_122b_seq8192_bs2()
    cfg.model_spec = model_registry(
        "122B-A10B", moe_comm_backend="minimal_async_ep"
    )
    cfg.dataloader.resize_fn = resize_to_patch_budget
    cfg.dataloader.max_patches = 1024
    return cfg


def qwen35_122b_seq8192_bs2_combined_bf16reduce() -> Trainer.Config:
    """EXPERIMENTAL: best config + bf16 FSDP gradient reduce-scatter.

    Profile of the combined config (job 640) shows ReduceScatter_Sum_f32 at
    1,230 ms of a 4,838 ms step -- 25% of the step and 65% of all NCCL time --
    while the model itself trains in bf16. Halving those bytes should save
    ~600 ms, i.e. ~12% step time, so roughly 12.6% MFU.

    *** THIS CHANGES TRAINING NUMERICS, IT IS NOT A FREE WIN. ***
    torchtitan types mixed_precision_reduce as Literal["float32"] on purpose;
    allowing "bfloat16" required widening that annotation. Reducing gradients
    across 8 shards in bf16 accumulates rounding error that the fp32 reduction
    exists to prevent. Pair every run of this with
    qwen35_122b_seq8192_bs2_combined at the SAME --debug.seed and compare loss
    curves before trusting it for anything but an MFU number.
    """
    cfg = qwen35_122b_seq8192_bs2_combined()
    cfg.training.mixed_precision_reduce = "bfloat16"
    return cfg