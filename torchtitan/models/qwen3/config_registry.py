# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

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
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
    NVFP4LinearConverter,
)
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import (
    FullAC,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry
from .model import Qwen3Model


def qwen3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
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
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_debugmodel_nvfp4() -> Trainer.Config:
    config = qwen3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Convert every decoder-layer Linear while leaving the lm_head in bf16.
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=["layers"],
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def qwen3_debugmodel_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = qwen3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Keep the last 15% of decoder layers and the lm_head in bf16.
    num_layers = len(cast(Qwen3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(
        num_layers,
        _NVFP4_BF16_TAIL_FRACTION,
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def qwen3_debugmodel_moe_param_groups() -> Trainer.Config:
    config = qwen3_moe_debug()
    config.optimizer = OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r"(?:tok_embeddings|output)\.",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 8e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            ParamGroupConfig(
                pattern=r"\.router\.gate\.",
                optimizer_name="Adam",
                optimizer_kwargs={"lr": 1e-4, "betas": (0.9, 0.95), "eps": 1e-8},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 8e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
    )
    return config


def qwen3_debugmodel_flex_flash() -> Trainer.Config:
    model_spec = model_registry("debugmodel", attn_backend="flex_flash")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
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
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_0_6b() -> Trainer.Config:
    model_spec = model_registry("0.6B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_1_7b() -> Trainer.Config:
    model_spec = model_registry("1.7B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_8b_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = sft_qwen3_8b_math()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    config.compile = CompileConfig(enable=True, components=["model"])
    # Keep the last 15% of decoder layers and the lm_head in bf16.
    num_layers = len(cast(Qwen3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(
        num_layers,
        _NVFP4_BF16_TAIL_FRACTION,
    )
    config.model_spec = model_registry(
        "8B",
        attn_backend="varlen",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=True,
            ),
        ],
    )
    return config


def qwen3_14b() -> Trainer.Config:
    model_spec = model_registry("14B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_30b_a3b() -> Trainer.Config:
    model_spec = model_registry("30B-A3B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_30b_a3b_8k_bs4() -> Trainer.Config:
    """qwen3_30b_a3b with seq_len 4096 -> 8192 and local_batch_size 2 -> 4.

    Together these put 32,768 tokens/GPU/step against the stock 8,192, a 4x
    increase in tokens per step. Everything else -- FullAC, FSDP-only, c4,
    dtype=float32, CUDA graphs on -- is inherited from qwen3_30b_a3b unchanged.

    Stock baseline for comparison (8x B200, job 1152, 400 steps):
        bs=2 seq=4096   9.66% MFU, 217.25 TFLOPs/GPU, 7,783 tok/s/GPU,
                        69.16GiB peak
    Note that baseline was taken WITHOUT --vboost=1 (the allocation_tuning
    spank plugin was broken on 2026-08-28), so compare this config's number
    against it only if this run is also unboosted.

    Memory projection. Params+grads+the two AdamW moments are all fp32, so the
    batch-independent term is 30,532,122,624 * 16 B/param / 8 ranks = 56.9GiB.
    The 69.16GiB baseline peak therefore leaves ~12.3GiB of activations and
    workspace at 8,192 tokens. Under FullAC stored activations are ~linear in
    tokens (the quadratic attention term is recomputed, not stored), so 4x
    tokens projects to ~49GiB, for a total of ~106GiB against the 178.35GiB
    usable. That should fit with ~70GiB of headroom, but the linear-in-tokens
    model is doing real work over a 4x jump -- the analogous gpt_oss ladder came
    in well under its own linear projection (see gptoss20b_bs16) -- so treat OOM
    as a live outcome. If it OOMs, back off local_batch_size to 3 before
    touching seq_len, since seq_len is the half of this change that alters the
    FLOPs mix rather than just the step size.

    Two things that make this NOT a pure throughput knob, unlike a plain batch
    increase:
      - Doubling seq_len doubles attention's per-token quadratic term, so
        num_flops_per_token rises and the tflops/MFU denominators are not the
        same as the baseline's. MFU stays comparable as a hardware-utilization
        figure, which is the point of the sweep, but tokens/sec and MFU no
        longer move together.
      - The 30B-A3B RoPE is built with max_seq_len=40960 (see _30b_a3b in
        __init__.py), so 8192 is well inside the trained range and needs no
        rope change.

    NOT YET MEASURED -- no MFU number recorded for this config as of
    2026-08-28.
    """
    config = qwen3_30b_a3b()
    config.training.seq_len = 8192
    config.training.local_batch_size = 4
    return config


def qwen3_30b_a3b_noac() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs4 with FullAC -> no activation checkpointing.

    seq_len=8192 and local_batch_size=4 are inherited unchanged; the ONLY
    difference from qwen3_30b_a3b_8k_bs4 is activation_checkpoint=None.
    ``None`` is the documented way to disable AC -- see
    ActivationCheckpointingConfig in distributed/activation_checkpoint.py,
    whose union is selective | full | memory-budget | None.

    NAMING: this was requested as qwen3_30b_a3b_noac_no_checkpointing, with the
    second change being "turn off checkpointing in float16". That change is a
    no-op, so the name was shortened per instruction. Checkpointing is ALREADY
    off and always has been: CheckpointManager.Config.enable defaults to False
    (components/checkpoint.py:224) and qwen3_30b_a3b never sets it -- it passes
    only interval, last_save_model_only and export_dtype="float16", all three of
    which are inert while enable is False. Verified empirically too: no *.distcp
    files and no step-* directories exist anywhere under outputs/ after any run
    in this sweep. The export_dtype="float16" in the base config is therefore
    dead configuration, not an active behaviour, and is left untouched here
    rather than removed, since removing it would edit the measured base config.

    Baselines to compare against (8x B200, unboosted clocks):
        stock       bs=2 seq=4096 FullAC  (job 1152)  9.66% MFU, 217.25 TF/GPU,
                                                      69.16GiB peak
        8k_bs4      bs=4 seq=8192 FullAC  (job 1156) ~17.9% MFU, ~404 TF/GPU,
                                                      83.92GiB peak
    Note both were taken WITHOUT --vboost=1 (broken spank plugin, 2026-08-28).

    EXPECT THIS TO OOM -- it is submitted to find the ceiling, not because it is
    projected to fit. Reasoning: the batch-independent term is 30,532,122,624 *
    16 B/param / 8 ranks = 56.9GiB, so 8k_bs4's 83.92GiB peak implies ~27GiB of
    activations and workspace WITH FullAC, which stores only one tensor per
    layer boundary and recomputes everything inside. Dropping AC stores every
    intermediate instead. The MoE block dominates that: at 32,768 tokens/GPU,
    top-8 routing and moe_hidden_dim=768 puts 32768*8*768*2 B = ~400MiB in a
    single expert intermediate, and there are several per layer across 48
    layers. That lands the extra somewhere in the tens-of-GiB range against
    178.35GiB usable, which makes this genuinely marginal rather than a
    comfortable fit -- the estimate is too loose to call either way, which is
    the point of running it.

    If it OOMs, the cheap knobs in order: drop local_batch_size 4 -> 2 (halves
    activations, keeps the 8192 context), then switch to SelectiveAC.Config()
    as the middle ground between full recompute and none.

    NOT YET MEASURED -- no MFU number recorded for this config as of
    2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs4()
    config.activation_checkpoint = None
    return config


def qwen3_30b_a3b_noac_bs1() -> Trainer.Config:
    """qwen3_30b_a3b_noac with local_batch_size 4 -> 1. seq_len stays 8192.

    The question this one answers: does no-AC run AT ALL at 8192 context, or is
    activation checkpointing mandatory at this sequence length regardless of
    batch size? bs=1 is the smallest batch that still exercises the real
    geometry, so an OOM here means no-AC is simply not viable at 8k and the next
    move is SelectiveAC rather than a smaller batch.

    Why the bs=4 attempt failed (job 1157, OOM on step 1, all 8 ranks):
        torch.OutOfMemoryError: Tried to allocate 2.00 GiB. GPU 0 has a total
        capacity of 178.35 GiB of which 1.41 GiB is free. Of the allocated
        memory 175.06 GiB is allocated by PyTorch.
    It died in models/common/token_dispatcher.py:159 combine, reached from
    moe.py:171 -> moe.py:448 -- i.e. in the MoE token-combine during the FORWARD
    pass, not in backward. That matters for extrapolating: it never accumulated
    a full 48-layer activation stack, so 175.06GiB is a FLOOR on what bs=4
    no-AC needs, not the peak. Quartering the batch quarters an unknown that is
    larger than the number we observed.

    Projection. The batch-independent term is 30,532,122,624 * 16 B/param / 8
    ranks = 56.9GiB. bs=4 therefore needed >=118GiB of activations before it
    died mid-forward. Scaling that floor linearly to bs=1 gives ~30GiB, for
    ~87GiB total -- comfortably inside 178.35GiB. Even if the true bs=4 peak was
    half again as large as the floor we saw, bs=1 still lands near ~101GiB. So
    this is expected to FIT, and the interesting number is what MFU it gets, not
    whether it survives.

    Comparison points (8x B200, unboosted clocks, same 8192 context):
        8k_bs4 FullAC (job 1156)  17.65% MFU, 397.03 TF/GPU, 83.92GiB peak
        8k_bs4 no-AC  (job 1157)  OOM on step 1
    Expect this to land WELL BELOW 17.65%: at bs=1 each expert sees ~1/4 the
    tokens, so every grouped expert GEMM shrinks in its M dimension and the
    optimizer step and FSDP all-gathers amortize over 4x fewer tokens. Removing
    recompute pulls the other way, but on the gpt_oss ladder batch size was
    worth far more than recompute savings. If it fits but lands low, that is a
    real result: it says no-AC only pays once the batch is large enough to feed
    the expert GEMMs, which at 8k context does not fit in 178GiB.

    NOT YET MEASURED -- no MFU number recorded for this config as of
    2026-08-28.
    """
    config = qwen3_30b_a3b_noac()
    config.training.local_batch_size = 1
    return config


def qwen3_30b_a3b_noac_bs2_nocudagraphs() -> Trainer.Config:
    """qwen3_30b_a3b_noac at local_batch_size 2, with CUDA graphs disabled.

    seq_len stays 8192 and activation checkpointing stays off; the two changes
    from qwen3_30b_a3b_noac are local_batch_size 4 -> 2 and
    disable_cuda_graphs False -> True.

    HISTORY: this function was previously named qwen3_30b_a3b_noac_bs2 and did
    NOT set disable_cuda_graphs. Under that name it was measured as job 1165 and
    OOMed -- see below. It was renamed and the flag added in place rather than
    forked into a new function, at the user's explicit request on 2026-08-28.
    The 1165 result is preserved here because it is the entire reason the flag
    is set; to reproduce that OOM exactly, run this config with
    --training.disable_cuda_graphs removed.

    Why CUDA graphs are disabled. As qwen3_30b_a3b_noac_bs2, job 1165 COMPLETED
    step 1 at 176.25GiB (98.82% of the card) and then OOMed on step 2, in the
    same MoE token-combine as every other no-AC failure
    (token_dispatcher.py:159 combine <- moe.py:171 <- moe.py:448):
        torch.OutOfMemoryError: Tried to allocate 1024.00 MiB. GPU 0 has a total
        capacity of 178.35 GiB of which 507.50 MiB is free. Of the allocated
        memory 175.98 GiB is allocated by PyTorch, with 2.11 GiB allocated in
        private pools (e.g., CUDA Graphs).
    It was ~517MiB short while CUDA graphs held 2.11GiB in private pools, i.e.
    roughly 4x the shortfall. Reclaiming that pool is the cheapest thing that
    could make this batch fit, and it is the only no-AC point with enough tokens
    per expert to have a real shot at beating the FullAC baseline.

    THAT REASONING WAS WRONG, AND THIS CONFIG DOES NOT FIT. Measured as job
    1169: OOM on step 2, in the same token_dispatcher.py:159 combine, on the
    same 1024.00MiB allocation.
        job 1165 (graphs ON):   step-1 peak 176.25GiB (98.82%), 507.50MiB free
                                at OOM, 175.98GiB allocated by PyTorch, of which
                                2.11GiB in private pools
        job 1169 (graphs OFF):  step-1 peak 176.54GiB (98.98%), 779.50MiB free
                                at OOM, 175.74GiB allocated by PyTorch, no
                                private pools
    Turning graphs off bought only 272MiB of extra free space, not 2.11GiB, and
    the reported peak went UP slightly. The error was reading "allocated in
    private pools (e.g., CUDA Graphs)" as reclaimable overhead. It is not: a
    graph's private pool holds the same activation tensors the step needs
    either way, and capture merely parks them in a separate allocator pool.
    Disabling graphs relocates that memory to the regular allocator instead of
    freeing it. Do not set this flag again expecting to recover memory -- it
    trades away graph replay for essentially nothing here.

    With 779.50MiB free against a 1024.00MiB request this is still ~245MiB
    short, and there is no comparable pool left to reclaim. no-AC at bs=2 does
    not fit at 8192 context, by any route tried.

    Measured ladder at 8192 context (8x B200, unboosted clocks):
        FullAC bs=4 (job 1156)  17.65% MFU, 397.03 TF/GPU,  83.92GiB  -- FITS
        no-AC  bs=1 (job 1163)  14.82% MFU, 333.42 TF/GPU, 137.91GiB  -- FITS
        no-AC  bs=2 (job 1165)  OOM on step 2,             176.25GiB peak
        no-AC  bs=4 (job 1157)  OOM on step 1,            >=175.06GiB

    Activation memory is NOT linear in batch size, which broke two earlier
    projections in opposite directions. Fitting the two measured no-AC peaks
    against the 56.9GiB batch-independent term (params + grads + the two AdamW
    moments, all fp32, sharded over 8 ranks) gives activations of 81.0GiB at
    8,192 tokens and 119.35GiB at 16,384 tokens -- +38.35GiB for a doubling, not
    +81GiB. That implies ~42.65GiB of batch-INDEPENDENT workspace plus
    ~4.68MiB/token. Use that two-point fit, not linear scaling, for anything
    further up this ladder; it projects no-AC bs=4 at ~196GiB of activations and
    ~253GiB total, which is why bs=4 is hopeless regardless of CUDA graphs.

    MEASURED 2026-08-28 as job 1169: OOM on step 2. Kept as the record of a
    closed line of attack, not as a runnable sweep point.

    This closes the no-AC branch at 8192 context. bs=1 (job 1163) is the only
    no-AC point that fits, and it loses to qwen3_30b_a3b_8k_bs4 (FullAC, 17.65%
    MFU at 83.92GiB) by 2.8 points of MFU while using 64% more memory for a
    quarter of the tokens per step. bs=2 does not fit with or without CUDA
    graphs, and the two-point fit above puts bs=4 at ~253GiB total. The next
    lever is qwen3_30b_a3b_8k_bs4_selac: SelectiveAC keeps the bs=4 batch that
    feeds the grouped expert GEMMs and spends the ~94GiB of headroom that FullAC
    leaves unused on storing only the ops that are expensive to recompute.
    """
    config = qwen3_30b_a3b_noac()
    config.training.local_batch_size = 2
    config.training.disable_cuda_graphs = True
    return config


def qwen3_30b_a3b_8k_bs4_selac() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs4 with FullAC -> SelectiveAC. bs=4, seq_len=8192.

    The only change from qwen3_30b_a3b_8k_bs4 is the AC policy; batch size,
    sequence length and everything else are inherited. SelectiveAC.Config() is
    used with its defaults, which is what the other qwen3 configs in this file
    use: it saves the outputs of ops that are expensive to recompute, recomputes
    every second matmul, and force-recomputes mms matching
    force_recompute_mm_shapes_by_fqns=["moe.router.gate"].

    Why this is the right next lever. The no-AC branch is closed -- it is only
    affordable at bs=1, which starves the grouped expert GEMMs:
        FullAC bs=4 (job 1156)  17.65% MFU, 397.03 TF/GPU,  83.92GiB  -- FITS
        no-AC  bs=1 (job 1163)  14.82% MFU, 333.42 TF/GPU, 137.91GiB  -- FITS
        no-AC  bs=2 (job 1165)  OOM step 2, 176.25GiB peak (graphs on)
        no-AC  bs=2 (job 1169)  OOM step 2, 176.54GiB peak (graphs off)
        no-AC  bs=4 (job 1157)  OOM step 1, >=175.06GiB
    Dropping AC entirely forces the batch down 4x to afford the activations, and
    that trade loses: -2.83 points of MFU for +53.99GiB. SelectiveAC is the only
    policy that buys back recompute time WITHOUT giving up batch size.

    Headroom available. FullAC bs=4 peaks at 83.92GiB of the 178.35GiB usable,
    so there is ~94GiB unspent. SelectiveAC should land somewhere between
    FullAC's 83.92GiB and the ~119GiB of activations that no-AC needs at 16,384
    tokens (measured, job 1165) -- and bs=4 here is 32,768 tokens, so the no-AC
    equivalent would be ~196GiB of activations by the two-point fit
    (~42.65GiB batch-independent workspace + ~4.68MiB/token). SelectiveAC has to
    recompute enough to stay under ~121GiB of activations to fit at all. That is
    a real constraint, not a formality: it saves roughly half the matmuls, so
    OOM is a live outcome and this may need force_recompute_mm_shapes_by_fqns
    widened beyond the router gate.

    What would count as a win: anything above 17.65% MFU. FullAC recomputes the
    entire block, so every layer's forward is done twice; SelectiveAC skips that
    second pass for the expensive ops while keeping the same 32,768 tokens/GPU
    feeding the expert GEMMs. If it fits and does NOT beat 17.65%, that says
    recompute was never the bottleneck at this shape and the next lever is the
    parallelism layout (expert_parallel_degree is 1 today, so every rank runs
    grouped GEMMs across all 128 experts).

    NOT YET MEASURED -- no MFU number recorded for this config as of
    2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs4()
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def qwen3_30b_a3b_8k_bs6_selac() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs4_selac with local_batch_size 4 -> 6. seq_len 8192.

    SelectiveAC is inherited unchanged; the only difference from
    qwen3_30b_a3b_8k_bs4_selac is the batch size, which puts 49,152 tokens/GPU
    against that config's 32,768.

    HISTORY: this function was previously named qwen3_30b_a3b_8k_bs8_selac and
    set local_batch_size=8. Under that name it was measured as job 1175 and
    OOMed. It was renamed and the batch size lowered in place, at the user's
    explicit request on 2026-08-28, rather than forked into a new function. To
    reproduce the OOM, set local_batch_size back to 8.

    Why bs=8 failed (job 1175): it COMPLETED step 1 at 176.45GiB (98.93% of the
    card) and OOMed on step 2 in distributed/cudagraph.py:381 flat_fwd_bwd:
        torch.OutOfMemoryError: Tried to allocate 4.64 GiB. GPU 0 has a total
        capacity of 178.35 GiB of which 2.73 GiB is free. Of the allocated
        memory 173.75 GiB is allocated by PyTorch, with 6.39 GiB allocated in
        private pools (e.g., CUDA Graphs).

    THE SELECTIVE-AC MEMORY MODEL IS NOW SOLVED, which is what job 1175 bought.
    Two points on the same AC policy, against the 56.9GiB batch-independent
    params+grads+AdamW term (30,532,122,624 * 16 B/param / 8 ranks):
        bs=4 (job 1173)  124.04GiB total ->  67.14GiB activations @ 32,768 tok
        bs=8 (job 1175) >=178.4GiB total  -> >=121.5GiB activations @ 65,536 tok
          (step-1 peak 176.45GiB plus the 1.91GiB it came up short on step 2)
    Fitting those gives ~14.0GiB of batch-independent workspace plus
    ~1.659MiB/token. Note how different that is from the no-AC fit
    (~42.65GiB fixed + ~0.747MiB/token): SelectiveAC has far LESS fixed
    workspace but a STEEPER per-token slope, because it stores per-token op
    outputs rather than large persistent buffers. Assuming the no-AC fixed term
    carried over to SelectiveAC is exactly why the bs=8 projection was wrong.

    PROJECTED TO FIT. 49,152 tokens -> ~93.7GiB activations, ~150.6GiB total,
    about 28GiB clear of the 178.35GiB usable. This is the first projection in
    this sweep derived from two measured points on the SAME activation
    checkpointing policy rather than from bracketing or a borrowed fixed term,
    so it deserves more confidence than its predecessors -- all of which missed
    (bs=1 no-AC by -51GiB, bs=2 no-AC by +43GiB, bs=8 selac by leaning fit when
    it OOMed, and the CUDA-graphs private-pool theory outright).

    EXPECT A SMALL GAIN AT BEST. Batch size was worth +7.99 points of MFU going
    4k/bs2 -> 8k/bs4, but the FullAC -> SelectiveAC change at bs=4 was worth only
    +0.51 (17.65% -> 18.16%). Returns are flattening, and bs=6 is only a 1.5x
    token increase, so this may land close to 18.16% rather than clearly above
    it. If it does, that is evidence the step is no longer activation- or
    recompute-bound and the next lever is expert_parallel_degree, which is 1
    today -- every rank runs grouped GEMMs across all 128 experts, so top-8
    routing leaves each expert only a slice of the tokens no matter how large
    the batch gets.

    If it OOMs, do NOT reach for --training.disable_cuda_graphs; that was tried
    on the no-AC branch (job 1169) and recovered only 272MiB, because a graph's
    private pool holds tensors the step needs anyway. Widen
    force_recompute_mm_shapes_by_fqns beyond the default ["moe.router.gate"]
    instead, or step down to bs=5.

    Measured ladder at 8192 context (8x B200, unboosted clocks):
        FullAC      bs=4 (job 1156)  17.65% MFU, 397.03 TF/GPU,  83.92GiB
        SelectiveAC bs=4 (job 1173)  18.16% MFU, 408.67 TF/GPU, 124.04GiB
        SelectiveAC bs=8 (job 1175)  OOM step 2,                176.45GiB peak
        no-AC       bs=1 (job 1163)  14.82% MFU, 333.42 TF/GPU, 137.91GiB
        no-AC       bs=2 (1165/1169) OOM step 2, graphs on and off
        no-AC       bs=4 (job 1157)  OOM step 1

    NOT YET MEASURED at bs=6 as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs4_selac()
    config.training.local_batch_size = 6
    return config


def qwen3_30b_a3b_8k_bs3_selac_noreshard() -> Trainer.Config:
    """SelectiveAC at bs=3, seq_len 8192, with fsdp_reshard_after_forward="never".

    Derived from qwen3_30b_a3b_8k_bs6_selac with local_batch_size lowered to 3
    (24,576 tokens/GPU) and FSDP resharding disabled.

    HISTORY: this function has been stepped down the batch ladder in place at
    the user's request rather than forked. It ran as
    qwen3_30b_a3b_8k_bs6_selac_noreshard (job 1177, OOM) and
    qwen3_30b_a3b_8k_bs5_selac_noreshard (job 1178, OOM) before this. Set
    local_batch_size back to 6 or 5 to reproduce either.

    The lever. FSDP frees the all-gathered parameters after forward and
    RE-GATHERS them in backward; "never" keeps them resident, removing an entire
    all-gather pass per step without changing the math or a single FLOP. It is
    the only lever in this sweep that targets communication rather than
    arithmetic.

    THE COST IS ~57GiB AND IT IS BATCH-INDEPENDENT. Parameters are all-gathered
    in bfloat16 (mixed_precision_param defaults to "bfloat16",
    config/configs.py:77), so a full unsharded copy is 30,532,122,624 * 2 B =
    56.87GiB per rank. Jobs 1177 and 1178 proved this empirically and in a way
    that is easy to misread:
        bs=6 (job 1177)  OOM, peak reached 176.51GiB (98.97%)
        bs=5 (job 1178)  OOM, peak reached 176.43GiB (98.92%)
    Cutting the batch by a sixth moved the observed peak by 0.08GiB, even though
    bs=5 has ~17GiB fewer activations. Both runs simply filled the card with
    resident parameters and died against the same ceiling. Those 176.x numbers
    are CENSORED measurements -- how far each got before failing, not what it
    needed -- so no increment can be fitted from them. An earlier attempt to do
    exactly that produced a bogus "+23GiB" figure and predicted bs=5 would fit
    at ~161GiB; it did not. Use the 56.87GiB theoretical value.

    PROJECTED TO BE MARGINAL. Using the solved SelectiveAC activation model
    (~14.0GiB batch-independent workspace + ~1.659MiB/token, from jobs 1173 and
    1175) plus the 56.9GiB fp32 params/grads/AdamW term:
        bs=3 base = 56.9 + 14.0 + 39.8 = ~110.7GiB, + ~57GiB -> ~168GiB
    against 178.35GiB usable, leaving only ~10GiB of margin. That is thin enough
    that OOM remains a live outcome, and thin enough that fragmentation could
    decide it.

    WHAT WOULD COUNT AS A WIN, and why it is a high bar: qwen3_30b_a3b_8k_bs6_selac
    measured 19.00% MFU (job 1176). Reaching bs=3 means giving up half that
    batch. bs=4 SelectiveAC measured 18.16% (job 1173), so bs=3 without the
    policy change would likely sit near 17.5%, and the removed all-gather would
    have to be worth ~1.5 points to break even. FSDP normally overlaps that
    collective with compute, so this is a genuine long shot -- but it is cheap
    to measure and it is the batch size where the policy first becomes
    affordable, so it is the right place to find out.

    Measured ladder at 8192 context (8x B200, unboosted clocks):
        FullAC      bs=4 (job 1156)  17.65% MFU, 397.03 TF/GPU,  83.92GiB
        SelectiveAC bs=4 (job 1173)  18.16% MFU, 408.67 TF/GPU, 124.04GiB
        SelectiveAC bs=6 (job 1176)  19.00% MFU, 427.44 TF/GPU, 154.21GiB
        SelectiveAC bs=8 (job 1175)  OOM step 2
        + noreshard bs=6 (job 1177)  OOM step 2
        + noreshard bs=5 (job 1178)  OOM step 2

    NOT YET MEASURED at bs=3 as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac()
    config.training.local_batch_size = 3
    config.parallelism.fsdp_reshard_after_forward = "never"
    return config


# ---------------------------------------------------------------------------
# 2026-08-28 experiment batch. Four independent levers, all branching from
# qwen3_30b_a3b_8k_bs6_selac (job 1176, 19.00% MFU, 427.44 TF/GPU, 154.21GiB),
# which is the best config measured so far. Each changes exactly ONE thing so
# the results are attributable.
#
# Baseline ladder they are all measured against (8x B200, 8192 ctx, unboosted):
#     stock       bs=2 4k FullAC       (job 1152)   9.66% MFU, 217.25 TF/GPU
#     8k_bs4      bs=4    FullAC       (job 1156)  17.65% MFU, 397.03 TF/GPU
#     8k_bs4_selac bs=4   SelectiveAC  (job 1173)  18.16% MFU, 408.67 TF/GPU
#     8k_bs6_selac bs=6   SelectiveAC  (job 1176)  19.00% MFU, 427.44 TF/GPU
#     bs3 selac noreshard (job 1179)               17.81% MFU, 400.73 TF/GPU
# ---------------------------------------------------------------------------


def qwen3_30b_a3b_8k_bs6_selac_compile() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs6_selac with torch.compile ENABLED.

    THE MODEL HAS NEVER BEEN COMPILED IN THIS SWEEP. CompileConfig.enable
    defaults to False (config/configs.py:288) and qwen3_30b_a3b never sets it,
    so every result from job 1152 through 1179 ran eager. This is easy to miss
    because the logs are full of Triton autotuning output for
    flex_attention -- but that is flex_attention generating its own kernels,
    which happens whether or not torch.compile is applied to the model. The only
    other "torch.compile" strings in those logs are pytree deprecation warnings.

    This is therefore the largest untested lever available, and it is orthogonal
    to everything tried so far: batch size, sequence length and AC policy all
    change WHAT work is done or when it is recomputed, while compilation changes
    how efficiently that work is emitted -- kernel fusion, fewer launches, less
    memory traffic between ops. On a MoE model with 48 layers of small
    elementwise work between the expert GEMMs, fusion has a lot to work with.

    compile.components defaults to ["model", "loss"], which is what is wanted
    here; only compile.enable needs setting.

    RISK: OOM. bs=6 SelectiveAC already peaks at 154.21GiB of 178.35GiB usable,
    leaving ~24GiB. Inductor generally REDUCES peak memory by fusing away
    intermediates, but it also allocates its own workspace and can change
    allocation patterns enough to fragment. If this OOMs, rerun at bs=4 (124.04
    GiB measured) rather than abandoning the lever -- an OOM here says nothing
    about whether compilation helps.

    Expect a longer startup: compiling 48 decoder layers takes minutes, and it
    happens before step 1. Do not mistake a long silent start for a hang; check
    outputs/structured_logs/*.jsonl, which is unbuffered.

    NOT YET MEASURED as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac()
    config.compile = CompileConfig(enable=True)
    return config


def qwen3_30b_a3b_8k_bs8_selac_compile() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs6_selac_compile with local_batch_size 6 -> 8.

    65,536 tokens/GPU, SelectiveAC, torch.compile enabled.

    This rung exists because COMPILATION REOPENED THE BATCH LADDER. bs=8 was
    tried uncompiled as qwen3_30b_a3b_8k_bs8_selac (job 1175) and OOMed on step
    2 at a 176.45GiB peak. Compilation then turned out to cut peak memory
    sharply while also being the single largest speed lever in the sweep:
        8k_bs6_selac         (job 1176)  19.00% MFU, 427.44 TF/GPU, 154.21GiB
        8k_bs6_selac_compile (job 1182)  25.15% MFU, 565.84 TF/GPU, 120.77GiB
    Same batch, same AC policy: +6.15 points of MFU for -33.44GiB. Inductor
    fuses away intermediates that the eager path materialises, so it is the only
    lever measured tonight that improved speed AND freed memory rather than
    trading one for the other.

    PROJECTED TO FIT COMFORTABLY. There is only one compiled data point, so the
    fixed/slope split cannot be solved yet; taking the pessimistic purely-linear
    reading of it, bs=6 compiled holds 120.77 - 56.9 = 63.87GiB of activations
    at 49,152 tokens, i.e. ~1.331MiB/token, so 65,536 tokens projects to
    ~85.2GiB of activations and ~142GiB total against 178.35GiB usable. Any
    real fixed component only lowers that. Note the compiled slope is well below
    the eager SelectiveAC slope of ~1.659MiB/token, which is the memory side of
    the same fusion effect.

    If this fits it also yields the second compiled point needed to solve the
    compiled memory model properly, and the ladder likely has another rung above
    it: bs=10 projects to ~163GiB on the same pessimistic reading.

    Expect a gain, but not a large one. Batch size was worth +0.84 points from
    bs=4 to bs=6 uncompiled, and returns were already flattening; compilation
    changes the efficiency of the emitted kernels, not the shape of the
    diminishing-returns curve in batch size. The value here is mostly in
    confirming that compile plus a large batch compose rather than interfere.

    NOT YET MEASURED as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac_compile()
    config.training.local_batch_size = 8
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs8_selac_compile with local_batch_size 8 -> 10.

    81,920 tokens/GPU, SelectiveAC, torch.compile enabled.

    COMPILATION RESTORED BATCH-SIZE SCALING, which is why this rung is worth
    running when the eager ladder had clearly flattened:
        eager  bs=6 (job 1176)  19.00% MFU, 154.21GiB
        eager  bs=7 (job 1184)  19.27% MFU, 169.12GiB   <- +0.27 for +14.9GiB
        compile bs=6 (job 1182) 24.74% MFU, 120.77GiB
        compile bs=8 (job 1188) 26.33% MFU, 139.82GiB   <- +1.59 for +19.1GiB
    Two steps of the eager ladder bought +0.27 points at 94.82% occupancy; two
    steps of the compiled ladder bought +1.59 at 78.40%. Fused kernels make the
    larger expert GEMMs pay off instead of saturating on memory traffic, so the
    curve that looked exhausted is not exhausted under compile.

    PROJECTED TO FIT. Two compiled points now give a proper fit instead of the
    single-point guess used for bs=8. Against the 56.9GiB batch-independent
    params/grads/AdamW term:
        bs=6 (job 1182)  120.77GiB total ->  63.87GiB activations @ 49,152 tok
        bs=8 (job 1188)  139.82GiB total ->  82.92GiB activations @ 65,536 tok
    That is +19.05GiB for +16,384 tokens, i.e. ~1.190MiB/token plus ~6.8GiB of
    batch-independent workspace. Extrapolating to 81,920 tokens gives ~102.1GiB
    of activations and ~159GiB total, roughly 19GiB clear of 178.35GiB usable.
    Note the compiled slope (~1.190MiB/token) is well under the eager
    SelectiveAC slope (~1.659MiB/token) -- the memory side of the same fusion
    effect -- and the compiled workspace term (~6.8GiB) is far smaller than
    eager's ~14.0GiB.

    This fit rests on two COMPLETED runs, which is the only kind that has
    predicted well in this sweep; every projection fitted from an OOM peak was
    wrong, because those peaks are censored measurements of how far a run got
    rather than what it needed.

    If it fits, bs=12 is the next rung (~183GiB projected, i.e. just over -- so
    bs=10 is likely the last one that fits).

    NOT YET MEASURED as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs8_selac_compile()
    config.training.local_batch_size = 10
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_bf16reduce() -> Trainer.Config:
    """Best config with the FSDP gradient reduce-scatter in bf16 instead of fp32.

    The only change from qwen3_30b_a3b_8k_bs10_selac_compile (job 1206: 26.72%
    MFU, 601.17 TF/GPU, 127,983 tok/s/node) is
    training.mixed_precision_reduce float32 -> bfloat16.

    NEVER TESTED IN THIS SWEEP. mixed_precision_reduce is not set by any qwen3
    config, so it resolved to "float32" in every job from 1152 to 1312. This is
    the last untried lever with a documented, quantified target.

    WHY IT SHOULD BE THE BIGGEST REMAINING ONE. This field only exists in this
    fork, and configs.py:85 explains why it was added: on Qwen3.5-122B the fp32
    gradient reduce-scatter measured 1,230ms of a 4,838ms step -- ~25% of the
    step and 65% of all NCCL time. Halving those bytes is a communication win, so
    unlike batch size it does not need more work per step to pay off, and unlike
    the AC and fusion experiments it does not touch compute at all. Nothing
    measured in this sweep has targeted gradient communication: no-AC and
    MemoryBudgetAC moved recompute, fsdp_reshard_after_forward="never" targeted
    the backward all-gather (and lost, because holding 56.87GiB of unsharded
    params forced the batch down), and MXFP8 moved GEMM precision.

    Expect a smaller share than the 122B figure. That measurement was on a much
    larger model where gradients are proportionally more expensive to reduce;
    30.5B params sharded over 8 ranks is a smaller reduce-scatter. Treat ~25% as
    an upper bound on the addressable fraction, not a prediction.

    THIS CHANGES TRAINING NUMERICS, and the field's own docstring is blunt about
    it: "Gradients reduce-scattered across 8 shards in bf16 accumulate rounding
    error that fp32 reduction is specifically there to avoid. Do not use it for a
    real run without comparing loss curves against a float32 control at the same
    --debug.seed." If this shows a throughput win, that seeded comparison is
    mandatory before use, not optional -- and the risk is higher here than for
    MXFP8 GEMMs, because reduction error compounds across steps rather than being
    re-quantized fresh each forward. Pair it with
    qwen3_30b_a3b_8k_bs10_selac_compile_seed42 as the fp32 control.

    MEASURED, AND IT IS THE ONE TO ADOPT. Matched 144-step window (13-156):
        job 1206  unseeded fp32 reduce   601.17 TF/GPU  26.72% MFU  161.77GiB
        job 1336  SEEDED   fp32 reduce   601.22 TF/GPU  26.72% MFU  161.77GiB
        job 1327  unseeded bf16 reduce   612.98 TF/GPU  27.24% MFU  160.78GiB
        job 1337  SEEDED   bf16 reduce   610.79 TF/GPU  27.15% MFU  160.78GiB
    So +1.6% tokens/sec and 26.72% -> 27.15% MFU, plus ~1GiB. Note how tightly
    the two fp32 controls agree (601.17 vs 601.22, both 26.72%) despite different
    seeds -- run-to-run throughput variation on this config is ~0.01%, so the
    gain is far outside noise.

    IT KEEPS MFU QUOTABLE, which the MXFP8 configs do not. mixed_precision_reduce
    does not set has_quantization, so metrics.py still divides by the bf16 peak
    legitimately -- nothing here changes GEMM precision. The MXFP8 wins are
    larger in tokens/sec but can only ever be reported as throughput. This is the
    only lever in the sweep that raises the headline MFU number.

    NUMERICS: CLEAN, and verified the way the field's own docstring demands. At
    seed=42, bf16 reduce against the fp32 control (jobs 1337 vs 1336):
        step  fp32      bf16      delta
          25  8.14452   8.12560   -0.019
          50  7.36861   7.41060   +0.042
          75  6.86892   6.86823   -0.001
         100  6.53468   6.48054   -0.054
         125  6.28686   6.36283   +0.076
         150  6.09964   6.17250   +0.073
    The sign alternates and there is no trend -- unbiased chaotic divergence from
    perturbed gradients, not systematic degradation. Contrast the MXFP8 configs,
    whose UNSEEDED gaps were monotonic and widening; those two signatures look
    genuinely different, which is why the seeded control was worth running.
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    config.training.mixed_precision_reduce = "bfloat16"
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_bf16reduce_seed42() -> Trainer.Config:
    """bf16 gradient reduce-scatter at seed=42, for the numerics comparison.

    Pairs with qwen3_30b_a3b_8k_bs10_selac_compile_seed42 (the fp32 control at
    the same seed). Reduction error compounds across steps, so this is the pair
    that actually prices the risk the field's docstring warns about.
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile_bf16reduce()
    config.debug.seed = 42
    return config


# ---------------------------------------------------------------------------
# MXFP8 on Qwen3-30B-A3B, branch qwen3_30b_a3b_mxfp8.
#
# Nothing in this sweep has run below bf16 -- every job from 1152 to 1287 was
# bf16 compute with fp32 master weights. The fork has MXFP8 experience on other
# models, and it is worth reading before interpreting these:
#   - gpt_oss 20b got +3.50% tokens/sec from MXFP8 on the DENSE linears
#     (commit 97d91a8e3).
#   - gpt_oss 120b could NOT use MXFP8 on the expert GEMMs at all: it died with
#     "AssertionError: K must be divisible by 128" inside torchao's
#     cutedsl_quantize_2d_1x32.py, because GPT-OSS has dim = hidden_dim = 2880
#     and 2880/128 = 22.5. pad_multiple pads the token (M) dimension, not the
#     contraction dimension K, so no knob fixes it.
#
# QWEN3-30B-A3B IS NOT SUBJECT TO THAT BLOCKER: dim = 2048 (2048/128 = 16) and
# moe_hidden_dim = 768 (768/128 = 6). Both contraction dims are clean multiples
# of 128, so the CuTeDSL path that was architecturally unusable on GPT-OSS
# should be usable here. torchao is 0.18.0 (>= 0.14 required) and B200 is sm_100.
#
# ...BUT THERE IS A SECOND BLOCKER THAT NO SHAPE AVOIDS. Clearing K % 128 is
# necessary and not sufficient. Job 1307 (the experts config below) died in
# torchao's CuTeDSL quantize kernel with:
#     cutedsl_quantize_2d_1x32.py:455 'cute_nvgpu.atom.tma_partition' op:
#     shared-memory operand stride at mode 1 is 16 bits, not a multiple of the
#     required 128-bit TMA smem alignment (element=16b, stride=1 elements)
# This was ALREADY investigated and settled in
# /data/dj-mat-torchtitan-mfu/RESULTS_KIMI3_MXFP8.md, "Finding 2: MXFP8 experts
# are blocked by a torchao/CuTeDSL bug, not by Kimi K3". That work established,
# with a standalone repro sweeping 12 cases:
#   - EVERY shape fails at both pad multiples, INCLUDING K=2048 -- exactly this
#     model's dim. The shape analysis above was never going to save us.
#   - It is not the cutlass-dsl dev build: stable 4.6.3 shadowed in via
#     PYTHONPATH fails all 12 cases too.
#   - The whole CuTeDSL quantize path is broken, not just the grouped/offset
#     variant. triton_to_mxfp8_dim0 works, MXFP8Linear fwd+bwd works, all
#     mxfp8_* grouped recipes fail. torchao 0.18.0 is already the latest release.
# The only remaining route is a hand-built Triton activation-quantize bypass
# feeding a pre-quantized MXTensor into _compute_fwd_sm100, which that document
# deliberately rejected: a custom quantization path with real numerics risk
# against a prototype torchao API, chasing a win the dense-half results argue
# against. DO NOT re-run the experts config expecting a different outcome; read
# RESULTS_KIMI3_MXFP8.md first.
#
# The DENSE half is a genuinely open question, and model-dependent:
#     gpt_oss 20b      +3.50% tokens/sec  (commit 97d91a8e3)
#     Kimi K3          -1.3% eager, -1.5% compiled  (jobs 1099, 1103)
#     Qwen3-30B-A3B    job 1306, this branch
# which is why qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn is worth running
# even though the experts path is closed.
#
# EP IS NOT REQUIRED, contrary to the gpt_oss_120b_mxfp8 docstring's claim that
# "EP must be enabled for the grouped-experts converter". Reading the code:
# MXFP8GroupedExpertsConverter.convert calls swap_token_dispatcher, which accepts
# a stock AllToAllTokenDispatcher.Config and swaps it for the padding-aware
# TorchAOTokenDispatcher.Config (components/quantization/utils.py:41-49). It only
# raises for dispatchers that cannot pad. Nothing checks expert_parallel_degree.
# That matters a lot here, because EP is unaffordable on this model -- standard
# EP OOMed at bs=6 (jobs 1186, 1207) and DeepEP ran but cost -3.78 points at
# bs=4 (job 1212). Quantizing at EP=1 sidesteps all of that.
#
# HOW TO READ THE RESULT: torchtitan computes MFU against the hardcoded bf16 peak
# (2.25e15 on B200), which is the wrong denominator once any GEMM is MXFP8, so
# metrics.py sets mfu=None and prints "mfu: N/A". num_flops_per_token is
# unchanged, so TFLOPs/GPU and tokens/sec remain directly comparable to the bf16
# baselines -- use those. summarize_tflops.sh was written for exactly this case.
#
# Baseline to beat (job 1206, bf16):
#     26.72% MFU, 601.17 TF/GPU, 15,998 tok/s/gpu, 127,983 tok/s/node, 161.77GiB
# ---------------------------------------------------------------------------


def qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn() -> Trainer.Config:
    """Best config + MXFP8 on the ATTENTION linears only. The safe rung.

    Mirrors the gpt_oss 20b change that measured +3.50% tokens/sec: quantize the
    dense linears and leave the MoE alone. No token-dispatcher swap, no padding
    requirement, so the only new machinery is MXFP8Linear in place of Linear.

    Attention is a bigger share of this model's work than its parameter count
    suggests, because MoE only activates 8 of 128 experts. Per layer the
    attention linears are the fused QKV (2048x5120 = 10.49M) plus wo
    (4096x2048 = 8.39M) = 18.87M, so 906M across 48 layers. The ACTIVE expert
    params are 8 experts x 3 matrices x 768x2048 x 48 layers = 1.81B. So
    attention is roughly 906M/2.72B = 33% of the active matmul work -- a third of
    the FLOPs, which is where a few percent end-to-end could come from.

    The router gate and lm_head stay bf16: fqns=["attention"] restricts the swap,
    and quantizing the router perturbs expert assignment while the 151936-wide
    lm_head feeds the loss.

    NOT YET MEASURED as of 2026-09-01.
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "30B-A3B",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention"],
            ),
        ],
    )
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn_lmhead() -> Trainer.Config:
    """MXFP8 on the attention linears AND lm_head. Extends the current best.

    qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn (job 1306) measured
    615.76 TF/GPU and 131,087 tok/s/node over a matched 144-step window, +2.4%
    over the bf16 best (job 1206, 601.17 TF/GPU, 26.72% MFU). This adds the one
    remaining dense GEMM it left in bf16.

    Enumerating every Linear.Config in this model gives four kinds:
        model.layers.N.attention.qkv_linear.wqkv   48   2048 x   5,120  = 503M
        model.layers.N.attention.wo                48   4096 x   2,048  = 403M
        model.lm_head                               1   2048 x 151,936  = 311M
        model.layers.N.moe.router.gate             48   2048 x     128  =  13M
    fqns=["attention"] reaches the first two, i.e. 906M of the 1,217M of dense
    Linear work. lm_head is the other 26% and is currently bf16. The router gate
    stays bf16 deliberately: it is 1% of the dense work and quantizing it
    perturbs expert assignment.

    THIS IS DIRECTLY EVIDENCED, not a guess. RESULTS_GPTOSS20B.md measured the
    same two steps on gpt_oss 20b:
        gptoss20b_mxfp8         MXFP8 attn            814.21 TF/GPU  +1.66%
        gptoss20b_mxfp8_lmhead  MXFP8 attn + lm_head  828.97 TF/GPU  +3.50%
    "lm_head roughly doubles the gain over attention alone, consistent with it
    being the largest dense GEMM." Loss was not degraded there (7.994 vs the
    control's 8.037 -- slightly better).

    Expect a SMALLER incremental gain here than gpt_oss's doubling, because the
    proportions differ. On gpt_oss the lm_head (2880 x 201,088 = 579M) was ~48%
    of dense Linear work; here it is 26%. So if the effect scales with share,
    +2.4% should become roughly +3.0-3.5%, not +4.8%.

    MEASURED (job 1310), matched 144-step window (steps 13-156) against the other
    two, which is the only fair comparison since job 1206 was cancelled at 144:
        bf16 baseline      (1206)  601.17 TF/GPU  127,983 tok/s/node    --
        MXFP8 attn         (1306)  615.76 TF/GPU  131,087 tok/s/node  +2.4%
        MXFP8 attn+lm_head (1310)  620.85 TF/GPU  132,171 tok/s/node  +3.3%
    So +3.3% over bf16 and +0.83% incremental over attention alone -- close to
    the +3.0-3.5% predicted from lm_head being 26% of dense Linear work here vs
    ~48% on gpt_oss.

    THE LOSS PENALTY IS THE OPEN QUESTION, AND IT IS NOT YET SETTLED. At matched
    steps the ordering is monotonic and the gap WIDENS with training:
        step  bf16      MXFP8 attn        MXFP8 attn+lm_head
          50  7.44559   7.46557 (+0.020)  7.50940 (+0.064)
         100  6.51796   6.54482 (+0.027)  6.58182 (+0.064)
         150  5.99936   6.04280 (+0.043)  6.08667 (+0.087)
    lm_head's penalty runs about 2x attention-only's throughout. That is the
    shape of slow drift rather than noise -- but debug.seed defaults to None
    (config/configs.py:349) and none of these runs set it, so the three jobs had
    different init and data order and part of the gap may be run-to-run
    variation. One run per config cannot separate the two. See the *_seed42
    configs below, which hold the seed fixed so any residual gap is
    attributable. Do not adopt this config for real training until that control
    has been read.
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "30B-A3B",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "lm_head"],
            ),
        ],
    )
    return config


# ---------------------------------------------------------------------------
# SEEDED CONTROLS for the MXFP8 loss question.
#
# debug.seed defaults to None (config/configs.py:349), so jobs 1206, 1306 and
# 1310 each had different random init and data ordering. Their loss gaps
# (+0.020 -> +0.043 for attn, +0.064 -> +0.087 for attn+lm_head, both widening
# with training) are therefore CONFOUNDED: the ordering is consistent enough to
# look like real quantization drift, but one unseeded run per config cannot
# separate drift from run-to-run variation.
#
# These three hold seed=42 fixed and are otherwise identical to their parents,
# so any residual loss gap between them is attributable to precision alone.
# Throughput conclusions do not need them -- seed does not affect tokens/sec --
# they exist purely to price the quality cost of the +2.4% and +3.3% wins.
#
# Run all three to the same step count and compare loss at matched steps.
# ---------------------------------------------------------------------------


def qwen3_30b_a3b_8k_bs10_selac_compile_seed42() -> Trainer.Config:
    """bf16 control at seed=42. Reference for the two MXFP8 seeded configs."""
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    config.debug.seed = 42
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn_seed42() -> Trainer.Config:
    """MXFP8 attention at seed=42. Pairs with qwen3_30b_a3b_8k_bs10_selac_compile_seed42."""
    config = qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn()
    config.debug.seed = 42
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn_lmhead_seed42() -> Trainer.Config:
    """MXFP8 attention + lm_head at seed=42. The config whose loss cost is in question."""
    config = qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8_attn_lmhead()
    config.debug.seed = 42
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_fp8_experts() -> Trainer.Config:
    """Float8 rowwise on the EXPERT grouped GEMMs. The big untouched prize.

    The experts are ~67% of this model's active matmul work (1.81B of 2.72B
    active params -- MoE activates 8 of 128 experts), and nothing has ever
    reached them below bf16. Attention-only MXFP8 bought +2.4% from the other
    33%; this targets the rest.

    Attention deliberately stays bf16 so this is single-variable against the
    bf16 best (job 1206, 601.17 TF/GPU, 26.72% MFU). If it works, stacking it
    with MXFP8 attn+lm_head is the obvious follow-up.

    WHY FLOAT8 AND NOT MXFP8 for the experts. The MXFP8 route is closed by a
    torchao/CuTeDSL TMA-alignment bug that no shape avoids -- job 1307 hit it
    here, and RESULTS_KIMI3_MXFP8.md proved it across 12 shapes including
    K=2048, across cutlass-dsl versions, and for the whole CuTeDSL quantize path
    rather than just the grouped variant. Float8 is a DIFFERENT code path:
    Float8GroupedExpertsConverter needs only 16-element alignment
    (PAD_MULTIPLE = 16, "16 byte alignment / 1 byte per elem") and dispatches
    through aten _scaled_grouped_mm, not the CuTeDSL quantize kernel.
    2048 % 16 == 0 and 768 % 16 == 0, so the shapes qualify.

    MEASURED: IT FAILS, AND THE ROUTE IS CLOSED. Job 1311 died at 0 steps with an
    async "CUDA error: unspecified launch failure"; job 1312 reran it under
    CUDA_LAUNCH_BLOCKING=1 and resolved it to
        RuntimeError: cutlass cannot run, error 7
    out of the aten _scaled_grouped_mm dispatch -- byte-identical to what
    RESULTS_GPTOSS20B.md recorded for gpt_oss. Conversion itself succeeds and
    logs "Converted GroupedExperts to use dynamic float8 rowwise quantization
    with scaled grouped GEMMs", so every documented pre-check passes and the
    failure is at execution.

    THIS GENERALISES THE PRIOR FINDING. RESULTS_GPTOSS20B.md scoped it to
    "gpt_oss's expert grouped-GEMM shapes" (K = 2880, not a power of two). The
    hypothesis here was that Qwen3's K = 2048 and 768 -- both powers of two --
    would land on a supported CUTLASS configuration. They do not:
        gpt_oss 20b        expert K=2880  N=2880   error 7
        Qwen3-30B-A3B      expert K=2048  N=768    error 7
    Two independent shape families, power-of-two and not, fail identically. The
    honest conclusion is that the Float8 grouped-GEMM path is broken on this
    stack (torch 2.15.0.dev20260817 / torchao 0.18.0) irrespective of shape, not
    that particular shapes are unsupported.

    KEPT AS A MINIMAL REPRODUCER rather than deleted. Both low-precision routes
    to the experts are now closed by two unrelated bugs -- MXFP8 by the CuTeDSL
    TMA-alignment failure above, Float8 by this -- which leaves 67% of the
    model's active matmul work at bf16. RESULTS_GPTOSS20B.md already names
    reporting this to torchao as the reasonable next step; this config plus job
    1312's log is the second data point for that report. Run it only to
    reproduce the bug, never expecting throughput.

    Note EP is NOT required, despite what gpt_oss_120b_fp8's docstring claims.
    Float8GroupedExpertsConverter.__init__ checks only torchao and SM89+; the
    RESULTS_GPTOSS20B "Correction to the 120b notes" section says the same. That
    matters because EP costs 3.78 points on this model (job 1212).

    MEASURED 2026-09-01: fails at 0 steps (jobs 1311, 1312).
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "30B-A3B",
        converters=[
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def qwen3_30b_a3b_8k_bs10_selac_compile_mxfp8() -> Trainer.Config:
    """Best config + MXFP8 on the attention linears AND the expert grouped GEMMs.

    The full-fat version, and the one with the real upside: the expert GEMMs are
    ~67% of the active matmul work (1.81B of 2.72B active params), so this is
    where MXFP8 pays if it pays at all. This is also the configuration that was
    IMPOSSIBLE on gpt_oss 120b -- see the block above for why Qwen3-30B-A3B's
    2048/768 dims clear the /128 contraction requirement that 2880 could not.

    pad_multiple=128, NOT the converter default of 32: the CuTeDSL quantization
    kernel on sm_100 requires 128 (the Config docstring says so, and deepseek_v3
    sets it for the same reason). The converter swaps the stock
    AllToAllTokenDispatcher for TorchAOTokenDispatcher, which does the padding;
    expert_parallel_degree stays 1.

    Kept at SelectiveAC rather than MemoryBudgetAC. The gpt_oss notes warn that
    the budget partitioner cannot size EP's process-group ScriptObject; that
    specific conflict does not apply at EP=1, but SelectiveAC is what every
    measured config in this sweep used, so holding it fixed keeps the comparison
    single-variable.

    MEMORY SHOULD IMPROVE, which may matter more than the speed. The baseline
    sits at 161.77GiB of 178.35GiB (90.71%), and MXFP8 activations and weights in
    the quantized regions are half the bytes of bf16. If that frees enough, bs=12
    -- previously projected at ~183GiB and therefore out of reach -- could come
    back into range, and batch size has been the single strongest lever on this
    model.

    Risks, in the order I expect them: a torchao/CuTeDSL kernel assertion on some
    shape (as on gpt_oss, though the dims say otherwise here); a torch.compile
    interaction with the quantized grouped GEMM; or numerics diverging enough
    that the loss curve looks wrong. Check loss against the bf16 baseline at the
    same step, not just throughput -- at step 400 job 1206-family runs were
    around 4.4.

    NOT YET MEASURED as of 2026-09-01.
    """
    config = qwen3_30b_a3b_8k_bs10_selac_compile()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "30B-A3B",
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

# ---------------------------------------------------------------------------
# MoE GATE/UP FUSION: TRIED AND REJECTED, 2026-08-31 (branch
# qwen3-30b-a3b-fused-grouped-experts, scrapped). No config kept.
#
# torchtitan/overrides/fused_swiglu.py ships FusedGroupedExperts, which replaces
# the stock w1_EFD/w3_EFD pair with a single w13 of shape
# (num_experts, hidden_dim, 2, dim) so ONE grouped GEMM computes both SwiGLU
# projections, plus a hand-written Triton torchtitan::silu_and_mul op. It is
# applied via override.imports and is used by the RL examples on a
# Qwen3-30B-A3B config, so it is known-compatible. Enable with:
#     config.override = OverrideConfig(
#         imports=["torchtitan.overrides.fused_swiglu.fused_grouped_experts"])
#
# Measured against qwen3_30b_a3b_8k_bs10_selac_compile over the SAME step range
# (13-156, n=144 each -- the baseline job 1206 was cancelled at 144 steps, and
# these runs drift down as gc outliers accumulate, so matched ranges matter):
#     baseline  (job 1206)  26.72% MFU, 601.17 TF/GPU, 161.77GiB
#     fused MoE (job 1287)  26.36% MFU, 593.17 TF/GPU, 158.43GiB
# i.e. -0.36 points and -8 TFLOPs, inside the 19.9-29.6% per-step spread, plus a
# real and consistent -3.34GiB. Job 1287's full 400-step average was 26.47%.
#
# The prediction that motivated it was that torch._grouped_mm is an opaque extern
# kernel, so inductor CANNOT merge the two up-projection GEMMs the way it merges
# plain nn.Linear work. That reasoning about inductor was right; the conclusion
# that the merge would therefore pay was wrong. At 128 experts / top-8 and
# 81,920 tokens/GPU, each expert GEMM already has enough M that doubling N from
# 768 to 1536 barely moves tensor-core occupancy, and 48 saved kernel launches
# per forward is noise against ~5s steps.
#
# Taken with the fused-QKV null result (see the block above), this is the second
# independent test showing that HAND-fusion in the model definition is not where
# the remaining MFU is once torch.compile is on. The memory saving is genuine but
# buys nothing -- 3.34GiB does not open a new batch rung (bs=12 needs ~183GiB).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FUSED QKV: ALREADY ON, AND WORTH NOTHING MEASURABLE UNDER COMPILE.
# Tested 2026-08-31 on scrapped branch qwen3_30b_a3b_nonfused; no configs kept,
# because they need a "30B-A3B_non_fused_qkv" model flavor that went with the
# branch. Recorded here so it is not re-investigated.
#
# Fusion was never off. _30b_a3b passes fuse_qkv=True, as does every real Qwen3
# flavor, so the 9.66% stock baseline and the 26.72% best config both had it.
# qwen3_debugmodel_non_fused_qkv is a COVERAGE test for the unfused path, not an
# optimisation to opt into. There was never any MFU to gain by "enabling" it.
#
# The reverse test, unfusing at matched batch with compile on:
#     bs=8  fused (job 1188)  26.33% MFU, 139.82GiB   | unfused (1214)  26.44%, 138.73GiB
#     bs=10 fused (job 1206)  26.72% MFU, 161.77GiB   | unfused (1215)  26.69%, 157.23GiB
# i.e. +0.11 and -0.03 points -- and the bs=8 pair came out with UNFUSED AHEAD.
# Per-step MFU spans 19.3-29.8% on these configs, so deltas of 0.03-0.11 are
# about 1% of the noise band. This is a null result, not a small effect. Earlier
# reads at 42-65 steps suggested unfused was 0.05-0.20 BEHIND; both reversed as
# steps accumulated, which is the cautionary detail worth keeping.
#
# Interpretation: torch.compile already captures the win. The expected cost of
# unfusing -- 96 extra small GEMMs per forward pass across 48 layers, two of them
# narrow at out=512 -- is worth ~0 once inductor fuses and schedules them. So
# this measures what HAND-fusion is worth ON TOP OF compilation. The eager-mode
# difference is probably real and was not measured; the same pair with
# compile.enable=False would isolate it, and is cheap.
#
# The one reproducible difference is memory, in the direction NOT predicted:
# unfused is LIGHTER by 1.09GiB at bs=8 and 4.54GiB at bs=10, plausibly because
# the fused path materialises the full 5120-wide output before splitting it into
# Q/K/V while unfused writes three independently-freeable tensors. Not enough to
# open a new batch rung (bs=12 needs ~183GiB), and not worth diverging from the
# upstream default for.
#
# Implementation note if this is ever redone: QKVLinear.Config lists only wq
# (2048x4096) and wkv (2048x512), which LOOKS like a combined KV projection. It
# is not -- wkv is a template that build() instantiates twice as separate wk and
# wv weights (models/common/attention.py:756-758), and forward runs three GEMMs.
# Parameter count matches the fused 2048x5120 exactly: 4096 + 512 + 512 = 5120.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EXPERT PARALLELISM: TRIED AND REJECTED, 2026-08-31. No configs kept, because
# they cannot run without the scrapped .venv-deepep, but the result is worth not
# re-deriving.
#
# EP=8 with the default "standard" AllToAllTokenDispatcher never ran at all:
# OOM before step 1 at bs=6, eager (job 1186) and compiled (job 1207). Its
# dispatch/combine buffers scale with the tokens arriving from ALL ranks. It also
# cannot use CUDA graphs -- torchtitan raises for any EP dispatcher that does a
# CPU sync (job 1183).
#
# DeepEP v2 (deep-ep 2.1.0, ElasticBuffer) DID run, in a throwaway venv on
# branch qwen3-30b-a3b-deepep-attempt. Its fixed-capacity buffer really is a
# leaner memory profile -- but expert parallelism still lost, at matched batch:
#     bs=4 SelectiveAC, no EP  (job 1173)  18.16% MFU, 408.67 TF/GPU, 124.04GiB
#     bs=4 SelectiveAC, EP=8   (job 1212)  14.38% MFU, 323.57 TF/GPU, 168.07GiB
# i.e. -3.78 points of MFU for +44GiB. bs=6 OOMed even with DeepEP (job 1213).
# Note the ElasticBuffer GROWS over the first ~30 steps (130.64 -> 168.07GiB on
# job 1212) as routing patterns vary, so an EP run that survives step 1 can still
# OOM later -- that is how 1213 died.
#
# Two further blockers if anyone revisits this:
#   - DeepEP is INCOMPATIBLE with torch.compile here. deepep::dispatch has a CUDA
#     impl and autograd registration but no fake/meta impl, so tracing raises
#     NotImplementedError (jobs 1210, 1211). Since compile is worth +5.70 points
#     on this model, eager EP starts far behind before the -3.78 above.
#   - A fake impl is not small: dispatch returns a received-token count that
#     depends on runtime routing, so it needs unbacked symints or DeepEP v2's
#     static no-host-sync layout (the ``cudagraphable`` flag on
#     DeepEPTokenDispatcher.Config).
#
# The build recipe, which took some finding, is preserved in the comments of
# /data/dj-mat-torchtitan-mfu/run_qwen3_deepep.sbatch. The short version: PyPI
# deep_ep 1.0.0 is an incomplete sdist and cannot build; install from
# git+https://github.com/deepseek-ai/DeepEP.git with --no-build-isolation, ninja
# installed, and NVSHMEM_DIR pointing at the nvidia-nvshmem-cu13 wheel.
# ---------------------------------------------------------------------------


def qwen3_30b_a3b_8k_bs6_selac_ep8() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs6_selac with expert_parallel_degree 1 -> 8.

    The only structural lever left: it changes the SHAPE of the expert GEMMs
    rather than how much work is done or where it is stored.

    Today every config runs expert_parallel_degree=1, so each rank materialises
    all 128 experts and runs grouped GEMMs across all of them. With top-8
    routing, each expert on each rank sees only about 8/128 of that rank's
    tokens -- at bs=6 that is 49,152 * 8 / 128 = ~3,072 tokens per expert, which
    is a thin M dimension for a 2048x768 GEMM. Expert parallelism instead gives
    each rank 128/8 = 16 experts and routes every rank's tokens to the rank
    that owns the chosen expert, so each expert GEMM sees roughly 8x the rows.
    Bigger M is exactly what has been driving every gain in this sweep; batch
    size was the crude way of buying it, and this is the direct way.

    The cost is an all-to-all dispatch and combine per MoE layer, replacing
    local compute with communication. Whether that trade wins is genuinely
    open -- on a single node with NVLink the all-to-all is cheap, which is the
    best case for this lever.

    Mesh constraint (config/configs.py:275): dp_shard * cp * tp == efsdp * ep.
    Here dp_shard=-1 resolves to 8, cp=1, tp=1, so 8 == efsdp * 8 gives
    efsdp=1 -- EP borrows all of the FSDP sharding in the expert region. That is
    legal, and expert weights end up sharded 8 ways by EP instead of by FSDP,
    so parameter memory should be roughly unchanged. Unlike every other lever
    tried tonight this one does not obviously COST memory, which is why it is
    worth trying at the full bs=6.

    CUDA GRAPHS MUST BE OFF, and that is a forced confound. Job 1183 (this
    config without the flag) died at startup:
        ValueError: CUDA graphs support only expert parallel token dispatcher
        configurations without CPU synchronization. Set HybridEP
        non_blocking_capacity_factor, or use MinimalAsyncEP, or set
        --training.disable_cuda_graphs. Unsupported token dispatcher:
        AllToAllTokenDispatcher.Config.
    The default "standard" comm backend uses AllToAllTokenDispatcher, whose
    dispatch does a CPU sync, so graph capture is impossible. Of the three
    escapes the error offers, disable_cuda_graphs is taken here because it needs
    no model_spec surgery. This means the run differs from
    qwen3_30b_a3b_8k_bs6_selac in TWO ways -- expert parallelism AND no CUDA
    graphs -- so a loss against the 19.00% baseline cannot be attributed to EP
    alone. Note this is a COMPATIBILITY requirement, unrelated to the
    memory-recovery theory that job 1169 disproved.

    If EP looks promising, the clean follow-up is
    model_registry("30B-A3B", moe_comm_backend="minimal_async_ep"), which the
    error names as cudagraph-compatible and which would restore graphs and make
    this a single-variable comparison. See build_token_dispatcher_config in
    models/common/config_utils.py:370 for the backend list; "deepep" and
    "hybridep" additionally require a DeepEP install.

    RETRY WITH COMPILE, 2026-08-28. Job 1186 (this config without compile) OOMed
    before step 1 in token_dispatcher.py:617 combine. The reasoning in the
    paragraph above -- that EP "does not obviously COST memory" -- was WRONG. It
    holds for parameters, which end up sharded 8 ways by EP instead of by FSDP,
    but it ignores activations: with ep=8 each rank owns 16 experts and receives
    the tokens routed to them from ALL 8 ranks, so the dispatch/combine buffers
    are large and scale with total tokens, not per-rank tokens. EP trades
    parameter memory for communication-buffer memory.
    torch.compile is therefore enabled here: it measured -33.44GiB at bs=6
    (154.21GiB eager, job 1176 -> 120.77GiB compiled, job 1182), which is the
    headroom the dispatch buffers need. It also makes this a THREE-variable run
    against the eager 19.00% baseline (EP + no CUDA graphs + compile), so read
    it against qwen3_30b_a3b_8k_bs6_selac_compile (24.74%, job 1182), which
    differs from it only by EP and the graph flag.

    NOT YET MEASURED with compile as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac()
    config.parallelism.expert_parallel_degree = 8
    config.training.disable_cuda_graphs = True
    config.compile = CompileConfig(enable=True)
    return config


def qwen3_30b_a3b_8k_bs7_selac() -> Trainer.Config:
    """qwen3_30b_a3b_8k_bs6_selac with local_batch_size 6 -> 7.

    Finishes the SelectiveAC batch ladder. bs=6 fits at 154.21GiB and bs=8 OOMed
    on step 2 (job 1175), so bs=7 is the only untested rung and it decides where
    the ceiling actually is.

    PROJECTED TO FIT, using the solved SelectiveAC model (~14.0GiB
    batch-independent workspace + ~1.659MiB/token, fitted from the two COMPLETED
    runs 1173 and 1176) plus the 56.9GiB fp32 params/grads/AdamW term:
        57,344 tokens -> ~92.9GiB activations -> ~163.8GiB total
    against 178.35GiB usable, so ~14.5GiB of margin. That model predicted bs=6
    within 2.4% and bs=3 within 4.3GiB, which is the best track record of any
    projection in this sweep -- the ones that failed were all fitted from OOM
    peaks, which are censored measurements rather than true requirements.

    EXPECT A SMALL GAIN. bs=4 -> bs=6 was worth +0.84 points (18.16% -> 19.00%),
    so a further 1/6 increase in tokens should be worth a few tenths at most.
    This is the cheap completion of a ladder rather than a promising lever, and
    if it lands flat that is itself the answer: batch size is exhausted and the
    remaining upside is in compilation or expert parallelism.

    NOT YET MEASURED as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac()
    config.training.local_batch_size = 7
    return config


def qwen3_30b_a3b_8k_bs6_membudget_compile() -> Trainer.Config:
    """bs=6, seq_len 8192, MemoryBudgetAC instead of SelectiveAC, compile ON.

    Replaces the hand-picked SelectiveAC op list with the compiler partitioner's
    own memory-vs-compute tradeoff (distributed/activation_checkpoint.py:290).
    memory_budget is set to 0.8: 0.0 is the activation memory of full AC and 1.0
    is the runtime-optimal strategy, so 0.8 leans towards keeping activations
    and recomputing little -- which is the direction the measured ladder points,
    since FullAC (17.65%) < SelectiveAC (18.16%) and only OOM stopped no-AC.

    This config changes TWO things at once relative to qwen3_30b_a3b_8k_bs6_selac
    (AC policy and compilation), which is deliberate but does mean it is not
    attributable on its own. MemoryBudgetAC REQUIRES a compiled model --
    trainer.py:171 raises ValueError otherwise -- so compilation is not
    optional here. Read it against qwen3_30b_a3b_8k_bs6_selac_compile, which
    holds compilation fixed and keeps SelectiveAC; the difference between those
    two isolates the AC policy under compilation.

    visualize_memory_budget_pareto is enabled: it dumps an SVG of expected
    runtime versus activation memory for every budget from 0 to 1 in 0.05 steps,
    into {dump_folder}/memory_budget_pareto. That maps the WHOLE tradeoff curve
    from a single run, which is far cheaper than one job per AC policy -- the
    approach that has consumed most of this sweep. Use the curve to pick the
    next budget rather than guessing.

    RISK: OOM, and less predictable than usual. The budget targets activation
    memory but the exact peak depends on what the partitioner chooses, and bs=6
    leaves only ~24GiB of headroom. If it OOMs, lower memory_budget (towards
    full AC) rather than lowering the batch -- the budget is precisely the knob
    for this.

    NOT YET MEASURED as of 2026-08-28.
    """
    config = qwen3_30b_a3b_8k_bs6_selac()
    config.compile = CompileConfig(enable=True)
    config.activation_checkpoint = MemoryBudgetAC.Config(
        # 0.8 OOMed before step 1 (job 1185, 175.47GiB allocated, ~0.5GiB
        # short): it leans towards KEEPING activations and consumed all the
        # headroom that SelectiveAC+compile had freed. 0.5 is the library
        # default and the documented way to back off -- lower the budget, not
        # the batch, since the budget is precisely this knob.
        memory_budget=0.5,
        # DISABLED. Job 1208 died in inductor with
        #   BackendCompilerFailed: ModuleNotFoundError: No module named 'matplotlib'
        # The pareto dump imports matplotlib, which is not installed in
        # /data/dj-mat-torchtitan-mfu/.venv, so enabling this fails the COMPILE
        # rather than just skipping the plot. Re-enable only after installing
        # matplotlib into that venv; the curve is worth having, since it maps
        # runtime vs activation memory for every budget 0..1 in one run.
        visualize_memory_budget_pareto=False,
    )
    return config


def qwen3_32b() -> Trainer.Config:
    model_spec = model_registry("32B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def qwen3_debugmodel_non_fused_qkv() -> Trainer.Config:
    # Reverse test: exercise the separate wq/wk/wv path now that fused QKV is
    # the debugmodel default.
    config = qwen3_debugmodel()
    config.model_spec = model_registry("debugmodel_non_fused_qkv")
    return config


def qwen3_moe_debug() -> Trainer.Config:
    model_spec = model_registry("debugmodel_moe")
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
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen3_moe_deepep() -> Trainer.Config:
    """Qwen3 debug MoE pretraining with the DeepEP v2 backend (compact training path), EP=4.

    The MoE expert dispatch uses the DeepEP v2 ElasticBuffer all-to-all; under autograd it
    takes the compact, host-synced, backward-able path. EP=4 (4 GPUs) so the dispatch is
    actually exercised (EP=1 falls back to local); the training shape determines the fixed
    per-rank buffer capacity. Numerics match the standard all-to-all backend (step-1 bitwise,
    reduction-order drift thereafter). Needs deep_ep v2 (ElasticBuffer) in the env.

    Local devgpu (no RDMA NIC) needs these env vars so the ElasticBuffer inits NVLink-only:
      - EP_DISABLE_GIN=1            skip the NCCL GIN / RDMA requirement (no RDMA NIC)
      - EP_REUSE_NCCL_COMM=0        avoid the ElasticBuffer null-device-comm segfault
      - NVSHMEM_REMOTE_TRANSPORT=none + NVSHMEM_DISABLE_MNNVL=1   intra-node NVLink only
      - LD_LIBRARY_PATH must include the deep_ep wheels' nvshmem + nccl lib dirs
    Then launch with NGPU=4 ./run_train.sh (none of this is needed on RDMA/RoCE hosts).
    """
    model_spec = model_registry("debugmodel_moe", moe_comm_backend="deepep")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(expert_parallel_degree=4),
        checkpoint=CheckpointManager.Config(
            interval=1000, last_save_model_only=False, export_dtype="float16"
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def sft_qwen3_8b_math() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K math dataset."""

    def process_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    model_spec = model_registry("8B", attn_backend="varlen")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_spec,
        optimizer=default_adamw(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=15,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=2048,
            steps=180,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
