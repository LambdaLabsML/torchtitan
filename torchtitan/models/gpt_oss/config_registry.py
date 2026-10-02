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
    OptimizersContainer,
    ParamGroupConfig,
    default_adamw,
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


# ---------------------------------------------------------------------------
# Take 2 (2026-10-02): the GPT-OSS-20B / Qwen3-30B-A3B take-2 wins and the
# applicable GB300 split/* units, ported onto the 120b best,
# gpt_oss_120b_bf16reduce_lr3e4 (job 2307: 661.16 mean / 684.26 median TF/GPU
# over steps 101-3820, 646.41 / 676.34 over steps 200-1500, 169.13GiB).
# Every config below is one variable off that config, read against a same-day
# control over a fixed step window (benchmarks/gpt_oss_120b/window_tflops.sh).
# ---------------------------------------------------------------------------


def _enable_expert_bias_grad_gemm(config: Trainer.Config) -> Trainer.Config:
    for layer in config.model_spec.model.layers:
        layer.moe.routed_experts.inner_experts.bias_grad_gemm = True
    return config


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
    """
    return _enable_expert_bias_grad_gemm(gpt_oss_120b_bf16reduce_lr3e4())
