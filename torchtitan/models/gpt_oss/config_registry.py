# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
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
