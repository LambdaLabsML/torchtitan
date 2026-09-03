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
