import os
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.deepseek_v3.mtp import apply_fsdp_to_mtp_decoder


import logging

logger = logging.getLogger(__name__)


def parallelize_deepseekv3(
    model: Decoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    if parallelism.fp8_expert_all_gather:
        # Before model.parallelize / FSDP, on the meta model: FSDP then shards
        # the wrapped parameter and finds the all-gather extension hooks.
        from torchtitan.distributed.fp8_allgather import (
            wrap_expert_weights_for_fp8_all_gather,
        )

        wrap_expert_weights_for_fp8_all_gather(model)
    model.parallelize(parallel_dims)

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if os.environ.get("FP8_DENSE_COMPILE", "0") == "1":
        # Compile each Float8Linear on its own (the block itself stays eager:
        # whole-block compile graph-breaks in the SPMD typecheck context) so
        # torchao's per-call amax/scale/cast kernels fuse around the fp8 GEMM.
        from torchtitan.quantization.float8 import Float8Linear

        n = 0
        for _name, module in model.named_modules():
            if Float8Linear is not None and isinstance(module, Float8Linear):
                module.compile(dynamic=False)
                n += 1
        logger.info("fp8 dense: compiled %d Float8Linear modules", n)

    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (mode="none", enable=False).
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    apply_fsdp_to_mtp_decoder(
        # pyrefly: ignore [bad-argument-type]
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )

    if parallelism.fp8_expert_all_gather:
        from torchtitan.distributed.fp8_allgather import debug_log_fsdp_expert_storage

        debug_log_fsdp_expert_storage(model)
    return model
