# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import functools
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.tools.logging import logger, warn_once


_layer_sac_count = 0


def _apply_layer_sac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply layer selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply layer selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with layer selective activation checkpointing applied.
    """
    global _layer_sac_count
    _layer_sac_count += 1
    ac_freq = int(ac_config.selective_ac_option)
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
    else:
        return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    """Apply selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        base_fqn (str, optional): The base fqn of the module. Defaults to None.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.

    Returns:
        nn.Module: The module with selective activation checkpointing applied.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    mm_recompute_shapes = set()
    if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
        for module_fqn, submod in module.named_modules():
            fqn = module_fqn
            if base_fqn is not None:
                fqn = f"{base_fqn}.{module_fqn}"
            if not any(
                filter_fqn in fqn
                for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
            ):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))
        logger.debug(
            f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
        )

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                if args[1].shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm
            to_save = func in op_sac_save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )

_LOGGED = {}
_NUM_RECOMPUTED = {}

def _apply_policy_ac(
    model: nn.Module,
    ac_config: ACConfig,
) -> nn.Module:
    keep_ratios = {
        name.replace("-", "."): ratio
        for name, ratio in ac_config.keep_ratios.items()
    }

    totals = {name: 0 for name, _ in keep_ratios.items()}
    def _count(parent, name, fqn, submodule):
        assert getattr(parent, name) == submodule
        target_pqn = None
        for pqn, _ in keep_ratios.items():
            if (pqn in fqn or pqn in submodule.__class__.__name__):
                target_pqn = pqn
                break
        if target_pqn is not None:
            totals[target_pqn] += 1
        for child_name, child in submodule.named_children():
            _count(submodule, child_name, fqn + "." + child_name, child)
    for name, child in model.named_children():
        _count(model, name, name, child)

    num_recomputes = {
        pqn: totals[pqn] - int(ratio * totals[pqn])
        for pqn, ratio in keep_ratios.items()
    }

    logger.info(f"AC Recompute First N Policy: {num_recomputes}")

    def _apply_module_ac(parent, name, fqn, submodule):
        assert getattr(parent, name) == submodule

        target_pqn = None
        for pqn, num in num_recomputes.items():
            if (pqn in fqn or pqn in submodule.__class__.__name__) and num > 0:
                target_pqn = pqn
                break
        
        if target_pqn is not None:
            if torch.distributed.get_rank() == 0:
                logger.info(f"Checkpointing `{fqn}` (via `{target_pqn}`)")
            num_recomputes[target_pqn] -= 1
            submodule = ptd_checkpoint_wrapper(
                submodule,
                preserve_rng_state=ac_config.preserve_rng_state,
                determinism_check=ac_config.determinism_check,
                early_stop=ac_config.early_stop,
                debug=ac_config.debug,
            )
            parent.register_module(name, submodule)
        else:
            for child_name, child in submodule.named_children():
                _apply_module_ac(submodule, child_name, fqn + "." + child_name, child)

    for name, child in model.named_children():
        _apply_module_ac(model, name, name, child)

def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply full activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with full activation checkpointing applied.
    """
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_op_sac_to_transformer_block_with_flex(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    """Apply SAC to the transformer block that uses FlexAttention.

    Args:
        module (nn.Module): The transformer block to apply SAC to.
        ac_config (ACConfig): The Activation Checkpoint config.
        base_fqn (str, optional): The base fqn of the module. Defaults to None.
        model_compile_enabled (bool): Whether model compilation is enabled.
            Defaults to False.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.

    Returns:
        nn.Module: The transformer block with SAC applied.
    """

    warn_once(
        logger,
        (
            "Flex Attention requires compilation for good performance.\n"
            "Thus, torch.compile is always used for Flex Attention, "
            "regardless of the compile.enable flag.\n"
            "However, when selective activation checkpointing (SAC) is enabled, "
            "torch.compile may be invalidated:\n"
            "1. If compile.enable is False, SAC will ignore any torch.compile "
            "inside the SAC region.\n"
            "2. If compile.enable is True but the transformer block contains an MoE module.\n\n"
            "For both cases, we will not wrap the entire TransformerBlock with SAC:\n"
            "   - For case 1: SAC will be used for MoE and FeedForward modules, "
            "while full AC will be used for the Attention module.\n"
            "   - For case 2: SAC will be applied to MoE and Attention modules if the block "
            "is sparse. But we still apply SAC to an entire dense block.\n"
        ),
    )

    def wrap_submodule(name: str, full_ac: bool = False) -> None:
        submodule = getattr(module, name)
        if full_ac:
            submodule = _apply_full_ac(submodule, ac_config)
        else:
            submodule = _apply_op_sac(
                submodule,
                ac_config,
                base_fqn=f"{base_fqn}.{name}" if base_fqn else name,
                op_sac_save_list=op_sac_save_list,
            )
        module.register_module(name, submodule)

    if hasattr(module, "moe"):
        wrap_submodule("moe", full_ac=False)
        if model_compile_enabled:
            wrap_submodule("attention", full_ac=False)
        else:
            wrap_submodule("attention", full_ac=True)
    else:
        if model_compile_enabled:
            module = _apply_op_sac(
                module,
                ac_config,
                base_fqn=base_fqn,
                op_sac_save_list=op_sac_save_list,
            )
        else:
            wrap_submodule("feed_forward", full_ac=False)
            wrap_submodule("attention", full_ac=True)
    return module


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    num_layers: int | None = None
) -> nn.Module:
    valid_ac_modes = ("full", "selective", "policy")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    if ac_config.mode == "policy":
        return _apply_policy_ac(module, ac_config, num_layers)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        op_sac_save_list = op_sac_save_list or set()
        if use_flex_attn:
            """
            For Flex Attention, we need to apply SAC carefully to avoid invalidating
            torch.compile. Any torch.compile inside the SAC region will be ignored,
            and any torch.compile outside the SAC region will also be ignored if the
            SAC region contains a graph break (e.g., MoE).

            TODO: remove this once SAC issues are resolved.
            """
            return _apply_op_sac_to_transformer_block_with_flex(
                module,
                ac_config,
                base_fqn=base_fqn,
                model_compile_enabled=model_compile_enabled,
                op_sac_save_list=op_sac_save_list,
            )
        else:
            return _apply_op_sac(
                module, ac_config, base_fqn=base_fqn, op_sac_save_list=op_sac_save_list
            )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Note that SAC, Flex Attention and model compilation have some conflicts.
    We explicitly ask the user to pass these configs to warn as the wrapping
    will be different.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        use_flex_attn (bool): Whether flex attention is enabled for the model.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.
    Returns:
        None
    """
    if ac_config.mode == "policy":
        _apply_policy_ac(model, ac_config)
    elif ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        if ac_config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        logger.info(f"Selected {ac_config.memory_budget} budget option")
    else:
        num_layers = len(model.layers)
        for layer_id, transformer_block in model.layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block,
                ac_config,
                base_fqn=f"layers.{layer_id}",
                model_compile_enabled=model_compile_enabled,
                use_flex_attn=use_flex_attn,
                op_sac_save_list=op_sac_save_list,
                num_layers=num_layers
            )
            model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
