"""Dense fp8/fp4 linear backed by Transformer Engine's ``te.Linear``.

Why: on GB300 (sm_103) TE's fused casts + cuBLASLt epilogues beat both
torchao's Float8Linear and the torchao-free custom fp8 linear on every dsv4
dense shape (job 837: per-set fwd+bwd 20.8 ms bf16, 14.95 custom, 14.22 TE
current-scaling, 13.50 MXFP8, 12.81 delayed-scaling, 10.88 NVFP4).

How: the torchtitan ``Linear`` keeps owning the parameter (so FSDP2, the
optimizer and the state dict are untouched); a ``te.Linear`` is created
lazily on the first CUDA forward, kept OUT of the module tree (no duplicate
parameter), and its ``weight`` is re-pointed at whatever Parameter object
FSDP2 exposes at forward time (sharded vs unsharded objects differ). TE's
autograd Function writes the weight grad into that same Parameter, which is
what FSDP2's post-backward reduce-scatters.

Recipe: ``TE_DENSE_RECIPE`` = mxfp8 (default) | delayed | current | nvfp4.
"""

from __future__ import annotations

import dataclasses
import os

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear

_RECIPE_NAME = os.environ.get("TE_DENSE_RECIPE", "mxfp8")
_recipe = None


def get_recipe():
    global _recipe
    if _recipe is None:
        from transformer_engine.common.recipe import (
            DelayedScaling,
            Float8CurrentScaling,
            MXFP8BlockScaling,
            NVFP4BlockScaling,
        )

        _recipe = {
            "mxfp8": MXFP8BlockScaling,
            "delayed": DelayedScaling,
            "current": Float8CurrentScaling,
            "nvfp4": NVFP4BlockScaling,
        }[_RECIPE_NAME]()
    return _recipe


class TELinear(Linear):
    """``Linear`` whose CUDA bf16 forward/backward run through ``te.Linear``."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds TELinear."""

    def __init__(self, config: Config):
        super().__init__(config)
        object.__setattr__(self, "_te_mod", None)  # unregistered on purpose

    def _te(self, w: torch.Tensor):
        import transformer_engine.pytorch as te

        mod = self.__dict__.get("_te_mod")
        if mod is None:
            mod = te.Linear(
                self.in_features,
                self.out_features,
                bias=False,
                params_dtype=w.dtype,
                device=w.device,
            )
            object.__setattr__(self, "_te_mod", mod)
        if mod.weight is not w:
            mod.weight = w  # our Parameter (FSDP2's current view of it)
        return mod

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if not input.is_cuda or input.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            return F.linear(input, w, self.bias)
        import transformer_engine.pytorch as te

        mod = self._te(w)
        x2 = input.reshape(-1, input.shape[-1])
        with te.autocast(enabled=True, recipe=get_recipe()):
            y = mod(x2)
        y = y.view(*input.shape[:-1], w.shape[0])
        if self.bias is not None:
            y = y + self.bias
        return y


def convert_linear_config(linear_config: Linear.Config) -> TELinear.Config:
    """Rebuild a Linear.Config as a TELinear.Config with the same fields."""
    kwargs = {
        f.name: getattr(linear_config, f.name)
        for f in dataclasses.fields(linear_config)
        if f.init
    }
    return TELinear.Config(**kwargs)
