# torch.compile on DeepSeek V4 Pro / GB300

Branch `deepseek_v4_compile_fix`. Config `deepseek_v4_pro_64xgb300_compile`.

## Model compile is blocked

Enabling compile with its default `components=["model", "loss"]` fails in
Inductor. Reproduced at debug scale on 4x GB300 with the plain
`deepseek_v4_debugmodel` and **empty `kernel_options`**, so it is not caused by
the pinned attention tiles from the baseline work:

```
torch._inductor.exc.InductorError: LoweringException:
AssertionError: convert FlexibleLayout to FixedLayout first
  at torchtitan/models/common/attention.py:304 in compiled_flex_attn
```

### Two fixes attempted, both reverted

**1. Skip the nested `torch.compile`.** `FlexAttention` holds a class-level
`_compiled_flex_attn = torch.compile(flex_attention, ...)`. Compiling the model
nests a second compile around it, so the direct call was gated on
`torch.compiler.is_compiling()`:

```python
flex_attn_fn = (
    flex_attention if torch.compiler.is_compiling()
    else FlexAttention._compiled_flex_attn
)
```

This is correct as far as it goes and did move the failure -- but only to the
next obstacle, `deepseek_v4/attention.py:_build_block_mask`. That builds a
data-dependent `BlockMask` with `scatter_add_`/`argsort` and closes over
`selected_mask` inside `dsa_mask_mod`, which Inductor also cannot lower.

**2. Graph-break the block-mask build** with `@torch._dynamo.disable` on
`_build_block_mask`. Dynamo refuses:

```
Unsupported: Skip inlining `torch.compiler.disable()`d function
```

The call sits under activation checkpointing and the spmd wrapper, which
require a single graph, so the break is not allowed there.

**Conclusion.** Compiling this model needs block-mask construction hoisted out
of the traced forward -- a structural change to `deepseek_v4/attention.py` that
belongs upstream, and one that should not be attempted without numerics
validation. Both patches were reverted; the branch carries no model-code
changes.

## What is enabled instead

`compile.components=["loss"]`, which passes (3 steps, loss 8.11 -> 4.98 at debug
scale) and is what upstream's own `deepseek_v3_671b` recipe selects.

Be clear about what it buys: **essentially no memory** (3.25 vs 3.23 GiB at debug
scale), because `ChunkedLossWrapper` already chunks the logits, and no
model-level fusion.

## Batch size

The microbatch goes 4096 -> **6144** tokens (1.5x context).

This spends headroom the **baseline already had**, not memory freed by compile:

| | |
|---|---|
| baseline peak | 244.94 of 276.50 GiB |
| free | 31.56 GiB |
| baseline activations+buffers @ 4096 tok | ~56.9 GiB |
| 1.5x by linear scaling | ~+28 GiB (fits, and the true cost is lower) |
| 2x (8192) by linear scaling | ~+57 GiB (does not fit) |

Linear scaling overstates the cost, because part of that 56.9 GiB is
token-independent -- FSDP all-gather buffers and allocator fragmentation. 2x is
still out of reach.

Non-multiples of `max_context_length` are accepted: config validation imposes
only `> 0`, and a 1.5x microbatch was confirmed to build and drive attention at
debug scale.

## Reproducing

```bash
WORKTREE=/mnt/dgxc/worktrees/dsv4-compile CONFIG=deepseek_v4_pro_64xgb300_compile \
  STEPS=25 sbatch gb300/dsv4_64xgb300.slurm
```
