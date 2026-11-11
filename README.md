<div align="center">

# torchtitan

#### A PyTorch native platform for training generative AI models

[![8 GPU Feature Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml?query=branch%3Amain)
[![8 GPU Model Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![forum](https://img.shields.io/badge/pytorch-forum-DE3412.svg)](https://discuss.pytorch.org/c/distributed/torchtitan/44)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)


</div>

`torchtitan` is under extensive development. To use the latest features of `torchtitan`, we recommend using the most recent PyTorch nightly.


## Latest News
- [2026/08] [TitanRL](torchtitan/experiments/rl) is a hackable RL stack for scaling and debugging. It reuses TorchTitan model definitions and kernels across training and vLLM generation and supports batch-invariant mode.
- [2025/11] AMD released an [optimized fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main) of `torchtitan` for AMD GPUs.
- [2025/10] We released `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases).
- [2025/10] SkyPilot now supports `torchtitan`! See the tutorial [here](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html).
- [2025/07] We published [instructions](/torchtitan/models/README.md) on how to add a model to `torchtitan`.
- [2025/04] Our paper was accepted by [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620).
- [2024/12] GPU MODE [lecture](https://www.youtube.com/watch?v=VYWRjcUqW6w) on torchtitan.
- [2024/07] [Presentation](https://pytorch2024.sched.com/event/1fHn3) at PyTorch Conference 2024.


## Overview

`torchtitan` is a PyTorch native platform designed for **rapid experimentation and large-scale training** of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, `torchtitan` provides a flexible foundation for developers to build upon. With `torchtitan` [extension points](docs/extension.md), one can easily create custom extensions tailored to specific needs.

Our mission is to accelerate innovation in the field of generative AI by empowering researchers and developers to explore new modeling architectures and infrastructure techniques.

The Guiding Principles when building `torchtitan`
* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code when applying multi-dimensional parallelism.
* Bias towards a clean, minimal codebase while providing basic reusable / swappable components.

`torchtitan` has been showcasing PyTorch's latest distributed training features, via support for pretraining Llama 3.1 LLMs of various sizes.

## Contributing

We look forward to your contributions!

* To accelerate contributions to and innovations around torchtitan, we host an [`experiments`](torchtitan/experiments) folder. New ideas should start there. To contribute, follow the [`experiments guidelines`](torchtitan/experiments/README.md).
* For fixes and contributions to core, follow these [`guidelines`](CONTRIBUTING.md).

## Llama 3.1 training

### Key features available

1. Multi-dimensional composable parallelisms
   - [FSDP2](docs/fsdp.md) with per-parameter sharding
   - [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
   - [Pipeline Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
   - [Context Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)
2. [Meta device](https://pytorch.org/docs/stable/meta.html) initialization
3. Per-op selective and full activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
   - [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
5. `torch.compile` support
6. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](torchtitan/components/quantization/float8.md))
7. [MXFP8 training for dense and MoE models](torchtitan/components/quantization/mxfp8.md) on Blackwell GPUs.
8. Supervised Fine-Tuning (SFT) with chat-formatted datasets
9. DDP and HSDP
10. [TorchFT](https://github.com/pytorch/torchft) integration
11. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries) and support for [custom datasets](docs/datasets.md)
12. Gradient accumulation, enabled by giving an additional `--training.global_batch_size` argument on the CLI
13. Flexible learning rate scheduler (warmup-stable-decay)
14. [BF16 optimizer states](docs/bf16_optimizer_states.md) for reduced memory usage
15. Loss, GPU memory, throughput (tokens/sec), TFLOPs, and MFU displayed and logged via [Tensorboard or Weights & Biases](/docs/metrics.md)
16. [Debugging tools](docs/debugging.md) including CPU/GPU profiling, memory profiling, Flight Recorder, etc.
17. All options easily configured via [Python config registry](torchtitan/models/llama3/config_registry.py) with `--module` and `--config` CLI flags
18. Structured logging: per-rank trace of key training phases; (see [`torchtitan/observability/structured_logger/README.md`](torchtitan/observability/structured_logger/README.md))
19. [Helper scripts](scripts/) to
    - download tokenizers from Hugging Face
    - convert original Llama 3 checkpoints into the expected DCP format
    - estimate FSDP/HSDP memory usage without materializing the model
    - run distributed inference with Tensor Parallel

We report [performance](benchmarks/llama3_h100_202412_torchtitan.md) on up to 512 GPUs, and verify [loss converging](docs/converging.md) correctness of various techniques.

### Dive into the code

You may want to see how the model is defined or how parallelism techniques are applied. For a guided tour, see these files first:
* [torchtitan/train.py](torchtitan/train.py) - the main training loop and high-level setup code
* [torchtitan/models/llama3/model.py](torchtitan/models/llama3/model.py) - the Llama 3.1 model definition
* [torchtitan/models/llama3/parallelize.py](torchtitan/models/llama3/parallelize.py) - helpers for applying Data Parallel, Tensor Parallel, activation checkpointing, and `torch.compile` to the model
* [torchtitan/distributed/pipeline_parallel.py](torchtitan/distributed/pipeline_parallel.py) - helpers for applying Pipeline Parallel to the model
* [torchtitan/components/checkpoint.py](torchtitan/components/checkpoint.py) - utils for saving/loading distributed checkpoints
* [torchtitan/components/quantization/float8.py](torchtitan/components/quantization/float8.py) - utils for applying Float8 techniques


## Installation

One can directly run the source code, or install `torchtitan` from a nightly build, or a stable release.

### From source

This method requires the nightly build of PyTorch, or the latest PyTorch built [from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

> **Note:** The nightly build of `torchdata` is required when using a PyTorch nightly. Install it from the nightly index as shown above.

> **Note:** You can run directly from the source tree. If you need to import `torchtitan` as a package from elsewhere, install it in editable mode without re-resolving dependencies: `pip install -e . --no-deps`.

### Nightly builds

This method requires the nightly build of PyTorch. You can replace `cu130` with another version of cuda or an AMD GPU (e.g. `rocm6.3`).

```sh
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Stable releases
One can install the latest [stable release](https://github.com/pytorch/torchtitan/releases) of `torchtitan` via `pip` or `conda`.
```sh
pip install torchtitan
```
```sh
conda install conda-forge::torchtitan
```
Note that each stable release pins the nightly versions of `torch` and `torchao`. Please see [release.md](docs/release.md) for more details.

### Downloading a tokenizer

`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box. To get started training these models, we need to download the tokenizer. Follow the instructions on the official [meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) repository to ensure you have access to the Llama model weights.

Once you have confirmed access, you can run the following command to download the Llama 3.1 tokenizer to your local machine.

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

### Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust `--nproc_per_node` in the torchrun command and `#SBATCH --gpus-per-task` in the SBATCH command section.


## Citation

We provide a detailed look into the parallelisms and optimizations available in `torchtitan`, along with summary advice on when to use various techniques.

[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)
```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```


## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.


## Recreating our runs

For recreation of our runs, please follow the instructions below. Each
workload has a baseline (the stock torchtitan config) and the most optimized
run that reached our peak TFLOP/s. Numbers are TFLOP/s per GPU.

**B200 runs** (1 to 6) use the `lambda-moe-tflops` branch and the launchers in
`benchmarks/`. The launchers expect the shared layout on the B200 cluster:
`/data/dj-mat-torchtitan-mfu/.venv` (venv), `/data/dj-mat-torchtitan-mfu/logs`
(job output), `/data/dj-mat-torchtitan-mfu/.hf` (C4 cache) and `~/.hf_token`
(Hugging Face token for streamed C4). `TT_REPO=$PWD` runs the tree you are in.
Tokenizers live in `assets/hf/<model>`, which git does not track; symlink them
into a fresh clone before submitting.

**GB300 runs** (7 and 8) use the `main` branch, where the DeepSeek-V4 work was
merged, and the launcher `gb300/dsv4_64xgb300.slurm`. It expects the GB300
cluster layout: `/mnt/dgxc/venvs/dsv4n` (venv), `/mnt/dgxc/pydeps-te` and
`/mnt/dgxc/pydeps-cudnn` (side-installed Transformer Engine 2.19 and cuDNN
frontend 1.29) and `/mnt/dgxc/cublas-new` (cuBLAS 13.8 preload).

### 1. GPT-OSS-20B on 8x B200, baseline

Stock `gpt_oss_20b` config: bs=1, seq 8192, FullAC, fp32 master weights.
If the run dies at step 2 with a CUDA-graph shape error, resubmit with
`TT_EXTRA="--training.disable_cuda_graphs"` (the only deviation from stock).

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
sbatch benchmarks/gpt_oss_20b/run_gptoss20b.sbatch
```

### 2. GPT-OSS-20B on 8x B200, best: 1,010 TF/GPU (job 7258)

MXFP8 linears and lm_head, full bf16 training, bf16 grad reduce,
SelectiveAC saving the grouped GEMMs, no reshard after forward, bs=5,
expert-bias gradient as a GEMM, contiguous-halves SwiGLU, TF32 fp32 matmuls,
Inductor pointwise autotune. 100 steps, read over steps 21-100.

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
TT_INDUCTOR="TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1" \
sbatch --export=ALL,TT_REPO=$PWD,TT_CONFIG=gptoss20b_mxfp8_lmhead_bf16reduce_bf16train_sacgmm_noreshard_bs5_biasgemm_swigluhalves,TORCHTITAN_FP32_MATMUL_PRECISION=tf32 \
  benchmarks/gpt_oss_20b/run_gptoss20b_take2.sbatch
```

### 3. GPT-OSS-120B on 8x B200, baseline

Stock `gpt_oss_120b` config with the two changes needed for it to run on one
8x B200 node at all: CUDA graphs off (varlen attention changes shape every
step) and full bf16 training (fp32 states do not fit in HBM).

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
sbatch benchmarks/gpt_oss_120b/train_gptoss120b_bf16.sbatch
```

### 4. GPT-OSS-120B on 8x B200, best: 872 TF/GPU mean, 876 median (job 7653)

EP=8, compile, SelectiveAC saving every block's grouped GEMMs, unsharded
experts skipped by FSDP, bs=2, full bf16 training, bf16 grad reduce, lr 3e-4,
TF32 fp32 matmuls, Inductor pointwise autotune. The number is the mean over
steps 1000-1500 of a 50-minute run with the normal 2000-step warmup
(`TT_STEPS=` empty keeps the config's step count; the 200-step default clamps
the warmup and reads lower). qos=dev is interactive-only, so the launcher
goes through `srun_dev.sh`.

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
TT_REPO=$PWD \
TT_CONFIG=gpt_oss_120b_bf16reduce_lr3e4_take2stack_skipexperts_sacgmmall \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 \
TT_INDUCTOR="TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1" \
TT_STEPS= TT_TIME=00:50:00 \
  benchmarks/gpt_oss_120b/srun_dev.sh gptoss120b_best
```

### 5. Qwen3-30B-A3B on 8x B200, baseline

Stock `qwen3_30b_a3b` config: bs=2, seq 4096, FullAC, fp32 master weights,
CUDA graphs on. The strict launcher passes nothing but the step count.

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
sbatch benchmarks/qwen3_30b_a3b/run_qwen3_30b_a3b_strict.sbatch
```

### 6. Qwen3-30B-A3B on 8x B200, best: 985 TF/GPU mean, 1,012 median (job 6180)

seq 8192, SelectiveAC, compile, bf16 grad reduce, MXFP8 attention and
lm_head, varlen flash attention, full bf16 training, dataloader workers,
bs=12, TF32 fp32 matmuls. 100 steps, read over steps 21-100.

```bash
cd /data/dj-mat-torchtitan-mfu/torchtitan   # branch lambda-moe-tflops
TT_EXTRA="--training.local_batch_size 12" \
sbatch --export=ALL,TT_REPO=$PWD,TT_CONFIG=qwen3_30b_a3b_8k_bs10_selac_compile_bf16reduce_mxfp8_attn_lmhead_varlen_bf16train_loaderworkers,TORCHTITAN_FP32_MATMUL_PRECISION=tf32 \
  benchmarks/qwen3_30b_a3b/run_qwen3_30b_a3b_take2.sbatch
```

### 7. DeepSeek-V4 Flash on 64x GB300, baseline

Stock `deepseek_v4_flash` config (seq 4096, EP=64, flex attention, fp32
master weights) on 16 nodes x 4 GB300. 30 steps, mean over steps 15 and up.

```bash
cd /mnt/dgxc/worktrees/dsv4-flash   # branch main
WORKTREE=$PWD CONFIG=deepseek_v4_flash STEPS=30 \
  sbatch --nodes=16 --time=01:00:00 gb300/dsv4_64xgb300.slurm
```

### 8. DeepSeek-V4 Flash on 128x GB300, best: 743.9 TF/GPU (job 257)

The full GB300 recipe: seq 8192, 9 sequences per GPU, EP=2, cuDNN fused DSA
forward and backward with the atomic dKV path, cuDNN fused indexer, dense-never
FSDP, MinimalAsyncEP with the TMA row copy, Transformer Engine fp8 dense
linears and fused mHC kernels, packed expert weights with direct FSDP
all-gather and reduce-scatter, full bf16 training. 200 steps on real C4,
32 nodes as HSDP (replicate 4 x shard 32), one rack per 8 or 16 nodes.
On 32 GPUs (8 nodes, one rack) the same recipe with forced balanced routing
(`..._tedense_9x_balanced`) reads 756 TF/GPU.

```bash
cd /mnt/dgxc/worktrees/dsv4-flash   # branch main
WORKTREE=$PWD \
EXTRA_PYTHONPATH=/mnt/dgxc/pydeps-te:/mnt/dgxc/pydeps-cudnn \
TORCHTITAN_FP32_MATMUL_PRECISION=tf32 TE_DENSE_RECIPE=delayed TE_REDUCE_AMAX=0 TORCHTITAN_TE_MHC=1 \
TORCHTITAN_DSA_PERSISTENT_WORKSPACE=0 FSDP_PREFETCH_DEPTH=2 \
MINIMAL_ASYNC_EP_COPY_BLOCK_M=16 MINIMAL_ASYNC_EP_COPY_WARPS=4 MINIMAL_ASYNC_EP_COPY_TMA=1 \
MINIMAL_ASYNC_EP_POOL_FACTOR=1.25 MOE_PACKED_EXPERT_WEIGHTS=1 FSDP_DIRECT_GATHER=1 FSDP_DIRECT_REDUCE_SCATTER=1 \
TORCHTITAN_DSA_DETERMINISTIC=0 CUBLAS_NEW=1 COMPRESSOR_BF16_GEMM=1 TE_MHC_BF16_GRAD_PHI=1 TORCHTITAN_MHC_GRAD_CHAIN=1 \
CONFIG=deepseek_v4_flash_8k_gb300_cudnn_full_ep2_densenever_cudnnidx_tedense_9x STEPS=200 TAG=hero200_128gpu \
  sbatch --nodes=32 --partition=all --time=01:00:00 gb300/dsv4_64xgb300.slurm \
    --parallelism.data_parallel_replicate_degree 4
```
