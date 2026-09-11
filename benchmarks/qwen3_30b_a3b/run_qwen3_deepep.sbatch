#!/bin/bash
# Qwen3-30B-A3B with the DeepEP v2 expert-parallel backend, 8x B200 single node.
#
# SEPARATE SCRIPT, SEPARATE VENV, SEPARATE BRANCH -- this whole path is a
# disposable experiment (branch qwen3-30b-a3b-deepep-attempt). It uses
# /data/dj-mat-torchtitan-mfu/.venv-deepep, a clone of .venv with deep_ep and
# ninja added, so nothing here can disturb the kimi3 / gpt_oss sweeps that share
# the main .venv. Delete the branch and .venv-deepep to scrap it.
#
# How deep_ep got installed (the PyPI route does NOT work):
#   1. PyPI `deep_ep` 1.0.0 is a source dist whose sdist is INCOMPLETE -- the
#      build dies with "fatal error: configs.cuh: No such file or directory"
#      because it does not ship its own csrc headers. Do not use it.
#   2. The official source does build:
#        uv pip install "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git" \
#            --no-build-isolation
#      giving deep-ep 2.1.0 (ElasticBuffer), which is the v2 API torchtitan's
#      DeepEPTokenDispatcher needs.
#   3. Prerequisites that had to be satisfied first:
#        - NVSHMEM: already present as the nvidia-nvshmem-cu13 wheel (3.4.5) in
#          the venv, but the build cannot find it without NVSHMEM_DIR pointing
#          at site-packages/nvidia/nvshmem.
#        - ninja: required because the extension builds with dlink=True.
#        - --no-build-isolation, because deep_ep does not declare torch as a
#          build dependency.
#
# The NVSHMEM_* / EP_* variables below come from the qwen3_moe_deepep docstring
# in torchtitan/models/qwen3/config_registry.py, which documents what an
# intra-node (no RDMA NIC required) ElasticBuffer init needs. This job is
# --nodes=1, so NVLink-only is the correct transport; on a multi-node RDMA run
# none of it is needed.
#
# Submit with:
#   sbatch --export=ALL,TT_CONFIG=qwen3_30b_a3b_8k_bs4_selac_compile_deepep_ep8 run_qwen3_deepep.sbatch
#SBATCH --job-name=tt-qwen3-deepep
#SBATCH --partition=b200full_1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/dj-mat-torchtitan-mfu/logs/qwen3-deepep-%j.out

set -x
: "${TT_CONFIG:?set TT_CONFIG to a deepep config_registry function name}"
: "${TT_STEPS:=400}"

REPO=/data/dj-mat-torchtitan-mfu/torchtitan
VENV=/data/dj-mat-torchtitan-mfu/.venv-deepep
SP="$VENV/lib/python3.12/site-packages"
source "$VENV/bin/activate"
cd "$REPO"

export PYTHONUNBUFFERED=1
export HF_HOME=/data/dj-mat-torchtitan-mfu/.hf

# deep_ep links against the nvshmem and nccl shared objects that ship inside the
# wheels; they are not on the default loader path.
export LD_LIBRARY_PATH="$SP/nvidia/nvshmem/lib:$SP/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_DIR="$SP/nvidia/nvshmem"

# Intra-node ElasticBuffer init (single node, NVLink only).
export EP_DISABLE_GIN=1              # no NCCL GIN / RDMA requirement
export EP_REUSE_NCCL_COMM=0          # avoid the null-device-comm segfault
export NVSHMEM_REMOTE_TRANSPORT=none # NVLink only, no remote transport
export NVSHMEM_DISABLE_MNNVL=1       # no multi-node NVLink

# HF auth: token from ~/.hf_token, read with xtrace off so it never reaches the
# log (this script runs under `set -x` into a world-readable file).
{ set +x; } 2>/dev/null
if [ -r "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    _hf_auth="yes (from ~/.hf_token)"
else
    _hf_auth="NO -- ~/.hf_token missing; expect HTTP 429 on c4"
fi
set -x

echo "===== RUN PROVENANCE ====="
date -Is
hostname
echo "slurm job: $SLURM_JOB_ID  nodelist: $SLURM_JOB_NODELIST"
echo "config: $TT_CONFIG  steps=$TT_STEPS"
echo "hf auth: $_hf_auth"
echo "venv: $VENV"
git -C "$REPO" rev-parse --short HEAD
git -C "$REPO" rev-parse --abbrev-ref HEAD
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import deep_ep; print('deep_ep', deep_ep.__version__, '| ElasticBuffer', hasattr(deep_ep,'ElasticBuffer'))"
python -c "
from torchtitan.models.qwen3.config_registry import $TT_CONFIG as c
t=c().training; p=c().parallelism
print('seq_len=%d local_batch_size=%d -> %d tokens/gpu/step' % (t.seq_len,t.local_batch_size,t.seq_len*t.local_batch_size))
print('ep=%d compile=%s graphs_disabled=%s' % (p.expert_parallel_degree,c().compile.enable,t.disable_cuda_graphs))
" 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=========================="

NGPU=8 MODULE=qwen3 CONFIG="$TT_CONFIG" ./run_train.sh \
    --training.steps "$TT_STEPS" \
    --metrics.log_freq 1
TRAIN_RC=$?

echo "===== TRAIN EXIT rc=$TRAIN_RC ====="
date -Is
exit "$TRAIN_RC"
