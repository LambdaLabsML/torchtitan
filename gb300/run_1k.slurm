#!/bin/bash
# GPT-OSS-120B push to 1000 TFLOP/s/GPU on 64x GB300. One config, one job.
#
# Descends from run_branch.slurm and adds the three things that sweep needed:
#   - TRITON_CACHE_DIR set INSIDE the script, per job. Passing it via
#     --export sends the literal "$SLURM_JOB_ID", which shares one cache
#     across jobs and causes "Cubin file saved by TritonBundler not found"
#     plus mid-run recompiles (145 in one job, 4 in the next). Differences
#     below ~2% are not reliable without this.
#   - MXFP8=1 puts the locally built torchao 0.19 (with the sm_103
#     _C_mxfp8 kernels) ahead of the venv's kernel-less 0.18.0 wheel.
#     See gb300/MXFP8_120B.md.
#   - PROFILE=n turns on the torch profiler at step n.
#
#   WORKTREE=/mnt/dgxc/worktrees/gptoss120b-1k \
#     CONFIG=gpt_oss_120b_gb300_ep8_sac_compile STEPS=100 \
#     sbatch gb300/run_1k.slurm
#
#SBATCH --job-name=g120b_1k
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --output=/mnt/dgxc/runs/1k-%j.out
set -uo pipefail

WORKTREE=${WORKTREE:-/mnt/dgxc/worktrees/gptoss120b-1k}
CONFIG=${CONFIG:?set CONFIG=<name from gpt_oss/config_registry.py>}
STEPS=${STEPS:-100}
TAG=${TAG:-$CONFIG}
MXFP8=${MXFP8:-0}
PROFILE=${PROFILE:-0}
# Steady-state window start. 100 by default to match the window every number in
# gb300/TUNING_RESULTS_120B.md uses, so rows stay comparable across run lengths.
# 120B needs it: throughput is still ramping at step 60 (784) and dips as late as
# step 100 (655 against a 790 neighbourhood), so a steps>=50 window measures the
# ramp rather than the config.
STEADY=${STEADY:-100}

cd "$WORKTREE"
source /mnt/dgxc/venvs/torchtitan/bin/activate
export PYTHONPATH="$WORKTREE"

# torchao 0.19 built from source against torch 2.14 for sm_103. Prepended so it
# wins over the venv's 0.18.0, which ships no compiled kernels on aarch64.
if [ "$MXFP8" = "1" ]; then
  export PYTHONPATH="/mnt/dgxc/overlays/torchao-tt:$PYTHONPATH"
fi

BRANCH=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD)
echo "###### $TAG  branch=$BRANCH  commit=$COMMIT  steps=$STEPS  mxfp8=$MXFP8  job=$SLURM_JOB_ID  $(date -u +%H:%M:%S) ######"

# Fail loudly rather than silently measuring /mnt/dgxc/torchtitan, and prove the
# mxfp8 kernels are really registered rather than quietly absent.
python - "$WORKTREE" "$MXFP8" <<'PY' || exit 3
import sys, torchtitan
want, mxfp8 = sys.argv[1], sys.argv[2]
got = torchtitan.__file__
print(f"###### torchtitan -> {got}", flush=True)
if not got.startswith(want.rstrip("/") + "/"):
    print(f"###### FATAL: expected import from {want}", flush=True)
    sys.exit(1)
if mxfp8 == "1":
    import torch, torchao
    from torchao.prototype.mx_formats.kernels import (
        _mxfp8_cuda_kernels_available,
        _triton_kernels_available,
    )
    # The failure this guards is silent: without the compiled _C_mxfp8 the ops
    # are simply not registered for CUDA, and the run either raises deep in the
    # first backward or quietly measures something that is not mxfp8.
    has_op = hasattr(torch.ops.torchao, "mxfp8_quantize")
    print(f"###### torchao {torchao.__version__} -> {torchao.__file__}", flush=True)
    print(f"###### cuda_kernels={_mxfp8_cuda_kernels_available} "
          f"triton_kernels={_triton_kernels_available} mxfp8_quantize_op={has_op}", flush=True)
    if not (_mxfp8_cuda_kernels_available and _triton_kernels_available and has_op):
        print("###### FATAL: mxfp8 kernels absent; run would not measure mxfp8", flush=True)
        sys.exit(1)
PY

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1
# Per-job, or jobs share a cache and recompile mid-run. Must be set here:
# $SLURM_JOB_ID is unknown at submit time.
export TRITON_CACHE_DIR="/mnt/dgxc/triton-cache/$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR"

PROF_ARGS=()
if [ "$PROFILE" != "0" ]; then
  PROF_ARGS=( --profiler.enable-profiling
              --profiler.profile-freq "$PROFILE"
              --profiler.save-traces-folder "/mnt/dgxc/traces/$TAG-$SLURM_JOB_ID" )
fi

LOG="/mnt/dgxc/runs/1k_${TAG}.log"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes "$SLURM_NNODES" --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train --module gpt_oss --config "$CONFIG" \
    --training.steps "$STEPS" --training.disable-cuda-graphs \
    "${PROF_ARGS[@]+"${PROF_ARGS[@]}"}" "$@" \
  > "$LOG" 2>&1
rc=$?
echo "###### END $TAG exit=$rc $(date -u +%H:%M:%S) ######"

# Steady-state mean, so rows are comparable. Steady state is the second half of
# the run: warmup and compile dominate the first steps.
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "step: +[0-9]+ .*" | \
  awk -v s="$STEADY" '{
    for(i=1;i<=NF;i++){
      if($i=="step:") st=$(i+1);
      if($i=="tflops:") tf=$(i+1);
      if($i=="memory:"){m=$(i+1); gsub(/GiB.*/,"",m)}
    }
    if(st>=s){n++; sum+=tf; if(m+0>pk) pk=m+0}
    last=tf
  }
  END{ if(n>0) printf "###### STEADY(>=%d) tflops_mean=%.1f over %d pts  cluster=%.2f PF  peak_mem=%.2f GiB (%.1f%%)  last=%.1f\n", s, sum/n, n, sum/n*64/1000, pk, pk/276.50*100, last;
       else print "###### STEADY no steps logged" }'
grep -c "Retry\|ConnectionError" "$LOG" | sed 's/^/###### dataloader retries+connerrs: /'
grep -c "Cubin file saved by TritonBundler not found" "$LOG" | sed 's/^/###### triton cubin misses: /'
[ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "AssertionError|RuntimeError|OutOfMemory|torch\..*Error|Error:" | tail -3 | cut -c1-220
exit $rc
