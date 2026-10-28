#!/bin/bash
# Run run_gptoss120b_take2.sbatch under qos=dev, which is interactive-only
# (sbatch is rejected: "QOS dev is interactive-only: use srun or salloc").
# srun blocks until the job ends, so this detaches it with setsid/nohup; the
# client's own output goes to logs/srun-clients/, the job's to the usual
# logs/gptoss120b-take2-<jobid>.out.
#
#   TT_REPO=<worktree> TT_CONFIG=<config> [TT_TIME=hh:mm:ss] [knobs...] srun_dev.sh <label>
LABEL=${1:?usage: srun_dev.sh <label>}
ROOT=/data/dj-mat-torchtitan-mfu
SCRIPT=$(dirname "$(readlink -f "$0")")/run_gptoss120b_take2.sbatch
setsid nohup srun --qos=dev --partition=b200full_1 --nodes=1 --ntasks=1 \
    --cpus-per-task=208 --gres=gpu:8 --exclusive --mem=0 --time=${TT_TIME:-00:20:00} \
    --cpu-bind=none \
    --job-name="t2-$LABEL" \
    --output="$ROOT/logs/gptoss120b-take2-%j.out" \
    bash "$SCRIPT" \
    > "$ROOT/logs/srun-clients/$LABEL.$(date +%s).log" 2>&1 < /dev/null &
echo "launched $LABEL (client pid $!)"
