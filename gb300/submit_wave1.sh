#!/bin/bash
# Wave 1: profile first, then the three levers in isolation.
#
# Chained with --dependency=afterany rather than left to FIFO. Every job here
# asks for all 16 nodes so they cannot overlap anyway, but the time limits
# differ, and backfill is free to reorder same-priority jobs when they do.
# The profile must run first for its result to aim anything.
#
# afterany, not afterok: a config that OOMs should not cancel the rest of the
# night.
set -euo pipefail
cd /mnt/dgxc/worktrees/gptoss120b-1k
W=/mnt/dgxc/worktrees/gptoss120b-1k

sub() {  # sub <tag> <config> <steps> <mxfp8> <time> [dep]
  local tag=$1 cfg=$2 steps=$3 mx=$4 tl=$5 dep=${6:-}
  local args=(--time="$tl" --job-name="$tag")
  [ -n "$dep" ] && args+=(--dependency=afterany:"$dep")
  WORKTREE=$W CONFIG=$cfg STEPS=$steps MXFP8=$mx TAG=$tag \
    sbatch "${args[@]}" gb300/run_1k.slurm | grep -oE '[0-9]+$'
}

# The profile. 15 steps is enough: it records step 12 only.
J1=$(sub prof_ref     gpt_oss_120b_1k_ref_profile   15 0 00:30:00)
echo "prof_ref            $J1"

# Control. Everything below is one variable against this, at the same 200 steps
# and the same steps>=100 steady-state window.
J2=$(sub ref          gpt_oss_120b_1k_ref          200 0 01:00:00 "$J1")
echo "ref                 $J2"

# MXFP8: aimed at ~72% of per-token FLOPs. Longer limit -- it compiles the
# quantised grouped GEMMs, and the first step pays for that.
J3=$(sub mxfp8        gpt_oss_120b_1k_mxfp8        200 1 01:15:00 "$J2")
echo "mxfp8               $J3"

# bf16 gradient reduce-scatter: aimed at the largest single collective.
J4=$(sub bf16reduce   gpt_oss_120b_1k_bf16reduce   200 0 01:00:00 "$J3")
echo "bf16reduce          $J4"

# Expert parallelism, two points. Sign unknown at this batch.
J5=$(sub ep8          gpt_oss_120b_1k_ep8          200 0 01:00:00 "$J4")
echo "ep8                 $J5"
J6=$(sub ep16         gpt_oss_120b_1k_ep16         200 0 01:00:00 "$J5")
echo "ep16                $J6"

echo "LAST=$J6"
