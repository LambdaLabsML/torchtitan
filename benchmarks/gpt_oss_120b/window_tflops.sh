#!/bin/bash
# Mean / median TF/GPU, tok/s/GPU and peak memory of a GPT-OSS-120B run over a
# fixed step window, so runs capped by walltime compare like for like.
#
# Usage: window_tflops.sh <log> [first_step=21] [last_step=200]
#
# The 120b throughput climbs for the first few hundred steps while the expert
# load balancer converges (job 2278 vs 2268), so a whole-run mean depends on
# how far the run got before --time stopped it. Read every A/B pair over the
# same window. Accepts torchtitan's thousands separator ("1,012.31").
LOG=${1:?usage: window_tflops.sh <log> [first_step] [last_step]}
LO=${2:-21}
HI=${3:-200}
sed -r 's/\x1B\[[0-9;]*[mGKH]//g' "$LOG" \
  | grep -E 'step: *[0-9]+.*tflops:' \
  | awk -v lo="$LO" -v hi="$HI" '
    {
      match($0, /step: *[0-9]+/); s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s)
      if (s+0 < lo || s+0 > hi) next
      match($0, /tflops: *[0-9.,]+/); t = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", t)
      match($0, /tps: *[0-9,]+/); p = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", p)
      if (match($0, /memory: *[0-9.]+GiB\([0-9.]+%\)/)) { m = substr($0, RSTART, RLENGTH); sub(/memory: */, "", m)
        g = m; sub(/GiB.*/, "", g); if (g+0 > maxg+0) { maxg = g; maxm = m } }
      n++; tf[n] = t; st += t; sp += p; last = s
    }
    END {
      if (n == 0) { print "no steps in window " lo "-" hi; exit 1 }
      asort(tf)
      med = (n % 2) ? tf[(n+1)/2] : (tf[n/2] + tf[n/2+1]) / 2
      printf "window %d-%d (%d points, last %d): TF/GPU mean %.2f median %.2f | tok/s/GPU %.0f | peak %s\n",
             lo, hi, n, last, st/n, med, sp/n, maxm
    }'
