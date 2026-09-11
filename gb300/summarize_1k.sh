#!/bin/bash
# Tabulate every 1k-push run from its job output. Reads the STEADY line the
# launcher emits, so all rows share one steady-state window.
printf "%-6s %-18s %-7s %9s %9s %10s %8s %s\n" JOB TAG EXIT "TFLOP/s" CLUSTER "PEAK_MEM" "vs787" NOTE
for f in $(ls -v /mnt/dgxc/runs/1k-*.out 2>/dev/null); do
  job=$(basename "$f" .out | sed 's/1k-//')
  tag=$(head -1 "$f" | awk '{print $2}')
  [ -z "$tag" ] && continue
  ex=$(grep -oE "exit=[0-9]+" "$f" | tail -1 | cut -d= -f2)
  tf=$(grep -oE "tflops_mean=[0-9.]+" "$f" | tail -1 | cut -d= -f2)
  pf=$(grep -oE "cluster=[0-9.]+" "$f" | tail -1 | cut -d= -f2)
  mem=$(grep -oE "peak_mem=[0-9.]+ GiB \([0-9.]+%\)" "$f" | tail -1 | sed 's/peak_mem=//')
  note=$(grep -oE "AssertionError.*|OutOfMemory.*|CUDA error: [a-z ]*|FATAL.*" "$f" | tail -1 | cut -c1-52)
  rel=""
  [ -n "$tf" ] && rel=$(awk -v t="$tf" 'BEGIN{printf "%+.1f%%", (t/786.9-1)*100}')
  printf "%-6s %-18s %-7s %9s %9s %10s %8s %s\n" "$job" "$tag" "${ex:-?}" "${tf:--}" "${pf:--}" "${mem:--}" "${rel:--}" "$note"
done
