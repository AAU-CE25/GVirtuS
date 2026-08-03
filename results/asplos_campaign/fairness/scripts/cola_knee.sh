#!/bin/bash
# cola_knee.sh -- finer grid between the load that always passes the SLO (0.50) and the one
# that only sometimes does (0.75), with windows scaled to >=120 offered requests per point
# instead of >=40. Answers the two open items of LLAMA-7B_RESULTS.md section 3d: no points
# around the knee, and windows too short for p99.
# N=8 runs first: if the queue is cut short, the most informative tenant count is complete.
set -u
LOG=/home/student.aau.dk/ll33pq/cola_knee.log
exec >> "$LOG" 2>&1
echo "=== cola_knee arrancada $(date +%F' '%H:%M:%S) ==="
for N in 8 4 2; do
  for REP in 1 2 3; do
    for S in bm bmmps ucx; do
      echo "--- $S N=$N rep=$REP $(date +%H:%M:%S) ---"
      bash ~/sweep_knee.sh "$S" "$N" "$REP" 2>&1 | grep -E "pods_up|ABORT|^\[kn_|DONE"
    done
  done
  echo "=== N=$N COMPLETO $(date +%H:%M:%S) ==="
done
echo "=== KNEE_DONE $(date +%H:%M:%S) ==="
