#!/usr/bin/env bash
set -u

MODE="${1:-tcp}"

RUNS="${RUNS:-1}"
WARMUPS="${WARMUPS:-0}"
GVIRTUS_HOME="${GVIRTUS_HOME:-/home/student.aau.dk/ll33pq/GVirtuS}"
LZ4_HOME="${LZ4_HOME:-/home/student.aau.dk/ll33pq/lz4-install}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"

case "$MODE" in
  tcp)
    unset GVIRTUS_CONFIG
    ;;
  rdma)
    export GVIRTUS_CONFIG="${GVIRTUS_HOME}/etc/properties_plain_rdma.json"
    ;;
  ucx)
    export GVIRTUS_CONFIG="${GVIRTUS_HOME}/etc/properties_ucx.json"
    ;;
  *)
    echo "Usage: $0 [tcp|rdma|ucx]"
    exit 1
    ;;
esac

export GVIRTUS_HOME
export LZ4_HOME
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_results/simple_${STAMP}_${MODE}"
LOG_DIR="$OUT_DIR/logs"
CSV="$OUT_DIR/results.csv"

mkdir -p "$LOG_DIR"

echo "timestamp,mode,phase,run,exit_code,wall_s,accuracy_pct,execution_s,per_image_s,log_file" > "$CSV"

run_one() {
  local phase="$1"
  local run_no="$2"
  local log_file="$LOG_DIR/${MODE}_${phase}_${run_no}.log"

  echo "[$MODE] $phase run $run_no..."

  local start_ns end_ns wall_s exit_code

  start_ns="$(date +%s%N)"

  GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-10000}" \
  STEADY_WARMUPS="${STEADY_WARMUPS:-0}" \
  STEADY_ITERS="${STEADY_ITERS:-1}" \
  STEADY_IMAGES="${STEADY_IMAGES:-1}" \
  python3 cnn.py > "$log_file" 2>&1

  exit_code=$?
  end_ns="$(date +%s%N)"
  wall_s="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e-s)/1000000000 }')"

  accuracy="$(grep -E 'Test Accuracy:' "$log_file" | tail -1 | sed -E 's/.*Test Accuracy:[[:space:]]*([0-9.]+)%.*/\1/' || true)"
  execution_s="$(grep -E 'Execution Time:' "$log_file" | tail -1 | sed -E 's/.*Execution Time:[[:space:]]*([0-9.]+).*/\1/' || true)"
  per_image_s="$(grep -E 'Average Time:' "$log_file" | tail -1 | sed -E 's/.*Average Time:[[:space:]]*([0-9.]+).*/\1/' || true)"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$(date -Iseconds)" "$MODE" "$phase" "$run_no" "$exit_code" "$wall_s" \
    "$accuracy" "$execution_s" "$per_image_s" "$log_file" >> "$CSV"

  if [[ "$exit_code" -eq 0 && -n "$execution_s" ]]; then
    echo "  OK wall=${wall_s}s exec=${execution_s}s per_image=${per_image_s}s acc=${accuracy}%"
  else
    echo "  FAILED exit_code=$exit_code wall=${wall_s}s log=$log_file"
    tail -40 "$log_file"
  fi
}

for i in $(seq 1 "$WARMUPS"); do
  run_one warmup "$i"
done

for i in $(seq 1 "$RUNS"); do
  run_one measure "$i"
done

echo
echo "Done."
echo "Main CSV: $CSV"
echo
echo "Summary measured only:"
awk -F, '
NR > 1 && $3 == "measure" && $5 == 0 && $8 != "" {
  n++
  wall += $6
  execs += $8
  perimg += $9
}
END {
  if (n > 0) {
    printf "n=%d mean_wall_s=%.6f mean_exec_s=%.6f mean_per_image_s=%.6f\n", n, wall/n, execs/n, perimg/n
  } else {
    print "No valid measured runs"
  }
}
' "$CSV"
