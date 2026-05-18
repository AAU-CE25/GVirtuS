#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./benchmark.sh tcp
#   ./benchmark.sh rdma
#   ./benchmark.sh ucx
#   ./benchmark.sh all
#
# Useful env:
#   RUNS=5 WARMUPS=1 ./benchmark.sh ucx
#   FRONTEND_CMD="./sample" ./benchmark.sh tcp
#   IFACES="ens1f1np1" ./benchmark.sh all
#   RUN_TIMEOUT=20 ./benchmark.sh rdma

MODE="${1:-all}"

RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-1}"
RUN_TIMEOUT="${RUN_TIMEOUT:-20}"
IFACES="${IFACES:-ens1f1np1}"
FRONTEND_CMD="${FRONTEND_CMD:-./sample}"

GVIRTUS_HOME="${GVIRTUS_HOME:-/home/student.aau.dk/ul11nh/gvirtus-install}"
GVIRTUS_REPO="${GVIRTUS_REPO:-/home/student.aau.dk/ul11nh/GVirtuS}"
GVIRTUS_CONFIG_DIR="${GVIRTUS_CONFIG_DIR:-$GVIRTUS_REPO/etc}"
OPENCV_HOME="${OPENCV_HOME:-/home/student.aau.dk/ul11nh/opencv-local}"
NPP_DIR="${NPP_DIR:-/home/student.aau.dk/ul11nh/.local/lib/python3.10/site-packages/nvidia/npp/lib}"
CUDNN_ROOT="${CUDNN_ROOT:-/home/student.aau.dk/ul11nh/cudnn-9.5.1}"
CUDNN_LIB="${CUDNN_LIB:-$CUDNN_ROOT/lib}"
LZ4_LIB="${LZ4_LIB:-/home/student.aau.dk/ul11nh/lz4-install/lib}"

CONFIG_TCP="${CONFIG_TCP:-$GVIRTUS_CONFIG_DIR/properties.json}"
CONFIG_RDMA="${CONFIG_RDMA:-$GVIRTUS_CONFIG_DIR/properties_plain_rdma.json}"
CONFIG_UCX="${CONFIG_UCX:-$GVIRTUS_CONFIG_DIR/properties_ucx.json}"

GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-30000}"

GV_LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_HOME/lib:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$LZ4_LIB:${LD_LIBRARY_PATH:-}"
GV_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcuda.so:$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_results/frontend_${TS}_${MODE}"
LOG_DIR="$OUT_DIR/logs"
COUNTER_DIR="$OUT_DIR/counters"

mkdir -p "$LOG_DIR" "$COUNTER_DIR"

RESULTS_CSV="$OUT_DIR/results.csv"
NIC_CSV="$OUT_DIR/nic_counters.csv"
META_FILE="$OUT_DIR/meta.txt"

echo "mode,run_type,run,status,exit_code,wall_s,inference_ms,total_images,accuracy,correct_predictions,predicted_class,confidence_pct,valid_output,log_file,timestamp" > "$RESULTS_CSV"
echo "mode,run_type,run,iface,rx_before,rx_after,rx_delta,tx_before,tx_after,tx_delta" > "$NIC_CSV"

cat > "$META_FILE" <<EOF
timestamp=$TS
host=$(hostname)
pwd=$(pwd)
runs=$RUNS
warmups=$WARMUPS
run_timeout=$RUN_TIMEOUT
ifaces=$IFACES
frontend_cmd=$FRONTEND_CMD
gvirtus_home=$GVIRTUS_HOME
gvirtus_repo=$GVIRTUS_REPO
gvirtus_config_dir=$GVIRTUS_CONFIG_DIR
opencv_home=$OPENCV_HOME
cudnn_root=$CUDNN_ROOT
npp_dir=$NPP_DIR
config_tcp=$CONFIG_TCP
config_rdma=$CONFIG_RDMA
config_ucx=$CONFIG_UCX
EOF

read_counter() {
  local iface="$1"
  local kind="$2"
  local path="/sys/class/net/$iface/statistics/${kind}_bytes"
  if [[ -r "$path" ]]; then
    cat "$path"
  else
    echo 0
  fi
}

dump_counters() {
  local mode="$1"
  local run_type="$2"
  local run="$3"
  local phase="$4"
  local file="$COUNTER_DIR/${mode}_${run_type}_${run}_${phase}.txt"

  : > "$file"
  for iface in $IFACES; do
    local rx tx
    rx="$(read_counter "$iface" rx)"
    tx="$(read_counter "$iface" tx)"
    echo "$iface rx=$rx tx=$tx" >> "$file"
  done
}

append_nic_delta() {
  local mode="$1"
  local run_type="$2"
  local run="$3"
  local before="$COUNTER_DIR/${mode}_${run_type}_${run}_before.txt"
  local after="$COUNTER_DIR/${mode}_${run_type}_${run}_after.txt"

  for iface in $IFACES; do
    local rx_b tx_b rx_a tx_a
    rx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || echo 0)"
    tx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || echo 0)"
    rx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || echo 0)"
    tx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || echo 0)"

    echo "$mode,$run_type,$run,$iface,$rx_b,$rx_a,$((rx_a-rx_b)),$tx_b,$tx_a,$((tx_a-tx_b))" >> "$NIC_CSV"
  done
}

config_for_mode() {
  case "$1" in
    tcp)  echo "$CONFIG_TCP" ;;
    rdma) echo "$CONFIG_RDMA" ;;
    ucx)  echo "$CONFIG_UCX" ;;
    *) echo "ERROR: unknown mode $1" >&2; exit 1 ;;
  esac
}

backend_hint() {
  local mode="$1"
  case "$mode" in
    tcp)
      cat <<EOF
GVIRTUS_CONFIG_FILE=properties.json GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev
EOF
      ;;
    rdma)
      cat <<EOF
GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_RDMA") GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev
EOF
      ;;
    ucx)
      cat <<EOF
GVIRTUS_UCX_DATAPATH=am \\
GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_UCX") \\
GVIRTUS_LOG_LEVEL=30000 \\
UCX_TLS=rc_mlx5,ud_mlx5,self \\
UCX_NET_DEVICES=mlx5_1:1 \\
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \\
UCX_IB_GID_INDEX=3 \\
UCX_RNDV_THRESH=inf \\
UCX_ZCOPY_THRESH=inf \\
UCX_LOG_LEVEL=warn \\
make run-gvirtus-backend-dev
EOF
      ;;
  esac
}

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -E "$pattern" "$file" | tail -1
}

run_one() {
  local mode="$1"
  local run_type="$2"
  local run="$3"
  local config
  config="$(config_for_mode "$mode")"

  local log="$LOG_DIR/${mode}_${run_type}_${run}.log"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"

  dump_counters "$mode" "$run_type" "$run" before

  local start_ns end_ns wall_s exit_code
  start_ns="$(date +%s%N)"

  if [[ "$mode" == "ucx" ]]; then
    timeout "$RUN_TIMEOUT" env \
      GVIRTUS_CONFIG="$config" \
      GVIRTUS_HOME="$GVIRTUS_HOME" \
      GVIRTUS_LOGLEVEL="$GVIRTUS_LOGLEVEL" \
      GVIRTUS_UCX_DATAPATH=am \
      UCX_TLS="${UCX_TLS:-rc_mlx5,ud_mlx5,self}" \
      UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_1:1}" \
      UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-rdmacm}" \
      UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-3}" \
      UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-inf}" \
      UCX_ZCOPY_THRESH="${UCX_ZCOPY_THRESH:-inf}" \
      UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}" \
      LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
      LD_PRELOAD="$GV_PRELOAD" \
      bash -lc "$FRONTEND_CMD" > "$log" 2>&1
    exit_code=$?
  else
    timeout "$RUN_TIMEOUT" env \
      GVIRTUS_CONFIG="$config" \
      GVIRTUS_HOME="$GVIRTUS_HOME" \
      GVIRTUS_LOGLEVEL="$GVIRTUS_LOGLEVEL" \
      LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
      LD_PRELOAD="$GV_PRELOAD" \
      bash -lc "$FRONTEND_CMD" > "$log" 2>&1
    exit_code=$?
  fi

  end_ns="$(date +%s%N)"
  wall_s="$(awk "BEGIN { printf \"%.6f\", ($end_ns - $start_ns) / 1000000000 }")"

  dump_counters "$mode" "$run_type" "$run" after
  append_nic_delta "$mode" "$run_type" "$run"

  local valid_output status
  if grep -q "Final Results:" "$log" && grep -q "Total images:" "$log" && grep -q "Saved total timings" "$log"; then
    valid_output="true"
  else
    valid_output="false"
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    status="OK"
  elif [[ "$valid_output" == "true" ]]; then
    status="SOFT-OK"
  else
    status="FAILED"
  fi

  local inference_ms total_images accuracy correct pred conf
  inference_ms="$(grep -Eo 'Time taken: [0-9.]+ ms' "$log" | tail -1 | awk '{print $3}')"
  total_images="$(grep -E 'Total images:' "$log" | tail -1 | awk -F': ' '{print $2}')"
  accuracy="$(grep -E 'Accuracy:' "$log" | tail -1 | awk -F': ' '{print $2}' | tr -d '%')"
  correct="$(grep -E 'Correct predictions:' "$log" | tail -1 | awk -F': ' '{print $2}')"
  pred="$(grep -Eo 'Predicted Class ID: [0-9]+' "$log" | tail -1 | awk '{print $4}')"
  conf="$(grep -Eo 'Confidence: [0-9.]+%' "$log" | tail -1 | awk '{print $2}' | tr -d '%')"

  inference_ms="${inference_ms:-}"
  total_images="${total_images:-}"
  accuracy="${accuracy:-}"
  correct="${correct:-}"
  pred="${pred:-}"
  conf="${conf:-}"

  echo "$mode,$run_type,$run,$status,$exit_code,$wall_s,$inference_ms,$total_images,$accuracy,$correct,$pred,$conf,$valid_output,$log,$timestamp" >> "$RESULTS_CSV"

  if [[ "$status" == "OK" ]]; then
    echo "  OK wall=${wall_s}s inference_ms=${inference_ms:-NA} images=${total_images:-NA} acc=${accuracy:-NA}%"
  elif [[ "$status" == "SOFT-OK" ]]; then
    echo "  SOFT-OK exit_code=$exit_code wall=${wall_s}s inference_ms=${inference_ms:-NA} images=${total_images:-NA} acc=${accuracy:-NA}%"
  else
    echo "  FAILED exit_code=$exit_code. See: $log"
  fi
}

run_mode() {
  local mode="$1"
  local config
  config="$(config_for_mode "$mode")"

  if [[ ! -f "$config" ]]; then
    echo "ERROR: missing config for $mode: $config"
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Mode: $mode"
  echo "Config: $config"
  echo "Frontend: $FRONTEND_CMD"
  echo "IFACES: $IFACES"
  echo "============================================================"
  echo "Start/restart backend with:"
  backend_hint "$mode"
  echo

  if [[ "${NO_PROMPT:-0}" != "1" ]]; then
    read -r -p "Press Enter when backend is ready..."
  fi

  local i
  for ((i=1; i<=WARMUPS; i++)); do
    echo "[$mode] warmup run $i..."
    run_one "$mode" "warmup" "$i"
  done

  for ((i=1; i<=RUNS; i++)); do
    echo "[$mode] measure run $i..."
    run_one "$mode" "measure" "$i"
  done
}

case "$MODE" in
  tcp) MODES="tcp" ;;
  rdma) MODES="rdma" ;;
  ucx) MODES="ucx" ;;
  all) MODES="tcp rdma ucx" ;;
  *)
    echo "Usage: $0 {tcp|rdma|ucx|all}"
    exit 1
    ;;
esac

echo "Benchmark output directory:"
echo "$OUT_DIR"

for m in $MODES; do
  run_mode "$m"
done

echo
echo "Done."
echo "Main CSV:    $RESULTS_CSV"
echo "NIC CSV:     $NIC_CSV"
echo "Logs:        $LOG_DIR"
echo "Counters:    $COUNTER_DIR"

echo
echo "Measured summary:"
awk -F',' '
NR == 1 { next }
$2 == "measure" && ($4 == "OK" || $4 == "SOFT-OK") {
  count[$1]++
  wall[$1]+=$6
  inf[$1]+=$7
}
END {
  for (m in count) {
    printf "%s: n=%d mean_wall_s=%.6f mean_inference_ms=%.3f\n", m, count[m], wall[m]/count[m], inf[m]/count[m]
  }
}
' "$RESULTS_CSV"

