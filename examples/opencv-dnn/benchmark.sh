#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-all}"
shift || true

FRONTEND_CMD="${FRONTEND_CMD:-./main}"
RUNS="${RUNS:-10}"
WARMUPS="${WARMUPS:-3}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"
IFACES="${IFACES:-lo}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-0.2}"
GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-30000}"
FAST_EXIT_MODES="${FAST_EXIT_MODES:-rdma}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  GVIRTUS_REPO_DEFAULT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  GVIRTUS_REPO_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

GVIRTUS_REPO="${GVIRTUS_REPO:-$GVIRTUS_REPO_DEFAULT}"
GVIRTUS_CONFIG_DIR="${GVIRTUS_CONFIG_DIR:-$GVIRTUS_REPO/etc}"

candidate_has_gvirtus_frontend() {
  local candidate="$1"
  [[ -f "$candidate/lib/frontend/libcuda.so" && \
     -f "$candidate/lib/frontend/libcudart.so" && \
     -f "$candidate/lib/frontend/libcublas.so" && \
     -f "$candidate/lib/frontend/libcudnn.so.9" && \
     -f "$candidate/lib/libgvirtus-frontend.so" ]]
}

detect_gvirtus_home() {
  if [[ -n "${GVIRTUS_HOME:-}" ]] && candidate_has_gvirtus_frontend "$GVIRTUS_HOME"; then
    echo "$GVIRTUS_HOME"
  elif candidate_has_gvirtus_frontend "$GVIRTUS_REPO/install"; then
    echo "$GVIRTUS_REPO/install"
  elif candidate_has_gvirtus_frontend "$HOME/gvirtus-install"; then
    echo "$HOME/gvirtus-install"
  elif candidate_has_gvirtus_frontend "/usr/local/gvirtus"; then
    echo "/usr/local/gvirtus"
  else
    echo ""
  fi
}

GVIRTUS_HOME="$(detect_gvirtus_home)"

if [[ -z "$GVIRTUS_HOME" ]]; then
  echo "ERROR: Could not detect GVIRTUS_HOME."
  echo "Set it explicitly, for example:"
  echo "  export GVIRTUS_HOME=/path/to/gvirtus-install"
  exit 1
fi

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
    echo "$CUDA_HOME"
  elif command -v nvcc >/dev/null 2>&1; then
    dirname "$(dirname "$(command -v nvcc)")"
  elif [[ -d "/usr/local/cuda" ]]; then
    echo "/usr/local/cuda"
  else
    echo ""
  fi
}

CUDA_HOME="$(detect_cuda_home)"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-}"

if [[ -z "$CUDA_LIB_DIR" && -n "$CUDA_HOME" ]]; then
  if [[ -d "$CUDA_HOME/lib64" ]]; then
    CUDA_LIB_DIR="$CUDA_HOME/lib64"
  elif [[ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]]; then
    CUDA_LIB_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
  fi
fi

OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
NPP_DIR="${NPP_DIR:-}"
CUDNN_ROOT="${CUDNN_ROOT:-}"
CUDNN_LIB="${CUDNN_LIB:-}"

if [[ -z "$CUDNN_LIB" && -n "$CUDNN_ROOT" ]]; then
  CUDNN_LIB="$CUDNN_ROOT/lib"
fi

LZ4_LIB="${LZ4_LIB:-}"

CONFIG_TCP="${CONFIG_TCP:-$GVIRTUS_CONFIG_DIR/properties.json}"
CONFIG_RDMA="${CONFIG_RDMA:-$GVIRTUS_CONFIG_DIR/properties_plain_rdma.json}"
CONFIG_UCX="${CONFIG_UCX:-$GVIRTUS_CONFIG_DIR/properties_ucx.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frontend)
      FRONTEND_CMD="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --warmups)
      WARMUPS="$2"
      shift 2
      ;;
    --timeout)
      RUN_TIMEOUT="$2"
      shift 2
      ;;
    --ifaces)
      IFACES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

append_existing_path() {
  local current="$1"
  local candidate="$2"

  if [[ -n "$candidate" && -d "$candidate" ]]; then
    if [[ -n "$current" ]]; then
      echo "$current:$candidate"
    else
      echo "$candidate"
    fi
  else
    echo "$current"
  fi
}

GV_LD_LIBRARY_PATH=""
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$GVIRTUS_HOME/lib")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$GVIRTUS_HOME/lib/frontend")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$OPENCV_HOME/lib")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$CUDNN_LIB")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$NPP_DIR")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$CUDA_LIB_DIR")"
GV_LD_LIBRARY_PATH="$(append_existing_path "$GV_LD_LIBRARY_PATH" "$LZ4_LIB")"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  GV_LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
fi

GV_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcuda.so:$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"

for lib in \
  "$GVIRTUS_HOME/lib/frontend/libcuda.so" \
  "$GVIRTUS_HOME/lib/frontend/libcudart.so" \
  "$GVIRTUS_HOME/lib/frontend/libcublas.so" \
  "$GVIRTUS_HOME/lib/frontend/libcudnn.so.9" \
  "$GVIRTUS_HOME/lib/libgvirtus-frontend.so"; do
  if [[ ! -f "$lib" ]]; then
    echo "ERROR: missing GVirtuS frontend library: $lib"
    echo "GVIRTUS_HOME is currently: $GVIRTUS_HOME"
    exit 1
  fi
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmark_results/frontend_${TS}_${MODE}"
LOG_DIR="$OUT_DIR/logs"
GPU_DIR="$OUT_DIR/gpu"
COUNTER_DIR="$OUT_DIR/counters"

mkdir -p "$LOG_DIR" "$GPU_DIR" "$COUNTER_DIR"

RESULTS_CSV="$OUT_DIR/results.csv"
PROCESS_CSV="$OUT_DIR/process_results.csv"
NIC_CSV="$OUT_DIR/nic_counters.csv"
GPU_CSV="$OUT_DIR/gpu_summary.csv"
META_FILE="$OUT_DIR/meta.txt"
SUMMARY_FILE="$OUT_DIR/summary.txt"

echo "mode,run_type,run,status,inference_ms,log_file,timestamp" > "$RESULTS_CSV"
echo "mode,status,exit_code,wall_s,valid_output,output_file,output_bytes,log_file,timestamp" > "$PROCESS_CSV"
echo "mode,iface,rx_before,rx_after,rx_delta,tx_before,tx_after,tx_delta" > "$NIC_CSV"
echo "mode,gpu_samples,gpu_util_avg,gpu_util_max,gpu_mem_avg_mib,gpu_mem_max_mib,gpu_power_avg_w,gpu_power_max_w,gpu_log_file" > "$GPU_CSV"

cat > "$META_FILE" <<EOF
timestamp=$TS
host=$(hostname)
pwd=$(pwd)
mode=$MODE
frontend_cmd=$FRONTEND_CMD
runs=$RUNS
warmups=$WARMUPS
run_timeout=$RUN_TIMEOUT
ifaces=$IFACES
gpu_sample_interval=$GPU_SAMPLE_INTERVAL
gvirtus_repo=$GVIRTUS_REPO
gvirtus_home=$GVIRTUS_HOME
gvirtus_config_dir=$GVIRTUS_CONFIG_DIR
config_tcp=$CONFIG_TCP
config_rdma=$CONFIG_RDMA
config_ucx=$CONFIG_UCX
opencv_home=$OPENCV_HOME
cuda_home=$CUDA_HOME
cuda_lib_dir=$CUDA_LIB_DIR
cudnn_root=$CUDNN_ROOT
cudnn_lib=$CUDNN_LIB
npp_dir=$NPP_DIR
lz4_lib=$LZ4_LIB
fast_exit_modes=$FAST_EXIT_MODES
EOF

usage() {
  cat <<EOF
Usage:
  ./benchmark.sh {tcp|rdma|ucx|all} [options]

Options:
  --frontend CMD      Frontend command to run, default: ./main
  --runs N            Measured runs inside one frontend process, default: 10
  --warmups N         Warmup runs inside one frontend process, default: 3
  --timeout SEC       Per-transport timeout, default: 120
  --ifaces "IFACES"   Space-separated network interfaces, default: lo

Examples:
  ./benchmark.sh tcp --frontend "./main"
  ./benchmark.sh rdma --frontend "./main"
  ./benchmark.sh ucx --frontend "./main"
  ./benchmark.sh all --frontend "./main"

Environment overrides:
  GVIRTUS_HOME=/path/to/gvirtus-install ./benchmark.sh all
  OPENCV_HOME=/path/to/opencv-install ./benchmark.sh all
  CUDA_HOME=/usr/local/cuda ./benchmark.sh all
  CUDNN_ROOT=/path/to/cudnn ./benchmark.sh all
  NPP_DIR=/path/to/npp/lib ./benchmark.sh all
  RUNS=10 WARMUPS=3 FRONTEND_CMD="./main" ./benchmark.sh all
EOF
}

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
  local phase="$2"
  local file="$COUNTER_DIR/${mode}_${phase}.txt"

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
  local before="$COUNTER_DIR/${mode}_before.txt"
  local after="$COUNTER_DIR/${mode}_after.txt"

  for iface in $IFACES; do
    local rx_b tx_b rx_a tx_a
    rx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || echo 0)"
    tx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || echo 0)"
    rx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || echo 0)"
    tx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || echo 0)"

    echo "$mode,$iface,$rx_b,$rx_a,$((rx_a-rx_b)),$tx_b,$tx_a,$((tx_a-tx_b))" >> "$NIC_CSV"
  done
}

config_for_mode() {
  case "$1" in
    tcp)  echo "$CONFIG_TCP" ;;
    rdma) echo "$CONFIG_RDMA" ;;
    ucx)  echo "$CONFIG_UCX" ;;
    *)
      echo "ERROR: unknown mode $1" >&2
      exit 1
      ;;
  esac
}

backend_hint() {
  local mode="$1"

  case "$mode" in
    tcp)
      cat <<EOF
cd "$GVIRTUS_REPO"
make stop-gvirtus || true
GVIRTUS_CONFIG_FILE=properties.json \\
GVIRTUS_LOG_LEVEL=30000 \\
make run-gvirtus-backend-dev
EOF
      ;;
    rdma)
      cat <<EOF
cd "$GVIRTUS_REPO"
make stop-gvirtus || true
GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \\
GVIRTUS_LOG_LEVEL=30000 \\
make run-gvirtus-backend-dev
EOF
      ;;
    ucx)
      cat <<EOF
cd "$GVIRTUS_REPO"
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \\
GVIRTUS_CONFIG_FILE=properties_ucx.json \\
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

start_gpu_sampler() {
  local file="$1"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    : > "$file"
    echo ""
    return
  fi

  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits 2>/dev/null
      sleep "$GPU_SAMPLE_INTERVAL"
    done
  ) > "$file" &
  echo "$!"
}

stop_gpu_sampler() {
  local pid="$1"
  if [[ -n "$pid" ]]; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" 2>/dev/null || true
  fi
}

append_gpu_summary() {
  local mode="$1"
  local file="$2"

  if [[ ! -s "$file" ]]; then
    echo "$mode,0,,,,,,,$file" >> "$GPU_CSV"
    return
  fi

  python3 - "$mode" "$file" "$GPU_CSV" <<'PY'
import csv
import sys
from statistics import mean

mode, file, out_csv = sys.argv[1:]

gpu_utils = []
gpu_mems = []
gpu_powers = []

with open(file, "r", errors="ignore") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 5:
            continue
        try:
            gpu_utils.append(float(parts[2]))
            gpu_mems.append(float(parts[3]))
            gpu_powers.append(float(parts[4]))
        except ValueError:
            continue

def avg(xs):
    return mean(xs) if xs else ""

def mx(xs):
    return max(xs) if xs else ""

row = [
    mode,
    len(gpu_utils),
    avg(gpu_utils),
    mx(gpu_utils),
    avg(gpu_mems),
    mx(gpu_mems),
    avg(gpu_powers),
    mx(gpu_powers),
    file,
]

with open(out_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
}

valid_output_for_log() {
  local log="$1"

  if grep -q "Detection finished. Results saved to output.jpg" "$log"; then
    return 0
  fi

  if grep -q "Final Results:" "$log" && grep -q "Total images:" "$log" && grep -q "Saved total timings" "$log"; then
    return 0
  fi

  return 1
}

append_bench_results() {
  local mode="$1"
  local log="$2"

  python3 - "$mode" "$log" "$RESULTS_CSV" <<'PY'
import csv
import re
import sys
from datetime import datetime

mode, log, out_csv = sys.argv[1:]
pat = re.compile(r"BENCH_RESULT,type=([^,]+),run=([0-9]+),inference_ms=([0-9.]+)")
timestamp = datetime.now().isoformat(timespec="seconds")

rows = []
with open(log, "r", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append([mode, m.group(1), m.group(2), "OK", m.group(3), log, timestamp])

with open(out_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

if not rows:
    sys.exit(1)
PY
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
  echo "Mode:        $mode"
  echo "Config:      $config"
  echo "Frontend:    $FRONTEND_CMD"
  echo "Warmups:     $WARMUPS"
  echo "Runs:        $RUNS"
  echo "Timeout:     $RUN_TIMEOUT"
  echo "IFACES:      $IFACES"
  echo "Output dir:  $OUT_DIR"
  echo "============================================================"
  echo
  echo "Start/restart backend with:"
  backend_hint "$mode"
  echo

  if [[ "${NO_PROMPT:-0}" != "1" ]]; then
    read -r -p "Press Enter when the $mode backend is ready..."
  fi

  local log="$LOG_DIR/${mode}.log"
  local gpu_log="$GPU_DIR/${mode}_gpu.csv"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"

  rm -f output.jpg
  dump_counters "$mode" before

  local gpu_pid
  gpu_pid="$(start_gpu_sampler "$gpu_log")"

  local start_ns end_ns wall_s exit_code
  start_ns="$(date +%s%N)"

  local fast_exit_env=()
  if [[ " $FAST_EXIT_MODES " == *" $mode "* ]]; then
    fast_exit_env=(GVIRTUS_FAST_EXIT_AFTER_RESULT=1)
  fi

  if [[ "$mode" == "ucx" ]]; then
    timeout "$RUN_TIMEOUT" env -u GVIRTUS_FAST_EXIT_AFTER_RESULT \
      "${fast_exit_env[@]}" \
      BENCH_WARMUPS="$WARMUPS" \
      BENCH_RUNS="$RUNS" \
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
      GV_PRELOAD="$GV_PRELOAD" \
      FRONTEND_CMD="$FRONTEND_CMD" \
      bash -lc 'export LD_PRELOAD="$GV_PRELOAD"; eval "exec $FRONTEND_CMD"' > "$log" 2>&1
    exit_code=$?
  else
    timeout "$RUN_TIMEOUT" env -u GVIRTUS_FAST_EXIT_AFTER_RESULT \
      "${fast_exit_env[@]}" \
      BENCH_WARMUPS="$WARMUPS" \
      BENCH_RUNS="$RUNS" \
      GVIRTUS_CONFIG="$config" \
      GVIRTUS_HOME="$GVIRTUS_HOME" \
      GVIRTUS_LOGLEVEL="$GVIRTUS_LOGLEVEL" \
      LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
      GV_PRELOAD="$GV_PRELOAD" \
      FRONTEND_CMD="$FRONTEND_CMD" \
      bash -lc 'export LD_PRELOAD="$GV_PRELOAD"; eval "exec $FRONTEND_CMD"' > "$log" 2>&1
    exit_code=$?
  fi

  end_ns="$(date +%s%N)"
  stop_gpu_sampler "$gpu_pid"

  wall_s="$(awk "BEGIN { printf \"%.6f\", ($end_ns - $start_ns) / 1000000000 }")"

  dump_counters "$mode" after
  append_nic_delta "$mode"
  append_gpu_summary "$mode" "$gpu_log"

  local valid_output status output_file output_bytes

  if valid_output_for_log "$log"; then
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

  output_file=""
  output_bytes=""

  if [[ -f output.jpg ]]; then
    output_file="output.jpg"
    output_bytes="$(stat -c%s output.jpg 2>/dev/null || echo "")"
    cp output.jpg "$OUT_DIR/${mode}_output.jpg" 2>/dev/null || true
  fi

  echo "$mode,$status,$exit_code,$wall_s,$valid_output,$output_file,$output_bytes,$log,$timestamp" >> "$PROCESS_CSV"

  if append_bench_results "$mode" "$log"; then
    :
  else
    echo "WARNING: no BENCH_RESULT lines found in $log"
  fi

  if [[ "$status" == "OK" ]]; then
    echo "  OK       wall=${wall_s}s output_bytes=${output_bytes:-NA}"
  elif [[ "$status" == "SOFT-OK" ]]; then
    echo "  SOFT-OK  exit=$exit_code wall=${wall_s}s output_bytes=${output_bytes:-NA}"
  else
    echo "  FAILED   exit=$exit_code wall=${wall_s}s log=$log"
    return 1
  fi

  grep "BENCH_RESULT" "$log" || true
}

write_summary() {
  python3 - "$RESULTS_CSV" "$PROCESS_CSV" "$NIC_CSV" "$GPU_CSV" "$SUMMARY_FILE" <<'PY'
import csv
import sys
from statistics import mean, median, pstdev

results_csv, process_csv, nic_csv, gpu_csv, summary_file = sys.argv[1:]

def fnum(x):
    try:
        if x == "":
            return None
        return float(x)
    except Exception:
        return None

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "min": min(xs),
        "max": max(xs),
        "std": pstdev(xs) if len(xs) > 1 else 0.0,
    }

results = []
with open(results_csv, newline="") as f:
    for r in csv.DictReader(f):
        if r["run_type"] == "measure" and r["status"] == "OK":
            results.append(r)

process = {}
try:
    with open(process_csv, newline="") as f:
        for r in csv.DictReader(f):
            process[r["mode"]] = r
except FileNotFoundError:
    pass

nic = {}
try:
    with open(nic_csv, newline="") as f:
        for r in csv.DictReader(f):
            nic.setdefault(r["mode"], []).append(r)
except FileNotFoundError:
    pass

gpu = {}
try:
    with open(gpu_csv, newline="") as f:
        for r in csv.DictReader(f):
            gpu[r["mode"]] = r
except FileNotFoundError:
    pass

by_mode = {}
for r in results:
    by_mode.setdefault(r["mode"], []).append(r)

lines = []
lines.append("Benchmark summary")
lines.append("=================")
lines.append("")

for mode in sorted(set(list(by_mode.keys()) + list(process.keys()))):
    lines.append(f"[{mode}]")

    pr = process.get(mode)
    if pr:
        lines.append(f"process_status={pr['status']}")
        lines.append(f"process_exit_code={pr['exit_code']}")
        lines.append(f"process_wall_s={pr['wall_s']}")
        lines.append(f"valid_output={pr['valid_output']}")
        if pr.get("output_bytes"):
            lines.append(f"output_bytes={pr['output_bytes']}")

    rs = by_mode.get(mode, [])
    inf = stats([fnum(r["inference_ms"]) for r in rs])
    lines.append(f"valid_measured_runs={len(rs)}")

    if inf:
        lines.append(
            "inference_ms: "
            f"median={inf['median']:.3f}, mean={inf['mean']:.3f}, "
            f"min={inf['min']:.3f}, max={inf['max']:.3f}, std={inf['std']:.3f}"
        )

    if mode in nic:
        for r in nic[mode]:
            lines.append(
                f"nic[{r['iface']}]: "
                f"rx_delta={r['rx_delta']}, tx_delta={r['tx_delta']}"
            )

    gr = gpu.get(mode)
    if gr:
        lines.append(f"gpu_samples={gr['gpu_samples']}")
        if gr.get("gpu_util_avg"):
            lines.append(f"gpu_util_avg_percent={float(gr['gpu_util_avg']):.2f}")
        if gr.get("gpu_util_max"):
            lines.append(f"gpu_util_peak_percent={float(gr['gpu_util_max']):.2f}")
        if gr.get("gpu_mem_max_mib"):
            lines.append(f"gpu_mem_peak_mib={float(gr['gpu_mem_max_mib']):.2f}")
        if gr.get("gpu_power_avg_w"):
            lines.append(f"gpu_power_avg_w={float(gr['gpu_power_avg_w']):.2f}")

    lines.append("")

with open(summary_file, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
PY
}

case "$MODE" in
  tcp) MODES="tcp" ;;
  rdma) MODES="rdma" ;;
  ucx) MODES="ucx" ;;
  all) MODES="tcp rdma ucx" ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Usage: $0 {tcp|rdma|ucx|all}"
    echo
    usage
    exit 1
    ;;
esac

echo "Benchmark output directory:"
echo "$OUT_DIR"

for m in $MODES; do
  run_mode "$m" || exit 1
done

echo
echo "Writing summary..."
write_summary

echo
echo "Done."
echo "Results CSV:        $RESULTS_CSV"
echo "Process CSV:        $PROCESS_CSV"
echo "NIC CSV:            $NIC_CSV"
echo "GPU CSV:            $GPU_CSV"
echo "Summary:            $SUMMARY_FILE"
echo "Logs:               $LOG_DIR"
echo "GPU samples:        $GPU_DIR"
echo "Counters:           $COUNTER_DIR"
