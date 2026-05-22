#!/usr/bin/env bash
set -uo pipefail

# OpenCV-DNN GVirtuS frontend benchmark with publication-ready metrics.
# Usage:
#   ./benchmark_opencv_dnn_full_metrics.sh tcp|rdma|ucx|all
#
# Typical final run:
#   RUNS=50 WARMUPS=5 RUN_TIMEOUT=120 IFACES="ens1f1np1 ens1f0np0 bond0" ./benchmark_opencv_dnn_full_metrics.sh tcp

MODE="${1:-all}"

RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-1}"
RUN_TIMEOUT="${RUN_TIMEOUT:-30}"
IFACES="${IFACES:-ens1f1np1}"
FRONTEND_CMD="${FRONTEND_CMD:-./sample}"
NO_PROMPT="${NO_PROMPT:-0}"
COLLECT_SYSTEM_METADATA="${COLLECT_SYSTEM_METADATA:-1}"

GVIRTUS_HOME="${GVIRTUS_HOME:-/home/student.aau.dk/ul11nh/GVirtuS}"
OPENCV_HOME="${OPENCV_HOME:-/home/student.aau.dk/ul11nh/opencv-local}"
NPP_DIR="${NPP_DIR:-/home/student.aau.dk/ul11nh/.local/lib/python3.10/site-packages/nvidia/npp/lib}"
CUDNN_ROOT="${CUDNN_ROOT:-/home/student.aau.dk/ul11nh/cudnn-9.5.1}"
CUDNN_LIB="${CUDNN_LIB:-$CUDNN_ROOT/lib}"
LZ4_HOME="${LZ4_HOME:-/home/student.aau.dk/ul11nh/lz4-install}"
LZ4_LIB="${LZ4_LIB:-$LZ4_HOME/lib}"

# Supports both TCP_CONFIG=... and CONFIG_TCP=... naming.
CONFIG_TCP="${TCP_CONFIG:-${CONFIG_TCP:-$GVIRTUS_HOME/etc/properties.json}}"
CONFIG_RDMA="${RDMA_CONFIG:-${CONFIG_RDMA:-$GVIRTUS_HOME/etc/properties_plain_rdma.json}}"
CONFIG_UCX="${UCX_CONFIG:-${CONFIG_UCX:-$GVIRTUS_HOME/etc/properties_ucx.json}}"

GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-30000}"

GV_LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_HOME/lib:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$LZ4_LIB:${LD_LIBRARY_PATH:-}"
GV_PRELOAD=""
for lib in \
  "$GVIRTUS_HOME/lib/frontend/libcuda.so" \
  "$GVIRTUS_HOME/lib/frontend/libcudart.so" \
  "$GVIRTUS_HOME/lib/frontend/libcublas.so" \
  "$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"
do
  if [[ -e "$lib" ]]; then
    if [[ -z "$GV_PRELOAD" ]]; then
      GV_PRELOAD="$lib"
    else
      GV_PRELOAD="$GV_PRELOAD:$lib"
    fi
  fi
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-benchmark_results/opencvdnn_full_metrics_${TS}_${MODE}}"
LOG_DIR="$OUT_DIR/logs"
COUNTER_DIR="$OUT_DIR/counters"
SYSTEM_DIR="$OUT_DIR/system"

mkdir -p "$LOG_DIR" "$COUNTER_DIR" "$SYSTEM_DIR"

RESULTS_CSV="$OUT_DIR/results.csv"
ROUTINE_CALLS_CSV="$OUT_DIR/routine_calls.csv"
ROUTINE_SUMMARY_CSV="$OUT_DIR/routine_summary.csv"
NIC_CSV="$OUT_DIR/nic_counters.csv"
META_FILE="$OUT_DIR/meta.txt"

python3 - <<PYCSVINIT
import csv
from pathlib import Path
headers = {
    "$RESULTS_CSV": [
        "timestamp","mode","phase","run","status","exit_code","wall_s",
        "inference_ms","total_images","accuracy_pct","correct_predictions",
        "predicted_class","confidence_pct","valid_output",
        "calls","gvirtus_in_B","gvirtus_out_B","nic_rx_B","nic_tx_B",
        "config","log_file","error","routine_counts"
    ],
    "$ROUTINE_CALLS_CSV": [
        "timestamp","mode","phase","run","call_index","routine","latency_ms","source"
    ],
    "$ROUTINE_SUMMARY_CSV": [
        "timestamp","mode","phase","run","routine","count","percent_of_run_calls"
    ],
    "$NIC_CSV": [
        "timestamp","mode","phase","run","iface","rx_before","rx_after","rx_delta","tx_before","tx_after","tx_delta"
    ],
}
for path, header in headers.items():
    with Path(path).open("w", newline="") as f:
        csv.writer(f).writerow(header)
PYCSVINIT

cat > "$META_FILE" <<EOFMETA
timestamp=$TS
host=$(hostname)
pwd=$(pwd)
runs=$RUNS
warmups=$WARMUPS
run_timeout=$RUN_TIMEOUT
ifaces=$IFACES
frontend_cmd=$FRONTEND_CMD
gvirtus_home=$GVIRTUS_HOME
opencv_home=$OPENCV_HOME
cudnn_root=$CUDNN_ROOT
npp_dir=$NPP_DIR
lz4_home=$LZ4_HOME
config_tcp=$CONFIG_TCP
config_rdma=$CONFIG_RDMA
config_ucx=$CONFIG_UCX
gvirtus_loglevel=$GVIRTUS_LOGLEVEL
EOFMETA

if [[ "$COLLECT_SYSTEM_METADATA" == "1" ]]; then
  {
    echo "hostname=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "kernel=$(uname -a)"
    echo "cmdline=$0 $*"
    echo
    echo "=== env relevant ==="
    env | grep -E '^(GVIRTUS|UCX|RUNS|WARMUPS|RUN_TIMEOUT|IFACES|OPENCV|CUDNN|LZ4|LD_)' | sort || true
    echo
    echo "=== ip -br addr ==="
    ip -br addr 2>/dev/null || true
    echo
    echo "=== rdma link ==="
    rdma link 2>/dev/null || true
    echo
    echo "=== ibdev2netdev ==="
    ibdev2netdev 2>/dev/null || true
    echo
echo "ldd skipped for benchmark stability" >> "$SYSTEM_DIR/frontend_ldd.txt" 2>/dev/null || true
    first_word="${FRONTEND_CMD%% *}"
    if [[ -x "$first_word" ]]; then
echo "ldd skipped for benchmark stability" >> "$SYSTEM_DIR/frontend_ldd.txt" 2>/dev/null || true
    fi
  } > "$SYSTEM_DIR/system_metadata.txt" 2>&1
fi


safe_ldd_frontend() {
  local cmd_bin
  cmd_bin="$(printf "%s\n" "$FRONTEND_CMD" | awk '{print $1}')"

  if [[ -x "$cmd_bin" ]] && file "$cmd_bin" 2>/dev/null | grep -q "ELF"; then
echo "ldd skipped for benchmark stability" >> "$SYSTEM_DIR/frontend_ldd.txt" 2>/dev/null || true
  else
    echo "Skipping ldd: FRONTEND_CMD first token is not an ELF binary: $cmd_bin"
    file "$cmd_bin" 2>/dev/null || true
  fi
}

read_counter() {
  local iface="$1"
  local kind="$2"
  local path="/sys/class/net/$iface/statistics/${kind}_bytes"
  if [[ -r "$path" ]]; then cat "$path"; else echo 0; fi
}

dump_counters() {
  local mode="$1" phase="$2" run="$3" when="$4"
  local file="$COUNTER_DIR/${mode}_${phase}_${run}_${when}.txt"
  : > "$file"
  for iface in $IFACES; do
    echo "$iface rx=$(read_counter "$iface" rx) tx=$(read_counter "$iface" tx)" >> "$file"
  done
}

append_nic_delta() {
  local timestamp="$1" mode="$2" phase="$3" run="$4"
  local before="$COUNTER_DIR/${mode}_${phase}_${run}_before.txt"
  local after="$COUNTER_DIR/${mode}_${phase}_${run}_after.txt"
  local totals="$COUNTER_DIR/${mode}_${phase}_${run}_total_delta.txt"
  local total_rx=0 total_tx=0

  for iface in $IFACES; do
    local rx_b tx_b rx_a tx_a rx_d tx_d
    rx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || true)"
    tx_b="$(grep "^$iface " "$before" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || true)"
    rx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*rx=([0-9]+).*/\1/' || true)"
    tx_a="$(grep "^$iface " "$after" 2>/dev/null | sed -E 's/.*tx=([0-9]+).*/\1/' || true)"
    rx_b="${rx_b:-0}"; tx_b="${tx_b:-0}"; rx_a="${rx_a:-0}"; tx_a="${tx_a:-0}"
    rx_d=$((rx_a-rx_b)); tx_d=$((tx_a-tx_b))
    total_rx=$((total_rx+rx_d)); total_tx=$((total_tx+tx_d))
    python3 - <<PYNIC
import csv
with open("$NIC_CSV", "a", newline="") as f:
    csv.writer(f).writerow(["$timestamp","$mode","$phase","$run","$iface","$rx_b","$rx_a","$rx_d","$tx_b","$tx_a","$tx_d"])
PYNIC
  done
  echo "rx=$total_rx tx=$total_tx" > "$totals"
}

config_for_mode() {
  case "$1" in
    tcp) echo "$CONFIG_TCP" ;;
    rdma) echo "$CONFIG_RDMA" ;;
    ucx) echo "$CONFIG_UCX" ;;
    *) echo "ERROR: unknown mode $1" >&2; exit 1 ;;
  esac
}

backend_hint() {
  local mode="$1"
  case "$mode" in
    tcp)
      echo "GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_TCP") GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev"
      ;;
    rdma)
      echo "GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_RDMA") GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev"
      ;;
    ucx)
      cat <<EOFUCXHINT
UCX-TCP backend example:
UCX_TLS=tcp UCX_NET_DEVICES=ens1f1np1 UCX_SOCKADDR_TLS_PRIORITY=tcp \\
UCX_PROTO_ENABLE=n UCX_RNDV_THRESH=inf UCX_ZCOPY_THRESH=inf \\
UCX_MAX_EAGER_RAILS=1 UCX_MAX_RNDV_RAILS=1 UCX_LOG_LEVEL=warn \\
GVIRTUS_UCX_DATAPATH=am GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_UCX") \\
GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev

UCX-RDMA backend example:
UCX_TLS=rc_x,tcp,self UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \\
UCX_SOCKADDR_TLS_PRIORITY=tcp UCX_PROTO_ENABLE=y UCX_LOG_LEVEL=warn \\
GVIRTUS_UCX_DATAPATH=am GVIRTUS_CONFIG_FILE=$(basename "$CONFIG_UCX") \\
GVIRTUS_LOG_LEVEL=30000 make run-gvirtus-backend-dev
EOFUCXHINT
      ;;
  esac
}

parse_and_append_metrics() {
  local timestamp="$1" mode="$2" phase="$3" run="$4" status="$5" exit_code="$6" wall_s="$7" config="$8" log="$9"
  local totals_file="$COUNTER_DIR/${mode}_${phase}_${run}_total_delta.txt"
  local nic_rx_B=0 nic_tx_B=0
  if [[ -f "$totals_file" ]]; then
    nic_rx_B="$(sed -E 's/.*rx=([0-9-]+).*/\1/' "$totals_file")"
    nic_tx_B="$(sed -E 's/.*tx=([0-9-]+).*/\1/' "$totals_file")"
  fi

  TIMESTAMP="$timestamp" MODE_VALUE="$mode" PHASE="$phase" RUN_VALUE="$run" STATUS_VALUE="$status" \
  EXIT_CODE="$exit_code" WALL_S="$wall_s" CONFIG_VALUE="$config" LOG_VALUE="$log" \
  NIC_RX_B="$nic_rx_B" NIC_TX_B="$nic_tx_B" RESULTS_CSV="$RESULTS_CSV" \
  ROUTINE_CALLS_CSV="$ROUTINE_CALLS_CSV" ROUTINE_SUMMARY_CSV="$ROUTINE_SUMMARY_CSV" \
  python3 - <<'PYMETRICS'
import csv, os, re
from collections import Counter
from pathlib import Path

log_path = Path(os.environ["LOG_VALUE"])
text = log_path.read_text(errors="replace") if log_path.exists() else ""
lines = text.splitlines()

def last_match(patterns):
    value = ""
    for line in lines:
        for pat in patterns:
            m = re.search(pat, line, re.I)
            if m: value = m.group(1).strip()
    return value

inference_ms = last_match([r"Time taken:\s*([0-9.]+)\s*ms", r"Inference(?: time)?(?: took)?:\s*([0-9.]+)\s*ms", r"inference_ms[=:]\s*([0-9.]+)"])
total_images = last_match([r"Total images:\s*([0-9]+)", r"total_images[=:]\s*([0-9]+)"])
accuracy = last_match([r"Accuracy:\s*([0-9.]+)\s*%?", r"accuracy[=:]\s*([0-9.]+)"])
correct = last_match([r"Correct predictions:\s*([^\r\n]+)", r"correct_predictions[=:]\s*([^\s,]+)"])
pred = last_match([r"Predicted Class ID:\s*([0-9]+)", r"predicted_class[=:]\s*([^\s,]+)"])
conf = last_match([r"Confidence:\s*([0-9.]+)\s*%", r"confidence(?:_pct)?[=:]\s*([0-9.]+)"])
valid_output = "true" if ("Final Results:" in text or total_images or accuracy or inference_ms) else "false"

received = []
for line in lines:
    m = re.search(r"Received routine\s+([A-Za-z0-9_]+)", line)
    if m: received.append((m.group(1), "received"))

routines = received
if not routines:
    for line in lines:
        for pat, src in [
            (r"AM routine '([^']+)' returned", "am_returned"),
            (r"Routine '([^']+)' returned", "returned"),
            (r"Called:\s*([A-Za-z0-9_]+)", "called"),
            (r"\[RDMA getstring\]\s+routine=\[?([A-Za-z0-9_]+)\]?", "rdma_getstring"),
        ]:
            m = re.search(pat, line)
            if m:
                routines.append((m.group(1), src)); break

# Best-effort communicator byte estimates from debug logs. NIC deltas are the authoritative byte metric.
gvirtus_out = gvirtus_in = write_seen = read_seen = 0
for line in lines:
    if "Write" in line:
        for pat in [r"Write\(AM\).*bytes=([0-9]+)", r"Called Write\([^)]*\).*Size:\s*([0-9]+)"]:
            m = re.search(pat, line)
            if m: gvirtus_out += int(m.group(1)); write_seen += 1; break
    if "Read" in line or "recv completed" in line:
        for pat in [r"Read\(AM\).*bytes=([0-9]+)", r"Called Read\([^)]*\).*Size:\s*([0-9]+)", r"recv completed byte_len=([0-9]+)"]:
            m = re.search(pat, line)
            if m: gvirtus_in += int(m.group(1)); read_seen += 1; break

gvirtus_in_value = str(gvirtus_in) if read_seen else ""
gvirtus_out_value = str(gvirtus_out) if write_seen else ""
counts = Counter(r for r, _ in routines)
routine_counts = ";".join(f"{k}:{counts[k]}" for k in sorted(counts))
error = ""
for pat in ["FAILED", "Exception", "Connection refused", "Destination", "Unsupported", "timeout", "Aborted", "Buffer::Get"]:
    if re.search(pat, text, re.I): error = pat; break

calls = len(routines)
ts, mode, phase, run = os.environ["TIMESTAMP"], os.environ["MODE_VALUE"], os.environ["PHASE"], os.environ["RUN_VALUE"]
with open(os.environ["RESULTS_CSV"], "a", newline="") as f:
    csv.writer(f).writerow([ts, mode, phase, run, os.environ["STATUS_VALUE"], os.environ["EXIT_CODE"], os.environ["WALL_S"], inference_ms, total_images, accuracy, correct, pred, conf, valid_output, calls, gvirtus_in_value, gvirtus_out_value, os.environ["NIC_RX_B"], os.environ["NIC_TX_B"], os.environ["CONFIG_VALUE"], str(log_path), error, routine_counts])
with open(os.environ["ROUTINE_CALLS_CSV"], "a", newline="") as f:
    w = csv.writer(f)
    for i, (routine, src) in enumerate(routines, start=1): w.writerow([ts, mode, phase, run, i, routine, "", src])
with open(os.environ["ROUTINE_SUMMARY_CSV"], "a", newline="") as f:
    w = csv.writer(f); denom = calls or 1
    for routine in sorted(counts): w.writerow([ts, mode, phase, run, routine, counts[routine], f"{100*counts[routine]/denom:.6f}"])
PYMETRICS
}

run_one() {
  local mode="$1" phase="$2" run="$3" config
  config="$(config_for_mode "$mode")"
  local log="$LOG_DIR/${mode}_${phase}_${run}.log"
  local timestamp="$(date --iso-8601=seconds)"
  dump_counters "$mode" "$phase" "$run" before

  local start_ns end_ns wall_s exit_code status
  start_ns="$(date +%s%N)"
  if [[ "$mode" == "ucx" ]]; then
    timeout "$RUN_TIMEOUT" env \
      GVIRTUS_CONFIG="$config" GVIRTUS_HOME="$GVIRTUS_HOME" GVIRTUS_LOGLEVEL="$GVIRTUS_LOGLEVEL" \
      GVIRTUS_UCX_DATAPATH="${GVIRTUS_UCX_DATAPATH:-am}" \
      UCX_TLS="${UCX_TLS:-tcp}" UCX_NET_DEVICES="${UCX_NET_DEVICES:-ens1f1np1}" \
      UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp}" UCX_PROTO_ENABLE="${UCX_PROTO_ENABLE:-n}" \
      UCX_RNDV_THRESH="${UCX_RNDV_THRESH:-inf}" UCX_ZCOPY_THRESH="${UCX_ZCOPY_THRESH:-inf}" \
      UCX_MAX_EAGER_RAILS="${UCX_MAX_EAGER_RAILS:-1}" UCX_MAX_RNDV_RAILS="${UCX_MAX_RNDV_RAILS:-1}" \
      UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}" UCX_WARN_UNUSED_ENV_VARS="${UCX_WARN_UNUSED_ENV_VARS:-n}" \
      LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" LD_PRELOAD="$GV_PRELOAD" \
      bash -c "$FRONTEND_CMD" > "$log" 2>&1
    exit_code=$?
  else
    timeout "$RUN_TIMEOUT" env \
      GVIRTUS_CONFIG="$config" GVIRTUS_HOME="$GVIRTUS_HOME" GVIRTUS_LOGLEVEL="$GVIRTUS_LOGLEVEL" \
      LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" LD_PRELOAD="$GV_PRELOAD" \
      bash -c "$FRONTEND_CMD" > "$log" 2>&1
    exit_code=$?
  fi
  end_ns="$(date +%s%N)"
  wall_s="$(awk "BEGIN { printf \"%.6f\", ($end_ns - $start_ns) / 1000000000 }")"
  dump_counters "$mode" "$phase" "$run" after
  append_nic_delta "$timestamp" "$mode" "$phase" "$run"

  if [[ "$exit_code" -eq 0 ]]; then status="OK"; elif grep -qE "Final Results:|Total images:|Accuracy:" "$log"; then status="SOFT-OK"; else status="FAILED"; fi
  parse_and_append_metrics "$timestamp" "$mode" "$phase" "$run" "$status" "$exit_code" "$wall_s" "$config" "$log"

  python3 - <<PYPRINT
import csv
r = list(csv.DictReader(open("$RESULTS_CSV")))[-1]
print(f"  {r['status']} wall={r['wall_s']}s inference_ms={r['inference_ms'] or 'NA'} images={r['total_images'] or 'NA'} acc={r['accuracy_pct'] or 'NA'}% calls={r['calls']} nic_tx={r['nic_tx_B']}B nic_rx={r['nic_rx_B']}B")
PYPRINT
}

run_mode() {
  local mode="$1" config
  config="$(config_for_mode "$mode")"
  if [[ ! -f "$config" ]]; then echo "ERROR: missing config for mode '$mode': $config"; exit 1; fi
  echo
  echo "============================================================"
  echo "Mode: $mode"
  echo "Config: $config"
  echo "Frontend: $FRONTEND_CMD"
  echo "Runs: $RUNS measured + $WARMUPS warmup"
  echo "IFACES: $IFACES"
  echo "============================================================"
  echo "Start/restart backend with:"
  backend_hint "$mode"
  echo
  if [[ "$NO_PROMPT" != "1" ]]; then read -r -p "Press Enter when backend is ready..."; fi
  local i
  for ((i=1; i<=WARMUPS; i++)); do echo "[$mode] warmup run $i..."; run_one "$mode" "warmup" "$i"; done
  for ((i=1; i<=RUNS; i++)); do echo "[$mode] measure run $i..."; run_one "$mode" "measure" "$i"; done
}

case "$MODE" in
  tcp) MODES="tcp" ;;
  rdma) MODES="rdma" ;;
  ucx) MODES="ucx" ;;
  all) MODES="tcp rdma ucx" ;;
  *) echo "Usage: $0 {tcp|rdma|ucx|all}"; exit 1 ;;
esac

echo "Benchmark output directory:"
echo "$OUT_DIR"
for m in $MODES; do run_mode "$m"; done

echo
echo "Done."
echo "Main CSV:              $RESULTS_CSV"
echo "Individual calls CSV:  $ROUTINE_CALLS_CSV"
echo "Routine summary CSV:   $ROUTINE_SUMMARY_CSV"
echo "NIC CSV:               $NIC_CSV"
echo "Logs:                  $LOG_DIR"
echo "Counters:              $COUNTER_DIR"
echo "System metadata:       $SYSTEM_DIR"

echo
echo "Measured summary:"
python3 - <<PYSUMMARY
import csv
from collections import defaultdict
from statistics import mean
rows = list(csv.DictReader(open("$RESULTS_CSV")))
by = defaultdict(list)
for r in rows:
    if r["phase"] == "measure" and r["status"] in {"OK", "SOFT-OK"}: by[r["mode"]].append(r)
for mode, rs in sorted(by.items()):
    def nums(k):
        vals=[]
        for r in rs:
            try:
                if r.get(k) not in (None, ""): vals.append(float(r[k]))
            except ValueError: pass
        return vals
    def m(k):
        vals=nums(k); return mean(vals) if vals else float('nan')
    print(f"{mode}: n={len(rs)} mean_wall_s={m('wall_s'):.6f} mean_inference_ms={m('inference_ms'):.3f} mean_calls={m('calls'):.1f} mean_nic_rx_B={m('nic_rx_B'):.1f} mean_nic_tx_B={m('nic_tx_B'):.1f}")
PYSUMMARY
