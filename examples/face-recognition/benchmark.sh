#!/usr/bin/env bash
set -euo pipefail

# Face-recognition GVirtuS full-metrics benchmark.
#
# Purpose:
#   Run the face-recognition benchmark many times and collect as much relevant
#   runtime information as possible, without measuring container build/setup.
#
# Timed region:
#   LD_PRELOAD=... python3 cnn.py
#
# Collected:
#   Application:
#     - Test Accuracy
#     - Execution Time
#     - Average Time / per-image time
#
#   GVirtuS frontend communicator/routine metrics:
#     - per-call routine name
#     - server_exec_ms
#     - send_ms
#     - recv_ms
#     - comm_ms = send_ms + recv_ms
#     - total_call_ms = server_exec_ms + send_ms + recv_ms
#     - in/out bytes
#     - per-routine sum/mean/median/p95
#
#   NIC/network metrics:
#     - sysfs before/after counters
#     - per-run sysfs deltas
#     - summed NIC deltas in results.csv
#     - optional ethtool -S before/after dumps
#     - optional rdma link/statistic before/after dumps
#
#   GPU/system snapshots:
#     - optional nvidia-smi query before/after
#     - uname, lscpu, nvidia-smi -L, ibv_devinfo, rdma link, ip link
#
# Usage:
#   RUNS=50 WARMUPS=3 ./benchmark_facerecon_full_metrics.sh tcp
#   RUNS=50 WARMUPS=3 ./benchmark_facerecon_full_metrics.sh rdma
#   RUNS=50 WARMUPS=3 ./benchmark_facerecon_full_metrics.sh ucx
#   RUNS=50 WARMUPS=3 ./benchmark_facerecon_full_metrics.sh all
#
# Useful options:
#   EXAMPLE_DIR=/path/to/facerecon
#   IFACES="ens1f1np1 ens1f0np0 bond0"
#   COLLECT_ETHTOOL=1
#   COLLECT_RDMA_STATS=1
#   COLLECT_GPU_STATS=1
#   BUILD_EXTENSION_ONCE=1
#   PAUSE_BETWEEN_MODES=0
#
# Note:
#   GVIRTUS_LOGLEVEL must be high enough for frontend lines like:
#   Routine 'cudaMemcpy' returned 0 | server_exec=... | send=... | recv=... | in=...B | out=...B

MODE_ARG="${1:-all}"

case "$MODE_ARG" in
    tcp|rdma|ucx)
        MODES="$MODE_ARG"
        ;;
    all)
        MODES="tcp rdma ucx"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE_ARG'"
        echo "Usage: $0 [tcp|rdma|ucx|all]"
        exit 1
        ;;
esac

GVIRTUS_HOME="${GVIRTUS_HOME:-/home/student.aau.dk/ul11nh/gvirtus-install}"
LZ4_HOME="${LZ4_HOME:-/home/student.aau.dk/ul11nh/lz4-install}"
EXAMPLE_DIR="${EXAMPLE_DIR:-$(pwd)}"

RUNS="${RUNS:-50}"
WARMUPS="${WARMUPS:-3}"
RUN_TIMEOUT="${RUN_TIMEOUT:-180}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-10000}"

BUILD_EXTENSION_ONCE="${BUILD_EXTENSION_ONCE:-0}"
PAUSE_BETWEEN_MODES="${PAUSE_BETWEEN_MODES:-1}"

TCP_CONFIG="${TCP_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
RDMA_CONFIG="${RDMA_CONFIG:-$GVIRTUS_HOME/etc/properties_plain_rdma.json}"
UCX_CONFIG="${UCX_CONFIG:-$GVIRTUS_HOME/etc/properties_ucx.json}"

# Override this to match your active DPU/NIC interfaces.
IFACES="${IFACES:-ens1f1np1 ens1f0np0 bond0}"

# These are enabled by default because you asked for as much data as possible.
# Set to 0 if a command is too slow/noisy on the machine.
COLLECT_SYSFS_NET="${COLLECT_SYSFS_NET:-1}"
COLLECT_ETHTOOL="${COLLECT_ETHTOOL:-1}"
COLLECT_RDMA_STATS="${COLLECT_RDMA_STATS:-1}"
COLLECT_GPU_STATS="${COLLECT_GPU_STATS:-1}"

CMD_TIMEOUT="${CMD_TIMEOUT:-8}"

OUT_ROOT="${OUT_ROOT:-$EXAMPLE_DIR/benchmark_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/facerecon_full_metrics_${STAMP}_${MODE_ARG}"

LOG_DIR="$OUT_DIR/logs"
RAW_DIR="$OUT_DIR/raw"
SYSFS_DIR="$RAW_DIR/sysfs_net"
ETHTOOL_DIR="$RAW_DIR/ethtool"
RDMA_DIR="$RAW_DIR/rdma"
GPU_DIR="$RAW_DIR/gpu"
SYSTEM_DIR="$OUT_DIR/system"

RESULTS_CSV="$OUT_DIR/results.csv"
CALLS_CSV="$OUT_DIR/routine_calls.csv"
ROUTINE_SUMMARY_CSV="$OUT_DIR/routine_summary.csv"
NIC_SYSFS_CSV="$OUT_DIR/nic_sysfs_deltas.csv"
GPU_CSV="$OUT_DIR/gpu_snapshots.csv"

mkdir -p "$LOG_DIR" "$SYSFS_DIR" "$ETHTOOL_DIR" "$RDMA_DIR" "$GPU_DIR" "$SYSTEM_DIR"

export GVIRTUS_HOME
export GVIRTUS_LOGLEVEL
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"

cd "$EXAMPLE_DIR" || {
    echo "ERROR: Could not cd into EXAMPLE_DIR=$EXAMPLE_DIR"
    exit 1
}

mode_config() {
    case "$1" in
        tcp)  echo "$TCP_CONFIG" ;;
        rdma) echo "$RDMA_CONFIG" ;;
        ucx)  echo "$UCX_CONFIG" ;;
        *)
            echo "ERROR: Unknown mode '$1'" >&2
            return 1
            ;;
    esac
}

show_endpoint() {
    local config="$1"
    "$PYTHON_BIN" - "$config" <<'PY' 2>/dev/null || true
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
e = data["communicator"][0]["endpoint"]
print(f'{e.get("suite")} / {e.get("protocol")} / {e.get("server_address")}:{e.get("port")}')
PY
}

build_extension_once_if_requested() {
    if [[ "$BUILD_EXTENSION_ONCE" != "1" ]]; then
        return 0
    fi

    if [[ ! -f "extension.cu" ]]; then
        echo "ERROR: BUILD_EXTENSION_ONCE=1 but extension.cu was not found in $EXAMPLE_DIR"
        exit 1
    fi

    echo "Building face-recognition CUDA extension once before measurements..."
    nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu -lcudart -lcublas
    echo "Build complete. Build time is not included in benchmark CSVs."
}

run_facerecon() {
    if [[ ! -f "cnn.py" ]]; then
        echo "ERROR: cnn.py not found in EXAMPLE_DIR=$EXAMPLE_DIR"
        return 127
    fi

    if [[ ! -f "libextension.so" ]]; then
        echo "ERROR: libextension.so not found. Build it beforehand or run with BUILD_EXTENSION_ONCE=1."
        return 127
    fi

    timeout "$RUN_TIMEOUT" env \
        LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so" \
        "$PYTHON_BIN" cnn.py
}

extract_accuracy() {
    local file="$1"
    grep -E 'Test Accuracy:' "$file" | tail -1 | sed -E 's/.*Test Accuracy:[[:space:]]*([0-9.]+)%.*/\1/' || true
}

extract_execution_time() {
    local file="$1"
    grep -E 'Execution Time:' "$file" | tail -1 | sed -E 's/.*Execution Time:[[:space:]]*([0-9.]+).*/\1/' || true
}

extract_average_time() {
    local file="$1"
    grep -E 'Average Time:' "$file" | tail -1 | sed -E 's/.*Average Time:[[:space:]]*([0-9.]+).*/\1/' || true
}

safe_cmd() {
    local out_file="$1"
    shift
    timeout "$CMD_TIMEOUT" "$@" > "$out_file" 2>&1 || true
}

safe_cmd_real_gpu() {
    local out_file="$1"
    shift

    # Run GPU/system tools without GVirtuS frontend interception.
    # Important: do NOT include $GVIRTUS_HOME/lib/frontend here.
    env \
        LD_PRELOAD="" \
        LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu" \
        timeout "$CMD_TIMEOUT" "$@" > "$out_file" 2>&1 || true
}


collect_system_metadata() {
    {
        echo "### date"
        date -Iseconds
        echo
        echo "### hostname"
        hostname || true
        echo
        echo "### uname"
        uname -a || true
        echo
        echo "### environment"
        echo "GVIRTUS_HOME=$GVIRTUS_HOME"
        echo "EXAMPLE_DIR=$EXAMPLE_DIR"
        echo "RUNS=$RUNS"
        echo "WARMUPS=$WARMUPS"
        echo "MODES=$MODES"
        echo "GVIRTUS_LOGLEVEL=$GVIRTUS_LOGLEVEL"
        echo "IFACES=$IFACES"
        echo "TCP_CONFIG=$TCP_CONFIG"
        echo "RDMA_CONFIG=$RDMA_CONFIG"
        echo "UCX_CONFIG=$UCX_CONFIG"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo
        echo "### git"
        git -C "$EXAMPLE_DIR" branch --show-current 2>/dev/null || true
        git -C "$EXAMPLE_DIR" rev-parse --short HEAD 2>/dev/null || true
    } > "$SYSTEM_DIR/run_info.txt"

    safe_cmd "$SYSTEM_DIR/lscpu.txt" lscpu
    safe_cmd "$SYSTEM_DIR/free_h.txt" free -h
    safe_cmd "$SYSTEM_DIR/lsblk.txt" lsblk
    safe_cmd "$SYSTEM_DIR/ip_addr.txt" ip addr
    safe_cmd "$SYSTEM_DIR/ip_link_stats.txt" ip -s link

    if command -v nvidia-smi >/dev/null 2>&1; then
        safe_cmd_real_gpu "$SYSTEM_DIR/nvidia_smi_L.txt" nvidia-smi -L
        safe_cmd_real_gpu "$SYSTEM_DIR/nvidia_smi_q.txt" nvidia-smi -q
    fi

    if command -v rdma >/dev/null 2>&1; then
        safe_cmd "$SYSTEM_DIR/rdma_link.txt" rdma link
        safe_cmd "$SYSTEM_DIR/rdma_statistic.txt" rdma statistic
    fi

    if command -v ibv_devinfo >/dev/null 2>&1; then
        safe_cmd "$SYSTEM_DIR/ibv_devinfo.txt" ibv_devinfo
    fi

    for iface in $IFACES; do
        if [[ -d "/sys/class/net/$iface" ]]; then
            safe_cmd "$SYSTEM_DIR/ethtool_${iface}.txt" ethtool "$iface"
            safe_cmd "$SYSTEM_DIR/ethtool_i_${iface}.txt" ethtool -i "$iface"
        fi
    done
}

net_value() {
    local iface="$1"
    local key="$2"
    local f="/sys/class/net/$iface/statistics/$key"
    if [[ -r "$f" ]]; then
        cat "$f"
    else
        echo 0
    fi
}

capture_sysfs_net_csv() {
    local file="$1"
    : > "$file"

    if [[ "$COLLECT_SYSFS_NET" != "1" ]]; then
        return 0
    fi

    local keys=(
        rx_bytes tx_bytes
        rx_packets tx_packets
        rx_errors tx_errors
        rx_dropped tx_dropped
        rx_fifo_errors tx_fifo_errors
        rx_frame_errors tx_carrier_errors
        rx_compressed tx_compressed
        multicast collisions
    )

    for iface in $IFACES; do
        if [[ ! -d "/sys/class/net/$iface" ]]; then
            continue
        fi
        for key in "${keys[@]}"; do
            echo "$iface,$key,$(net_value "$iface" "$key")" >> "$file"
        done
    done
}

write_sysfs_net_deltas() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local before_file="$4"
    local after_file="$5"

    "$PYTHON_BIN" - "$mode" "$phase" "$run_no" "$before_file" "$after_file" "$NIC_SYSFS_CSV" <<'PY'
import csv
import sys

mode, phase, run_no, before_file, after_file, out_csv = sys.argv[1:7]

def read(path):
    rows = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                iface, key, val = parts
                try:
                    rows[(iface, key)] = int(val)
                except Exception:
                    rows[(iface, key)] = 0
    except FileNotFoundError:
        pass
    return rows

b = read(before_file)
a = read(after_file)

with open(out_csv, "a", newline="") as f:
    w = csv.writer(f)
    for iface, key in sorted(set(b) | set(a)):
        before = b.get((iface, key), 0)
        after = a.get((iface, key), 0)
        w.writerow([mode, phase, run_no, iface, key, before, after, after - before])
PY
}

sum_sysfs_delta() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local key="$4"

    awk -F, -v m="$mode" -v p="$phase" -v r="$run_no" -v k="$key" '
        $1==m && $2==p && $3==r && $5==k { s += $8 }
        END { print s+0 }
    ' "$NIC_SYSFS_CSV"
}

collect_ethtool_snapshot() {
    local prefix="$1"

    if [[ "$COLLECT_ETHTOOL" != "1" ]]; then
        return 0
    fi

    for iface in $IFACES; do
        if [[ -d "/sys/class/net/$iface" ]] && command -v ethtool >/dev/null 2>&1; then
            safe_cmd "$ETHTOOL_DIR/${prefix}_${iface}.txt" ethtool -S "$iface"
        fi
    done
}

collect_rdma_snapshot() {
    local prefix="$1"

    if [[ "$COLLECT_RDMA_STATS" != "1" ]]; then
        return 0
    fi

    if command -v rdma >/dev/null 2>&1; then
        safe_cmd "$RDMA_DIR/${prefix}_rdma_link.txt" rdma link
        safe_cmd "$RDMA_DIR/${prefix}_rdma_statistic.txt" rdma statistic
        safe_cmd "$RDMA_DIR/${prefix}_rdma_resource.txt" rdma resource
    fi
}

collect_gpu_snapshot() {
    local prefix="$1"
    local mode="$2"
    local phase="$3"
    local run_no="$4"
    local point="$5"

    if [[ "$COLLECT_GPU_STATS" != "1" ]]; then
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    local out="$GPU_DIR/${prefix}_nvidia_smi.csv"

    nvidia-smi \
        --query-gpu=timestamp,index,name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,power.limit,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits \
        > "$out" 2>/dev/null || true

    "$PYTHON_BIN" - "$mode" "$phase" "$run_no" "$point" "$out" "$GPU_CSV" <<'PY'
import csv
import sys

mode, phase, run_no, point, in_file, out_file = sys.argv[1:7]

with open(out_file, "a", newline="") as out:
    w = csv.writer(out)
    try:
        with open(in_file) as f:
            for line in f:
                parts = [x.strip() for x in line.rstrip("\n").split(",")]
                if len(parts) < 16:
                    continue
                w.writerow([mode, phase, run_no, point] + parts[:16])
    except FileNotFoundError:
        pass
PY
}

parse_log_to_csvs() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local log_file="$4"
    local timestamp="$5"
    local exit_code="$6"
    local wall_s="$7"
    local accuracy="$8"
    local execution_s="$9"
    local per_image_s="${10}"
    local config="${11}"
    local nic_rx_bytes="${12}"
    local nic_tx_bytes="${13}"
    local nic_rx_packets="${14}"
    local nic_tx_packets="${15}"
    local nic_rx_errors="${16}"
    local nic_tx_errors="${17}"
    local nic_rx_dropped="${18}"
    local nic_tx_dropped="${19}"

    "$PYTHON_BIN" - \
        "$mode" "$phase" "$run_no" "$log_file" "$timestamp" "$exit_code" "$wall_s" \
        "$accuracy" "$execution_s" "$per_image_s" "$config" \
        "$nic_rx_bytes" "$nic_tx_bytes" "$nic_rx_packets" "$nic_tx_packets" \
        "$nic_rx_errors" "$nic_tx_errors" "$nic_rx_dropped" "$nic_tx_dropped" \
        "$RESULTS_CSV" "$CALLS_CSV" "$ROUTINE_SUMMARY_CSV" <<'PY'
import csv
import math
import re
import statistics
import sys
from collections import defaultdict

(
    mode,
    phase,
    run_no,
    log_file,
    timestamp,
    exit_code,
    wall_s,
    accuracy,
    execution_s,
    per_image_s,
    config,
    nic_rx_bytes,
    nic_tx_bytes,
    nic_rx_packets,
    nic_tx_packets,
    nic_rx_errors,
    nic_tx_errors,
    nic_rx_dropped,
    nic_tx_dropped,
    results_csv,
    calls_csv,
    summary_csv,
) = sys.argv[1:23]

def to_ms(value, unit):
    v = float(value)
    unit = (unit or "s").lower()
    if unit == "s":
        return v * 1000.0
    if unit == "ms":
        return v
    if unit in ("us", "µs"):
        return v / 1000.0
    if unit == "ns":
        return v / 1_000_000.0
    return v * 1000.0

def percentile(vals, p):
    vals = sorted(vals)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
pat = re.compile(
    rf"Routine '([^']+)'\s+returned\s+(-?\d+).*?"
    rf"server_exec=({num})\s*(s|ms|us|µs|ns)?\s*\|\s*"
    rf"send=({num})\s*(s|ms|us|µs|ns)?\s*\|\s*"
    rf"recv=({num})\s*(s|ms|us|µs|ns)?\s*\|\s*"
    rf"in=(\d+)B\s*\|\s*out=(\d+)B"
)

calls = []
try:
    with open(log_file, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue

            routine, status, server_v, server_u, send_v, send_u, recv_v, recv_u, in_b, out_b = m.groups()

            server_ms = to_ms(server_v, server_u)
            send_ms = to_ms(send_v, send_u)
            recv_ms = to_ms(recv_v, recv_u)
            comm_ms = send_ms + recv_ms
            total_ms = server_ms + comm_ms

            calls.append({
                "routine": routine,
                "status": int(status),
                "server_exec_ms": server_ms,
                "send_ms": send_ms,
                "recv_ms": recv_ms,
                "comm_ms": comm_ms,
                "total_call_ms": total_ms,
                "in_bytes": int(in_b),
                "out_bytes": int(out_b),
            })
except FileNotFoundError:
    pass

with open(calls_csv, "a", newline="") as f:
    w = csv.writer(f)
    for call_index, c in enumerate(calls, start=1):
        w.writerow([
            timestamp,
            mode,
            phase,
            run_no,
            call_index,
            c["routine"],
            c["status"],
            f'{c["server_exec_ms"]:.6f}',
            f'{c["send_ms"]:.6f}',
            f'{c["recv_ms"]:.6f}',
            f'{c["comm_ms"]:.6f}',
            f'{c["total_call_ms"]:.6f}',
            c["in_bytes"],
            c["out_bytes"],
            config,
            log_file,
        ])

by_routine = defaultdict(list)
for c in calls:
    by_routine[c["routine"]].append(c)

with open(summary_csv, "a", newline="") as f:
    w = csv.writer(f)
    for routine in sorted(by_routine):
        rows = by_routine[routine]

        server_vals = [r["server_exec_ms"] for r in rows]
        send_vals = [r["send_ms"] for r in rows]
        recv_vals = [r["recv_ms"] for r in rows]
        comm_vals = [r["comm_ms"] for r in rows]
        total_vals = [r["total_call_ms"] for r in rows]

        w.writerow([
            timestamp,
            mode,
            phase,
            run_no,
            routine,
            len(rows),
            sum(r["in_bytes"] for r in rows),
            sum(r["out_bytes"] for r in rows),
            f"{sum(server_vals):.6f}",
            f"{sum(send_vals):.6f}",
            f"{sum(recv_vals):.6f}",
            f"{sum(comm_vals):.6f}",
            f"{sum(total_vals):.6f}",
            f"{statistics.mean(server_vals):.6f}",
            f"{statistics.mean(send_vals):.6f}",
            f"{statistics.mean(recv_vals):.6f}",
            f"{statistics.mean(comm_vals):.6f}",
            f"{statistics.mean(total_vals):.6f}",
            f"{statistics.median(total_vals):.6f}",
            f"{percentile(total_vals, 0.95):.6f}",
            config,
            log_file,
        ])

total_calls = len(calls)
failed_calls = sum(1 for c in calls if c["status"] != 0)
total_in = sum(c["in_bytes"] for c in calls)
total_out = sum(c["out_bytes"] for c in calls)
total_server = sum(c["server_exec_ms"] for c in calls)
total_send = sum(c["send_ms"] for c in calls)
total_recv = sum(c["recv_ms"] for c in calls)
total_comm = total_send + total_recv
total_call = total_server + total_comm

routine_summary = ";".join(
    f"{routine}:{len(rows)}"
    for routine, rows in sorted(by_routine.items())
)

# Useful for UCX close hangs: exit 124 can still be analytically useful if app output exists.
soft_ok = int(exit_code == "0" or (exit_code == "124" and accuracy and execution_s))

with open(results_csv, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        timestamp,
        mode,
        phase,
        run_no,
        exit_code,
        soft_ok,
        wall_s,
        accuracy,
        execution_s,
        per_image_s,
        total_calls,
        failed_calls,
        total_in,
        total_out,
        f"{total_server:.6f}",
        f"{total_send:.6f}",
        f"{total_recv:.6f}",
        f"{total_comm:.6f}",
        f"{total_call:.6f}",
        nic_rx_bytes,
        nic_tx_bytes,
        nic_rx_packets,
        nic_tx_packets,
        nic_rx_errors,
        nic_tx_errors,
        nic_rx_dropped,
        nic_tx_dropped,
        config,
        log_file,
        routine_summary,
    ])
PY
}

echo "timestamp,mode,phase,run,exit_code,soft_ok,wall_s,accuracy_pct,execution_s,per_image_s,gvirtus_calls,failed_gvirtus_calls,gvirtus_in_bytes,gvirtus_out_bytes,total_server_exec_ms,total_send_ms,total_recv_ms,total_comm_ms,total_call_ms,nic_rx_bytes,nic_tx_bytes,nic_rx_packets,nic_tx_packets,nic_rx_errors,nic_tx_errors,nic_rx_dropped,nic_tx_dropped,config,frontend_log,routine_summary" > "$RESULTS_CSV"
echo "timestamp,mode,phase,run,call_index,routine,status,server_exec_ms,send_ms,recv_ms,comm_ms,total_call_ms,in_bytes,out_bytes,config,frontend_log" > "$CALLS_CSV"
echo "timestamp,mode,phase,run,routine,calls,in_bytes,out_bytes,sum_server_exec_ms,sum_send_ms,sum_recv_ms,sum_comm_ms,sum_total_call_ms,mean_server_exec_ms,mean_send_ms,mean_recv_ms,mean_comm_ms,mean_total_call_ms,median_total_call_ms,p95_total_call_ms,config,frontend_log" > "$ROUTINE_SUMMARY_CSV"
echo "mode,phase,run,iface,counter,before,after,delta" > "$NIC_SYSFS_CSV"
echo "mode,phase,run,point,timestamp,gpu_index,name,uuid,driver_version,pstate,temperature_gpu,utilization_gpu,utilization_memory,memory_total_mb,memory_used_mb,memory_free_mb,power_draw_w,power_limit_w,clocks_sm_mhz,clocks_mem_mhz" > "$GPU_CSV"

collect_system_metadata
build_extension_once_if_requested

echo "Benchmark output directory:"
echo "$OUT_DIR"
echo

for mode in $MODES; do
    config="$(mode_config "$mode")"

    if [[ ! -f "$config" ]]; then
        echo "ERROR: Missing config for mode '$mode': $config"
        exit 1
    fi

    export GVIRTUS_CONFIG="$config"

    echo "============================================================"
    echo "Mode: $mode"
    echo "Config: $GVIRTUS_CONFIG"
    echo "Endpoint: $(show_endpoint "$GVIRTUS_CONFIG")"
    echo "Runs: $RUNS measured + $WARMUPS warmup"
    echo "IFACES: $IFACES"
    echo "============================================================"

    if [[ "$PAUSE_BETWEEN_MODES" == "1" ]]; then
        echo "Start/restart the backend with this same config, then press Enter."
        echo "Backend command:"
        echo "  GVIRTUS_CONFIG=$GVIRTUS_CONFIG GVIRTUS_LOGLEVEL=$GVIRTUS_LOGLEVEL gvirtus-backend"
        read -r
    fi

    total_runs=$((WARMUPS + RUNS))

    for i in $(seq 1 "$total_runs"); do
        if (( i <= WARMUPS )); then
            phase="warmup"
            run_no="$i"
        else
            phase="measure"
            run_no=$((i - WARMUPS))
        fi

        log_file="$LOG_DIR/${mode}_${phase}_${run_no}.log"
        before_net="$SYSFS_DIR/${mode}_${phase}_${run_no}_before.csv"
        after_net="$SYSFS_DIR/${mode}_${phase}_${run_no}_after.csv"
        prefix="${mode}_${phase}_${run_no}"

        echo "[$mode] $phase run $run_no..."

        capture_sysfs_net_csv "$before_net"
        collect_ethtool_snapshot "${prefix}_before"
        collect_rdma_snapshot "${prefix}_before"
        collect_gpu_snapshot "${prefix}_before" "$mode" "$phase" "$run_no" "before"

        start_ns="$(date +%s%N)"
        set +e
        run_facerecon > "$log_file" 2>&1
        exit_code=$?
        set -e
        end_ns="$(date +%s%N)"

        collect_gpu_snapshot "${prefix}_after" "$mode" "$phase" "$run_no" "after"
        collect_rdma_snapshot "${prefix}_after"
        collect_ethtool_snapshot "${prefix}_after"
        capture_sysfs_net_csv "$after_net"

        write_sysfs_net_deltas "$mode" "$phase" "$run_no" "$before_net" "$after_net"

        wall_s="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e-s)/1000000000 }')"
        accuracy="$(extract_accuracy "$log_file")"
        execution_s="$(extract_execution_time "$log_file")"
        per_image_s="$(extract_average_time "$log_file")"
        timestamp="$(date -Iseconds)"

        nic_rx_bytes="$(sum_sysfs_delta "$mode" "$phase" "$run_no" rx_bytes)"
        nic_tx_bytes="$(sum_sysfs_delta "$mode" "$phase" "$run_no" tx_bytes)"
        nic_rx_packets="$(sum_sysfs_delta "$mode" "$phase" "$run_no" rx_packets)"
        nic_tx_packets="$(sum_sysfs_delta "$mode" "$phase" "$run_no" tx_packets)"
        nic_rx_errors="$(sum_sysfs_delta "$mode" "$phase" "$run_no" rx_errors)"
        nic_tx_errors="$(sum_sysfs_delta "$mode" "$phase" "$run_no" tx_errors)"
        nic_rx_dropped="$(sum_sysfs_delta "$mode" "$phase" "$run_no" rx_dropped)"
        nic_tx_dropped="$(sum_sysfs_delta "$mode" "$phase" "$run_no" tx_dropped)"

        parse_log_to_csvs \
            "$mode" "$phase" "$run_no" "$log_file" \
            "$timestamp" "$exit_code" "$wall_s" \
            "${accuracy:-}" "${execution_s:-}" "${per_image_s:-}" "$GVIRTUS_CONFIG" \
            "$nic_rx_bytes" "$nic_tx_bytes" "$nic_rx_packets" "$nic_tx_packets" \
            "$nic_rx_errors" "$nic_tx_errors" "$nic_rx_dropped" "$nic_tx_dropped"

        last_row="$(tail -n 1 "$RESULTS_CSV")"
        soft_ok="$(echo "$last_row" | awk -F, '{print $6}')"
        calls="$(echo "$last_row" | awk -F, '{print $11}')"
        comm_ms="$(echo "$last_row" | awk -F, '{print $18}')"
        total_ms="$(echo "$last_row" | awk -F, '{print $19}')"

        if [[ "$soft_ok" == "1" ]]; then
            echo "  OK exit=$exit_code wall=${wall_s}s acc=${accuracy:-NA}% exec=${execution_s:-NA}s per_image=${per_image_s:-NA}s calls=${calls} comm_ms=${comm_ms} total_call_ms=${total_ms} nic_tx=${nic_tx_bytes}B nic_rx=${nic_rx_bytes}B"
        else
            echo "  FAILED exit_code=$exit_code wall=${wall_s}s calls=${calls}. See: $log_file"
        fi
    done

    echo
done

echo "Done."
echo "Main CSV:              $RESULTS_CSV"
echo "Individual calls CSV:  $CALLS_CSV"
echo "Routine summary CSV:   $ROUTINE_SUMMARY_CSV"
echo "NIC sysfs delta CSV:   $NIC_SYSFS_CSV"
echo "GPU snapshot CSV:      $GPU_CSV"
echo "Raw snapshots:         $RAW_DIR"
echo "System metadata:       $SYSTEM_DIR"
echo

echo "Measured-run summary:"
"$PYTHON_BIN" - "$RESULTS_CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

rows_by_mode = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["phase"] == "measure" and row["soft_ok"] == "1":
            rows_by_mode[row["mode"]].append(row)

for mode, rows in sorted(rows_by_mode.items()):
    def vals(col):
        out = []
        for x in rows:
            try:
                out.append(float(x[col] or 0))
            except Exception:
                pass
        return out

    print(
        f"{mode}: "
        f"n={len(rows)} "
        f"mean_acc={statistics.mean(vals('accuracy_pct')):.4f}% "
        f"mean_exec_s={statistics.mean(vals('execution_s')):.6f} "
        f"mean_per_image_s={statistics.mean(vals('per_image_s')):.6f} "
        f"mean_calls={statistics.mean(vals('gvirtus_calls')):.1f} "
        f"mean_send_ms={statistics.mean(vals('total_send_ms')):.6f} "
        f"mean_recv_ms={statistics.mean(vals('total_recv_ms')):.6f} "
        f"mean_comm_ms={statistics.mean(vals('total_comm_ms')):.6f} "
        f"mean_total_call_ms={statistics.mean(vals('total_call_ms')):.6f} "
        f"mean_gvirtus_in_B={statistics.mean(vals('gvirtus_in_bytes')):.1f} "
        f"mean_gvirtus_out_B={statistics.mean(vals('gvirtus_out_bytes')):.1f} "
        f"mean_nic_rx_B={statistics.mean(vals('nic_rx_bytes')):.1f} "
        f"mean_nic_tx_B={statistics.mean(vals('nic_tx_bytes')):.1f}"
    )
PY
