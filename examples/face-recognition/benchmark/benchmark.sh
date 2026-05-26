#!/usr/bin/env bash
set -u

# GVirtuS frontend benchmark runner.
#
# Usage:
#   ./benchmark.sh tcp
#   ./benchmark.sh rdma
#   ./benchmark.sh ucx
#   ./benchmark.sh all
#
# Examples:
#   RUNS=20 WARMUPS=3 ./benchmark.sh rdma
#   IFACES="ens1f1np1" RUNS=30 ./benchmark.sh rdma
#   IFACES="bond0" RUNS=30 ./benchmark.sh tcp
#   PAUSE_BETWEEN_MODES=0 ./benchmark.sh tcp
#
# Optional:
#   BACKEND_LOG=/tmp/gvirtus-backend.log ./benchmark.sh rdma

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

RUNS="${RUNS:-10}"
WARMUPS="${WARMUPS:-2}"

# DEBUG is needed to parse GVirtuS call counts and in/out bytes from frontend logs.
GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-10000}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_ONCE="${BUILD_ONCE:-1}"
PAUSE_BETWEEN_MODES="${PAUSE_BETWEEN_MODES:-1}"
CMD_TIMEOUT="${CMD_TIMEOUT:-5}"
COLLECT_COUNTER_DUMPS="${COLLECT_COUNTER_DUMPS:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"

TCP_CONFIG="${TCP_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
RDMA_CONFIG="${RDMA_CONFIG:-$GVIRTUS_HOME/etc/properties_plain_rdma.json}"
UCX_CONFIG="${UCX_CONFIG:-$GVIRTUS_HOME/etc/properties_ucx.json}"

# Choose relevant DPU/NIC interfaces. Override per mode if desired.
# RDMA on your machine was ens1f1np1 / 25.25.25.2.
IFACES="${IFACES:-ens1f1np1 ens1f0np0 bond0}"

OUT_ROOT="${OUT_ROOT:-$EXAMPLE_DIR/benchmark_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/frontend_${STAMP}_${MODE_ARG}"
LOG_DIR="$OUT_DIR/logs"
COUNTER_DIR="$OUT_DIR/counters"
ROUTINE_CSV="$OUT_DIR/routines.csv"
NIC_CSV="$OUT_DIR/nic_counters.csv"
CSV="$OUT_DIR/results.csv"

mkdir -p "$LOG_DIR" "$COUNTER_DIR"

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
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
e = data["communicator"][0]["endpoint"]
print(f'{e.get("suite")} / {e.get("protocol")} / {e.get("server_address")}:{e.get("port")}')
PY
}

build_face_recognition_once() {
    if [[ "$BUILD_ONCE" != "1" ]]; then
        return 0
    fi

    if [[ -f "extension.cu" && -f "cnn.py" ]]; then
        echo "Building face-recognition CUDA extension once..."
        nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu -lcudart -lcublas
        if [[ $? -ne 0 ]]; then
            echo "ERROR: nvcc build failed"
            exit 1
        fi
    fi
}

run_example() {
    local py="${PYTHON_BIN:-python3}"

    if [[ -z "$py" ]]; then
        py="python3"
    fi

    if ! command -v "$py" >/dev/null 2>&1; then
        echo "ERROR: Python command not found: $py"
        return 127
    fi

    if [[ -n "${BENCH_CMD:-}" ]]; then
        timeout "${RUN_TIMEOUT:-120}" bash -lc "$BENCH_CMD"
        return $?
    fi

    if [[ -f "cnn.py" && -f "libextension.so" ]]; then
        timeout "${RUN_TIMEOUT:-120}" env \
            LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so" \
            "$py" cnn.py
        return $?
    fi

    if [[ -f "./run.sh" ]]; then
        sed -i 's/
$//' ./run.sh
        timeout "${RUN_TIMEOUT:-120}" bash ./run.sh
        return $?
    fi

    echo "ERROR: No BENCH_CMD set, no cnn.py/libextension.so, and no run.sh found."
    return 127
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

parse_gvirtus_metrics() {
    local file="$1"

    "$PYTHON_BIN" - "$file" <<'PY'
import re, sys
path = sys.argv[1]

call_count = 0
in_bytes = 0
out_bytes = 0
routines = {}

# Example:
# DEBUG - Routine 'cudaMemcpy' returned 0 | server_exec=0s | send=0s | recv=0s | in=29B | out=168B
pat = re.compile(r"Routine '([^']+)' returned .*?\|\s*.*?in=([0-9]+)B\s*\|\s*out=([0-9]+)B")

try:
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            routine = m.group(1)
            ib = int(m.group(2))
            ob = int(m.group(3))
            call_count += 1
            in_bytes += ib
            out_bytes += ob
            routines[routine] = routines.get(routine, 0) + 1
except FileNotFoundError:
    pass

routine_summary = ";".join(f"{k}:{v}" for k, v in sorted(routines.items()))
print(f"{call_count},{in_bytes},{out_bytes},{routine_summary}")
PY
}

write_routine_rows() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local log_file="$4"

    "$PYTHON_BIN" - "$mode" "$phase" "$run_no" "$log_file" "$ROUTINE_CSV" <<'PY'
import csv, re, sys
mode, phase, run_no, log_file, out_csv = sys.argv[1:6]
pat = re.compile(r"Routine '([^']+)' returned .*?in=([0-9]+)B\s*\|\s*out=([0-9]+)B")

stats = {}
try:
    with open(log_file, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            name = m.group(1)
            ib = int(m.group(2))
            ob = int(m.group(3))
            s = stats.setdefault(name, [0, 0, 0])
            s[0] += 1
            s[1] += ib
            s[2] += ob
except FileNotFoundError:
    pass

with open(out_csv, "a", newline="") as f:
    w = csv.writer(f)
    for routine, (calls, ib, ob) in sorted(stats.items()):
        w.writerow([mode, phase, run_no, routine, calls, ib, ob])
PY
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

snapshot_counters() {
    local prefix="$1"

    if [[ "${COLLECT_COUNTER_DUMPS:-0}" != "1" ]]; then
        return 0
    fi

    {
        echo "### date"
        date -Iseconds
        echo
        echo "### interfaces"
        for iface in $IFACES; do
            echo "--- $iface ---"
            timeout "$CMD_TIMEOUT" ip -s link show dev "$iface" 2>&1 || true
            echo
        done
        echo "### rdma link"
        timeout "$CMD_TIMEOUT" rdma link 2>&1 || true
        echo
        echo "### rdma statistic"
        timeout "$CMD_TIMEOUT" rdma statistic 2>&1 || true
    } > "$COUNTER_DIR/${prefix}_ip_rdma.txt"

    for iface in $IFACES; do
        timeout "$CMD_TIMEOUT" ethtool -S "$iface" > "$COUNTER_DIR/${prefix}_ethtool_${iface}.txt" 2>&1 || true
    done
}

write_nic_deltas() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local before_file="$4"
    local after_file="$5"

    "$PYTHON_BIN" - "$mode" "$phase" "$run_no" "$before_file" "$after_file" "$NIC_CSV" <<'PY'
import csv, sys
mode, phase, run_no, before_file, after_file, out_csv = sys.argv[1:7]

def read_rows(path):
    rows = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 5:
                    continue
                iface, rx_b, tx_b, rx_p, tx_p = parts
                rows[iface] = tuple(map(int, [rx_b, tx_b, rx_p, tx_p]))
    except FileNotFoundError:
        pass
    return rows

b = read_rows(before_file)
a = read_rows(after_file)

with open(out_csv, "a", newline="") as f:
    w = csv.writer(f)
    for iface in sorted(set(b) | set(a)):
        brx, btx, brxp, btxp = b.get(iface, (0,0,0,0))
        arx, atx, arxp, atxp = a.get(iface, (0,0,0,0))
        w.writerow([
            mode, phase, run_no, iface,
            arx - brx,
            atx - btx,
            arxp - brxp,
            atxp - btxp,
        ])
PY
}

capture_sysfs_net_csv() {
    local file="$1"
    : > "$file"

    for iface in $IFACES; do
        rx_b="$(net_value "$iface" rx_bytes)"
        tx_b="$(net_value "$iface" tx_bytes)"
        rx_p="$(net_value "$iface" rx_packets)"
        tx_p="$(net_value "$iface" tx_packets)"
        echo "$iface,$rx_b,$tx_b,$rx_p,$tx_p" >> "$file"
    done
}

sum_nic_delta_field() {
    local mode="$1"
    local phase="$2"
    local run_no="$3"
    local field="$4"

    # field numbers in nic_counters.csv:
    # 5 rx_bytes_delta, 6 tx_bytes_delta, 7 rx_packets_delta, 8 tx_packets_delta
    awk -F, -v m="$mode" -v p="$phase" -v r="$run_no" -v f="$field" '
        $1==m && $2==p && $3==r { s += $f }
        END { print s+0 }
    ' "$NIC_CSV"
}

echo "timestamp,mode,phase,run,exit_code,wall_s,accuracy_pct,execution_s,per_image_s,gvirtus_calls,gvirtus_in_bytes,gvirtus_out_bytes,nic_rx_bytes,nic_tx_bytes,nic_rx_packets,nic_tx_packets,config,frontend_log,backend_log,routine_summary" > "$CSV"
echo "mode,phase,run,routine,calls,in_bytes,out_bytes" > "$ROUTINE_CSV"
echo "mode,phase,run,iface,rx_bytes_delta,tx_bytes_delta,rx_packets_delta,tx_packets_delta" > "$NIC_CSV"

cat > "$OUT_DIR/run_info.txt" <<INFO
timestamp=$STAMP
example_dir=$EXAMPLE_DIR
gvirtus_home=$GVIRTUS_HOME
runs=$RUNS
warmups=$WARMUPS
modes=$MODES
loglevel=$GVIRTUS_LOGLEVEL
python=$PYTHON_BIN
build_once=$BUILD_ONCE
ifaces=$IFACES
tcp_config=$TCP_CONFIG
rdma_config=$RDMA_CONFIG
ucx_config=$UCX_CONFIG
bench_cmd=${BENCH_CMD:-AUTO}
backend_log=${BACKEND_LOG:-}
INFO

git -C /home/student.aau.dk/ul11nh/GVirtuS-Project branch --show-current >> "$OUT_DIR/run_info.txt" 2>/dev/null || true
git -C /home/student.aau.dk/ul11nh/GVirtuS-Project rev-parse --short HEAD >> "$OUT_DIR/run_info.txt" 2>/dev/null || true

build_face_recognition_once

echo "Benchmark output directory:"
echo "$OUT_DIR"
echo

for mode in $MODES; do
    config="$(mode_config "$mode")" || exit 1

    if [[ ! -f "$config" ]]; then
        echo "ERROR: Missing config for mode '$mode': $config"
        exit 1
    fi

    export GVIRTUS_CONFIG="$config"

    echo "============================================================"
    echo "Mode: $mode"
    echo "Config: $GVIRTUS_CONFIG"
    echo "Endpoint: $(show_endpoint "$GVIRTUS_CONFIG")"
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
        before_net="$COUNTER_DIR/${mode}_${phase}_${run_no}_before_net.csv"
        after_net="$COUNTER_DIR/${mode}_${phase}_${run_no}_after_net.csv"

        echo "[$mode] $phase run $run_no..."

        capture_sysfs_net_csv "$before_net"
        snapshot_counters "${mode}_${phase}_${run_no}_before"

        start_ns="$(date +%s%N)"
        run_example > "$log_file" 2>&1
        exit_code=$?
        end_ns="$(date +%s%N)"

        snapshot_counters "${mode}_${phase}_${run_no}_after"
        capture_sysfs_net_csv "$after_net"

        write_nic_deltas "$mode" "$phase" "$run_no" "$before_net" "$after_net"

        wall_s="$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.6f", (e-s)/1000000000 }')"
        accuracy="$(extract_accuracy "$log_file")"
        execution_s="$(extract_execution_time "$log_file")"
        per_image_s="$(extract_average_time "$log_file")"

        gvirtus_metrics="$(parse_gvirtus_metrics "$log_file")"
        gvirtus_calls="$(echo "$gvirtus_metrics" | cut -d, -f1)"
        gvirtus_in_bytes="$(echo "$gvirtus_metrics" | cut -d, -f2)"
        gvirtus_out_bytes="$(echo "$gvirtus_metrics" | cut -d, -f3)"
        routine_summary="$(echo "$gvirtus_metrics" | cut -d, -f4-)"

        write_routine_rows "$mode" "$phase" "$run_no" "$log_file"

        nic_rx_bytes="$(sum_nic_delta_field "$mode" "$phase" "$run_no" 5)"
        nic_tx_bytes="$(sum_nic_delta_field "$mode" "$phase" "$run_no" 6)"
        nic_rx_packets="$(sum_nic_delta_field "$mode" "$phase" "$run_no" 7)"
        nic_tx_packets="$(sum_nic_delta_field "$mode" "$phase" "$run_no" 8)"

        timestamp="$(date -Iseconds)"
        backend_log="${BACKEND_LOG:-}"

        # Keep routine_summary safe for CSV by replacing commas just in case.
        routine_summary="${routine_summary//,/;}"

        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$timestamp" "$mode" "$phase" "$run_no" "$exit_code" "$wall_s" \
            "$accuracy" "$execution_s" "$per_image_s" \
            "$gvirtus_calls" "$gvirtus_in_bytes" "$gvirtus_out_bytes" \
            "$nic_rx_bytes" "$nic_tx_bytes" "$nic_rx_packets" "$nic_tx_packets" \
            "$GVIRTUS_CONFIG" "$log_file" "$backend_log" "$routine_summary" \
            >> "$CSV"

        if [[ "$exit_code" -eq 0 ]]; then
            echo "  OK wall=${wall_s}s exec=${execution_s:-NA}s per_image=${per_image_s:-NA}s acc=${accuracy:-NA}% calls=${gvirtus_calls} in=${gvirtus_in_bytes}B out=${gvirtus_out_bytes}B nic_tx=${nic_tx_bytes}B nic_rx=${nic_rx_bytes}B"
        elif [[ "$exit_code" -eq 124 && -n "$accuracy" && -n "$execution_s" ]]; then
            echo "  SOFT-OK exit_code=124 after valid output; likely UCX close hang. exec=${execution_s}s per_image=${per_image_s:-NA}s acc=${accuracy}% calls=${gvirtus_calls} in=${gvirtus_in_bytes}B out=${gvirtus_out_bytes}B"
        else
            echo "  FAILED exit_code=$exit_code. See: $log_file"
        fi
    done

    echo
done

echo "Done."
echo "Main CSV:      $CSV"
echo "Routine CSV:   $ROUTINE_CSV"
echo "NIC CSV:       $NIC_CSV"
echo "Counter dumps: $COUNTER_DIR"
echo

echo "Simple summary, measured runs only:"
awk -F, '
NR > 1 && $3 == "measure" && ($5 == 0 || ($5 == 124 && $7 != "" && $8 != "")) {
    mode=$2
    wall[mode]+=$6
    execs[mode]+=$8
    perimg[mode]+=$9
    calls[mode]+=$10
    inb[mode]+=$11
    outb[mode]+=$12
    nrx[mode]+=$13
    ntx[mode]+=$14
    n[mode]++
}
END {
    for (m in n) {
        printf "%s: n=%d mean_wall_s=%.6f mean_exec_s=%.6f mean_per_image_s=%.6f mean_calls=%.1f mean_gvirtus_in_B=%.1f mean_gvirtus_out_B=%.1f mean_nic_rx_B=%.1f mean_nic_tx_B=%.1f\n",
            m, n[m], wall[m]/n[m], execs[m]/n[m], perimg[m]/n[m],
            calls[m]/n[m], inb[m]/n[m], outb[m]/n[m], nrx[m]/n[m], ntx[m]/n[m]
    }
}
' "$CSV"
