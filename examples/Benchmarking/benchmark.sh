#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_SCRIPT_VERSION="2026-05-19c"

# GVirtuS frontend benchmark harness.
# Assumption: the GVirtuS backend is already running with the same connector mode/config.
# Output by default: <this script directory>/benchmark_results/<session>/results.csv + static.csv + metadata.json + logs/*.log
#
# Recommended explicit benchmark keys for examples:
#   BENCHMARK_RESULT_MS=...
#   BENCHMARK_INFERENCE_MS=...
#   BENCHMARK_PREPROCESS_MS=...
#   BENCHMARK_POSTPROCESS_MS=...
#   BENCHMARK_FPS=...
#   BENCHMARK_DETECTIONS=...
#   BENCHMARK_FACES=...
# The script also parses simple_matrix STAGE_*_MS and RESULT_CHECK output.

usage() {
    cat <<'USAGE'
Usage:
  ./benchmark.sh --mode <tcp|rdma|ucx|ucx_tcp|ucx_rdma> [options]

Defaults:
  --warmups 3
  --runs 10
  --examples simple_matrix,face_recon,opencv_dnn,opencv_yolo
  --out benchmark_results

Options:
  --mode MODE              Connector mode to record/apply: tcp, rdma, ucx, ucx_tcp, ucx_rdma
  --warmups N              Warmup iterations per example, not used in averages
  --runs N                 Measured iterations per example
  --examples CSV           Comma-separated examples; hyphen or underscore both work
  --out DIR                Output root directory. Relative paths are under this script's directory.
  --matrix-n N|all         MATRIX_N for simple_matrix; all = 256 512 1024 2048 4096 8192 16384
  --ucx-tls VALUE          Override UCX_TLS, e.g. tcp,self or rc_mlx5,ud_mlx5,self
  --ucx-net-devices VALUE  Override UCX_NET_DEVICES, e.g. mlx5_1:1
  --ucx-datapath VALUE     Override GVIRTUS_UCX_DATAPATH, e.g. am
  --backend-container NAME Backend container name to check; warning only
  --repo-root DIR          GVirtuS repo root; auto-detected by default
  -h, --help               Show this help

Environment overrides for frontend commands:
  SIMPLE_MATRIX_CMD='make -C /path/to/GVirtuS run-simple-matrix-test'
  FACE_RECON_CMD='make -C /path/to/GVirtuS run-face-recon-test'
  OPENCV_DNN_CMD='make -C /path/to/GVirtuS run-opencv-dnn-test'
  OPENCV_YOLO_CMD='make -C /path/to/GVirtuS run-opencv-yolo-test'

Examples:
  ./benchmark.sh --mode tcp --matrix-n all
  ./benchmark.sh --mode ucx_tcp --matrix-n all
  ./benchmark.sh --mode ucx_rdma --matrix-n 4096 --ucx-net-devices mlx5_1:1
  ./benchmark.sh --mode tcp --examples simple_matrix,opencv-yolo
USAGE
}

ORIGINAL_PWD="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

MODE="${MODE:-ucx_tcp}"
WARMUPS="${WARMUPS:-3}"
RUNS="${RUNS:-10}"
EXAMPLES_CSV="${EXAMPLES:-simple_matrix,face_recon,opencv_dnn,opencv_yolo}"
OUT_ROOT="${OUT_DIR:-benchmark_results}"
MATRIX_N="${MATRIX_N:-512}"
MATRIX_N_ALL_VALUES="${MATRIX_N_ALL_VALUES:-256 512 1024 2048 4096 8192 16384}"
USER_SHORT="$(whoami | cut -d'@' -f1 | tr -d '.')"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-gvirtus-${USER_SHORT}}"
REPO_ROOT="${REPO_ROOT:-}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --warmups) WARMUPS="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --examples) EXAMPLES_CSV="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        --matrix-n) MATRIX_N="$2"; shift 2 ;;
        --ucx-tls) UCX_TLS="$2"; shift 2 ;;
        --ucx-net-devices) UCX_NET_DEVICES="$2"; shift 2 ;;
        --ucx-datapath) GVIRTUS_UCX_DATAPATH="$2"; shift 2 ;;
        --backend-container) BACKEND_CONTAINER="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$WARMUPS" =~ ^[0-9]+$ && "$RUNS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --warmups and --runs must be non-negative integers" >&2
    exit 2
fi

if [[ "$RUNS" -eq 0 ]]; then
    echo "ERROR: --runs must be at least 1" >&2
    exit 2
fi

if [[ "$MATRIX_N" != "all" ]] && ! [[ "$MATRIX_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --matrix-n must be a positive integer or 'all'" >&2
    exit 2
fi

normalize_example() {
    printf '%s' "$1" | tr '[:upper:]-' '[:lower:]_'
}

find_repo_root() {
    local start="$1"
    local dir
    dir="$(cd "$start" 2>/dev/null && pwd -P)" || return 1

    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/Makefile" && -d "$dir/examples" && -d "$dir/etc" ]]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done

    return 1
}

detect_repo_root() {
    local detected

    if [[ -n "${REPO_ROOT:-}" ]]; then
        if [[ ! -f "$REPO_ROOT/Makefile" ]]; then
            echo "ERROR: --repo-root '$REPO_ROOT' does not look like the GVirtuS repo root." >&2
            exit 2
        fi
        REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
    else
        detected="$(find_repo_root "$PWD" || true)"
        if [[ -z "$detected" ]]; then
            detected="$(find_repo_root "$SCRIPT_DIR" || true)"
        fi
        if [[ -z "$detected" ]]; then
            echo "ERROR: Could not auto-detect GVirtuS repo root." >&2
            echo "Pass --repo-root /home/student.aau.dk/ul11nh/GVirtuS" >&2
            exit 2
        fi
        REPO_ROOT="$detected"
    fi

    export REPO_ROOT SCRIPT_DIR
}

apply_connector_mode() {
    case "$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]')" in
        tcp|plain_tcp)
            MODE="plain_tcp"
            CONNECTOR_SUITE="tcp"
            export GVIRTUS_CONFIG_FILE="${GVIRTUS_CONFIG_FILE:-${TCP_CONFIG_FILE:-properties.json}}"
            export GVIRTUS_UCX_DATAPATH="${GVIRTUS_UCX_DATAPATH:-}"
            export UCX_TLS="${UCX_TLS:-}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES:-}"
            export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-}"
            ;;
        rdma|plain_rdma)
            MODE="plain_rdma"
            CONNECTOR_SUITE="rdma"
            export GVIRTUS_CONFIG_FILE="${GVIRTUS_CONFIG_FILE:-${RDMA_CONFIG_FILE:-properties_plain_rdma.json}}"
            export GVIRTUS_UCX_DATAPATH="${GVIRTUS_UCX_DATAPATH:-}"
            export UCX_TLS="${UCX_TLS:-}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES:-}"
            export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-}"
            ;;
        ucx|ucx_tcp)
            MODE="ucx_tcp"
            CONNECTOR_SUITE="ucx"
            export GVIRTUS_CONFIG_FILE="${GVIRTUS_CONFIG_FILE:-${UCX_CONFIG_FILE:-properties_ucx.json}}"
            export GVIRTUS_UCX_DATAPATH="${GVIRTUS_UCX_DATAPATH:-am}"
            export UCX_TLS="${UCX_TLS:-tcp,self}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES:-}"
            export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp}"
            ;;
        ucx_rdma)
            MODE="ucx_rdma"
            CONNECTOR_SUITE="ucx"
            export GVIRTUS_CONFIG_FILE="${GVIRTUS_CONFIG_FILE:-${UCX_CONFIG_FILE:-properties_ucx.json}}"
            export GVIRTUS_UCX_DATAPATH="${GVIRTUS_UCX_DATAPATH:-am}"
            export UCX_TLS="${UCX_TLS:-rc_mlx5,ud_mlx5,self}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_1:1}"
            export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-rdmacm}"
            ;;
        *)
            echo "ERROR: Unsupported --mode '$MODE'" >&2
            usage >&2
            exit 2
            ;;
    esac

    export MATRIX_N
    export CONNECTOR_SUITE
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_backend_warning_only() {
    if command_exists docker; then
        if docker ps --format '{{.Names}}' | grep -qx "$BACKEND_CONTAINER"; then
            echo "Backend container detected: $BACKEND_CONTAINER"
        else
            echo "WARNING: Backend container '$BACKEND_CONTAINER' not detected. Continuing because the backend may be running manually or under another name." >&2
        fi
    fi
}

write_static_metadata() {
    RESULTS_CSV="$RESULTS_CSV" STATIC_CSV="$STATIC_CSV" METADATA_JSON="$METADATA_JSON" python3 - <<'PY'
import csv
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "script_version": os.environ.get("BENCHMARK_SCRIPT_VERSION", ""),
    "session_id": os.environ.get("SESSION_ID", ""),
    "hostname": socket.gethostname(),
    "repo_root": os.environ.get("REPO_ROOT", ""),
    "script_dir": os.environ.get("SCRIPT_DIR", ""),
    "platform": platform.platform(),
    "kernel": run("uname -a"),
    "git_branch": run(f"git -C {os.environ.get('REPO_ROOT', '.')!r} rev-parse --abbrev-ref HEAD"),
    "git_commit": run(f"git -C {os.environ.get('REPO_ROOT', '.')!r} rev-parse HEAD"),
    "git_status_short": run(f"git -C {os.environ.get('REPO_ROOT', '.')!r} status --short"),
    "docker_version": run("docker --version"),
    "nvidia_smi": run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "ib_devices": run("ls -1 /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//'"),
    "mode": os.environ.get("MODE", ""),
    "connector_suite": os.environ.get("CONNECTOR_SUITE", ""),
    "gv_config_file": os.environ.get("GVIRTUS_CONFIG_FILE", ""),
    "gv_ucx_datapath": os.environ.get("GVIRTUS_UCX_DATAPATH", ""),
    "ucx_tls": os.environ.get("UCX_TLS", ""),
    "ucx_net_devices": os.environ.get("UCX_NET_DEVICES", ""),
    "ucx_sockaddr_tls_priority": os.environ.get("UCX_SOCKADDR_TLS_PRIORITY", ""),
    "matrix_n": os.environ.get("MATRIX_N", ""),
    "warmups": os.environ.get("WARMUPS", ""),
    "runs": os.environ.get("RUNS", ""),
    "examples": os.environ.get("EXAMPLES_CSV", ""),
}

with open(os.environ["STATIC_CSV"], "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    for key, value in metadata.items():
        writer.writerow([key, value])

with open(os.environ["METADATA_JSON"], "w") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
PY
}

append_result_row() {
    RESULTS_CSV="$RESULTS_CSV" LOG_FILE="$LOG_FILE" COMMAND_STR="$COMMAND_STR" EXAMPLE="$EXAMPLE" RUN_TYPE="$RUN_TYPE" RUN_INDEX="$RUN_INDEX" WARMUP_FLAG="$WARMUP_FLAG" EXIT_CODE="$EXIT_CODE" ELAPSED_MS="$ELAPSED_MS" python3 - <<'PY'
import csv
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

columns = [
    "timestamp_utc", "session_id", "hostname", "git_branch", "git_commit",
    "mode", "connector_suite", "gv_config_file", "gv_ucx_datapath",
    "ucx_tls", "ucx_net_devices", "ucx_sockaddr_tls_priority",
    "example", "run_type", "run_index", "warmup", "exit_code", "elapsed_ms",
    "command", "log_file",
    "matrix_n", "benchmark_result_ms",
    "stage_malloc_ms", "stage_cudamalloc_ms", "stage_h2d_ms",
    "stage_cublas_create_ms", "stage_gemm_ms", "stage_d2h_ms", "stage_cleanup_ms",
    "result_check", "inference_ms", "preprocess_ms", "postprocess_ms", "fps",
    "faces", "objects", "detections", "extra_metrics_json",
]


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except Exception:
        return ""


def first(patterns, text, flags=re.IGNORECASE | re.MULTILINE):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return ""

log_path = Path(os.environ["LOG_FILE"])
text = log_path.read_text(errors="replace") if log_path.exists() else ""
row = {c: "" for c in columns}
row.update({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "session_id": os.environ.get("SESSION_ID", ""),
    "hostname": socket.gethostname(),
    "git_branch": run(f"git -C {os.environ.get('REPO_ROOT', '.')!r} rev-parse --abbrev-ref HEAD"),
    "git_commit": run(f"git -C {os.environ.get('REPO_ROOT', '.')!r} rev-parse HEAD"),
    "mode": os.environ.get("MODE", ""),
    "connector_suite": os.environ.get("CONNECTOR_SUITE", ""),
    "gv_config_file": os.environ.get("GVIRTUS_CONFIG_FILE", ""),
    "gv_ucx_datapath": os.environ.get("GVIRTUS_UCX_DATAPATH", ""),
    "ucx_tls": os.environ.get("UCX_TLS", ""),
    "ucx_net_devices": os.environ.get("UCX_NET_DEVICES", ""),
    "ucx_sockaddr_tls_priority": os.environ.get("UCX_SOCKADDR_TLS_PRIORITY", ""),
    "example": os.environ.get("EXAMPLE", ""),
    "run_type": os.environ.get("RUN_TYPE", ""),
    "run_index": os.environ.get("RUN_INDEX", ""),
    "warmup": os.environ.get("WARMUP_FLAG", ""),
    "exit_code": os.environ.get("EXIT_CODE", ""),
    "elapsed_ms": os.environ.get("ELAPSED_MS", ""),
    "command": os.environ.get("COMMAND_STR", ""),
    "log_file": str(log_path),
    "matrix_n": os.environ.get("MATRIX_N", ""),
})

explicit_map = {
    "benchmark_result_ms": [r"^BENCHMARK_RESULT_MS=([0-9.]+)"],
    "matrix_n": [r"^BENCHMARK_MATRIX_N=([0-9]+)", r"\bOK\s+N=([0-9]+)"],
    "result_check": [r"^RESULT_CHECK=([^\s]+)"],
    "inference_ms": [
        r"^BENCHMARK_INFERENCE_MS=([0-9.]+)",
        r"\binference(?:_time)?(?:_ms)?\s*[:=]\s*([0-9.]+)\s*ms?",
    ],
    "preprocess_ms": [r"^BENCHMARK_PREPROCESS_MS=([0-9.]+)", r"\bpreprocess(?:_ms)?\s*[:=]\s*([0-9.]+)\s*ms?"],
    "postprocess_ms": [r"^BENCHMARK_POSTPROCESS_MS=([0-9.]+)", r"\bpostprocess(?:_ms)?\s*[:=]\s*([0-9.]+)\s*ms?"],
    "fps": [r"^BENCHMARK_FPS=([0-9.]+)", r"\bfps\s*[:=]\s*([0-9.]+)"],
    "faces": [r"^BENCHMARK_FACES=([0-9]+)", r"\bfaces?\s*[:=]\s*([0-9]+)"],
    "objects": [r"^BENCHMARK_OBJECTS=([0-9]+)", r"\bobjects?\s*[:=]\s*([0-9]+)"],
    "detections": [r"^BENCHMARK_DETECTIONS=([0-9]+)", r"\bdetections?\s*[:=]\s*([0-9]+)"],
}
for key, patterns in explicit_map.items():
    value = first(patterns, text)
    if value:
        row[key] = value

for stage, value in re.findall(r"^STAGE_([A-Z0-9_]+)_MS=([0-9.]+)", text, flags=re.MULTILINE):
    col = f"stage_{stage.lower()}_ms"
    if col in row:
        row[col] = value

extra = {}
known_upper = {
    "BENCHMARK_RESULT_MS", "BENCHMARK_MATRIX_N", "RESULT_CHECK",
    "BENCHMARK_INFERENCE_MS", "BENCHMARK_PREPROCESS_MS", "BENCHMARK_POSTPROCESS_MS",
    "BENCHMARK_FPS", "BENCHMARK_FACES", "BENCHMARK_OBJECTS", "BENCHMARK_DETECTIONS",
}
for key, value in re.findall(r"^([A-Z][A-Z0-9_]{2,})=([^\r\n]+)", text, flags=re.MULTILINE):
    if key not in known_upper and (key.startswith("BENCHMARK_") or key.startswith("STAGE_") or key.startswith("RESULT_") or key.startswith("FACE_") or key.startswith("YOLO_") or key.startswith("DNN_") or key.startswith("OPENCV_")):
        extra[key] = value.strip()
row["extra_metrics_json"] = json.dumps(extra, sort_keys=True) if extra else ""

results_path = Path(os.environ["RESULTS_CSV"])
new_file = not results_path.exists()
with results_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    if new_file:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
    local example="$1"
    local run_type="$2"
    local run_index="$3"
    local command_str="$4"
    local warmup_flag="$5"
    local matrix_n_value="${6:-$MATRIX_N}"

    local log_file
    if [[ "$example" == "simple_matrix" ]]; then
        log_file="${LOG_DIR}/${example}_${run_type}_${run_index}_N${matrix_n_value}.log"
    else
        log_file="${LOG_DIR}/${example}_${run_type}_${run_index}.log"
    fi

    local label
    if [[ "$example" == "simple_matrix" ]]; then
        label="[$example][N=$matrix_n_value][$run_type $run_index]"
    else
        label="[$example][$run_type $run_index]"
    fi

    if is_verbose; then
        echo "$label $command_str"
    else
        printf '%-48s ' "$label"
    fi

    local start_ns end_ns exit_code elapsed_ms
    start_ns="$(date +%s%N)"
    set +e
    MATRIX_N="$matrix_n_value" bash -lc "$command_str" >"$log_file" 2>&1
    exit_code=$?
    set -e
    end_ns="$(date +%s%N)"
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

    export LOG_FILE="$log_file"
    export COMMAND_STR="$command_str"
    export MATRIX_N="$matrix_n_value"
    export EXAMPLE="$example"
    export RUN_TYPE="$run_type"
    export RUN_INDEX="$run_index"
    export WARMUP_FLAG="$warmup_flag"
    export EXIT_CODE="$exit_code"
    export ELAPSED_MS="$elapsed_ms"
    append_result_row

    if [[ "$exit_code" -eq 0 ]]; then
        if ! is_verbose; then
            printf 'OK   %sms
' "$elapsed_ms"
        fi
    else
        if ! is_verbose; then
            printf 'FAIL %sms
' "$elapsed_ms"
        fi
        echo "ERROR: $example $run_type $run_index failed with exit code $exit_code"
        echo "       Log: $(relpath "$log_file")"
    fi
}

detect_repo_root
apply_connector_mode

SESSION_ID="${SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${MODE}}"
case "$OUT_ROOT" in
    /*) OUT_DIR="${OUT_ROOT}/${SESSION_ID}" ;;
    *) OUT_DIR="${SCRIPT_DIR}/${OUT_ROOT}/${SESSION_ID}" ;;
esac
LOG_DIR="${OUT_DIR}/logs"
RESULTS_CSV="${OUT_DIR}/results.csv"
STATIC_CSV="${OUT_DIR}/static.csv"
METADATA_JSON="${OUT_DIR}/metadata.json"
mkdir -p "$LOG_DIR"

export BENCHMARK_SCRIPT_VERSION MODE WARMUPS RUNS EXAMPLES_CSV SESSION_ID OUT_DIR LOG_DIR RESULTS_CSV STATIC_CSV METADATA_JSON REPO_ROOT SCRIPT_DIR


# Terminal output controls.
BENCHMARK_VERBOSE="${BENCHMARK_VERBOSE:-0}"

is_verbose() {
    [[ "${BENCHMARK_VERBOSE:-0}" == "1" || "${BENCHMARK_VERBOSE:-0}" == "true" ]]
}

hr() {
    printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '-'
}

relpath() {
    local path="$1"
    if [[ "$path" == "$REPO_ROOT"* ]]; then
        printf '.%s' "${path#$REPO_ROOT}"
    elif [[ "$path" == "$BENCHMARK_DIR"* ]]; then
        printf '.%s' "${path#$BENCHMARK_DIR}"
    else
        printf '%s' "$path"
    fi
}

print_kv() {
    printf '  %-18s %s\n' "$1:" "$2"
}

print_runtime_summary() {
    echo
    hr
    echo "GVirtuS Benchmark"
    hr
    print_kv "Mode" "$MODE"
    print_kv "Examples" "$EXAMPLES_CSV"
    if [[ "$EXAMPLES_CSV" == *simple_matrix* ]]; then
        print_kv "Matrix sizes" "$MATRIX_N"
    else
        print_kv "Matrix sizes" "-"
    fi
    print_kv "Runs" "$RUNS"
    print_kv "Warmups" "$WARMUPS"
    print_kv "Config" "$(relpath "$HOST_GVIRTUS_CONFIG")"
    print_kv "Results" "$(relpath "$RESULTS_CSV")"

    if [[ "$MODE" == ucx_* ]]; then
        print_kv "UCX TLS" "${UCX_TLS:-}"
        print_kv "UCX devices" "${UCX_NET_DEVICES:-}"
        print_kv "UCX priority" "${UCX_SOCKADDR_TLS_PRIORITY:-}"
        print_kv "UCX modules" "$(relpath "${UCX_MODULE_DIR:-}")"
    fi

    if [[ "$EXAMPLES_CSV" == *opencv* ]]; then
        print_kv "OpenCV prefix" "$OPENCV_PREFIX"
    fi

    if is_verbose; then
        print_kv "Repo root" "$REPO_ROOT"
        print_kv "Benchmark dir" "$BENCHMARK_DIR"
        print_kv "GVirtuS home" "$HOST_GVIRTUS_HOME"
        print_kv "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH:-}"
    fi

    hr
}
# Host-side GVirtuS runtime used by non-Docker examples.
# Keep this mode-dependent:
#   tcp  -> properties.json
#   rdma -> properties_plain_rdma.json
HOST_GVIRTUS_HOME="${HOST_GVIRTUS_HOME:-$REPO_ROOT}"
HOST_GVIRTUS_CONFIG="${HOST_GVIRTUS_CONFIG:-$REPO_ROOT/etc/$GVIRTUS_CONFIG_FILE}"
OPENCV_PREFIX="${OPENCV_PREFIX:-$HOME/opencv-local}"

export HOST_GVIRTUS_HOME HOST_GVIRTUS_CONFIG OPENCV_PREFIX


# Mode-specific UCX defaults.
# These are only defaults; users can still override them from the shell.
if [[ "$MODE" == "ucx_tcp" ]]; then
    export UCX_TLS="${UCX_TLS:-tcp,self}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-ens1f1np1}"
    export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp}"
    export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-info}"
    export UCX_MODULE_DIR="${UCX_MODULE_DIR:-$REPO_ROOT/lib/ucx}"
elif [[ "$MODE" == "ucx_rdma" ]]; then
    export UCX_TLS="${UCX_TLS:-rc_mlx5,ud_mlx5,self}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_1:1}"
    export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-rdmacm}"
    export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-info}"
    export UCX_MODULE_DIR="${UCX_MODULE_DIR:-$REPO_ROOT/lib/ucx}"
fi


# Repo-local runtime library path for host-side examples.
prepend_ld_library_path() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0

    case ":${LD_LIBRARY_PATH:-}:" in
        *":$dir:"*) ;;
        *)
            if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
                export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
            else
                export LD_LIBRARY_PATH="$dir"
            fi
            ;;
    esac
}

prepend_ld_library_path "$OPENCV_PREFIX/lib"
prepend_ld_library_path "$HOST_GVIRTUS_HOME/lib/frontend"
prepend_ld_library_path "$HOST_GVIRTUS_HOME/lib/ucx"
prepend_ld_library_path "$HOST_GVIRTUS_HOME/lib"

print_runtime_summary

if [[ ! -f "$HOST_GVIRTUS_HOME/lib/libgvirtus-frontend.so" ]]; then
    if is_verbose; then
        echo "Runtime note: repo-local GVirtuS frontend libs not found in $(relpath "$HOST_GVIRTUS_HOME/lib")." >&2
        echo "              If host examples fail, rebuild/copy runtime libs from the backend container." >&2
    fi
fi

if [[ ! -f "$HOST_GVIRTUS_CONFIG" ]]; then
    echo "WARNING: Host GVirtuS config not found: $HOST_GVIRTUS_CONFIG" >&2
fi

check_backend_warning_only
write_static_metadata

if [[ ! -f "$REPO_ROOT/etc/${GVIRTUS_CONFIG_FILE}" ]]; then
    echo "WARNING: $REPO_ROOT/etc/${GVIRTUS_CONFIG_FILE} was not found. If this is not intended, fix --mode or GVIRTUS_CONFIG_FILE." >&2
fi

declare -A EXAMPLE_CMDS
EXAMPLE_CMDS["simple_matrix"]="${SIMPLE_MATRIX_CMD:-make -C \"$REPO_ROOT\" run-simple-matrix-test}"
EXAMPLE_CMDS["face_recon"]="${FACE_RECON_CMD:-cd \"$REPO_ROOT/examples/face-recognition\" && GVIRTUS_HOME=\"$HOST_GVIRTUS_HOME\" GVIRTUS_CONFIG=\"$HOST_GVIRTUS_CONFIG\" bash ./run.sh}"
EXAMPLE_CMDS["opencv_dnn"]="${OPENCV_DNN_CMD:-cd \"$REPO_ROOT/examples/opencv-dnn\" && GVIRTUS_HOME=\"$HOST_GVIRTUS_HOME\" GVIRTUS_CONFIG=\"$HOST_GVIRTUS_CONFIG\" OPENCV_PREFIX=\"$OPENCV_PREFIX\" bash ./run.sh}"
EXAMPLE_CMDS["opencv_yolo"]="${OPENCV_YOLO_CMD:-cd \"$REPO_ROOT/examples/opencv-yolo\" && GVIRTUS_HOME=\"$HOST_GVIRTUS_HOME\" GVIRTUS_CONFIG=\"$HOST_GVIRTUS_CONFIG\" OPENCV_PREFIX=\"$OPENCV_PREFIX\" bash ./run.sh}"

IFS=',' read -r -a requested_examples <<< "$EXAMPLES_CSV"

for raw_example in "${requested_examples[@]}"; do
    example="$(normalize_example "${raw_example// /}")"
    command_str="${EXAMPLE_CMDS[$example]:-}"

    if [[ -z "$command_str" ]]; then
        echo "WARNING: Unknown example '$raw_example' normalized as '$example'. Skipping." >&2
        continue
    fi

    matrix_values=("$MATRIX_N")
    if [[ "$example" == "simple_matrix" && "$MATRIX_N" == "all" ]]; then
        # shellcheck disable=SC2206
        matrix_values=($MATRIX_N_ALL_VALUES)
    fi

    for matrix_n_value in "${matrix_values[@]}"; do
        for ((i = 1; i <= WARMUPS; i++)); do
            run_one "$example" "warmup" "$i" "$command_str" "true" "$matrix_n_value"
        done

        for ((i = 1; i <= RUNS; i++)); do
            run_one "$example" "measured" "$i" "$command_str" "false" "$matrix_n_value"
        done
    done
done

echo
hr
echo "Benchmark complete"
hr
print_kv "Results CSV" "$(relpath "$RESULTS_CSV")"
print_kv "Static CSV" "$(relpath "$STATIC_CSV")"
print_kv "Metadata JSON" "$(relpath "$METADATA_JSON")"
print_kv "Logs" "$(relpath "$LOG_DIR")"

if is_verbose; then
    echo
    echo "Suggested quick checks:"
    echo "  column -s, -t < \"$RESULTS_CSV\" | less -S"
    echo "  awk -F, 'NR==1 || $14==\"measured\" {print}' \"$RESULTS_CSV\" | column -s, -t | less -S"
else
    echo
    echo "Tip: rerun with BENCHMARK_VERBOSE=1 for full commands and debug paths."
fi
