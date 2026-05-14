#!/bin/bash
# ============================================================================
# Benchmark Runner
#
# Runs data_copy_bench and/or matrix_mul_bench across configurable N scales,
# number of runs, and saves results to CSV files.
#
# Usage:
#   ./run_benchmarks.sh [OPTIONS]
#
# Options:
#   --test       Which tests to run: "data_copy", "matrix_mul", or "all" (default: all)
#   --runs       Number of repetitions per N value (default: 10)
#   --data-sizes Comma-separated list of byte sizes for data_copy test
#                (default: 1024,8192,65536,262144,1048576,4194304,16777216)
#   --matrix-ns  Comma-separated list of N dimensions for matrix_mul test
#                (default: 64,128,256,512,1024,2048)
#   --output-dir Directory for CSV output (default: ./results)
#   --tag        Tag appended to output filenames (e.g., "tcp" or "ucx")
#   --transport  Transport for data_copy test: "tcp" or "ucx" (default: tcp)
#   --server     Server IP/hostname for data_copy test (default: 24.24.24.1)
#   --port       Server port for data_copy test (default: 5555)
#   --help       Show this help message
#
# Example:
#   ./run_benchmarks.sh --test all --runs 20 --tag ucx --transport ucx --server 25.25.25.1
# ============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
TEST="all"
RUNS=10
DATA_SIZES="1024,8192,65536,262144,1048576,4194304,16777216"
MATRIX_NS="64,128,256,512,1024,2048"
OUTPUT_DIR="./results"
TAG=""
TRANSPORT="tcp"
SERVER="24.24.24.1"
PORT="5555"

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)       TEST="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --data-sizes) DATA_SIZES="$2"; shift 2 ;;
        --matrix-ns)  MATRIX_NS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --transport)  TRANSPORT="$2"; shift 2 ;;
        --server)     SERVER="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --help)
            head -32 "$0" | tail -29
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUFFIX="${TAG:+_${TAG}}_${TIMESTAMP}"

echo "============================================="
echo " GVirtuS UCX Benchmark Runner"
echo "============================================="
echo " Test:        ${TEST}"
echo " Runs:        ${RUNS}"
echo " Transport:   ${TRANSPORT}"
echo " Server:      ${SERVER}:${PORT}"
echo " Data sizes:  ${DATA_SIZES}"
echo " Matrix Ns:   ${MATRIX_NS}"
echo " Output dir:  ${OUTPUT_DIR}"
echo " Tag:         ${TAG:-<none>}"
echo "============================================="
echo ""

# ─── Test 1: Data Copy (pure transport, no CUDA) ─────────────────────────────
run_data_copy() {
    local OUTFILE="${OUTPUT_DIR}/data_copy${SUFFIX}.csv"
    echo "run,n_bytes,send_us,recv_us,roundtrip_us" > "${OUTFILE}"

    IFS=',' read -ra SIZES <<< "${DATA_SIZES}"
    for size in "${SIZES[@]}"; do
        echo "[data_copy] transport=${TRANSPORT} N=${size} bytes, ${RUNS} runs..."
        # Run as client, connecting to the echo server
        "${SCRIPT_DIR}/data_copy_bench" client "${TRANSPORT}" "${SERVER}" "${PORT}" "${size}" "${RUNS}" \
            | tail -n +2 >> "${OUTFILE}"
    done

    echo "[data_copy] Results saved to: ${OUTFILE}"
    echo ""
}

# ─── Test 2: Matrix Multiplication (CUDA via GVirtuS) ────────────────────────
run_matrix_mul() {
    local OUTFILE="${OUTPUT_DIR}/matrix_mul${SUFFIX}.csv"
    echo "run,n,matrix_bytes,h2d_us,compute_us,d2h_us,total_us" > "${OUTFILE}"

    IFS=',' read -ra NS <<< "${MATRIX_NS}"
    for n in "${NS[@]}"; do
        echo "[matrix_mul] N=${n}, ${RUNS} runs..."
        "${SCRIPT_DIR}/matrix_mul_bench" "${n}" "${RUNS}" | tail -n +2 >> "${OUTFILE}"
    done

    echo "[matrix_mul] Results saved to: ${OUTFILE}"
    echo ""
}

# ─── Execute selected tests ──────────────────────────────────────────────────
case "${TEST}" in
    data_copy)
        run_data_copy
        ;;
    matrix_mul)
        run_matrix_mul
        ;;
    all)
        run_data_copy
        run_matrix_mul
        ;;
    *)
        echo "ERROR: Unknown test '${TEST}'. Use: data_copy, matrix_mul, or all."
        exit 1
        ;;
esac

echo "============================================="
echo " Benchmark complete!"
echo " Output: ${OUTPUT_DIR}/"
echo "============================================="
