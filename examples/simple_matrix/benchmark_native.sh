#!/bin/bash
set -euo pipefail

# Native CUDA benchmark — no GVirtuS, links directly against system CUDA/cuBLAS.
# Use this as the baseline to measure GVirtuS overhead.

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MATRIX_SIZES=${MATRIX_SIZES:-"128 256 512 1024 2048 4096 8192 16384"}
ITERATIONS=${ITERATIONS:-50}
WARMUP=${WARMUP:-5}
MODE_LABEL=${MODE_LABEL:-native}
VARIANT_LABEL=${VARIANT_LABEL:-baseline}

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

cd "${SCRIPT_DIR}"

if ! grep -q "CSV," simple_matrix.cu; then
    echo "ERROR: simple_matrix.cu does not contain CSV output." >&2
    exit 1
fi

echo "Compiling simple_matrix (native CUDA)..."
nvcc simple_matrix.cu -o simple_matrix_native \
    -O2 \
    -L"${CUDA_HOME}/lib64" \
    -lcuda -lcudart -lcublas

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/benchmark_results}"
OUT_FILE="${OUT_DIR}/simple_matrix_${MODE_LABEL}_${VARIANT_LABEL}.csv"

mkdir -p "${OUT_DIR}"
echo "mode,variant,size,iters,gpu_ms,host_ms" > "${OUT_FILE}"

for n in ${MATRIX_SIZES}; do
    echo "  Running size=${n} iters=${ITERATIONS} warmup=${WARMUP}..."
    output=$(MATRIX_SIZE="${n}" ITERATIONS="${ITERATIONS}" WARMUP="${WARMUP}" \
        ./simple_matrix_native 2>&1) || {
        echo "ERROR: simple_matrix_native failed for size ${n}" >&2
        echo "${output}" >&2
        exit 1
    }
    line=$(printf '%s\n' "${output}" | awk -F, '/^CSV,/{print $0}')
    if [[ -z "${line}" ]]; then
        echo "ERROR: No CSV line produced for size ${n}" >&2
        echo "${output}" >&2
        exit 1
    fi
    echo "${MODE_LABEL},${VARIANT_LABEL},${line#CSV,}" >> "${OUT_FILE}"
    # Print human-readable line too
    printf '%s\n' "${output}" | grep -v "^CSV,"
done

echo ""
echo "Results written to: ${OUT_FILE}"
echo ""
echo "--- CSV contents ---"
cat "${OUT_FILE}"
