#!/bin/bash
set -euo pipefail

GVIRTUS_HOME=${GVIRTUS_HOME:-/opt/GVirtuS}
EXAMPLES_DIR="${GVIRTUS_HOME}/examples"

MATRIX_SIZES=${MATRIX_SIZES:-"1024 2048 4096"}
ITERATIONS=${ITERATIONS:-50}
WARMUP=${WARMUP:-5}
MODE_LABEL=${MODE_LABEL:-custom}
VARIANT_LABEL=${VARIANT_LABEL:-custom}
export LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH:-}"

mkdir -p "${EXAMPLES_DIR}"
cd "${EXAMPLES_DIR}"

if ! grep -q "CSV," simple_matrix.cu; then
    echo "ERROR: simple_matrix.cu does not contain CSV output. Sync the updated source file." >&2
    exit 1
fi

nvcc simple_matrix.cu -o simple_matrix -g --cudart=shared \
  -L"${GVIRTUS_HOME}/lib/frontend" \
  -L"${GVIRTUS_HOME}/lib" \
  -lcuda -lcudart -lcublas

run_sizes() {
    local out_dir="${OUT_DIR:-${PWD}/benchmark_results}"
    local out_file="${out_dir}/simple_matrix_${MODE_LABEL}_${VARIANT_LABEL}.csv"

    mkdir -p "${out_dir}"
    echo "mode,variant,size,iters,gpu_ms,host_ms,h2d_us,d2h_us,h2d_ms,d2h_ms,h2d_GBps,d2h_GBps" > "${out_file}"
    for n in ${MATRIX_SIZES}; do
        local line
        local output
        local status=0
        output=$(MATRIX_SIZE="${n}" ITERATIONS="${ITERATIONS}" WARMUP="${WARMUP}" ./simple_matrix 2>&1) || status=$?
        line=$(printf '%s\n' "${output}" | awk -F, '/^CSV,/{print $0}')
        if [[ -z "${line}" ]]; then
            echo "ERROR: No CSV line produced for size ${n} (exit=${status})" >&2
            echo "--- simple_matrix output ---" >&2
            echo "${output}" >&2
            exit 1
        fi
        echo "${MODE_LABEL},${VARIANT_LABEL},${line#CSV,}" >> "${out_file}"
    done

    echo "Wrote ${out_file}"
}

run_sizes
