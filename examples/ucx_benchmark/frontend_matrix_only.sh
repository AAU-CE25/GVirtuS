#!/bin/bash
# Single-run matrix_mul through GVirtuS UCX — smoke test.
# Compiles matrix_mul_bench against the GVirtuS frontend stub and runs it once
# with a small matrix.
set -e

export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}

BENCH_DIR="${GVIRTUS_HOME}/examples/ucx_benchmark"
cd "${BENCH_DIR}"

N="${N:-128}"
RUNS="${RUNS:-1}"

echo "=== Building matrix_mul_bench ==="
nvcc -O2 matrix_mul_bench.cu -o /tmp/matrix_mul_bench \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lcublas

echo "=== Running matrix_mul_bench N=${N} runs=${RUNS} via GVirtuS (UCX) ==="
echo "    config: ${GVIRTUS_HOME}/etc/properties.json"
cat "${GVIRTUS_HOME}/etc/properties.json"
echo

/tmp/matrix_mul_bench "${N}" "${RUNS}"
