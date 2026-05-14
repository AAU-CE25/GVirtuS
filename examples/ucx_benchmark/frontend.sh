#!/bin/bash
# Compile and run benchmarks as a GVirtuS frontend client.
# This script is executed inside the benchmark container.
set -e

export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}

BENCH_DIR="${GVIRTUS_HOME}/examples/ucx_benchmark"
cd "${BENCH_DIR}"

echo "=== Compiling benchmarks ==="

# Test 1: Pure data copy (no CUDA) — TCP always available, UCX if libs present
g++ -O2 -o data_copy_bench data_copy_bench.cpp -lpthread 2>/dev/null && \
    echo "  [OK] data_copy_bench (TCP only)"

# Try building with UCX support
if g++ -O2 -DUSE_UCX -o data_copy_bench_ucx data_copy_bench.cpp -lucp -lucs -lpthread 2>/dev/null; then
    echo "  [OK] data_copy_bench_ucx (TCP + UCX)"
    # Use the UCX-enabled binary as default
    cp data_copy_bench_ucx data_copy_bench
else
    echo "  [SKIP] UCX libs not found, data_copy_bench supports TCP only"
fi

# Test 2: Matrix mul (CUDA via GVirtuS)
nvcc matrix_mul_bench.cu -o matrix_mul_bench \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lcublas
echo "  [OK] matrix_mul_bench"

chmod +x run_benchmarks.sh

echo "=== Running benchmarks ==="

./run_benchmarks.sh \
    --test "${BENCH_TEST:-all}" \
    --runs "${BENCH_RUNS:-10}" \
    --data-sizes "${BENCH_DATA_SIZES:-1024,8192,65536,262144,1048576,4194304,16777216}" \
    --matrix-ns "${BENCH_MATRIX_NS:-64,128,256,512,1024,2048}" \
    --output-dir "${BENCH_OUTPUT_DIR:-./results}" \
    --tag "${BENCH_TAG:-default}" \
    --transport "${BENCH_TRANSPORT:-tcp}" \
    --server "${BENCH_SERVER:-24.24.24.1}" \
    --port "${BENCH_PORT:-5555}"
