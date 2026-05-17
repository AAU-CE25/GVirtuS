#!/bin/bash
# Matrix-multiplication sweep across N values through GVirtuS.
# Compiles matrix_mul_bench against the GVirtuS frontend stub and runs it
# once per N in BENCH_NS, appending all rows into a single tagged CSV.
#
# Inputs (env vars):
#   BENCH_NS    — comma-separated list of N values (default: 128,256,512,1024,2048,4096)
#   RUNS        — repetitions per N (default: 10)
#   BENCH_TAG   — transport tag used in result filename (default: unknown)
#   RESULTS_DIR — output directory (default: results/)
#
# Output:
#   ${RESULTS_DIR}/matrix_sweep_${BENCH_TAG}_${TS}.csv
set -e

export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-30000}   # WARN by default to keep CSV clean

BENCH_DIR="${GVIRTUS_HOME}/examples/ucx_benchmark"
cd "${BENCH_DIR}"

BENCH_NS="${BENCH_NS:-128,256,512,1024,2048,4096,8192,16384}"
RUNS="${RUNS:-10}"
BENCH_TAG="${BENCH_TAG:-unknown}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_DIR}/results}"
mkdir -p "${RESULTS_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${RESULTS_DIR}/matrix_sweep_${BENCH_TAG}_${TS}.csv"

echo "=== Building matrix_mul_bench ==="
nvcc -O2 matrix_mul_bench.cu -o /tmp/matrix_mul_bench \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lcublas

echo "=== Matrix sweep (tag=${BENCH_TAG}, runs=${RUNS}, Ns=${BENCH_NS}) ==="
echo "    config: ${GVIRTUS_HOME}/etc/properties.json"
cat "${GVIRTUS_HOME}/etc/properties.json"
echo
echo "    output: ${OUT}"

# Master CSV header (adds transport column on top of matrix_mul_bench's CSV).
echo "transport,run,n,matrix_bytes,h2d_us,compute_us,d2h_us,total_us" > "${OUT}"

IFS=',' read -ra NS <<< "${BENCH_NS}"
for N in "${NS[@]}"; do
    N="${N// /}"
    [[ -z "${N}" ]] && continue
    echo "--- N=${N} ---"
    # matrix_mul_bench prints its own header on stdout; strip it and prefix tag.
    /tmp/matrix_mul_bench "${N}" "${RUNS}" \
        | tail -n +2 \
        | awk -v t="${BENCH_TAG}" '{print t","$0}' \
        | tee -a "${OUT}"
done

echo
echo "=== Done. Results: ${OUT} ==="
