#!/bin/bash
# run_async_benchmark.sh
# Compiles and runs test_async_overlap.cu in both native and GVirtuS modes
# Place this alongside test_async_overlap.cu in ~/GVirtuS/tests/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="test_async_overlap"
SOURCE="test_async_overlap.cu"
OUTDIR="${SCRIPT_DIR}"
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
RESULT_FILE="${OUTDIR}/async_benchmark_${TIMESTAMP}.csv"

# Header
echo "mode,scenario,avg_ms,speedup_vs_seq,overlap_pct,timestamp" > "$RESULT_FILE"

run_and_parse() {
    local mode="$1"
    local binary="$2"
    local ts
    ts=$(date -Iseconds)

    echo ""
    echo "========================================="
    echo "  Running: $mode"
    echo "========================================="

    output=$("$binary" 2>&1)
    echo "$output"

    # Parse CSV block from output
    in_csv=0
    while IFS= read -r line; do
        if [[ "$line" == "CSV:" ]]; then
            in_csv=1
            continue
        fi
        if [[ $in_csv -eq 1 && "$line" == scenario* ]]; then
            continue  # skip header
        fi
        if [[ $in_csv -eq 1 && -n "$line" ]]; then
            echo "${mode},${line},${ts}" >> "$RESULT_FILE"
        fi
    done <<< "$output"
}

# -----------------------------------------------
# Step 1: Compile
# -----------------------------------------------
echo "Compiling ${SOURCE}..."
nvcc -O2 -arch=native \
    -o "${OUTDIR}/${BINARY}_native" \
    "${SCRIPT_DIR}/${SOURCE}"
echo "Native binary ready."

# GVirtuS version: link against GVirtuS frontend stubs
if [ -f /usr/local/gvirtus/lib/libcudart.so ]; then
    echo "Compiling GVirtuS version..."
    nvcc -O2 -arch=native \
        -L/usr/local/gvirtus/lib \
        -Wl,-rpath,/usr/local/gvirtus/lib \
        -o "${OUTDIR}/${BINARY}_gvirtus" \
        "${SCRIPT_DIR}/${SOURCE}"
    GVIRTUS_BINARY="${OUTDIR}/${BINARY}_gvirtus"
    HAS_GVIRTUS=1
else
    echo "WARNING: GVirtuS lib not found at /usr/local/gvirtus/lib — skipping GVirtuS run."
    HAS_GVIRTUS=0
fi

# -----------------------------------------------
# Step 2: Native run
# -----------------------------------------------
run_and_parse "native" "${OUTDIR}/${BINARY}_native"

# -----------------------------------------------
# Step 3: GVirtuS run (requires backend on es-dpu-01)
# -----------------------------------------------
if [[ $HAS_GVIRTUS -eq 1 ]]; then
    echo ""
    echo "NOTE: Make sure GVirtuS backend is running on es-dpu-01 before continuing."
    echo "Press ENTER to run GVirtuS test, or Ctrl+C to skip."
    read -r
    GVIRTUS_CONFIG="${GVIRTUS_CONFIG:-/usr/local/gvirtus/etc/gvirtus.json}"
    GVIRTUS_FRONTEND_CONFIG="$GVIRTUS_CONFIG" \
        run_and_parse "gvirtus" "$GVIRTUS_BINARY"
fi

echo ""
echo "========================================="
echo "  Done! Results saved to:"
echo "  $RESULT_FILE"
echo "========================================="
cat "$RESULT_FILE"