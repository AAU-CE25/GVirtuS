#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════════════════════
# Unified entrypoint for Spark simple_matrix benchmark
# ══════════════════════════════════════════════════════════════════════════════
#
# Determines execution mode from the first argument passed to simple_matrix.py:
#   local   → Run with real CUDA libs (needs --runtime=nvidia)
#   docker  → Same as local, inside Docker
#   gvirtus → Use GVirtuS frontend stubs (no local GPU needed)
#
# The key difference is WHERE the JVM finds CUDA:
#   local/docker: /usr/local/cuda/lib64 (real CUDA from nvidia base image)
#   gvirtus:      /opt/GVirtuS/lib/frontend (GVirtuS stubs → remote GPU)
# ══════════════════════════════════════════════════════════════════════════════

GVIRTUS_HOME="${GVIRTUS_HOME:-/opt/GVirtuS}"
RAPIDS_JAR="/app/jars/rapids-4-spark_2.12-26.02.1.jar"
NATIVE_DIR="/tmp/rapids-native"

# ── Extract RAPIDS native libs from JAR (needed for both modes) ──
if [[ -f "$RAPIDS_JAR" ]] && [[ ! -d "$NATIVE_DIR" ]]; then
    mkdir -p "$NATIVE_DIR"
    unzip -q -j "$RAPIDS_JAR" "amd64/Linux/*.so" -d "$NATIVE_DIR" 2>/dev/null || true
    
    if [[ -f "$NATIVE_DIR/libnvcomp.so" ]]; then
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.5"
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.4"
    fi
    
    [[ -f "$NATIVE_DIR/libcudf.so" ]] && echo "Extracted RAPIDS native libs to $NATIVE_DIR" >&2
fi

# ── Handle --test flag (GVirtuS connectivity test, no Spark) ──
if [[ "$1" == "--test" ]]; then
    export LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so"
    export LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib:${NATIVE_DIR}"
    exec python3 gvirtus_test.py
fi

# ── GVirtuS mode: set env vars for Spark config (config.py reads these) ──
# Detect if first arg is "gvirtus" (it's the env arg to simple_matrix.py)


cd /app/src
exec python3 simple_matrix.py "$@"
