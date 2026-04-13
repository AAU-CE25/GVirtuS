#!/bin/bash
set -e

# ── GVirtuS Frontend Environment ──
# Environment variables are set in Dockerfile:
#   GVIRTUS_HOME, GVIRTUS_LOGLEVEL, LD_LIBRARY_PATH, LD_PRELOAD
#
# Usage:
#   docker run ... <image> --test                              # Test GVirtuS connectivity only
#   docker run ... <image> --mode cpu --overwrite yes          # CPU only (no GPU)
#   docker run ... <image> --mode rapids --overwrite yes       # CPU then RAPIDS

# ── Extract RAPIDS native libs from JAR ──
# RAPIDS JNI needs libnvcomp.so and libcudf.so in LD_LIBRARY_PATH
RAPIDS_JAR="/app/jars/rapids-4-spark_2.12-26.02.1.jar"
NATIVE_DIR="/tmp/rapids-native"

if [[ -f "$RAPIDS_JAR" ]] && [[ ! -d "$NATIVE_DIR" ]]; then
    mkdir -p "$NATIVE_DIR"
    unzip -q -j "$RAPIDS_JAR" "amd64/Linux/*.so" -d "$NATIVE_DIR" 2>/dev/null || true
    
    # Create versioned symlinks that libcudf.so expects
    if [[ -f "$NATIVE_DIR/libnvcomp.so" ]]; then
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.5"
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.4"
    fi
    
    if [[ -f "$NATIVE_DIR/libcudf.so" ]]; then
        echo "Extracted RAPIDS native libs to $NATIVE_DIR" >&2
    fi
fi

# Add RAPIDS native libs to LD_LIBRARY_PATH (AFTER GVirtuS stubs)
# GVirtuS must be first so libcudart.so/libcuda.so use GVirtuS stubs
if [[ -d "$NATIVE_DIR" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NATIVE_DIR}"
fi

cd /app/src

# NOTE: We do NOT set LD_PRELOAD here because it affects ALL processes
# including Spark's shell scripts (find-spark-home, spark-class, etc.)
# which breaks them. Instead, we pass LD_PRELOAD via PYSPARK_SUBMIT_ARGS
# or spark.executor.extraLibraryPath so only the JVM loads GVirtuS stubs.

# Export for use in Spark config (config.py will use these)
export GVIRTUS_LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so"

# Check for --test flag - for test, we DO need LD_PRELOAD directly
if [[ "$1" == "--test" ]]; then
    export LD_PRELOAD="${GVIRTUS_LD_PRELOAD}"
    exec python3 gvirtus_test.py
fi

# Pass all arguments to simple_matrix.py
# LD_PRELOAD will be set via Spark's extraLibraryPath config
exec python3 simple_matrix.py gvirtus "$@"

