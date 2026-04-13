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
for arg in "$@"; do
    if [[ "$arg" == "gvirtus" ]]; then
        # ── Hide real CUDA libs so the linker can only find GVirtuS stubs ──
        # cuDF JNI's native libs (libcudf.so, libcudf_jni.so) are linked against
        # versioned sonames like libcudart.so.12 and libcuda.so.1. Without this,
        # the dynamic linker finds the REAL cuda libs from the base image instead
        # of GVirtuS stubs, and since there's no GPU driver mounted, CUDA fails
        # with cudaErrorInsufficientDriver.
        GVIRTUS_FRONTEND="${GVIRTUS_HOME}/lib/frontend"

        echo "GVirtuS mode: hiding real CUDA libs and creating stub symlinks..." >&2

        # 1. Move real CUDA runtime libs out of the way
        if [[ -d /usr/local/cuda/lib64 ]]; then
            mkdir -p /usr/local/cuda/lib64/real-backup
            for lib in libcudart.so* libcuda.so* libcublas.so* libcufft.so* \
                       libcurand.so* libcusolver.so* libcusparse.so* libcudnn.so* \
                       libnvrtc.so* libnvidia-ml.so*; do
                if ls /usr/local/cuda/lib64/$lib 1>/dev/null 2>&1; then
                    mv /usr/local/cuda/lib64/$lib /usr/local/cuda/lib64/real-backup/ 2>/dev/null || true
                fi
            done
            # Also check targets/x86_64-linux/lib (some CUDA images put libs here)
            if [[ -d /usr/local/cuda/targets/x86_64-linux/lib ]]; then
                mkdir -p /usr/local/cuda/targets/x86_64-linux/lib/real-backup
                for lib in libcudart.so* libcuda.so* libcublas.so* libcufft.so* \
                           libcurand.so* libcusolver.so* libcusparse.so* libcudnn.so* \
                           libnvrtc.so* libnvidia-ml.so*; do
                    if ls /usr/local/cuda/targets/x86_64-linux/lib/$lib 1>/dev/null 2>&1; then
                        mv /usr/local/cuda/targets/x86_64-linux/lib/$lib \
                           /usr/local/cuda/targets/x86_64-linux/lib/real-backup/ 2>/dev/null || true
                    fi
                done
            fi
        fi

        # 2. Create unversioned symlinks in GVirtuS frontend dir
        #    (some loaders look for libcudart.so without version)
        cd "$GVIRTUS_FRONTEND"
        for lib in libcudart libcublas libcufft libcurand libcusolver libcusparse libcudnn; do
            # Find the highest-versioned .so file for this lib
            latest=$(ls ${lib}.so.* 2>/dev/null | grep -v '\.so\.[0-9]*\.' | head -1)
            if [[ -n "$latest" ]] && [[ ! -e "${lib}.so" ]]; then
                ln -sf "$(basename "$latest")" "${lib}.so"
            fi
        done
        # libcuda.so → libcuda.so.1 (driver API stub)
        if [[ -e "libcuda.so.1" ]] && [[ ! -e "libcuda.so" ]]; then
            ln -sf libcuda.so.1 libcuda.so
        fi
        cd /app/src

        # 3. Rebuild ldconfig cache so it finds GVirtuS stubs first
        echo "${GVIRTUS_FRONTEND}" > /etc/ld.so.conf.d/gvirtus.conf
        ldconfig 2>/dev/null || true

        # 4. Set library paths — NO LD_PRELOAD needed!
        #    With real CUDA libs hidden, LD_LIBRARY_PATH + ldconfig is sufficient.
        #    LD_PRELOAD causes GVirtuS frontend constructor to fire in EVERY
        #    process (JVM threads, shell scripts, etc.), creating spurious TCP
        #    connections and interfering with JVM startup.
        unset LD_PRELOAD 2>/dev/null || true
        unset GVIRTUS_LD_PRELOAD 2>/dev/null || true
        export LD_LIBRARY_PATH="${GVIRTUS_FRONTEND}:${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH:-}"
        
        if [[ -d "$NATIVE_DIR" ]]; then
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NATIVE_DIR}"
        fi
        
        echo "GVirtuS mode: stubs at ${GVIRTUS_FRONTEND}" >&2
        echo "GVirtuS mode: real CUDA libs moved to backup dirs" >&2
        break
    fi
done

cd /app/src
exec python3 simple_matrix.py "$@"
