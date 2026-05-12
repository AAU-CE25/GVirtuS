#!/bin/bash
set -e

# --- Set environment variables ---
export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH:-}

# If requested RDMA resources are not available in the container, force UCX to TCP mode.
need_tcp_fallback=0

if [[ "${UCX_TLS:-}" == *"mlx5"* ]] || [[ "${UCX_NET_DEVICES:-}" == *"mlx5"* ]] || [[ "${UCX_SOCKADDR_TLS_PRIORITY:-}" == *"rdmacm"* ]]; then
    if [[ ! -e /dev/infiniband/rdma_cm ]]; then
        need_tcp_fallback=1
    fi

    if [[ -n "${UCX_NET_DEVICES:-}" ]]; then
        ucx_ib_dev="${UCX_NET_DEVICES%%:*}"
        if [[ -n "${ucx_ib_dev}" ]] && [[ ! -d "/sys/class/infiniband/${ucx_ib_dev}" ]]; then
            need_tcp_fallback=1
        fi
    fi

    if [[ ! -d /sys/class/infiniband ]] || [[ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
        need_tcp_fallback=1
    fi
fi

if [[ "${need_tcp_fallback}" -eq 1 ]]; then
    echo "WARNING: Requested UCX RDMA settings are unavailable in container. Falling back to UCX tcp,self."
    export UCX_TLS="tcp,self"
    unset UCX_NET_DEVICES
    export UCX_SOCKADDR_TLS_PRIORITY="tcp"
fi

# --- Navigate to examples folder ---
cd "${GVIRTUS_HOME}/examples" || { echo "Failed to enter ${GVIRTUS_HOME}/examples"; exit 1; }

# --- Compile the benchmark ---
nvcc benchmark.cu -o benchmark \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lcublas

# --- Run the benchmark ---
echo "=== GVirtuS Matrix Multiplication Benchmark ==="
./benchmark
