#!/bin/bash
set -euo pipefail

GVIRTUS_HOME=${GVIRTUS_HOME:-/opt/GVirtuS}
EXAMPLES_DIR="${GVIRTUS_HOME}/examples"

MATRIX_SIZES=${MATRIX_SIZES:-"128 256 512 1024 1536 2048"}
ITERATIONS=${ITERATIONS:-10}
WARMUP=${WARMUP:-1}
MODES=${MODES:-"tcp ucx-tcp ucx-rdma ucx-mixed"}
PROMPT=${PROMPT:-1}

UCX_TCP_DEV=${UCX_TCP_DEV:-ens1f1np1}
UCX_RDMA_DEV=${UCX_RDMA_DEV:-mlx5_1:1}
UCX_MIXED_DEVS=${UCX_MIXED_DEVS:-${UCX_RDMA_DEV},${UCX_TCP_DEV}}
UCX_GID_INDEX=${UCX_GID_INDEX:-3}

export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-30000}
export LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH:-}"

mkdir -p "${EXAMPLES_DIR}"
cd "${EXAMPLES_DIR}"

nvcc simple_matrix.cu -o simple_matrix -g --cudart=shared \
  -L"${GVIRTUS_HOME}/lib/frontend" \
  -L"${GVIRTUS_HOME}/lib" \
  -lcuda -lcudart -lcublas

run_sizes() {
    local mode="$1"
    local out_dir="${OUT_DIR:-${PWD}/benchmark_results}"
    local out_file="${out_dir}/simple_matrix_${mode}.csv"

    mkdir -p "${out_dir}"
    echo "mode,size,iters,avg_ms,gflops" > "${out_file}"

    for n in ${MATRIX_SIZES}; do
        local line
        line=$(MATRIX_SIZE="${n}" ITERATIONS="${ITERATIONS}" WARMUP="${WARMUP}" ./simple_matrix | \
            awk -F, '/^CSV,/{print $0}')
        if [[ -z "${line}" ]]; then
            echo "ERROR: No CSV line produced for size ${n}" >&2
            exit 1
        fi
        echo "${mode},${line#CSV,}" >> "${out_file}"
    done

    echo "Wrote ${out_file}"
}

prompt_backend() {
    local mode="$1"
    if [[ "${PROMPT}" != "1" ]]; then
        return 0
    fi

    echo "Start backend for mode: ${mode} and press Enter to continue."
    read -r
}

set_mode_env() {
    local mode="$1"

    case "${mode}" in
        tcp)
            export GVIRTUS_CONFIG="${GVIRTUS_CONFIG_TCP:-/opt/GVirtuS/etc/properties.json}"
            unset UCX_TLS UCX_NET_DEVICES UCX_LOG_LEVEL UCX_SOCKADDR_TLS_PRIORITY UCX_IB_GID_INDEX
            ;;
        ucx-tcp)
            export GVIRTUS_CONFIG="${GVIRTUS_CONFIG_UCX:-/opt/GVirtuS/etc/properties_ucx.json}"
            export UCX_TLS=${UCX_TLS_TCP:-tcp,self}
            export UCX_NET_DEVICES=${UCX_NET_DEVICES_TCP:-${UCX_TCP_DEV}}
            export UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY_TCP:-tcp}
            export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
            unset UCX_IB_GID_INDEX
            ;;
        ucx-rdma)
            export GVIRTUS_CONFIG="${GVIRTUS_CONFIG_UCX:-/opt/GVirtuS/etc/properties_ucx.json}"
            export UCX_TLS=${UCX_TLS_RDMA:-rc_mlx5,ud_mlx5,self}
            export UCX_NET_DEVICES=${UCX_NET_DEVICES_RDMA:-${UCX_RDMA_DEV}}
            export UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY_RDMA:-rdmacm}
            export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
            export UCX_IB_GID_INDEX=${UCX_GID_INDEX}
            ;;
        ucx-mixed)
            export GVIRTUS_CONFIG="${GVIRTUS_CONFIG_UCX:-/opt/GVirtuS/etc/properties_ucx.json}"
            export UCX_TLS=${UCX_TLS_MIXED:-rc_mlx5,ud_mlx5,tcp,self}
            export UCX_NET_DEVICES=${UCX_NET_DEVICES_MIXED:-${UCX_MIXED_DEVS}}
            export UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY_MIXED:-tcp}
            export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
            export UCX_IB_GID_INDEX=${UCX_GID_INDEX}
            ;;
        *)
            echo "Unknown mode: ${mode}" >&2
            exit 1
            ;;
    esac
}

for mode in ${MODES}; do
    set_mode_env "${mode}"
    prompt_backend "${mode}"
    run_sizes "${mode}"
done
