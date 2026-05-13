#!/bin/bash
# benchmark_focused.sh — targeted re-run at sizes 4096, 8192, 16384
# Usage: ./benchmark_focused.sh [MODE_GROUP] [runs]
#   MODE_GROUP: plain_tcp | ucx_tcp | ucx_rdma | tcp | rdma | all (default)

set -euo pipefail

GROUP=${1:-all}      
RUNS=${2:-5}        
WARMUP_RUNS=3
SIZES=(4096 8192 16384)
case "$GROUP" in
    tcp)                                    MODES="plain_tcp ucx_tcp" ;;
    rdma)                                   MODES="ucx_rdma" ;;
    all)                                    MODES="plain_tcp ucx_tcp ucx_rdma" ;;
    plain_tcp|ucx_tcp|ucx_rdma)             MODES="$GROUP" ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Usage: $0 [tcp|rdma|all|plain_tcp|ucx_tcp|ucx_rdma] [runs]"
        exit 1
        ;;
esac

OUTPUT="examples/simple_matrix/benchmark_results/focused_${GROUP}_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p "$(dirname "$OUTPUT")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAFE_USER=$(whoami 2>/dev/null | cut -d'@' -f1 | tr -d '.' || echo "$UID")
CONTAINER_NAME="simple_matrix_test_container-${SAFE_USER}"

# ── CSV header ────────────────────────────────────────────────────
echo "mode,matrix_n,run,warmup,elapsed_ms,exit_code,\
ucx_tls,ucx_net_devices,\
malloc_ms,cudamalloc_ms,h2d_ms,cublas_create_ms,gemm_ms,d2h_ms,cleanup_ms,\
result_check,timestamp" > "$OUTPUT"

# ── Transport config (same as benchmark.sh) ───────────────────────
declare -A TLS_MAP=([plain_tcp]="" [ucx_tcp]="tcp,self" [ucx_rdma]="rc_mlx5,ud_mlx5,self")
declare -A DEV_MAP=([plain_tcp]="" [ucx_tcp]="ens1f1np1" [ucx_rdma]="mlx5_1:1")
declare -A GID_MAP=([plain_tcp]="" [ucx_tcp]="" [ucx_rdma]="3")
declare -A CFG_MAP=([plain_tcp]="properties.json" [ucx_tcp]="properties_ucx.json" [ucx_rdma]="properties_ucx.json")
declare -A CM_MAP=([plain_tcp]="tcp" [ucx_tcp]="tcp" [ucx_rdma]="rdmacm")
declare -A DATAPATH_MAP=([plain_tcp]="am" [ucx_tcp]="am" [ucx_rdma]="am")

parse_field() { echo "$1" | grep "^${2}=" | tail -1 | cut -d= -f2; }

for MODE in $MODES; do
    TLS="${TLS_MAP[$MODE]}"
    DEV="${DEV_MAP[$MODE]}"
    GID="${GID_MAP[$MODE]}"
    CFG="${CFG_MAP[$MODE]}"
    CM="${CM_MAP[$MODE]}"
    DATAPATH="${DATAPATH_MAP[$MODE]}"

    MAKE_ARGS_BASE=(
        GVIRTUS_UCX_DATAPATH="$DATAPATH"
        GVIRTUS_CONFIG_FILE="$CFG"
        UCX_SOCKADDR_TLS_PRIORITY="$CM"
        UCX_LOG_LEVEL=warn
        SIMPLE_MATRIX_GPU_FLAGS="--gpus all"
        MATRIX_N="4096"          # warmup always at smallest target size
    )
    [ -n "$TLS" ] && MAKE_ARGS_BASE+=( UCX_TLS="$TLS" )
    [ -n "$DEV" ] && MAKE_ARGS_BASE+=( UCX_NET_DEVICES="$DEV" )
    [ -n "$GID" ] && MAKE_ARGS_BASE+=( UCX_IB_GID_INDEX="$GID" )

    echo ""
    echo "===== MODE: $MODE ====="
    echo "  [WARMUP] ${WARMUP_RUNS}x N=4096 — discarded"
    for ((w=1; w<=WARMUP_RUNS; w++)); do
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        sleep 1
        RUN_LOG=$(make -C "$REPO_ROOT" run-simple-matrix-test "${MAKE_ARGS_BASE[@]}" 2>&1)
        ELAPSED_MS=$(parse_field "$RUN_LOG" "BENCHMARK_RESULT_MS")
        TIMESTAMP=$(date --iso-8601=seconds)
        printf '%s\n' \
          "${MODE},4096,${w},true,${ELAPSED_MS:-N/A},0,\"${TLS:-none}\",\"${DEV:-none}\",,,,,,,,,${TIMESTAMP}" \
          >> "$OUTPUT"
        echo "    warmup $w/${WARMUP_RUNS}: ${ELAPSED_MS:-N/A}ms"
        sleep 1
    done

    for N in "${SIZES[@]}"; do
        echo ""
        echo "  ── N=${N} ──"
        for ((i=1; i<=RUNS; i++)); do
            echo -n "    run $i/$RUNS ... "
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            sleep 1

            MAKE_ARGS=(
                GVIRTUS_UCX_DATAPATH="$DATAPATH"
                GVIRTUS_CONFIG_FILE="$CFG"
                UCX_SOCKADDR_TLS_PRIORITY="$CM"
                UCX_LOG_LEVEL=warn
                SIMPLE_MATRIX_GPU_FLAGS="--gpus all"
                MATRIX_N="$N"
            )
            [ -n "$TLS" ] && MAKE_ARGS+=( UCX_TLS="$TLS" )
            [ -n "$DEV" ] && MAKE_ARGS+=( UCX_NET_DEVICES="$DEV" )
            [ -n "$GID" ] && MAKE_ARGS+=( UCX_IB_GID_INDEX="$GID" )

            RUN_LOG=$(make -C "$REPO_ROOT" run-simple-matrix-test "${MAKE_ARGS[@]}" 2>&1)
            EXIT_CODE=$?

            ELAPSED_MS=$(parse_field "$RUN_LOG" "BENCHMARK_RESULT_MS")
            ACTUAL_N=$(parse_field   "$RUN_LOG" "BENCHMARK_MATRIX_N")
            T_MALLOC=$(parse_field   "$RUN_LOG" "STAGE_MALLOC_MS")
            T_CUDAMALLOC=$(parse_field "$RUN_LOG" "STAGE_CUDAMALLOC_MS")
            T_H2D=$(parse_field      "$RUN_LOG" "STAGE_H2D_MS")
            T_CREATE=$(parse_field   "$RUN_LOG" "STAGE_CUBLAS_CREATE_MS")
            T_GEMM=$(parse_field     "$RUN_LOG" "STAGE_GEMM_MS")
            T_D2H=$(parse_field      "$RUN_LOG" "STAGE_D2H_MS")
            T_CLEANUP=$(parse_field  "$RUN_LOG" "STAGE_CLEANUP_MS")
            T_CHECK=$(parse_field    "$RUN_LOG" "RESULT_CHECK")
            TIMESTAMP=$(date --iso-8601=seconds)

            echo "${ELAPSED_MS:-N/A}ms (exit=${EXIT_CODE})"

            printf '%s\n' \
              "${MODE},${ACTUAL_N:-$N},${i},false,${ELAPSED_MS:-N/A},${EXIT_CODE},\"${TLS:-none}\",\"${DEV:-none}\",${T_MALLOC},${T_CUDAMALLOC},${T_H2D},${T_CREATE},${T_GEMM},${T_D2H},${T_CLEANUP},\"${T_CHECK}\",${TIMESTAMP}" \
              >> "$OUTPUT"

            sleep 0.5
        done
        sleep 1
    done
done

echo ""
echo "Done → $OUTPUT"