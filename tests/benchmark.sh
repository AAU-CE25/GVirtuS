#!/bin/bash
# benchmark.sh — GVirtuS UCX transport benchmark
# Usage: ./benchmark.sh [tcp|rdma|all] [runs_per_mode]
# tcp  → plain_tcp, ucx_tcp, hybrid  (needs TCP/mixed backend)
# rdma → rdma                         (needs RDMA-only backend)
# hybrid → hybrid                     (needs RDMA+TCP backend)

GROUP=${1:-all}
RUNS=${2:-10}
OUTPUT="benchmark_results_${GROUP}_$(date +%Y%m%d_%H%M%S).csv"

echo "mode,run,elapsed_ms,exit_code,ucx_tls,ucx_net_devices,timestamp" > "$OUTPUT"

declare -A TLS_MAP=(
    [plain_tcp]=""
    [ucx_tcp]="tcp,self"
    [rdma]="rc_mlx5,ud_mlx5,self"
    [hybrid]="rc_mlx5,tcp,self"
)
declare -A DEV_MAP=(
    [plain_tcp]=""
    [ucx_tcp]="ens1f1np1"
    [rdma]="mlx5_1:1"
    [hybrid]="mlx5_1:1,ens1f1np1"
)
declare -A GID_MAP=(
    [plain_tcp]=""
    [ucx_tcp]=""
    [rdma]="3"
    [hybrid]="3"
)
declare -A CFG_MAP=(
    [plain_tcp]="properties.json"
    [ucx_tcp]="properties_ucx.json"
    [rdma]="properties_ucx.json"
    [hybrid]="properties_ucx.json"
)
declare -A CM_MAP=(
    [plain_tcp]="tcp"
    [ucx_tcp]="tcp"
    [rdma]="rdmacm"
    [hybrid]="rdmacm"
)

case "$GROUP" in
    tcp)         MODES="plain_tcp ucx_tcp" ;;
    rdma)        MODES="rdma" ;;
    hybrid)      MODES="hybrid" ;;          
    rdma_hybrid) MODES="rdma hybrid" ;;
    all)         MODES="plain_tcp ucx_tcp rdma hybrid" ;;
    *)           echo "Unknown group: $GROUP. Use tcp, rdma, hybrid, rdma_hybrid, or all."; exit 1 ;;
esac

[ -z "$MODES" ] && { echo "ERROR: No modes selected for group '$GROUP'"; exit 1; }

for MODE in $MODES; do
    TLS="${TLS_MAP[$MODE]}"
    DEV="${DEV_MAP[$MODE]}"
    GID="${GID_MAP[$MODE]}"
    CFG="${CFG_MAP[$MODE]}"
    CM="${CM_MAP[$MODE]}"

    echo ""
    echo "========================================="
    echo "  MODE: $MODE  (TLS=${TLS:-none}, CFG=$CFG)"
    echo "========================================="

    for ((i=1; i<=RUNS; i++)); do
        echo -n "  Run $i/$RUNS ... "

        ENV_ARGS=( env
            GVIRTUS_UCX_DATAPATH=am
            GVIRTUS_CONFIG="/opt/GVirtuS/etc/${CFG}"
            UCX_SOCKADDR_TLS_PRIORITY="$CM"
            UCX_LOG_LEVEL=warn )
        [ -n "$TLS" ] && ENV_ARGS+=( UCX_TLS="$TLS" )
        [ -n "$DEV" ] && ENV_ARGS+=( UCX_NET_DEVICES="$DEV" )
        [ -n "$GID" ] && ENV_ARGS+=( UCX_IB_GID_INDEX="$GID" )

        RUN_LOG=$( "${ENV_ARGS[@]}" make run-simple-matrix-test 2>&1 )

        EXIT_CODE=$?
        ELAPSED_MS=$(echo "$RUN_LOG" | grep "BENCHMARK_RESULT_MS" | cut -d= -f2)
        ELAPSED_MS=${ELAPSED_MS:-"N/A"}
        TIMESTAMP=$(date --iso-8601=seconds)

        echo "${ELAPSED_MS}ms (exit=$EXIT_CODE)"
        echo "$MODE,$i,$ELAPSED_MS,$EXIT_CODE,${TLS:-none},${DEV:-none},$TIMESTAMP" >> "$OUTPUT"

        sleep 2
    done
done

echo ""
echo "========================================="
echo "  Done! Results saved to: $OUTPUT"
echo "========================================="