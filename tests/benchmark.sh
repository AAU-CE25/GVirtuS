#!/bin/bash
# benchmark.sh — GVirtuS UCX transport benchmark
# Usage: ./benchmark.sh [MODE_GROUP] [runs_per_mode]
#
# MODE_GROUP options:
#   tcp        — plain_tcp + ucx_tcp
#   rdma       — plain_rdma + ucx_rdma
#   all        — all four modes (default)
#   plain_tcp  — single mode
#   ucx_tcp    — single mode
#   plain_rdma — single mode
#   ucx_rdma   — single mode

GROUP=${1:-all}
RUNS=${2:-10}
OUTPUT="benchmark_results_${GROUP}_$(date +%Y%m%d_%H%M%S).csv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAFE_USER=$(whoami 2>/dev/null | cut -d'@' -f1 | tr -d '.' || echo "$UID")
CONTAINER_NAME="simple_matrix_test_container-${SAFE_USER}"

echo "mode,run,elapsed_ms,exit_code,ucx_tls,ucx_net_devices,matrix_n,malloc_ms,h2d_ms,cublas_create_ms,gemm_ms,d2h_ms,cleanup_ms,timestamp" > "$OUTPUT"

# ── Transport configuration ───────────────────────────────────────────────────
declare -A TLS_MAP=(
    [plain_tcp]=""
    [ucx_tcp]="tcp,self"
    [plain_rdma]="rc_mlx5,ud_mlx5,self"
    [ucx_rdma]="rc_mlx5,ud_mlx5,self"
)
declare -A DEV_MAP=(
    [plain_tcp]=""
    [ucx_tcp]="ens1f1np1"
    [plain_rdma]="mlx5_1:1"
    [ucx_rdma]="mlx5_1:1"
)
declare -A GID_MAP=(
    [plain_tcp]=""
    [ucx_tcp]=""
    [plain_rdma]="3"
    [ucx_rdma]="3"
)
declare -A CFG_MAP=(
    [plain_tcp]="properties.json"
    [ucx_tcp]="properties_ucx.json"
    [plain_rdma]="properties_plain_rdma.json"
    [ucx_rdma]="properties_ucx.json"
)
declare -A CM_MAP=(
    [plain_tcp]="tcp"
    [ucx_tcp]="tcp"
    [plain_rdma]="rdmacm"
    [ucx_rdma]="rdmacm"
)
declare -A DATAPATH_MAP=(
    [plain_tcp]="am"
    [ucx_tcp]="am"
    [plain_rdma]="rdma"
    [ucx_rdma]="am"
)

# ── Mode group selector ───────────────────────────────────────────────────────
case "$GROUP" in
    tcp)                                    MODES="plain_tcp ucx_tcp" ;;
    rdma)                                   MODES="plain_rdma ucx_rdma" ;;
    all)                                    MODES="plain_tcp ucx_tcp plain_rdma ucx_rdma" ;;
    plain_tcp|ucx_tcp|plain_rdma|ucx_rdma)  MODES="$GROUP" ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Usage: $0 [tcp|rdma|all|plain_tcp|ucx_tcp|plain_rdma|ucx_rdma] [runs]"
        exit 1
        ;;
esac

[ -z "$MODES" ] && { echo "ERROR: No modes selected for group '$GROUP'"; exit 1; }

# ── Helper: extract KEY=value from a log string ───────────────────────────────
parse_field() {
    local log="$1" key="$2"
    echo "$log" | grep "^${key}=" | cut -d= -f2
}

# ── Benchmark loop ────────────────────────────────────────────────────────────
for MODE in $MODES; do
    TLS="${TLS_MAP[$MODE]}"
    DEV="${DEV_MAP[$MODE]}"
    GID="${GID_MAP[$MODE]}"
    CFG="${CFG_MAP[$MODE]}"
    CM="${CM_MAP[$MODE]}"
    DATAPATH="${DATAPATH_MAP[$MODE]}"

    echo ""
    echo "========================================="
    echo "  MODE: $MODE  (TLS=${TLS:-none}, CFG=$CFG)"
    echo "========================================="

    for ((i=1; i<=RUNS; i++)); do
        echo -n "  Run $i/$RUNS ... "

        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        sleep 1

        MAKE_ARGS=(
            GVIRTUS_UCX_DATAPATH="$DATAPATH"
            GVIRTUS_CONFIG_FILE="$CFG"
            UCX_SOCKADDR_TLS_PRIORITY="$CM"
            UCX_LOG_LEVEL=warn
            SIMPLE_MATRIX_GPU_FLAGS="--gpus all"
        )
        [ -n "$TLS" ] && MAKE_ARGS+=( UCX_TLS="$TLS" )
        [ -n "$DEV" ] && MAKE_ARGS+=( UCX_NET_DEVICES="$DEV" )
        [ -n "$GID" ] && MAKE_ARGS+=( UCX_IB_GID_INDEX="$GID" )

        RUN_LOG=$( make -C "$REPO_ROOT" run-simple-matrix-test "${MAKE_ARGS[@]}" 2>&1 )
        EXIT_CODE=$?

        # Total wall time
        ELAPSED_MS=$(parse_field "$RUN_LOG" "BENCHMARK_RESULT_MS")
        ELAPSED_MS=${ELAPSED_MS:-"N/A"}

        # Matrix size confirmed from binary stdout
        ACTUAL_N=$(parse_field "$RUN_LOG" "BENCHMARK_MATRIX_N")
        ACTUAL_N=${ACTUAL_N:-"unknown"}

        # Per-stage timings (empty string if binary is old version without stages)
        T_MALLOC=$(parse_field  "$RUN_LOG" "STAGE_MALLOC_MS")
        T_H2D=$(parse_field     "$RUN_LOG" "STAGE_H2D_MS")
        T_CREATE=$(parse_field  "$RUN_LOG" "STAGE_CUBLAS_CREATE_MS")
        T_GEMM=$(parse_field    "$RUN_LOG" "STAGE_GEMM_MS")
        T_D2H=$(parse_field     "$RUN_LOG" "STAGE_D2H_MS")
        T_CLEANUP=$(parse_field "$RUN_LOG" "STAGE_CLEANUP_MS")

        TIMESTAMP=$(date --iso-8601=seconds)

        echo "${ELAPSED_MS}ms (exit=$EXIT_CODE) [N=$ACTUAL_N] malloc=${T_MALLOC}ms h2d=${T_H2D}ms create=${T_CREATE}ms gemm=${T_GEMM}ms d2h=${T_D2H}ms"

        echo "$MODE,$i,$ELAPSED_MS,$EXIT_CODE,${TLS:-none},${DEV:-none},$ACTUAL_N,${T_MALLOC},${T_H2D},${T_CREATE},${T_GEMM},${T_D2H},${T_CLEANUP},$TIMESTAMP" >> "$OUTPUT"

        sleep 2
    done
done

echo ""
echo "========================================="
echo "  Done! Results saved to: $OUTPUT"
echo "========================================="