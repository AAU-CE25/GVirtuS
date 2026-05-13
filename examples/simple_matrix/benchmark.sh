#!/bin/bash
# benchmark.sh — GVirtuS UCX transport benchmark
# Usage: ./benchmark.sh [MODE_GROUP] [runs_per_mode]
#
# MODE_GROUP options:
#   tcp        — plain_tcp + ucx_tcp
#   rdma       — plain_rdma + ucx_rdma
#   all        — all four modes (default)
#   plain_tcp | ucx_tcp | plain_rdma | ucx_rdma  — single mode
#
# Matrix sizes ramp geometrically from 8 → 16384.
# Each size is run RUNS times; the first WARMUP_RUNS at the smallest
# size are discarded (cache/QP warmup, Criterion.rs methodology).

GROUP=${1:-all}
RUNS=${2:-5}
WARMUP_RUNS=3
OUTPUT="examples/simple_matrix/benchmark_results/benchmark_results_${GROUP}_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p "$(dirname "$OUTPUT")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAFE_USER=$(whoami 2>/dev/null | cut -d'@' -f1 | tr -d '.' || echo "$UID")
CONTAINER_NAME="simple_matrix_test_container-${SAFE_USER}"

# Geometric ramp: 8 → 16384
SIZES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384)

# ── CSV header ────────────────────────────────────────────────────────────────
echo "mode,matrix_n,run,warmup,elapsed_ms,exit_code,\
ucx_tls,ucx_net_devices,\
malloc_ms,cudamalloc_ms,h2d_ms,cublas_create_ms,gemm_ms,d2h_ms,cleanup_ms,\
result_check,timestamp" > "$OUTPUT"

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
parse_field() { echo "$1" | grep "^${2}=" | tail -1 | cut -d= -f2; }

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

    # ── Phase 1: Warmup ──────────────────────────────────────────
    # Run WARMUP_RUNS iterations at the smallest matrix size and
    # discard results. This lets the RDMA QP connect, CPU/GPU caches
    # warm up, and cuBLAS initialise its internal state — preventing
    # first-connection and JIT-compilation penalties from bleeding
    # into the measured data. (Criterion.rs methodology)
    WARMUP_N=${SIZES[0]}
    echo "  [WARMUP] ${WARMUP_RUNS}x N=${WARMUP_N} — results discarded"
    MAKE_ARGS_BASE=(
        GVIRTUS_UCX_DATAPATH="$DATAPATH"
        GVIRTUS_CONFIG_FILE="$CFG"
        UCX_SOCKADDR_TLS_PRIORITY="$CM"
        UCX_LOG_LEVEL=warn
        SIMPLE_MATRIX_GPU_FLAGS="--gpus all"
        MATRIX_N="$WARMUP_N"
    )
    [ -n "$TLS" ] && MAKE_ARGS_BASE+=( UCX_TLS="$TLS" )
    [ -n "$DEV" ] && MAKE_ARGS_BASE+=( UCX_NET_DEVICES="$DEV" )
    [ -n "$GID" ] && MAKE_ARGS_BASE+=( UCX_IB_GID_INDEX="$GID" )

    for ((w=1; w<=WARMUP_RUNS; w++)); do
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        sleep 1
        RUN_LOG=$(make -C "$REPO_ROOT" run-simple-matrix-test \
            "${MAKE_ARGS_BASE[@]}" 2>&1)
        ELAPSED_MS=$(parse_field "$RUN_LOG" "BENCHMARK_RESULT_MS")
        echo "    warmup $w/${WARMUP_RUNS}: ${ELAPSED_MS:-N/A}ms (discarded)"
        # Log warmup runs flagged so they can be filtered out in analysis
        TIMESTAMP=$(date --iso-8601=seconds)
        printf '%s\n' \
          "${MODE},4096,${w},true,${ELAPSED_MS:-N/A},0,\"${TLS:-none}\",\"${DEV:-none}\",,,,,,,,,${TIMESTAMP}" \
          >> "$OUTPUT"
        sleep 1
    done

    # ── Phase 2: Geometric ramp-up ───────────────────────────────
    # Sizes double each step (plus a 10000 cap), exposing O(n²)
    # memory scaling, H2D/D2H bandwidth saturation, and GEMM
    # compute-bound vs. memory-bound crossover points.
    for N in "${SIZES[@]}"; do
        echo ""
        echo "  ── N=${N} ──"

        TIMES=()

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

            RUN_LOG=$(make -C "$REPO_ROOT" run-simple-matrix-test \
                "${MAKE_ARGS[@]}" 2>&1)
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

            ELAPSED_MS=${ELAPSED_MS:-N/A}
            ACTUAL_N=${ACTUAL_N:-$N}

            echo "${ELAPSED_MS}ms (exit=${EXIT_CODE}) | \
malloc=${T_MALLOC}ms cudamalloc=${T_CUDAMALLOC}ms \
h2d=${T_H2D}ms create=${T_CREATE}ms \
gemm=${T_GEMM}ms d2h=${T_D2H}ms"

            printf '%s\n' \
              "${MODE},${ACTUAL_N:-$N},${i},false,${ELAPSED_MS:-N/A},${EXIT_CODE},\"${TLS:-none}\",\"${DEV:-none}\",${T_MALLOC},${T_CUDAMALLOC},${T_H2D},${T_CREATE},${T_GEMM},${T_D2H},${T_CLEANUP},\"${T_CHECK}\",${TIMESTAMP}" \
              >> "$OUTPUT"

            [[ "$ELAPSED_MS" =~ ^[0-9]+(\.[0-9]+)?$ ]] && TIMES+=("$ELAPSED_MS")

            # Brief pause: lets RDMA completion queues drain between
            # runs at the same size
            sleep 0.5
        done

        # ── Per-size summary (median + range) ────────────────────
        if [ ${#TIMES[@]} -gt 0 ]; then
            IFS=$'\n' SORTED=($(sort -n <<<"${TIMES[*]}")); unset IFS
            COUNT=${#SORTED[@]}
            MID=$(( COUNT / 2 ))
            MEDIAN=${SORTED[$MID]}
            MIN=${SORTED[0]}
            MAX=${SORTED[$((COUNT-1))]}
            echo "    → median=${MEDIAN}ms  min=${MIN}ms  max=${MAX}ms  (n=${COUNT})"
        fi

        # Pause between sizes: allows RDMA buffers to drain and
        # prevents thermal state from previous workload carrying over
        sleep 1
    done
done

echo ""
echo "========================================="
echo "  Done! Results: $OUTPUT"
echo "========================================="