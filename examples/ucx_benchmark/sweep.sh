#!/usr/bin/env bash
# Sweep data_copy_bench across multiple payload sizes for a single transport.
# Run the matching server first on the peer node, then run this on the client.
#
# Usage:
#   ./sweep.sh <tcp|ucx-tcp|ucx-rdma> <server_ip> <port> [out_csv] [runs_per_size]
#
# Example (on client, after starting server on peer):
#   ./sweep.sh ucx-rdma  24.24.24.2 7777 results_ucx_rdma.csv 20
#   ./sweep.sh ucx-tcp   24.24.24.2 7777 results_ucx_tcp.csv  20
#   ./sweep.sh tcp       24.24.24.2 7777 results_tcp.csv      20
#
# To produce one combined CSV across all three transports, point them at the
# same out_csv file (header is written only once).

set -euo pipefail

LABEL="${1:-}"
SERVER="${2:-}"
PORT="${3:-}"
OUT="${4:-results_$(date +%Y%m%d_%H%M%S).csv}"
RUNS="${5:-20}"

if [[ -z "$LABEL" || -z "$SERVER" || -z "$PORT" ]]; then
    sed -n '2,15p' "$0"
    exit 1
fi

cd "$(dirname "$0")"

case "$LABEL" in
    tcp)
        BIN=./build/data_copy_bench
        TRANSPORT=tcp
        ENV_PREFIX=""
        ;;
    ucx-tcp)
        BIN=./build/data_copy_bench_ucx
        TRANSPORT=ucx
        ENV_PREFIX="UCX_TLS=tcp UCX_IB_GID_INDEX=1"
        ;;
    ucx-rdma)
        BIN=./build/data_copy_bench_ucx
        TRANSPORT=ucx
        ENV_PREFIX="UCX_TLS=rc_verbs,tcp UCX_IB_GID_INDEX=1"
        ;;
    *)
        echo "Unknown label: $LABEL (use tcp | ucx-tcp | ucx-rdma)" >&2
        exit 1
        ;;
esac

if [[ ! -x "$BIN" ]]; then
    echo "Missing binary: $BIN \u2014 build first with: make tcp / make ucx" >&2
    exit 1
fi

# Payload sizes: 4 KiB up to 64 MiB (powers of 4)
SIZES=(
    4096        # 4 KiB
    16384       # 16 KiB
    65536       # 64 KiB
    262144      # 256 KiB
    1048576     # 1 MiB
    4194304     # 4 MiB
    16777216    # 16 MiB
    67108864    # 64 MiB
)

# Write combined CSV header if file does not exist
if [[ ! -s "$OUT" ]]; then
    echo "transport,size_bytes,run,n_bytes,send_us,recv_us,roundtrip_us" > "$OUT"
fi

echo "[sweep] label=$LABEL  server=$SERVER:$PORT  runs/size=$RUNS  out=$OUT"

for SIZE in "${SIZES[@]}"; do
    echo "[sweep]  size=$SIZE bytes ..."
    # Run client; it prints its own "run,n_bytes,..." CSV header on stdout.
    # Strip that header line, prepend transport+size columns, append to OUT.
    # stderr (server-wireup messages) is left visible.
    OUTPUT=$(eval $ENV_PREFIX "$BIN" client "$TRANSPORT" "$SERVER" "$PORT" "$SIZE" "$RUNS")
    echo "$OUTPUT" \
        | tail -n +2 \
        | awk -v t="$LABEL" -v s="$SIZE" 'BEGIN{OFS=","} {print t,s,$0}' \
        >> "$OUT"
done

echo "[sweep] done -> $OUT"
echo
echo "Quick summary (median roundtrip per size):"
awk -F, -v lbl="$LABEL" '
    NR>1 && $1==lbl { v[$2] = v[$2] " " $7 }
    END {
        for (s in v) {
            n = split(v[s], arr, " ")
            # arr[1] is empty due to leading space
            cnt = 0
            for (i=1; i<=n; i++) if (arr[i] != "") nums[++cnt] = arr[i]+0
            # sort
            for (i=1; i<=cnt; i++) for (j=i+1; j<=cnt; j++) if (nums[i]>nums[j]) { t=nums[i]; nums[i]=nums[j]; nums[j]=t }
            med = (cnt % 2) ? nums[(cnt+1)/2] : (nums[cnt/2]+nums[cnt/2+1])/2
            printf "  %10d B  median %8.1f us\n", s, med
            delete nums
        }
    }
' "$OUT" | sort -n -k2
