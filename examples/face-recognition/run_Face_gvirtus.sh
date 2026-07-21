#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-ucx}"

export GVIRTUS_HOME="${GVIRTUS_HOME:-/home/student.aau.dk/ll33pq/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-/home/student.aau.dk/ll33pq/lz4-install}"

case "$MODE" in
  tcp)
    unset GVIRTUS_CONFIG
    ;;
  rdma)
    export GVIRTUS_CONFIG="$GVIRTUS_HOME/etc/properties_plain_rdma.json"
    ;;
  ucx)
    export GVIRTUS_CONFIG="$GVIRTUS_HOME/etc/properties_ucx.json"
    ;;
  *)
    echo "Usage: $0 [tcp|rdma|ucx]"
    exit 1
    ;;
esac

export GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-10000}"
export STEADY_WARMUPS="${STEADY_WARMUPS:-0}"
export STEADY_ITERS="${STEADY_ITERS:-1}"
export STEADY_IMAGES="${STEADY_IMAGES:-1}"

export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_HOME/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "GVIRTUS_HOME=$GVIRTUS_HOME"
echo "GVIRTUS_CONFIG=${GVIRTUS_CONFIG:-<unset>}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

exec ./run.sh python3 cnn.py
