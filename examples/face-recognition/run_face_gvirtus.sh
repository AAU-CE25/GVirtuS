#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GVIRTUS_HOME=/home/student.aau.dk/ll33pq/GVirtuS
export LZ4_HOME=/home/student.aau.dk/ll33pq/lz4-install
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"

exec env \
  GVIRTUS_LOGLEVEL="${GVIRTUS_LOGLEVEL:-10000}" \
  STEADY_WARMUPS="${STEADY_WARMUPS:-0}" \
  STEADY_ITERS="${STEADY_ITERS:-1}" \
  STEADY_IMAGES="${STEADY_IMAGES:-1}" \
  python3 cnn.py
