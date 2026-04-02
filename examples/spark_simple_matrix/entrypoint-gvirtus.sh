#!/bin/bash
set -e

# ── GVirtuS Frontend Environment ──
# This script sets up the environment so CUDA calls are intercepted
# by GVirtuS frontend and forwarded to a remote backend.
#
# Usage:
#   docker run ... <image> --mode cpu --overwrite yes        # CPU only (no GPU)
#   docker run ... <image> --mode rapids --overwrite yes      # CPU then RAPIDS

export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}

# Prepend GVirtuS frontend libraries to LD_LIBRARY_PATH
# This makes the linker use GVirtuS stubs instead of real CUDA libs
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

echo "=== GVirtuS Frontend Environment ==="
echo "GVIRTUS_HOME:     ${GVIRTUS_HOME}"
echo "GVIRTUS_LOGLEVEL: ${GVIRTUS_LOGLEVEL}"
echo "LD_LIBRARY_PATH:  ${LD_LIBRARY_PATH}"
echo "Properties file:  ${GVIRTUS_HOME}/etc/properties.json"
echo "===================================="

cd /app/src

# Pass all arguments to simple_matrix.py
exec python3 simple_matrix.py gvirtus "$@"

