#!/bin/bash
set -e
export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
cd /opt/GVirtuS/examples
nvcc simple_matrix.cu -o simple_matrix --cudart=shared \
    -L${GVIRTUS_HOME}/lib/frontend -L${GVIRTUS_HOME}/lib \
    -lcuda -lcudart -lcublas 2>&1
MATRIX_SIZE="${MATRIX_SIZE:-16384}" ITERATIONS="${ITERATIONS:-3}" WARMUP="${WARMUP:-2}" \
    ./simple_matrix
