#!/bin/bash
set -e
export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

cd "${GVIRTUS_HOME}/examples"

nvcc percall_bench.cu -o percall_bench --cudart=shared \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lpthread

N=${THREADS:-1}
M=${ITERS:-4}
./percall_bench "$N" "$M"
