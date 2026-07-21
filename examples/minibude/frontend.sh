#!/bin/bash
#
# frontend.sh — compile miniBUDE against the GVirtuS frontend stubs and run it.
# Intended to run INSIDE the GVirtuS frontend container.
#
# miniBUDE is a compute-bound molecular-docking proxy: a few long kernels, little
# host<->device traffic. It is the "compute-bound extreme" of the suite —
# transport- and dispatch-insensitive (~native over GVirtuS), so it validates
# that the async dispatcher introduces NO regression on compute-bound work.
#
# Env overrides:
#   CUDA_ARCH  GPU arch (default sm_89 = Ada / L40S)
#   DECK       input deck (default data/bm1)
#   ITER       iterations (default 8)
#   GVIRTUS_LOGLEVEL  default 40000 (ERROR)
#   GVIRTUS_ASYNC_DISPATCH  0 or 1 (expected ~identical for miniBUDE)
#
set -e
cd "$(dirname "$0")/miniBUDE"

export GVIRTUS_HOME=${GVIRTUS_HOME:-/opt/GVirtuS}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-40000}
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

CUDA_ARCH=${CUDA_ARCH:-sm_89}; DECK=${DECK:-data/bm1}; ITER=${ITER:-8}

if [ ! -d build/generated ]; then
    echo "miniBUDE build/generated not found. Run ./setup.sh on the host first."; exit 1
fi

echo "Compiling miniBUDE against GVirtuS frontend stubs (arch ${CUDA_ARCH}) ..."
nvcc -x cu -DCUDA -DUSE_PPWI=1 -I src/cuda -I build/generated \
    -std=c++17 -extended-lambda -use_fast_math -restrict -arch=${CUDA_ARCH} \
    -O3 -DNDEBUG --cudart shared src/main.cpp \
    -L${GVIRTUS_HOME}/lib/frontend -L${GVIRTUS_HOME}/lib -lcudart -lcuda \
    -o build/cuda-bude-gvirtus

echo "Running miniBUDE over GVirtuS: --deck ${DECK} --iter ${ITER}"
./build/cuda-bude-gvirtus --deck "${DECK}" --iter "${ITER}"
