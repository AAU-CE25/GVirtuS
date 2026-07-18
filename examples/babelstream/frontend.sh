#!/bin/bash
#
# frontend.sh — compile BabelStream against the GVirtuS frontend stubs and run it.
# Intended to run INSIDE the GVirtuS frontend container (GVIRTUS_HOME installed).
#
# Env overrides:
#   CUDA_ARCH   GPU arch for nvcc (default sm_89 = Ada / L40S)
#   SIZE        number of array elements (default 33554432 = 33.5M doubles)
#   ITERS       measured iterations (default 100)
#   GVIRTUS_LOGLEVEL  default 40000 (ERROR) to keep benchmark output clean
#
set -e

export GVIRTUS_HOME=${GVIRTUS_HOME:-/opt/GVirtuS}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-40000}
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

CUDA_ARCH=${CUDA_ARCH:-sm_89}
SIZE=${SIZE:-33554432}
ITERS=${ITERS:-100}

cd "${GVIRTUS_HOME}/examples" || { echo "Failed to enter ${GVIRTUS_HOME}/examples"; exit 1; }

if [ ! -d BabelStream ]; then
    echo "BabelStream source not found under $(pwd)/BabelStream."
    echo "Run ./setup.sh on the host first (it clones + adapts BabelStream)."
    exit 1
fi

# Compile the CUDA model against the GVirtuS frontend stubs (NOT the real CUDA runtime).
# nvml_shim.cpp replaces -lnvidia-ml (cosmetic NVML only; see nvml_shim.cpp).
nvcc -O3 -std=c++17 --extended-lambda -DCUDA -DGRID_STRIDE -arch=${CUDA_ARCH} \
    -I BabelStream/src -I BabelStream/src/cuda \
    BabelStream/src/main.cpp BabelStream/src/cuda/CUDAStream.cu nvml_shim.cpp \
    -L${GVIRTUS_HOME}/lib/frontend -L${GVIRTUS_HOME}/lib \
    -lcudart -lcuda \
    -o cuda-stream

echo "Running BabelStream over GVirtuS: -s ${SIZE} -n ${ITERS} (arch ${CUDA_ARCH})"
./cuda-stream -s "${SIZE}" -n "${ITERS}"
