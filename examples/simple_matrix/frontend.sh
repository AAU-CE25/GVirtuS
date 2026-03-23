#!/bin/bash
set -e  # Exit immediately if a command fails

# --- Set environment variables ---
export GVIRTUS_HOME=/opt/GVirtuS
export EXTRA_NVCCFLAGS='--cudart=shared'
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

# --- Navigate to the examples folder ---
cd "${GVIRTUS_HOME}/examples" || { echo "Failed to enter ${GVIRTUS_HOME}/examples"; exit 1; }

# --- Compile the CUDA program ---
nvcc simple_matrix.cu -o simple_matrix \
    -L${GVIRTUS_HOME}/lib/frontend \
    -L${GVIRTUS_HOME}/lib/ \
    -lcuda -lcudart -lcublas 

# --- Run the compiled program ---
./simple_matrix
