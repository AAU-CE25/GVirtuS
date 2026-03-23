#!/usr/bin/env bash
# docker/conf/spark-env.sh

export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/etc/gvirtus/gvirtus.properties

# Intercept CUDA calls → route to GVirtuS → 24.24.24.1:2222
export LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so"
export LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}"