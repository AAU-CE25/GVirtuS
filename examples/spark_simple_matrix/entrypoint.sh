#!/bin/bash
set -e

GVIRTUS_HOME="${GVIRTUS_HOME:-/opt/GVirtuS}"
RAPIDS_JAR="/app/jars/rapids-4-spark_2.13-26.04.0.jar"
FRONTEND_LIB="${GVIRTUS_HOME}/lib/frontend"

# GVirtuS stubs are 12.2 — compatible with the cuda12 RAPIDS JAR
export LD_LIBRARY_PATH="${FRONTEND_LIB}:${GVIRTUS_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd /app/src
exec python3 simple_matrix.py "$@"
