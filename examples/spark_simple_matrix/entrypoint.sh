#!/bin/bash
set -e

GVIRTUS_HOME="${GVIRTUS_HOME:-/opt/GVirtuS}"
RAPIDS_JAR="/app/jars/rapids-4-spark_2.13-26.04.0.jar"

cd /app/src
exec python3 simple_matrix.py "$@"
