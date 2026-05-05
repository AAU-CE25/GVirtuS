#!/bin/bash
set -e

GVIRTUS_HOME="${GVIRTUS_HOME:-/opt/GVirtuS}"
RAPIDS_JAR="/app/jars/rapids-4-spark_2.12-26.02.1.jar"

cd /app/src
exec python3 simple_matrix.py "$@"
