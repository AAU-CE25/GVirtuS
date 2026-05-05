#!/bin/bash
set -e

# ══════════════════════════════════════════════════════════════════════════════
# Unified entrypoint for Spark simple_matrix benchmark
# ══════════════════════════════════════════════════════════════════════════════
#
# Determines execution mode from the first argument passed to simple_matrix.py:
#   local   → Run with real CUDA libs (needs --runtime=nvidia)
#   docker  → Same as local, inside Docker
#   gvirtus → Use GVirtuS frontend stubs (no local GPU needed)
#
# The key difference is WHERE the JVM finds CUDA:
#   local/docker: /usr/local/cuda/lib64 (real CUDA from nvidia base image)
#   gvirtus:      /opt/GVirtuS/lib/frontend (GVirtuS stubs → remote GPU)
# ══════════════════════════════════════════════════════════════════════════════

GVIRTUS_HOME="${GVIRTUS_HOME:-/opt/GVirtuS}"
RAPIDS_JAR="/app/jars/rapids-4-spark_2.12-26.02.1.jar"

# ── GVirtuS mode: set env vars for Spark config (config.py reads these) ──
# Detect if first arg is "gvirtus" (it's the env arg to simple_matrix.py)


cd /app/src
exec python3 simple_matrix.py "$@"
