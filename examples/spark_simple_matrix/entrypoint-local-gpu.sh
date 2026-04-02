#!/bin/bash
set -e

# Entrypoint for Docker local GPU mode
# Runs Spark simple_matrix benchmark with local GPU access
#
# Usage:
#   docker run ... <image> --mode cpu         # CPU only
#   docker run ... <image> --mode rapids      # RAPIDS GPU only

cd /app/src

exec python3 simple_matrix.py docker "$@"

