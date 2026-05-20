#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"

export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/gvirtus-install}"
export GVIRTUS_CONFIG="${GVIRTUS_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
export OPENCV_PREFIX="${OPENCV_PREFIX:-$HOME/opencv-local}"

export PKG_CONFIG_PATH="$OPENCV_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

pkg-config --modversion opencv4

nvcc main.cu -o main -g \
  -L "$GVIRTUS_HOME/lib/frontend" \
  -L "$GVIRTUS_HOME/lib" \
  $(pkg-config --cflags --libs opencv4) \
  -lcublas -lcudnn

./main
