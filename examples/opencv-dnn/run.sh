#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"

export GVIRTUS_HOME="${GVIRTUS_HOME:-$REPO_ROOT}"
export GVIRTUS_CONFIG="${GVIRTUS_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"

prepend_ld_library_path() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0

    case ":${LD_LIBRARY_PATH:-}:" in
        *":$dir:"*) ;;
        *)
            if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
                export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
            else
                export LD_LIBRARY_PATH="$dir"
            fi
            ;;
    esac
}

prepend_ld_library_path "$OPENCV_PREFIX/lib"
prepend_ld_library_path "$GVIRTUS_HOME/lib/frontend"
prepend_ld_library_path "$GVIRTUS_HOME/lib/ucx"
prepend_ld_library_path "$GVIRTUS_HOME/lib"

export OPENCV_PREFIX="${OPENCV_PREFIX:-$HOME/opencv-local}"
export PKG_CONFIG_PATH="$OPENCV_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

pkg-config --modversion opencv4

nvcc main.cu -o sample \
  -L "$GVIRTUS_HOME/lib/frontend" \
  -L "$GVIRTUS_HOME/lib" \
  -lcuda -lcublas -lcudnn -lcudart \
  $(pkg-config --cflags --libs opencv4)

./sample
