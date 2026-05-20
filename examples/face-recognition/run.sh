#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"

export GVIRTUS_HOME="${GVIRTUS_HOME:-$REPO_ROOT}"
export GVIRTUS_CONFIG="${GVIRTUS_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
# Deterministic runtime/link paths for GVirtuS frontend shims.
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$GVIRTUS_HOME/lib/ucx:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$GVIRTUS_HOME/lib/ucx:${LIBRARY_PATH:-}"

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

prepend_ld_library_path "$GVIRTUS_HOME/lib/frontend"
prepend_ld_library_path "$GVIRTUS_HOME/lib/ucx"
prepend_ld_library_path "$GVIRTUS_HOME/lib"


nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu  -lcudart -lcublas
ldd libextension.so
env LD_PRELOAD=${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so \
python3 cnn.py
