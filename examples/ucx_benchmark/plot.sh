#!/usr/bin/env bash
# Wrapper: runs plot_sweep.py inside the local .venv if present.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -x "$DIR/.venv/bin/python" ]]; then
    echo "[plot] No venv found — creating $DIR/.venv ..." >&2
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" install --quiet --upgrade pip
    "$DIR/.venv/bin/pip" install --quiet -r "$DIR/requirements.txt"
fi

exec "$DIR/.venv/bin/python" "$DIR/plot_sweep.py" "$@"
