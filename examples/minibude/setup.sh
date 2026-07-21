#!/bin/bash
#
# setup.sh — fetch miniBUDE and generate its build metadata so the GVirtuS
# variant can be compiled. Run ONCE on the host (from this directory).
#
# miniBUDE's CMake configure generates headers under build/generated (deck sizes
# etc.) that the single-file GVirtuS compile in frontend.sh includes. We run the
# configure step here but do the actual (GVirtuS-linked) compile in frontend.sh.
#
set -e
cd "$(dirname "$0")"

MINIBUDE_REPO="https://github.com/UoB-HPC/miniBUDE.git"
CUDA_ARCH="${CUDA_ARCH:-sm_89}"

if [ ! -d miniBUDE ]; then
    echo "[setup] cloning miniBUDE ..."
    git clone --depth 1 "${MINIBUDE_REPO}" miniBUDE
else
    echo "[setup] miniBUDE already present, skipping clone."
fi

echo "[setup] configuring miniBUDE (CUDA model) to generate build/generated ..."
cmake -S miniBUDE -B miniBUDE/build \
    -DMODEL=cuda \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH#sm_}" \
    -DCMAKE_BUILD_TYPE=Release || true
# We only need the generated headers; a full native build is optional.
cmake --build miniBUDE/build --target help >/dev/null 2>&1 || true

echo "[setup] done. Build/run the GVirtuS variant with frontend.sh (see README.md)."
