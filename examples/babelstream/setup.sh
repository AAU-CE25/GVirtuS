#!/bin/bash
#
# setup.sh — fetch BabelStream and apply the small GVirtuS adaptation.
#
# Run this ONCE on the host (from this directory) before building/running the
# example. It clones BabelStream next to this script and adapts the CUDA
# reduction buffer so it works over GVirtuS (see README.md, "GVirtuS adaptation").
#
set -e
cd "$(dirname "$0")"

BABELSTREAM_REPO="https://github.com/UoB-HPC/BabelStream.git"

if [ ! -d BabelStream ]; then
    echo "[setup] cloning BabelStream ..."
    git clone --depth 1 "${BABELSTREAM_REPO}" BabelStream
else
    echo "[setup] BabelStream already present, skipping clone."
fi

echo "[setup] applying GVirtuS adaptation to cuda/CUDAStream.cu ..."
python3 - <<'PY'
f = "BabelStream/src/cuda/CUDAStream.cu"
s = open(f).read()
orig = s

# The dot-reduction partial-sums buffer is allocated as pinned host memory and
# written DIRECTLY by the kernel (zero-copy). GVirtuS implements cudaHostAlloc as
# a frontend-local malloc, so that host address is invalid on the backend GPU ->
# CUDA error 700. Fix: put the buffer in device memory + explicit copy-back.
s = s.replace(
    "sums = alloc_host<T>(dot_num_blocks);",
    "sums = alloc_device<T>(dot_num_blocks); // GVirtuS: device mem (zero-copy pinned host not remotable)")
s = s.replace(
    "free_host(sums);",
    "free_device(sums); // GVirtuS")
old = "  T sum = 0.0;\n  for (intptr_t i = 0; i < dot_num_blocks; ++i) sum += sums[i];"
new = ("  std::vector<T> h_sums(dot_num_blocks); // GVirtuS: explicit D2H of partial sums\n"
       "  CU(cudaMemcpy(h_sums.data(), sums, sizeof(T) * dot_num_blocks, cudaMemcpyDeviceToHost));\n"
       "  T sum = 0.0;\n"
       "  for (intptr_t i = 0; i < dot_num_blocks; ++i) sum += h_sums[i];")
s = s.replace(old, new)
if "#include <vector>" not in s:
    s = s.replace('#include "CUDAStream.h"', '#include <vector>\n#include "CUDAStream.h"', 1)

open(f, "w").write(s)
if s == orig:
    print("[setup] WARNING: no changes applied (already adapted or upstream layout changed).")
else:
    print("[setup] adaptation applied.")
PY

echo "[setup] done. Build/run with the frontend.sh script (see README.md)."
