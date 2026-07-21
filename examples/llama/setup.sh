#!/bin/bash
#
# setup.sh — fetch + build llama.cpp (CUDA backend) and a small test model, so
# this example can run over GVirtuS. Run ONCE on the host (from this directory).
#
# llama.cpp is built with `--cudart shared` + BUILD_SHARED_LIBS so it links the
# CUDA runtime dynamically and can be redirected onto the GVirtuS frontend stubs
# at run time (LD_LIBRARY_PATH). Static cudart cannot be LD-redirected.
#
set -e
cd "$(dirname "$0")"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
CUDA_ARCH="${CUDA_ARCH:-89}"   # 89 = Ada / L40S

if [ ! -d llama.cpp ]; then
    echo "[setup] cloning llama.cpp ..."
    git clone --depth 1 "${LLAMA_REPO}" llama.cpp
else
    echo "[setup] llama.cpp already present, skipping clone."
fi

echo "[setup] building llama.cpp with CUDA (shared cudart) ..."
cmake -S llama.cpp -B llama.cpp/build_cuda \
    -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_CUDA_FLAGS="--cudart shared"
cmake --build llama.cpp/build_cuda -j"$(nproc)" --target llama-bench llama-cli

mkdir -p models
if [ ! -f models/tinyllama-1.1b-q4.gguf ]; then
    echo "[setup] downloading TinyLlama-1.1B Q4_K_M (~668 MB) ..."
    wget -q -O models/tinyllama-1.1b-q4.gguf "${MODEL_URL}"
else
    echo "[setup] model already present, skipping download."
fi

echo "[setup] done. Run with frontend.sh (see README.md)."
