#!/bin/bash
#
# frontend.sh — run llama.cpp inference over GVirtuS (frontend stubs).
# Intended to run INSIDE the GVirtuS frontend container.
#
# The most transport-/RPC-revealing GVirtuS workload: token generation is
# thousands of tiny sequential kernel launches, each a synchronous RPC — unless
# the async dispatcher is enabled (GVIRTUS_ASYNC_DISPATCH=1), which makes the
# stream-ordered launches/copies fire-and-forget and ~doubles decode throughput.
#
# Env overrides:
#   MODEL       gguf path (default models/tinyllama-1.1b-q4.gguf)
#   NGL         layers offloaded to GPU (default 99 = all)
#   PROMPT_N    prompt-eval tokens for llama-bench -p (default 8)
#   GEN_N       generation tokens for llama-bench -n (default 16)
#   REPS        repetitions -r (default 3)
#   GVIRTUS_ASYNC_DISPATCH  0 (sync) or 1 (async) — the headline knob
#   GVIRTUS_LOGLEVEL        default 40000 (ERROR); keep lean for clean timing
#
set -e

export GVIRTUS_HOME=${GVIRTUS_HOME:-/opt/GVirtuS}
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-40000}
export GGML_CUDA_DISABLE_GRAPHS=1   # GVirtuS has only partial CUDA-graph support
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:$(dirname "$0")/llama.cpp/build_cuda/bin:${LD_LIBRARY_PATH}

MODEL=${MODEL:-$(dirname "$0")/models/tinyllama-1.1b-q4.gguf}
NGL=${NGL:-99}; PROMPT_N=${PROMPT_N:-8}; GEN_N=${GEN_N:-16}; REPS=${REPS:-3}
BIN=$(dirname "$0")/llama.cpp/build_cuda/bin/llama-bench

if [ ! -x "${BIN}" ]; then
    echo "llama-bench not found. Run ./setup.sh on the host first."; exit 1
fi

echo "Running llama-bench over GVirtuS (async=${GVIRTUS_ASYNC_DISPATCH:-unset}):"
echo "  model=${MODEL} -ngl ${NGL} -p ${PROMPT_N} -n ${GEN_N} -r ${REPS}"
"${BIN}" -m "${MODEL}" -ngl "${NGL}" -p "${PROMPT_N}" -n "${GEN_N}" -r "${REPS}"
