# llama.cpp inference over GVirtuS

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a portable LLM inference engine. This example
runs its **CUDA backend through GVirtuS**: the client links the GVirtuS frontend stubs instead of the
real CUDA runtime, so every CUDA call is forwarded to a remote backend that owns the GPU.

> **Why it's the key GVirtuS benchmark:** LLM **token generation** is thousands of tiny *sequential*
> kernel launches per token, each a synchronous RPC over GVirtuS. It is the most RPC-/latency-bound
> real workload in the suite, and therefore the one that most exposes — and most benefits from — the
> **async dispatcher**. With `GVIRTUS_ASYNC_DISPATCH=1`, stream-ordered launches and `cudaMemcpyAsync`
> become fire-and-forget, roughly **doubling** decode throughput (see `docs/benchmarking/`).

Model: **TinyLlama-1.1B-Chat Q4_K_M** (668 MB GGUF). Built with `--cudart shared` +
`BUILD_SHARED_LIBS` so it links the GVirtuS stubs. `GGML_CUDA_DISABLE_GRAPHS=1` (GVirtuS has only
partial CUDA-graph support).

## Files
| File | Purpose |
|------|---------|
| `setup.sh` | Clone + build llama.cpp (CUDA, shared cudart) and download the test model (run once, on host). |
| `frontend.sh` | Run `llama-bench` over the GVirtuS frontend stubs (inside the frontend container). |
| `backend.sh` | Launch the GVirtuS backend with `properties_ucx.json`. |
| `Dockerfile` | Builds a self-contained image (GVirtuS + llama.cpp + model). |

## Quick start
```bash
# 0) One-time: build llama.cpp + fetch the model (on the host, in this directory)
cd examples/llama && ./setup.sh

# 1) Backend node (owns the GPU): start the GVirtuS backend
#    (e.g. `make run-gvirtus-backend-dev`, which serves properties_ucx.json)

# 2) Frontend node (GPU-less client): run inference over GVirtuS
#    sync path:
GVIRTUS_ASYNC_DISPATCH=0 bash examples/llama/frontend.sh
#    async dispatcher (the headline result):
GVIRTUS_ASYNC_DISPATCH=1 bash examples/llama/frontend.sh
```

Tunables (env): `MODEL`, `NGL` (default 99), `PROMPT_N`/`GEN_N`/`REPS` (llama-bench `-p`/`-n`/`-r`),
`CUDA_ARCH` (default 89 = Ada/L40S), `GVIRTUS_LOGLEVEL` (default 40000 = ERROR). Transport is selected
via `etc/properties_ucx.json` + the `UCX_*` variables (TCP / RDMA / RDMA+GPUDirect).

## Notes
- Use **`llama-bench`** (self-exits) for clean numbers; `llama-cli` blocks on stdin after generation
  (append `< /dev/null`) — see the benchmark docs.
- Build the frontend from the **same GVirtuS source** as the backend (a stale frontend desyncs the
  wire protocol).
- If a run faults, restart the backend before the next run to clear the persistent CUDA context.
- Measured result (TinyLlama-1.1B, RDMA): token-gen tg16 **87 → 187 t/s (~2.15×)** with the async
  dispatcher, correctness preserved. Data: `benchmarks/llama-{sync,async}/`.
