# llama — baseline (clean GVirtuS)

**Placeholder — to be run later.**

`-baseline` = **clean/stock GVirtuS** (upstream, without this project's optimizations:
no async dispatcher, no local push/pop config, no cached `cudaGetDevice`/`cudaGetLastError`).
It is the *unoptimized remoting reference* used to quantify the full improvement stack.

- `../llama-sync/`  — GVirtuS with the frontend RPC optimizations, `GVIRTUS_ASYNC_DISPATCH=0`.
- `../llama-async/` — same build with `GVIRTUS_ASYNC_DISPATCH=1`.
- `-baseline` (here) — stock GVirtuS, to measure the delta from zero.

## How to run (later)
Build a clean/stock GVirtuS (or check out a pre-optimization commit), then:
```bash
GVIRTUS_LOGLEVEL=40000 llama-bench -m tinyllama-1.1b-q4.gguf -ngl 99 -p 8 -n 16 -r 3
# GGML_CUDA_DISABLE_GRAPHS=1 ; RDMA transport ; capture stdout CSV here.
```
Save the raw `llama-bench` output + backend `docker logs` evidence in this folder.
