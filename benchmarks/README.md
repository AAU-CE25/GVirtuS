# GVirtuS benchmarks — result data

Raw result data + plot scripts for the GVirtuS (GUSTO) benchmark campaign on the AAU two-node
UCX testbed. The **write-up and analysis** live in [`../docs/benchmarking/`](../docs/benchmarking/);
this tree holds the reproducible artifacts (CSVs, logs, plots) behind those numbers.

## Layout — one folder per benchmark × mode

| benchmark | `-async` | `-sync` | `-baseline` |
|-----------|----------|---------|-------------|
| **llama** (TinyLlama-1.1B decode, RPC-bound) | `llama-async/` | `llama-sync/` | `llama-baseline/` |
| **miniBUDE** (compute-bound) | `miniBUDE-async/` | `miniBUDE-sync/` | `miniBUDE-baseline/` |
| **BabelStream** (memory-bandwidth) | `BabelStream-async/` | `BabelStream-sync/` | `BabelStream-baseline/` |
| **simple-matrix** (cuBLAS SGEMM, bulk transfer) | `simple-matrix-async/` | `simple-matrix-sync/` | `simple-matrix-baseline/` |

- **`-async`** — GVirtuS with the async dispatcher on (`GVIRTUS_ASYNC_DISPATCH=1`).
- **`-sync`** — GVirtuS with the frontend RPC optimizations, async **off** (`=0`). Includes the older
  transport/optimization-progression campaign data (all measured on the synchronous path).
- **`-baseline`** — **clean/stock GVirtuS** (no async dispatcher, no RPC opts). Placeholder READMEs;
  to be measured later to quantify the full improvement stack from zero.

## Other folders
- **`_summary/`** — cross-benchmark summary CSVs + plots (async-vs-sync speedup by workload).
- **`transport-characterization/`** — measurements that characterize the *transport* rather than one
  app: per-RPC latency distributions (`latency/`, TCP vs RDMA vs GPUDirect CDFs/percentiles) and raw
  H2D/D2H bandwidth (`transfer/`).

## Standing rules (see the benchmark-dev skill)
Measure at `GVIRTUS_LOGLEVEL=40000` (ERROR), fresh backend between phases, verify GPU freed on both
nodes, every number reproducible from a captured artifact. No number without a source.
