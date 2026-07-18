# GVirtuS Benchmarking (InfoComm 2027)

> ## 🛠️ NEW HERE? START WITH THE SKILL: [`../benchmark-dev/SKILL.md`](../benchmark-dev/SKILL.md)
> The **benchmark-dev skill** is the operating manual: testbed layout (es-dpu-01 backend / es-dpu-02
> frontend), SSH access, how to run each test suite, known problems, where everything lives, and the
> five standing rules (clean up, back up with evidence, don't overclaim, challenge assumptions,
> verify GPUDirect). Read it before running anything.

> # ⚠️ READ FIRST: [`00-CRITICAL-verify-gpudirect.md`](00-CRITICAL-verify-gpudirect.md)
> **ALWAYS MAKE SURE GPUDIRECT IS ACTUALLY WORKING. NEVER ASSUME WITHOUT BACKING.**
> Verify every transport-mode claim against backend logs / a known-good app before reporting.
> We were burned twice by assuming instead of verifying.

This folder documents the expanded benchmark suite for evaluating the UCX-based
GVirtuS communicator (project **GUSTO**) across bare-metal and remote-virtualized
configurations. It is the living record of the testing effort; each phase adds a
numbered document here.

> **Why this exists:** the paper's current evaluation (cuBLAS SimpleMatrix +
> OpenCV-DNN/YOLO) is too narrow for an HPC venue. We add recognized HPC proxy
> apps and an LLM inference workload, measured across all transport modes.

## Document index

| Doc | Contents |
|-----|----------|
| [`00-CRITICAL-verify-gpudirect.md`](00-CRITICAL-verify-gpudirect.md) | **MANDATORY RULE + checklist: always verify GPUDirect/transport is actually engaged (backend log proof) before reporting any result. Never assume.** |

| Doc | Contents |
|-----|----------|
| [`01-testbed-setup.md`](01-testbed-setup.md) | Two-node testbed: hardware, network, SSH access, branch, backend/frontend launch, transport presets, Phase 0 bring-up + smoke test results. |
| [`02-babelstream.md`](02-babelstream.md) | BabelStream: what it measures, bare-metal L40S baseline, the diagnosis journey (stale-image red herring + real `cudaHostAlloc` zero-copy limitation), the fix, and confirmation it runs over GVirtuS. |
| [`03-transport-sweep.md`](03-transport-sweep.md) | Full transport sweep (4 configs × 9 sizes × 5 kernels = **180 points**): methodology, findings (TCP < RDMA ≈ GPUDirect, control-path amortization toward bare metal), caveats. Data + plots under [`data/babelstream/`](data/babelstream/). |
| [`04-transfer-bandwidth.md`](04-transfer-bandwidth.md) | H2D/D2H `cudaMemcpy` bandwidth sweep (the transport data path). **RDMA ~3× TCP**; **GPUDirect ~2.3× D2H speedup for bulk transfers** (≥4 MiB). Corrects an earlier benchmark artifact ("crash ≥8 MiB" was an oversized-reused-buffer bug, not GPUDirect). Data + plots under [`data/transfer/`](data/transfer/). |
| [`05-gvirtus-bugs-and-knobs.md`](05-gvirtus-bugs-and-knobs.md) | GVirtuS-internal issues found during benchmarking: `RMA_ZEROCOPY` what/why/OOM, buffer-reuse RMA edge case, backend listener non-recovery, frontend GPUDirect probe, `cudaHostAlloc` zero-copy. Workarounds + future-work fixes. |
| [`06-minibude.md`](06-minibude.md) | miniBUDE (compute-bound molecular-docking proxy): **GVirtuS overhead < 0.2 % vs bare metal across all transports** (216 GFLOP/s). Transport-insensitive throughput; transport only shows in one-time setup cost. Data + plots under [`data/minibude/`](data/minibude/). |
| [`07-llama.md`](07-llama.md) | Llama LLM inference (llama.cpp, TinyLlama-1.1B): **runs over GVirtuS after fixing `cudaDeviceGetPCIBusId` + host-alloc alignment**; token gen **~79× slower** than native (RDMA) — the RPC-bound extreme, confirming the synchronous-dispatch bottleneck. **Transport DOES matter here: RDMA ~2.2× TCP** (latency-bound); GPUDirect == plain RDMA. Data + plots under [`data/llama/`](data/llama/). |
| [`08-recommended-improvements.md`](08-recommended-improvements.md) | Prioritized GVirtuS improvements from the campaign. **Improvement #1 PROTOTYPED + KEPT + VALIDATED:** local `__cudaPush/PopCallConfiguration` (thread-local, no RPC) → **llama tg +1.45×, small BabelStream +1.64×, miniBUDE unchanged**, correctness preserved. Plus recommendations: cache `cudaGetDevice`/`cudaGetLastError`, async dispatch, launch batching / graph capture, RMA rcache fix. Data under [`data/prototype_pushpop_summary.csv`](data/prototype_pushpop_summary.csv). |
| [`09-measurement-roadmap.md`](09-measurement-roadmap.md) | **What to measure next for an INFOCOM-grade paper.** Gap: we have means, not distributions. Roadmap: per-RPC **latency percentiles p50/p90/p99/p99.9 + CDFs / tails** (biggest gap), latency decomposition, throughput–latency saturation curves, small-message RPC/s ceiling, **multi-tenancy scaling + fairness (Jain) + isolation**, TTFT/ITL for LLM, CPU/registration efficiency, statistical rigor (CIs), external baseline, design ablations. Concrete first step: env-gated latency trace in `Frontend::Execute()`. |
| [`10-latency-distributions.md`](10-latency-distributions.md) | **First distribution-level result (INFOCOM headline).** Per-RPC latency for 44k RPCs/transport via new `GVIRTUS_LATENCY_TRACE` instrumentation. Control-plane p99: **TCP 1328 µs vs RDMA 56 µs (24×)**; p99.9 44×; **tail ratio p99/p50: TCP 22.9× vs RDMA 1.33×**. RDMA's win is near-elimination of tail-latency variance, not lower median. Explains the llama TCP slowdown. GPUDirect ≈ RDMA on control plane. Data + CDF/percentile plots under [`data/latency/`](data/latency/). |
| [`11-matrix-vs-paper.md`](11-matrix-vs-paper.md) | **SimpleMatrix (cuBLAS SGEMM, N=16000) vs the report (N=16384).** Reproduces the report within the size margin: GPUDirect **358 ms** (report 389), UCX RDMA zerocopy **497 ms** (report 548), GEMM ~157 ms (report ~168). **GPUDirect provably engages at ~1 GB/transfer → ~1.4× over RDMA** (the bulk-transfer workload, unlike llama). Win is D2H; requires `RMA_ZEROCOPY=1`. Also documents that early "multi-minute" runs were zombie/leak contamination, not a regression. Data under [`data/matrix/`](data/matrix/). |
| [`12-async-dispatch-suite.md`](12-async-dispatch-suite.md) | **Async dispatcher (rec #3) A/B across the whole suite.** `GVIRTUS_ASYNC_DISPATCH=1` makes stream-ordered `cudaLaunchKernel` &c fire-and-forget. Benefit scales with launch-boundedness: **llama +117% (2.17×)**, **BabelStream small sizes +20…31%**, transitional +6%, and **~0% for bandwidth/compute/transfer-bound** (large BabelStream, miniBUDE, transfer_bw2, simple_matrix). No regressions. Also: the current build is ~5.6× faster on small BabelStream than the pre-opt tables, and two pre-existing bugs (BabelStream 2²² size, backend GPU leak). Data under [`data/_async_suite/`](data/_async_suite/). |

**Runnable example:** [`examples/babelstream/`](../../examples/babelstream/) — first-class
GVirtuS example (mirrors `simple_matrix`): `setup.sh` (clone + GVirtuS adaptation),
`frontend.sh`, `backend.sh`, `nvml_shim.cpp`, `Dockerfile`, `README.md`. Run with
`make run-babelstream-test`.

## Related top-level docs
- [`../../BENCHMARK_PLAN.md`](../../BENCHMARK_PLAN.md) — full plan: capability audit,
  per-app CUDA requirements, risk table, phased rollout, metrics, multitenancy.
- [`../GPUDIRECT.md`](../GPUDIRECT.md) — GPUDirect design + quickstart transport launchers.
- [`../UCX_GUIDE.md`](../UCX_GUIDE.md), [`../UCX_PROPERTIES_GUIDE.md`](../UCX_PROPERTIES_GUIDE.md),
  [`../UCX_OPTIMIZATIONS.md`](../UCX_OPTIMIZATIONS.md) — UCX transport configuration.

## Test matrix (target)

Each application is run in 5 configurations:

1. **Bare metal** — native CUDA, no GVirtuS (baseline).
2. **GVirtuS UCX / TCP** — `UCX_TLS=tcp,self` (transport baseline).
3. **GVirtuS UCX / RDMA** — RoCEv2, no GPUDirect (host-staged).
4. **GVirtuS UCX / RDMA + GPUDirect** — NIC ↔ GPU memory directly.
5. _(optional)_ **Legacy plain TCP/RDMA connectors** for regression vs the paper.

Applications: **BabelStream, miniBUDE, XSBench, CloverLeaf** (HPC proxies) +
**Llama** (LLM inference). Later: **multitenancy** (≥2 frontends → 1 backend).

## Status snapshot

- **Phase 0 (testbed bring-up): DONE.** Backend builds & runs on es-dpu-01;
  `simple_matrix` frontend from es-dpu-02 passes over UCX (`check=pass`, `max_abs_err=0`).
