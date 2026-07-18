# 11 — SimpleMatrix (cuBLAS SGEMM) vs the project report

Status: **reproduces the report's matrix results; validates the testbed + optimizations.**
Last updated: 2026-07-18.

## Why this test
The report's headline transport result is the **SimpleMatrix** benchmark (dense `cublasSgemm` on
two square N×N single-precision matrices, timing **H2D + GEMM + D2H** per internal iteration). It
is the cleanest bulk-transfer workload and the one place GPUDirect should *provably* engage
(each matrix at N≈16000 is ~1 GB per transfer). We re-ran it on the current testbed to (a) confirm
GPUDirect actually works end-to-end at large payloads and (b) cross-check our numbers against the
report.

We ran **N=16000** (the report used **N=16384** — 2.5% larger side, ~5% more elements, so our
numbers should be slightly *lower* than the report's at the same config). Metric = `avg_host_ms`
(wall time of H2D A + H2D B + GEMM + D2H C per iteration), which matches the report's
"H2D + GEMM + D2H mean". `check=pass` (max_abs_err=0) on every run.

## Results vs report

| configuration | our N=16000 `avg_host_ms` | report N=16384 (ms) | our GEMM ms | report GEMM ms |
|---------------|--------------------------:|--------------------:|------------:|---------------:|
| **UCX RDMA + GPUDirect** (zerocopy) | **358.4** | **389.5** | 157.9 | ~168 |
| **UCX RDMA** (zerocopy)             | **496.9** | **548.4** | 156.8 | ~168 |
| UCX RDMA (no zerocopy / staged)     | 698.9     | — (not a report config) | 156.9 | — |
| Baremetal (reference)               | not re-measured | 306.0 | — | ~168 |
| UCX TCP (reference)                 | not re-measured | 2210.7 | — | — |
| Plain RDMA, pre-project (reference) | not re-measured | 2265.6 | — | — |

Data: `data/matrix/matrix_vs_paper.csv`. Report source: `_pdftxt/page_78..79,84` (Tables 7.2, 7.3,
7.5, 7.6).

## Findings

### 1. Our numbers reproduce the report (within the N=16000 vs 16384 margin)
- **GPUDirect: 358.4 ms (ours) vs 389.5 ms (report).** Ours is ~8% lower, consistent with the
  slightly smaller matrix. ✓
- **UCX RDMA (zerocopy): 496.9 ms (ours) vs 548.4 ms (report).** Again ~9% lower, same margin. ✓
- **GEMM: ~157 ms (ours) vs ~168 ms (report)** — the pure compute stage matches closely, confirming
  it's the same workload on the same-class GPU (L40S) and that GEMM is *not* the bottleneck
  (~157 ms of a ~360–500 ms core), exactly as the report concludes.

### 2. GPUDirect provably works at large payloads — ~1.4× over UCX RDMA
GPUDirect cuts the core workload from 496.9 → 358.4 ms (**1.39×**), matching the report's
548.4 → 389.5 ms (**1.41×**). Unlike the llama workload (doc 07/10, where the data path is idle and
GPUDirect ≡ RDMA), SimpleMatrix moves ~1 GB per transfer, so the GPU↔NIC data path is genuinely
exercised and GPUDirect's benefit is real and measurable here. **This is the workload that
demonstrates NIC→GPU DMA**, complementing doc 04's H2D/D2H micro-benchmark.

### 3. The win is D2H, and RMA_ZEROCOPY is required to see it
Per-transfer profiling (`[GVS PROFILE] cudaMemcpy payload=976MB`) shows H2D at ~110 ms (~8.9 GB/s)
but **D2H highly variable on the staged path — up to ~960 ms for a single 976 MB copy**. Enabling
`GVIRTUS_RMA_ZEROCOPY=1` removes the staging memcpy and drops the core workload from 698.9 →
496.9 ms; GPUDirect then removes the backend-side GPU→host copy on D2H for a further drop to
358.4 ms. This mirrors the report's analysis (Table 7.3): >92% of the core workload is data
movement, D2H is the imbalance the optimized UCX design targets, and both H2D and D2H need a
dedicated bulk-transfer path.

## Operational notes (see the standing cleanup rule + doc 05)
- **GPUDirect requires `GVIRTUS_GPUDIRECT=1` on the backend AND `GVIRTUS_RMA_ZEROCOPY=1`** (GPUDirect
  needs zero-copy for GPU-resident buffers). The backend launcher was made to honour both via env.
- The frontend is logically GPU-less; its GPUDirect probe falls back to host slots, which is fine —
  the backend-side GPU shadow still accelerates D2H.
- **Cleanup is essential (standing rule).** Early "multi-minute" SimpleMatrix runs were not a code
  regression — they were accumulated **zombie frontend processes** (from `timeout`-killed runs) and
  an **orphaned backend container leaking GPU memory**, which clogged the backend and starved
  GPUDirect registration. After killing zombies + a fresh backend restart, the identical N=16000
  run completed in **~6–8 s**. Always clean up between runs; verify GPU memory is freed on both nodes.

## Reproduce
Backend (es-dpu-01): `GVIRTUS_GPUDIRECT={0|1} GVIRTUS_RMA_ZEROCOPY=1 bash /tmp/gvirtus-backend-run.sh`,
wait for `listener created` in `docker logs` (the entrypoint rebuilds from source first). Frontend
(es-dpu-02, `gvirtus-fe-kz08ey`): `cd /gvirtus/examples/simple_matrix; ./simple_matrix 2048 1 1`
(warmup) then `./simple_matrix 16000 3 1`; read `avg_host_ms` and `check`. **Kill any leftover
`simple_matrix` procs afterwards.** Verify GPUDirect on the backend log (`GPUDirect=enabled`).
