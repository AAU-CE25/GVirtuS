# Sync vs Async — before/after the async dispatcher

Direct A/B of the **synchronous** path (the behaviour **before the async dispatcher**, reproduced by
`GVIRTUS_ASYNC_DISPATCH=0` on the current build) versus the **async** path (`=1`), for the two most
relevant workloads. Fresh backend, RDMA, ERROR log level (2026-07-19).

## Results

| benchmark | metric | sync (before) | async (after) | change |
|-----------|--------|--------------:|--------------:|-------:|
| **llama** | token gen tg16 (tok/s) | 87.16 | **187.43** | **+115% (2.15×)** |
| **llama** | prompt eval pp8 (tok/s) | 558.23 | **1204.48** | **+116% (2.16×)** |
| **simple_matrix** | SGEMM (ms, ↓) | 156.36 | 156.59 | ~0% |
| **simple_matrix** | host per-iter (ms, ↓) | 737.70 | 725.80 | ~0% (−1.6%, noise) |

## Takeaways
- **llama: async is ~2.15× faster** than sync — the async dispatcher (fire-and-forget kernel launches
  + async `cudaMemcpyAsync`) more than doubles remote LLM decode throughput. vs bare metal (634.9
  tg16): sync ~7.3× slower → async ~3.4× slower.
- **simple_matrix: async ≈ sync** — it is transfer-bound (cuBLAS SGEMM + synchronous `cudaMemcpy`),
  so the async dispatcher correctly has no effect. This is the expected non-regression outcome for a
  non-launch-bound workload.
- **No regression:** the sync numbers on the current build (with the async dispatcher + all three
  `cudaMemcpyAsync` phases) match the historical sync baseline (llama tg16 87.16 vs 87.35). The new
  code, when gated off, leaves the synchronous path unchanged.

## Files
- `sync_vs_async_summary.csv` — the numbers above + exact commands/config.
- `llama_sync_raw.log`, `llama_async_raw.log` — raw llama-bench output (evidence).
- `matrix_sync_vs_async_raw.csv` — raw simple_matrix rows (`config,n,iters,sgemm_ms,host_ms`).
