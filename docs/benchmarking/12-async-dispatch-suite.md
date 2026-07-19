# 12 — Async dispatcher: effect across the benchmark suite

Status: **measured A/B (async off vs on) across BabelStream, miniBUDE, simple_matrix, transfer_bw2,
and llama.** Last updated: 2026-07-18.

This doc reports how the async dispatcher (rec #3, `GVIRTUS_ASYNC_DISPATCH=1`, see
[`08-recommended-improvements.md`](08-recommended-improvements.md)) affects each workload in the
suite. All GVirtuS runs used a **fresh backend at ERROR log level** (DEBUG logging ~halves throughput
— see doc 07), GPUDirect enabled (`GVIRTUS_GPUDIRECT=1 GVIRTUS_RMA_ZEROCOPY=1`), frontend linked to
the GVirtuS cudart stub. Raw data: `data/_async_suite/`.

## Headline: the async benefit scales with launch-boundedness (as predicted)

| benchmark | regime | metric | async off | async on | **async effect** |
|-----------|--------|--------|----------:|---------:|-----------------:|
| **llama** (TinyLlama-1.1B) | launch/RPC-bound | token gen tg16 (t/s) | 87.35 | 189.86 | **+117% (2.17×)** |
| **BabelStream** @256K elems | launch-overhead-bound | Triad (GB/s) | 176.0 | 230.9 | **+31%** |
| **BabelStream** @512K elems | launch-overhead-bound | Triad (GB/s) | 353.1 | 457.7 | **+30%** |
| **BabelStream** @1M elems | launch-overhead-bound | Triad (GB/s) | 702.1 | 845.8 | **+20%** |
| **BabelStream** @2M elems | transitional | Triad (GB/s) | 1393.7 | 1472.1 | +6% |
| **BabelStream** @16M elems | bandwidth-bound | Triad (GB/s) | 648.4 | 641.4 | ~0% |
| **miniBUDE** bm1 | compute-bound | GFLOP/s | 216.53 | 216.53 | **0%** |
| **transfer_bw2** @256MB | transfer-bound | D2H (GB/s) | 8.79 | 8.69 | ~0% |
| **simple_matrix** n=16000 | transfer-bound | host time (ms, ↓) | 567.5 | 564.3 | ~0% |

**Interpretation.** The dispatcher only removes **per-launch RPC round-trips** for stream-ordered
calls (`cudaLaunchKernel` &c). So the win is large where those round-trips dominate (llama decode;
BabelStream at small sizes, which are launch-overhead-bound) and vanishes where the cost is compute
(miniBUDE), bandwidth (large BabelStream), or bulk `cudaMemcpy` (transfer_bw2, simple_matrix — which
issue no app-level kernel launches at all: cuBLAS runs the GEMM internally on the backend). This is
exactly the roofline expectation and confirms the prediction in doc 08.

**No regressions.** On every non-launch-bound workload, async on == async off within noise, so the
gate can be left on safely; and it is off by default.

## Per-benchmark notes

- **BabelStream** (`data/_async_suite/babel_async_raw.csv`): async gives **+20…31%** on the three
  smallest sizes (256K–1M elements), +6% at 2M, and ~0% from 8M up (bandwidth-bound). Copy/Mul/Add/
  Triad/Dot all follow the same trend. This complements rec #1's earlier small-size BabelStream win.
- **miniBUDE** (`data/_async_suite/minibude_async.csv`): 216.53 GFLOP/s both ways, `valid: true` —
  compute-bound, transport- and dispatch-insensitive (≈ native, matches doc 06).
- **transfer_bw2** (`data/_async_suite/transfer_async.csv`): H2D/D2H bandwidth identical off vs on
  across the whole 4 KB–256 MB sweep — pure `cudaMemcpy`, not on the async path.
- **simple_matrix** (`data/_async_suite/matrix_async.csv`): host and SGEMM times identical off vs on
  — cuBLAS SGEMM + synchronous `cudaMemcpy`, no app-level launches to overlap.

## Secondary finding — the current build is much faster than the pre-optimization tables

The current build (rec #1/#2 local Push/Pop + cached queries, plus the zero-copy Active-Message path)
is **~5.6× faster on small-size BabelStream** than the numbers in
`data/babelstream/babelstream_summary_gbps.csv` (Triad @256K: old `gvirtus-rdma` **30.2 GB/s** →
current sync **176 GB/s**), *before* async. Those older multi-transport BabelStream tables predate the
optimization stack and are stale for GVirtuS; the current sync/async numbers here supersede them for
the GVirtuS column. (Baremetal is unchanged.) miniBUDE, transfer_bw2, and simple_matrix are within
noise of their existing docs, so those are not restated.

## Pre-existing issues observed during the sweep (not caused by async)

Both reproduce with async **off**, so they are baseline GVirtuS behaviour, not dispatcher bugs:

1. **BabelStream at exactly 4,194,304 elements (2²²) fails** (`cudaMalloc`/transfer error, rc=1) on a
   *fresh* backend, while 2²¹ and 2²³ succeed — a size-specific bug worth a follow-up.
2. **Backend GPU memory leaks across client connections.** After many sequential client runs the
   backend GPU climbed to **45/46 GB** and `cudaMalloc` began failing ("malloc fail"); a fresh backend
   restart cleared it. Known issue (doc 05). Restart the backend between heavy sweeps.

## Reproduce
Driver: `/tmp/run_suite.sh <babel|minibude|transfer|matrix> <0|1>` on es-dpu-02 (sets the GVirtuS UCX
env + `GVIRTUS_ASYNC_DISPATCH`, calls the `~/benchmarks/harness/*.sh` scripts). Backend fresh at
ERROR via `/tmp/gvirtus-backend-run-err.sh` on es-dpu-01. Restart the backend between long sweeps to
avoid the GPU leak.

---

## Addendum — `cudaMemcpyAsync` made async (all 3 directions)

The v1 dispatcher excluded `cudaMemcpyAsync`; it is now async in three phases (all gated by
`GVIRTUS_ASYNC_DISPATCH`, validated with `examples/testing/test_memcpyasync_phase{1,2,3}.cu` on RDMA,
correctness bit-identical to the sync path, no regression on llama):

- **Phase 1 — D2D + small H2D (fire-and-forget).** D2D carries only pointers; small H2D (< 64 KB, AM
  path) returns no data. Both go fire-and-forget like `cudaLaunchKernel`. Verified on the wire
  (backend logs `cudaMemcpyAsync … no response sent`).
- **Phase 2 — large H2D via RMA-slot flow control (ack-free).** The RMA data path round-robins a
  bounded set of remote slots (configurable: `GVIRTUS_RMA_SLOTS`, `GVIRTUS_RMA_SLOT_CAP_MB`). The
  frontend tracks in-flight slots and, when the ring would wrap onto a slot the backend hasn't
  consumed, **demotes that one copy to synchronous** — which drains all prior slots (strict in-order
  FIFO). No extra network messages; correct backpressure. Test: 24 distinct large-H2D buffers with 8
  slots → **21 fire-and-forget + 3 sync demotions, all buffers intact** (slot reuse would corrupt).
- **Phase 3 — D2H via deferred completion.** A D2H into a **pinned** dst (`cudaMallocHost`/
  `cudaHostAlloc`, tracked) is sent in stream order but its reply is collected at the next
  synchronization point (`DrainPendingD2H`, drained at the start of the next synchronous receive —
  in-order FIFO guarantees the reply precedes it). **No backend change needed**: the backend's D2H
  handler copies into a pageable buffer, which makes the copy synchronous and correctly
  stream-ordered before it replies. Pageable dst stays synchronous (real CUDA fills it synchronously).
  Test verifies a deferred D2H captures device state **at its stream position** (before a later kernel
  overwrites the source).

**Regression note (no perf loss by design):** async either overlaps or applies correct backpressure;
the ack-free Phase-2 design adds no extra round-trips, and Phase-3 replaces the sync reply with the
same reply collected later. A static-init/destruction-order fiasco in the pinned-allocation registry
(hit because the communicator's `init_rx_pool` calls `cudaHostAlloc` during static init) was fixed
with immortal function-local singletons.
