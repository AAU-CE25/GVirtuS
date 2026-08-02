---
title: "ASPLOS claim map — paper claim → evidence → status"
date: "2026-08-02"
---

Status vocabulary, used strictly:

- **verified** — the artifact reconstructs the number from raw data with the recorded script,
  commit and config, at the stated repetition count.
- **partial** — the result is real but under-replicated, or reconstructible only in part.
- **unsupported** — cited but not reconstructible from what exists today.
- **pending** — this campaign has not reached it yet.

A claim does not stay **verified** if the artifact cannot rebuild it. Repetition counts are
process-level independent runs, never intra-run samples.

# A. Data-path placement and throughput

| # | claim | figure/table | raw | script | reps | status |
|---|---|---|---|---|---|---|
| A1 | GPU-resident RMA reaches 92.5–93.0 % of the network ceiling | — | `results/pool_provisioning*` | — | — | pending (inherited; not re-measured, to be re-cited with its raw) |
| A2 | GPU RMA ≈1.78–1.80× host RMA at 1 GiB | — | — | — | — | pending |
| A3 | Admission thresholds are direction- and memory-kind-dependent, not scalar | — | `docs/gusto_raw_2026-08-02/` | — | — | **pending — Phase 5.2 re-runs the boundary points** |
| A4 | Quadrant policy reaches the measured oracle in isolation (×3.15) | — | 2026-08-01 sweep | — | — | partial (inherited) |

**Carried risk on A3/A4.** Phase 0 F-01: below 4 MiB, RMA admission requires
`GVIRTUS_RMA_SCALAR_FLOOR` or `GVIRTUS_RMA_POLICY` to be set explicitly. Any boundary point
re-run without that measures the AM path. Phase 5.2 asserts `admit_rma > 0` per run.

# B. Semantics preserved across remoting

| # | claim | evidence | status |
|---|---|---|---|
| B1 | Order within a stream is preserved | — | pending (Phase 2.1) |
| B2 | Cross-stream dependency via event is preserved | — | pending (Phase 2.2) |
| B3 | Per-thread default streams behave correctly | `results/asplos_campaign/ptds/` | **partial — single-thread passes; multi-thread untested** |
| B4 | Deferred errors surface at the next host-visible point | — | pending (Phase 2.4) |
| B5 | Host-visible operations are never made fire-and-forget | — | pending (Phase 2.5) |
| B6 | Stream/event lifetime is respected | — | pending (Phase 2.6) |
| B7 | CUDA Graph lifecycle is complete and leak-free | — | pending (Phase 2.7) |

**B3 caveat, recorded now.** The `_ptsz` surface has 5 real entry points and 49 silent stubs
(F-03). Any claim of PTDS conformance must state which entry points it covers. As written
today, a PTDS-compiled application calling `cudaEventRecord`, `cudaStreamWaitEvent` or
`cudaGraphLaunch` receives `cudaErrorNotSupported` with no diagnostic.

# C. Lifetime protocol correctness

| # | claim | evidence | status |
|---|---|---|---|
| C1 | Allocation-lifetime invalidation prevents corruption that pointer-value caching allows | `pointer_keyed` ablation, 65 280 bad bytes vs 0 | partial (inherited; re-run under Phase 3) |
| C2 | Generation checking prevents premature slot release on stale ACKs | `hold_ack` with/without guard | partial (inherited) |
| C3 | Epoch checking rejects ACKs from a retired epoch | `ack_epoch_dropped=1`, ablation → 0 | **partial** — guard demonstrated reachable; **0 harmful deliveries in 186 opportunities**, so necessity is as an invariant, not as an observed save |
| C4 | Slot reuse is safe under connection + epoch + slot id + generation | — | pending (Phase 3) |

# D. Failure behaviour

| # | claim | evidence | status |
|---|---|---|---|
| D1 | Backend death is survived with a diagnosable error and clean teardown | 1 s exit, clean teardown (inherited) | partial — not yet at 50 reps/scenario |
| D2 | Client death returns all backend resources | GPU 435→435 MiB (inherited) | partial |
| D3 | No exception escapes a UCX callback | — | **unsupported** — identified as the most plausible route to killing the backend, never verified |
| D4 | Listener re-accepts promptly after a failure | — | pending (Phase 4) |

# E. End-to-end workloads

| # | claim | evidence | status |
|---|---|---|---|
| E1 | cuDF gains 13.6–15.4 % over the same UCX protocol with a host bounce | prior campaign | partial — Phase 5.3 completes under-replicated cells |
| E2 | TinyLlama 2×2 separates async (fewer waits) from Graphs (fewer operations) | prior campaign | partial — Phase 5.1 repeats to ≥5, target 10 |
| E3 | XSBench / BabelStream / miniBUDE amortise remoting; CloverLeaf is RPC-latency-bound | prior campaigns | partial (inherited, not re-measured by instruction) |
| E4 | llama abort rate is below 10.9 % (95 % confidence) | 26 runs, 0 aborts | **partial and explicitly bounded** — this rejects the original 1/9 by 0.23 points, and that 1/9 was a single event. **Not** evidence that the defect is fixed. |

# F. Overheads

| # | claim | evidence | status |
|---|---|---|---|
| F1 | Lifetime bookkeeping is a small fraction of transfer cost | reservation/ACK microbench | pending (Phase 6) |
| F2 | Instrumentation does not dominate the measured overhead | — | pending (Phase 6, on/off arms) |
| F3 | GPUDirect completion implies CUDA visibility before slot reuse | — | **pending (Phase 7)** — must name the concrete guarantee of this driver + UCX, not assume PUT-complete ⇒ kernel-visible |

# G. Reproducibility

| # | claim | evidence | status |
|---|---|---|---|
| G1 | Every figure regenerates from raw data | — | pending (Phase 10) |
| G2 | Frontend and backend are at the same commit | — | **unsupported today** — the hosts have divergent histories and dpu-01 has 13 uncommitted files (F-04) |
