---
title: "ASPLOS correctness & evaluation campaign — results"
date: "2026-08-02"
branch: "exp/asplos-correctness-campaign"
base_commit: "3b05300f70a35bd80134bb248529e90d430b63e2"
---

# 1. Phase 0 — inventory and baseline

## 1.0 Baseline gate: PASSED

Single-threaded, over UCX/RoCEv2 to the live backend on `25.25.25.2:32222`, pool
8 × 32 MiB, `PREALLOC=1`:

| step | result |
|---|---|
| control-plane connect | ok |
| 1 MiB H2D (`cudaMemcpy` and `cudaMemcpyAsync`) | ok |
| `cudaStreamSynchronize` on null / explicit / legacy `0x1` / per-thread `0x2` | **all ok** |
| 1 MiB D2H | ok |
| teardown | clean, `ack_gen_mismatch=0`, `ack_epoch_dropped=0` |
| exit code | 0 |

Raw: `results/asplos_campaign/ptds/baseline_singlethread.log`.

This **confirms**, now with a preserved raw file, the 2026-08-02 morning finding that the
single-threaded PTDS reproducer does not fail. The morning session drew that conclusion but
saved no raw output for it; per this campaign's own rule the claim was unsupported until now.

It also **corrects** a statement written into that reproducer's own source comments:
`cudaStreamSynchronize_ptsz` was said not to exist. It exists, at
`plugins/cudart/frontend/CudaRt_stream.cpp:225`.

## 1.1 F-01 — `GVIRTUS_RMA_MIN_BYTES` does not gate RMA admission

**Severity: methodology-critical.** Not a crash; it silently decides which data path an
experiment measures.

The frontend's placement decision is `prefer_rma()` in
`include/gvirtus/communicators/RmaPolicy.h:126`. Under the default policy (`Scalar`) it
compares the payload against `scalar_floor_bytes()`, which reads **`GVIRTUS_RMA_SCALAR_FLOOR`,
default 4 MiB**. `GVIRTUS_RMA_MIN_BYTES` is passed into `prefer_rma` as the `pool_floor`
argument and is **discarded — the parameter is unnamed in the signature**. In the same
function, `UcxCommunicator.cpp:4222`, a second variable `kRmaMinBytesUnused` is computed from
that env var and never read.

The split is deliberate and documented in `RmaPolicy.h` (a scalar-vs-quadrant A/B driven by
one shared variable would put both arms behind the same gate, measuring nothing). The defect
is not the split; it is that the operator-facing documentation — including
`docs/HANDOFF_2026-08-02_tarde.md` §3.2, which lists `GVIRTUS_RMA_MIN_BYTES` as read by
"both" with effect "RMA floor" — states the opposite of what the frontend does.

**Control, three arms, same binary, same 1 MiB payloads, same backend:**

| arm | admit_rma | admit_am | rma_pct |
|---|---:|---:|---:|
| A: as documented (`GVIRTUS_RMA_MIN_BYTES=8192` only) | **0** | 25 | **0.00 %** |
| B: A + `GVIRTUS_RMA_SCALAR_FLOOR=8192` | 4 | 21 | 16.00 % |
| C: A + `GVIRTUS_RMA_POLICY=quadrant` | 4 | 21 | 16.00 % |

Raw: `results/asplos_campaign/environment/f01_admission_control.log`.

**Blast radius, checked rather than assumed.** Of the harnesses in `~`, only
`llama_abort_repro.sh` sets a policy or scalar floor; six set `GVIRTUS_RMA_MIN_BYTES` without
one (`gusto_noregresion.sh`, `gusto_sat_run.sh`, `gusto_epoch_run.sh`,
`gusto_epochgrow_run.sh`, `gusto_firsttouch_ab.sh`, `gusto_tsan_run.sh`).

I expected this to have made the saturation and epoch campaigns inert. **It did not.** Their
benches default to payloads above the 4 MiB scalar floor, so RMA was admitted:

| raw file | counters |
|---|---|
| `sat_S1_s2_ms50.log` | `admit_rma=320 admit_am=1146 rma_pct=21.83` |
| `sat_S2_s1_ms50.log` | `admit_rma=320 admit_am=1146 rma_pct=21.83` |
| `sat_S3_s1_ms35000.log` | `admit_rma=3 admit_am=23 rma_pct=11.54` |
| `sat_S4_s1_ctrl.log` | `admit_rma=320 admit_am=1146 rma_pct=21.83` |
| `epoch_ep_full.log` | `admit_rma=61 admit_am=196 rma_pct=23.74` |
| `epoch_ep_noepoch.log` | `admit_rma=61 admit_am=196 rma_pct=23.74` |
| `epoch_eg_idx.log` | `admit_rma=96 admit_am=309 rma_pct=23.70` |

So the prior campaign's conclusions stand on this point. They stand **by the accident of the
bench defaults (8/64 MiB in `concgrow`), not by design** — nothing in those harnesses would
have caught a smaller payload silently routing to AM.

**Consequence for this campaign.** Phase 5.2 probes 4–64 KiB, i.e. entirely below the default
floor. Every admission-boundary point must set the policy explicitly, and every run must
assert a nonzero `admit_rma` before its number is used. Added to the Phase 5.2 procedure.

## 1.2 F-02 — the pool banner reports a different pool than the one under study

`gusto_validate_pool_cfg()` (`UcxCommunicator.cpp:1968`) prints

```
[GUSTO CFG] pool efectivo: slots=2 cap=1.1 MiB shadow=no total=2.1 MiB
```

while the same run's teardown reports `slots_total=8` and the backend was configured
8 × 32 MiB. Both are true: the banner is called from `init_rx_pool()`
(`UcxCommunicator.cpp:2084`) and describes the **frontend's own RX pool**, which is
deliberately 2 slots for the request/response pattern. The backend TX slot pool — the object
every pool experiment is about — is never printed by it.

The banner's own comment says it exists so that "'I set it' and 'it read it' are not
indistinguishable from outside, which is exactly how an entire campaign ended up measuring the
AM path while believing it measured the pool". The guard is real; it just guards the other
pool. `docs/HANDOFF_2026-08-02_tarde.md` §3.2 tells the reader to trust it for the pool
configuration. **Status: reporting defect, fix pending (Phase 9).**

## 1.3 F-03 — the `_ptsz` surface is 5 real entry points against 49 silent stubs

`plugins/cudart/frontend/CudaRt_stubs_compat.cpp` defines 49 `_ptsz` symbols as
zero-argument functions returning `cudaErrorNotSupported` (71). `STUB_LOG` compiles to a
no-op unless `GVIRTUS_LOG_STUB_CALLS` is defined, so **they are silent by default**.

Implemented for real (forwarding to the legacy entry point):

`cudaLaunchKernel_ptsz`, `cudaMemcpyAsync_ptsz`, `cudaMemsetAsync_ptsz`,
`cudaStreamQuery_ptsz`, `cudaStreamSynchronize_ptsz`.

Stubbed, among the 49, and directly relevant to the workloads in the paper:

`cudaGraphLaunch_ptsz`, `cudaGraphUpload_ptsz`, `cudaEventRecord_ptsz`,
`cudaEventRecordWithFlags_ptsz`, `cudaStreamWaitEvent_ptsz`, `cudaLaunchHostFunc_ptsz`,
`cudaMemcpy2DAsync_ptsz`, `cudaMemcpy3DAsync_ptsz`, `cudaMemcpyToSymbolAsync_ptsz`,
`cudaMemcpyFromSymbolAsync_ptsz`, `cudaMallocAsync_ptsz`, `cudaFreeAsync_ptsz`.

A translation unit compiled with `nvcc --default-stream per-thread` has its `cudaEventRecord`,
`cudaStreamWaitEvent` and `cudaGraphLaunch` calls redirected by the CUDA headers to the
`_ptsz` names. Against this frontend those calls return 71 without a log line.

This is a **candidate root cause for the llama abort** — llama.cpp uses both CUDA graphs and
the per-thread default stream — but it is **not yet demonstrated**, and the single-threaded
reproducer above passes precisely because it exercises none of the stubbed entry points.
Phase 1 tests it directly. **Status: lead, unproven.**

Second-order concern, also unproven: the stubs are declared with an empty parameter list. In
C that is a call with unspecified arguments, so the ABI happens to work for the return-71
path, but it defeats any type checking at the call site and would misbehave under a
control-flow-integrity build.

## 1.4 F-04 — `AblationGate.h` diverges, and the stale copy is on the side that prints

`src/` and `plugins/` are byte-identical between the two hosts. `include/` is not:
`include/gvirtus/communicators/AblationGate.h` differs, and the difference is exactly the
2026-08-02 afternoon fix that stops the gate announcing "no injection" during runs that do
inject (`hold_ack` / `epoch_ack` / `epoch_ack_idx` / `slow_ack` are read directly by
`UcxCommunicator.cpp`, not by the enum).

The fix is on **dpu-02**. `slow_ack` is a **backend** fault, so the banner is emitted by
**dpu-01**, which carries the pre-fix copy. The correction never reached the host that needed
it. dpu-01 additionally has 13 tracked files modified and uncommitted.

Preserved before any change: `es-dpu-01:~/asplos_preserve_0802/dpu01_uncommitted.diff`
(1581 lines, 13 files, +925/−74) plus copies of both untracked headers.

## 1.5 Environment notes carried forward

- **Different NVIDIA drivers per host** (backend 580.95.05, frontend 560.35.05). Phase 7's
  visibility statements are about the backend driver only.
- **`es-dpu-02:/home` at 95 %, 15 GiB free.** Bulk raw goes to dpu-01; soak needs a space
  check first.
- The 44 orphan `syssample.py` processes that contaminated the 2026-08-02 measurements are
  **gone** (`pgrep` count 0). That confound is closed.
- `lib/` and `build/` agree by md5 for the four communicator/frontend libraries, and
  `lib/frontend/libcudart.so.12` — the shim the applications actually `LD_PRELOAD` — matches
  `build/plugins/cudart/libcudart.so.12`. The load path is clean.
- `lib/frontend/` also holds nine `libgvirtus-plugin-*.so` dated 11–12 May. These are backend
  handler plugins that the frontend does not load; they are stale clutter and a trap for a
  reader. Flagged for Phase 10, not touched now.
- A `UCX endpoint close failed: Invalid parameter` appears on backend-side teardown. Noted for
  Phase 4; not yet investigated.
