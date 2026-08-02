---
title: "ASPLOS correctness & evaluation campaign — log"
date: "2026-08-02"
---

Chronological journal. One block per unit of work. Newest last.

---

## 2026-08-02 — Block 0.1 — Inventory: git, hosts, environment

**Objetivo.** Establish exactly what is on disk on both hosts before editing anything.

**Hipótesis.** The two source trees are in sync, as `docs/HANDOFF_2026-08-02_tarde.md` §4
states ("dpu-01 has the same sources synced but uncommitted").

**Comandos.**
```
secure-machine-access es-dpu-0{1,2} 'cd ~/GVirtuS && git branch --show-current; git log --oneline; git status --porcelain'
find src include plugins -name '*.cpp' -o -name '*.h' -o -name '*.cu' | sort | xargs md5sum | md5sum   # per host
bash ~/capture_env.sh                                                                                   # per host
```

**Resultado.**
- Both hosts on `exp/lazy-pool-v2` but with **divergent histories** (dpu-02 `3b05300`, three
  commits ahead; dpu-01 `1afa114`, with 13 tracked files modified and uncommitted).
- `src/` identical (`00ca4d25…`), `plugins/` identical (`86eb6306…`), **`include/` differs**.
- The only real divergence is `include/gvirtus/communicators/AblationGate.h`, and the host
  carrying the older copy is the one that emits the affected banner. → **F-04**.
- Environment captured for both hosts. Drivers differ (580.95.05 backend / 560.35.05
  frontend); `es-dpu-02:/home` at 95 %.
- The 44 orphan samplers reported open in the previous handoff are gone (count 0).

**Evidencia.** `results/asplos_campaign/environment/{env_dpu01.txt,env_dpu02.txt}`;
`es-dpu-01:~/asplos_preserve_0802/dpu01_uncommitted.diff`.

**Interpretación.** The handoff's sync claim is **not** true at the byte level. It was true
for `src/` and `plugins/`, which is where the reader would look, and false for the one header
that governs fault injection.

**Problemas abiertos.** dpu-01's 13 modified files are preserved but not committed; the
branch has not yet been created there.

**Siguiente acción.** Baseline smoke test before any edit.

---

## 2026-08-02 — Block 0.2 — Baseline smoke test and two anomalies

**Objetivo.** Gate the campaign: prove connect / H2D / sync / D2H / teardown works.

**Hipótesis.** The system is healthy; this is a formality.

**Comandos.** `examples/rmatest/ptds_repro` in `ll33pq/cudf_gvirtus_dyncudf:cuda12.6` against
the live backend, `GVIRTUS_RMA_MIN_BYTES=8192`, GPUDirect and zerocopy on.

**Resultado.** **Gate passed** — all four stream classes ok, exit 0, clean teardown. Two
anomalies in the same output that the gate would not have caught:
1. `admit_rma=0 admit_am=25 rma_pct=0.00` — no operation took the RMA path at all.
2. `[GUSTO CFG] pool efectivo: slots=2 cap=1.1 MiB` against `slots_total=8` in the teardown.

**Evidencia.** `results/asplos_campaign/ptds/baseline_singlethread.log`.

**Interpretación.** A passing smoke test that exercises 0 % of the data path under study is
the failure mode this campaign exists to eliminate. Both anomalies were run down rather than
noted — see Block 0.3.

**Siguiente acción.** Explain both before proceeding.

---

## 2026-08-02 — Block 0.3 — Root-causing the two anomalies

**Objetivo.** Explain `rma_pct=0` and the pool-banner mismatch.

**Hipótesis.** (a) the placement policy, not the pool, rejected the transfers; (b) the banner
describes a different pool.

**Código leído.** `include/gvirtus/communicators/RmaPolicy.h`,
`src/communicators/ucx/UcxCommunicator.cpp:{1968,2084,4201-4262}`.

**Comandos.** Three-arm control, same binary and payloads, varying only the policy env:
```
(A) GVIRTUS_RMA_MIN_BYTES=8192
(B) A + GVIRTUS_RMA_SCALAR_FLOOR=8192
(C) A + GVIRTUS_RMA_POLICY=quadrant
```

**Resultado.** A → `admit_rma=0 / rma_pct=0.00`; B → `4 / 16.00 %`; C → `4 / 16.00 %`.
Both hypotheses confirmed. → **F-01**, **F-02**.

**Evidencia.** `results/asplos_campaign/environment/f01_admission_control.log`.

**Interpretación.** `GVIRTUS_RMA_MIN_BYTES` gates nothing on the frontend; the gate is
`GVIRTUS_RMA_SCALAR_FLOOR` (default 4 MiB). The handoff's knob table is wrong on this row.

I then checked whether this had silently invalidated the previous campaign's saturation and
epoch cells. **It had not** — those benches default to 8/64 MiB payloads, above the scalar
floor, and their raws show `rma_pct` 11–24 %. Recording the negative result explicitly: my
first reading of the blast radius was too pessimistic, and the earlier conclusions stand.

**Problemas abiertos.** Six harnesses set `GVIRTUS_RMA_MIN_BYTES` and no policy. They were
saved by their payload sizes, not by construction. Phase 5.2 works entirely below 4 MiB and
must set the policy explicitly and assert `admit_rma > 0` per run.

**Siguiente acción.** Map the PTDS surface, since Phase 1 is the next priority.

---

## 2026-08-02 — Block 0.4 — The `_ptsz` surface

**Objetivo.** Locate every path that handles special stream sentinels.

**Hipótesis** (inherited from the previous session's reproducer comments): GVirtuS implements
only three `_ptsz` variants, and `cudaStreamSynchronize_ptsz` does not exist.

**Comandos.** `grep -rn 'ptsz|PerThread|streamLegacy' --include=*.cpp --include=*.h src include plugins`;
`nm -D --defined-only lib/frontend/libcudart.so.12`.

**Resultado.**
- `cudaStreamPerThread` and `cudaStreamLegacy` appear **nowhere** in the source. The sentinels
  `0x1`/`0x2` travel as ordinary opaque handles: the frontend sends them with
  `AddDevicePointerForArguments`, the backend reads them with `Get<cudaStream_t>()` and calls
  the CUDA runtime with the same raw value — so `0x2` resolves against the **backend worker
  thread's** per-thread stream, not the client thread's.
- **5 real `_ptsz` entry points, 49 silent stubs** returning `cudaErrorNotSupported`. → **F-03**.
- The inherited hypothesis is **wrong on one point**: `cudaStreamSynchronize_ptsz` exists
  (`CudaRt_stream.cpp:225`) and forwards to the legacy implementation. Corrected in the
  results document.

**Interpretación.** The stubs include `cudaGraphLaunch_ptsz`, `cudaEventRecord_ptsz` and
`cudaStreamWaitEvent_ptsz`. A PTDS-compiled translation unit reaches them through the CUDA
headers and gets 71 with no log line. That is a plausible route to the llama abort — llama.cpp
uses graphs and PTDS together — and it is **unproven**: the single-threaded reproducer passes
because it touches none of them.

**Problemas abiertos.** Whether the backend's frontend-thread → worker-thread mapping is
stable enough that `0x2` is *accidentally* correct for the 5 implemented entry points is not
established. That is the Phase 1 question.

**Siguiente acción.** Phase 1: build `tests/semantic/ptds_conformance.cu` across the matrix
1/2/8/32 threads, native and Gusto, including the stubbed entry points.
