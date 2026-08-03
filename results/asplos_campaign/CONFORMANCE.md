---
title: "Semantic conformance: three failures already in the data, and why none of them can be read yet"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

Phase 2 of the campaign. Its acceptance criterion is *"every listed property has a row; 0
mismatches on the full system; injected-wrong variants fail and only they."*

**Status: the suite is built and compiles; execution is pending a free GPU.** What follows is
already established: an audit of the 58 conformance rows that exist, which contains **three
failures nobody has written up**, and the reason none of them can currently be interpreted.

# 1. What the existing data already says

`results/asplos_campaign/ptds/` holds 94 files and 58 `CSVROW` rows from
`ptds_conformance.cu`, across two compilations -- `handle` (legacy entry points, sentinel `0x2`
passed explicitly) and `ptsz` (compiled `--default-stream per-thread`, which redirects to the
`_ptsz` surface).

| property | variant | passes | verdict |
|---|---|---:|---|
| `order_ptds` | handle / ptsz | **11/11** / **7/7** | ordering holds |
| `order_null` | handle / ptsz | 1/1 / 5/5 | holds |
| `order_legacy` | handle / ptsz | 1/1 / 5/5 | holds |
| `order_explicit` | handle / ptsz | 1/1 / 5/5 | holds |
| `event_crossstream` | handle | 3/3 | holds |
| `event_crossstream` | **ptsz** | **5/7** | **fails intermittently**, "unrecognized error code" |
| `driver_ptds` | handle | 3/3 | holds |
| `driver_ptds` | **ptsz** | **0/3** | **fails always**, `cudaErrorNotSupported` |
| `graph_ptds` | **handle and ptsz** | **0/3 and 0/3** | **fails in BOTH**, `cudaErrorStreamCaptureInvalidated` |

**Stream ordering is solid** -- 30 of 30 rows across four stream kinds and both compilations.
That is the property the whole remoting design most obviously risks, and it holds.

**The three failures are the finding, and they are not equivalent:**

1. **`driver_ptds` under `ptsz`: 0 of 3, `cudaErrorNotSupported` (71).** This is the signature of
   a **silent stub** -- the driver-API `_ptsz` surface returning "not supported" rather than
   forwarding. It matches F-03 of the inventory: the `_ptsz` surface was 5 real entry points
   against 49 stubs. The runtime-API side was fixed in phase 1 (`CudaRt_ptsz.cpp`, 17 forwards);
   **the driver-API side was not**, and this row is the evidence.
2. **`event_crossstream` under `ptsz`: 5 of 7, "unrecognized error code".** Intermittent, and
   the error is not even a named CUDA code -- which usually means a value crossed the wire that
   the client could not map. Intermittency plus an unmappable code is the more worrying of the
   three.
3. **`graph_ptds`: 0 of 3 in *both* compilations, `cudaErrorStreamCaptureInvalidated`.** Failing
   in `handle` too is what makes this one different: the `handle` variant does not touch the
   `_ptsz` surface at all.

# 2. Why none of the three can be interpreted today

**There is no native control.** The file named `baseline_singlethread` is not one: it runs
`ptds_repro` **through GVirtuS** (its own JSON records `GVIRTUS_RMA_MIN_BYTES` and an
`LD_PRELOAD`), and it emits no `CSVROW` rows at all.

Without a native arm running the same binary, a failure is ambiguous in exactly the way that
matters:

| observed | possible readings |
|---|---|
| fails on Gusto | a conformance defect **or** a test bug **or** a CUDA-version behaviour |
| fails on Gusto and native | a test bug or CUDA behaviour -- **not** a remoting defect |
| passes native, fails Gusto | **a conformance defect**, unambiguously |

`graph_ptds` is the sharp case. It fails in the `handle` variant, which exercises no `_ptsz`
entry point, so either graph capture is broken through remoting **or the test's capture
sequence is itself invalid**. A 30-second native run separates those. Until it is run, this
document does **not** claim a graph-capture defect, and neither should the paper.

# 3. What was built for it

**`tests/semantic/semantic_conformance.cu`** (new; `ptds_conformance.cu` is left untouched
because 58 published rows are tied to its binary). Same `CSVROW` contract, so one analysis reads
both. It targets what *this* architecture risks rather than CUDA in general:

| property | what it catches |
|---|---|
| `roundtrip_pinned` / `roundtrip_pageable` | byte fidelity at **nine sizes straddling all four placement thresholds** (H2D pinned 8 KiB, H2D pageable 1 MiB, D2H pinned 1 MiB, D2H pageable 2 MiB, plus the 4 MiB scalar floor). A placement bug appears only in the size class where the path changes, so a single-size test would never see it |
| `pinned_equiv` | the same bytes through pinned and pageable memory must agree. Between 8 KiB and 1 MiB the two take **different transports** (RMA against active messages), so this compares the fast path against the slow one with the answer known |
| `d2d_fidelity` | device-to-device, which on this backend goes through the GPU scratch -- the root cause of a real corruption in an earlier campaign |
| `memset_ordered` | `cudaMemsetAsync` ordered against a later kernel on the same stream |
| `event_query` | `cudaEventQuery` returns Success after synchronising; elapsed time finite and non-negative |
| `error_sticky` | `cudaGetLastError` **clears** and `cudaPeekAtLastError` does **not**; an invalid launch config yields the documented code |

Both compilations build clean.

**`~/conformidad_run.sh`** runs any suite in **three arms** -- `native`, `gusto_handle`,
`gusto_ptsz` -- with the reading rule stated in the script itself, plus an optional fourth arm
`GVS_ABLATE=pointer_keyed` as the injected-wrong control. That control is not decoration: the
criterion demands *"injected-wrong variants fail and only they"*, and without it "everything
passes" cannot distinguish a correct system from a test that proves nothing.

# 4. What is claimed, and what is not

**May be claimed now.** That stream ordering holds across four stream kinds and both
compilations, 30 rows, 0 mismatches. That the driver-API `_ptsz` surface returns
`cudaErrorNotSupported` and is therefore still stubbed, which the phase-1 fix did not cover.

**May not be claimed.** Anything about `graph_ptds` or `event_crossstream` until the native arm
runs. In particular the paper must **not** state that graph capture is broken through remoting:
it fails in a variant that does not use the `_ptsz` surface, so a test bug is at least as likely
until a control says otherwise.

**Pending, and it is the whole of the remaining work**: run the three arms, add the fourth
(injected-wrong), and re-read the three failures against the native column.

Data: `results/asplos_campaign/ptds/` (58 rows audited above);
`results/asplos_campaign/semantic_conformance/` (empty until execution). Suite
`tests/semantic/semantic_conformance.cu`, runner `~/conformidad_run.sh`.
