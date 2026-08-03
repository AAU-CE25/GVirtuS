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

# 3b. Executed 2026-08-03 -- one defect found and fixed, one confirmed, one dissolved

The knee sweep was cut after its N=4 block to free both GPUs, and the suite ran.

## The defect the new suite found, and the fix

First run, single thread, three arms: **native 7/7 and `gusto_handle` 7/7 passed; `gusto_ptsz`
failed 4 of 7** -- and precisely the four properties that use **synchronous** memory calls.
`memset_ordered` passed because it uses the *Async* variants. Every failure carried
`cudaErrorNotSupported` (**71**) at the H2D call: the silent-stub signature.

The native arm is what made this readable in one run. A failure in only the `ptsz` compilation,
with native and `handle` clean, cannot be a test bug.

`nm -u` on the test binary named the missing symbols exactly:

    U cudaMemcpy_ptds@libcudart.so.12
    U cudaMemset_ptds@libcudart.so.12

**The synchronous per-thread surface uses the `_ptds` suffix, not `_ptsz`.** Phase 1 fixed
`_ptsz` -- the *asynchronous* suffix, 17 forwards -- and left `_ptds` untouched. Those symbols
were not absent: they were **eight silent stubs** in `CudaRt_stubs_compat.cpp`, each returning
`CUDART_STUB_NOT_SUPPORTED`. A program compiled `--default-stream per-thread` therefore received
71 from `cudaMemcpy` and, if it did not check the return code, carried on with uncopied data.

**Fix:** eight real forwards added to `CudaRt_ptsz.cpp` (`cudaMemcpy_ptds`, `cudaMemcpy2D_ptds`,
`cudaMemcpy3D_ptds`, `cudaMemcpyToSymbol_ptds`, `cudaMemcpyFromSymbol_ptds`, `cudaMemset_ptds`,
`cudaMemset2D_ptds`, `cudaMemset3D_ptds`) and the eight corresponding stubs retired -- commented
out with the reason rather than deleted. They take no stream argument, so the forward is direct:
the frontend's implementation is already host-synchronous, which is the observable contract of a
blocking copy.

**After the fix: 21/21 at one thread and 28/28 at eight**, across native, `gusto_handle` and
`gusto_ptsz`. The test failed before and passes after, which is the campaign's standing rule.

## The three old failures, now resolved against a native control

Re-ran `ptds_conformance.cu` through the same three arms -- the control §2 said was missing:

| property | native | `gusto_handle` | `gusto_ptsz` | verdict |
|---|---|---|---|---|
| `order_ptds` / `order_null` / `order_legacy` / `order_explicit` | PASS | PASS | PASS | conformant |
| **`event_crossstream`** | PASS | PASS | **PASS** | **dissolved** -- it was the same code 71, fixed above |
| **`driver_ptds`** | PASS | PASS | **FAIL, 71** | **defect of the driver-API `_ptds` surface**, still stubbed |
| **`graph_ptds`** | **PASS** | **FAIL** | **FAIL** | **a real conformance defect of the remoting layer** |

**`graph_ptds` is the one that needed the control and got it.** §2 refused to call it a defect
because it failed in the `handle` variant too, which touches no per-thread surface -- a test bug
was at least as likely. **Native now passes it**, so the test is sound and
**`cudaErrorStreamCaptureInvalidated` is produced by the remoting layer**: stream capture does
not survive the round trip. This is a genuine, newly-established defect and it is *not* fixed.

## 3c. The two remaining defects, attacked

### `driver_ptds`: fixed

Same shape as the `_ptds` defect above, one API down. The test failed at
`cuStreamSynchronize(CU_STREAM_PER_THREAD)` with **801, `CUDA_ERROR_NOT_SUPPORTED`**, and
`CudaDr_compat_stubs.cpp` held **56 driver `_ptsz` stubs** against just two real forwards.

**Only two were implemented**, and the restraint is the point: `cuStreamSynchronize_ptsz` and
`cuStreamQuery_ptsz`, the two whose **base function actually exists** in this frontend. Of the
eight candidates checked, `cuStreamWaitEvent`, `cuMemcpyAsync`, `cuMemcpyDtoDAsync_v2` and
`cuMemsetD8Async` are **not implemented at all**, so forwarding them would replace an honest
"not supported" with a failure further down and harder to diagnose. The other 52 stubs stay.

**`driver_ptds` now passes in all three arms.**

Build note, because it cost twenty minutes: the driver plugin **cannot be built on the host** --
`CudaDr.h` includes `GL/gl.h`, which is not installed -- so it is built inside
`aauce25/gvirtus-dev`. That is also why `build/plugins/cudadr/` was full of root-owned objects,
and why removing them broke `flags.make` until `cmake` regenerated it.

### `graph_ptds`: localised, not fixed

A four-cell probe (`tests/semantic/graphprobe.cu`) captures the same stream with different
contents:

| capture contains | native | Gusto |
|---|---|---|
| nothing | pass | **pass** |
| a kernel only | pass | **pass** |
| **a memcpy only** | pass | **FAIL** `cudaErrorStreamCaptureInvalidated` |
| kernel + memcpy | pass | **FAIL** |

**Stream capture survives remoting. `cudaMemcpyAsync` inside a capture does not.** The error is
raised by the copy itself -- `cudaGetLastError` immediately after already reports it -- not by
`cudaStreamEndCapture`. This also explains why the CUDA-graphs work on llama holds: llama
captures **kernels**.

**The cause, and why the fix is design work rather than a patch.** The frontend has no CUDA
context, so `BeginCapture` is forwarded and the capture lives on the **backend**. The backend's
memcpy handler synchronises -- `cudaStreamSynchronize(0)`, `cudaStreamSynchronize(stream)` and
`cudaDeviceSynchronize()` all appear in `CudaRtHandler_memory.cpp` -- and **synchronising a
stream that is being captured invalidates the capture**. Suppressing those calls while a capture
is active is the obvious fix, and it is not safe as written: those synchronisations are what
enforce the slot-lifetime invariants of `CONTRACTS.md` §4. Under capture the copy is **deferred
to graph launch**, so the source slot would have to stay reserved from capture until launch --
a lifetime the current protocol does not model at all.

So: **fixable, with a scoped change to the lifetime protocol; not fixable by deleting a
synchronise.** Recorded as measured-and-localised, not fixed.

## The injected-wrong control: it did not fire, and that is reported rather than glossed

The criterion asks that wrong variants fail **and only they**. A fourth arm ran with
`GVS_ABLATE=pointer_keyed`. It **activated** -- the banner is in the log and the registration
cache ran address-keyed, 50 hits against 110 misses -- and **all 28 properties still passed**.

The counters say why: **9 RMA reservations in the whole run.** These properties do not
free-then-reallocate at the same address with transfers above the RMA floor often enough to
exercise the ablated mechanism. **So this suite does not satisfy the "only they fail" half of
the criterion**, and saying otherwise would be the exact error the criterion exists to prevent.

The control does exist, one document over: `SLOT_LIFETIME_RESULTS.md` §B.1 drives
`pointer_keyed` against realloc-at-the-same-VA from the same binary and gets **6 of 8 expected
failures in both directions**, against 0 of 8 for the full protocol. **The criterion is met at
campaign level, not inside this suite.**

## A test-invocation error of mine, recorded

The first `ptds` run through the new runner reported every property failing **including
native**, with `cudaErrorInvalidConfiguration`. Not a system defect: the two suites have
different signatures -- `semantic_conformance <threads> <iters> <seed>` against
`ptds_conformance <threads> <iters> <BYTES> <seed>` -- and the runner passed the seed in the
bytes position, launching one-byte kernels. **The native arm is what exposed it**: a defect that
appears in native too is the harness, not the system. The runner now builds its argument list
per suite, with the reason in a comment.

# 4. What is claimed, and what is not

*(Superseded by §3b, which ran the control. Kept because it records what was and was not
claimable before the measurement, and the second paragraph is exactly the claim the control
would have refuted had it gone the other way.)*

**Claimed after execution:**

- **Stream ordering is conformant.** Four stream kinds, both compilations, native control clean.
- **The synchronous `_ptds` surface was eight silent stubs returning 71, and is now
  implemented.** Found by this suite, localised by `nm -u`, fixed, and the suite goes 21/21 and
  28/28 afterwards.
- **`event_crossstream` is conformant**; its earlier intermittent failure was the same defect.
- **The driver-API `_ptsz` surface was 56 stubs; the two whose base exists are now implemented**
  and `driver_ptds` passes in all three arms. The other 52 are left stubbed on purpose -- their
  base functions do not exist, so a forward would hide the real gap.
- **Stream capture does not survive remoting.** `graph_ptds` passes native and fails **both**
  Gusto arms with `cudaErrorStreamCaptureInvalidated`. Newly established, not fixed.

**Still may not be claimed:**

- That this suite satisfies *"injected-wrong variants fail and only they"*. It does not: the
  `pointer_keyed` arm activates and every property still passes, because the run makes only 9
  RMA reservations. The criterion is met by `SLOT_LIFETIME_RESULTS.md` §B.1, elsewhere.
- Any cause for the graph-capture defect. It is measured, not diagnosed.
- Anything about threads beyond 1 and 8, or about the `ptds` suite at 8 threads (run at 1 only).

Data: `results/asplos_campaign/ptds/` (58 rows audited above);
`results/asplos_campaign/semantic_conformance/` (empty until execution). Suite
`tests/semantic/semantic_conformance.cu`, runner `~/conformidad_run.sh`.
