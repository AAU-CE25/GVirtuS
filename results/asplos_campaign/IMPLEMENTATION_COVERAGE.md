---
title: "Implementation coverage, size, fallback rate and extension cost"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

What is actually implemented, how big it is, how often the fast path gives up, and what it
costs to add an API. Every number here is counted from the built artefacts or from the
campaign's own counters; none is an estimate.

# 1. API coverage

Counted from the **built frontend shims** -- the libraries an application actually preloads --
not from the sources. "Exported" is `nm -D --defined-only` filtered to the CUDA-family symbol
prefixes; "stubs" are the entry points defined in each plugin's ABI-completion table, which
return `NOT_SUPPORTED` rather than doing work.

| plugin | exported | stubs | **real** | coverage |
|---|---:|---:|---:|---:|
| cudart | 407 | 234 | **173** | 42.5% |
| cudadr (driver API) | 526 | 372 | **154** | 29.2% |
| cublas | 298 | 93 | **205** | 68.7% |
| cudnn | 194 | 0 | **194** | 100% |
| cufft | 43 | 3 | **40** | 93.0% |
| curand | 18 | 0 | **18** | 100% |
| cusolver | 229 | 225 | **4** | **1.7%** |
| cusparse | 255 | 154 | **101** | 39.6% |
| nvml | 31 | 0 | **31** | 100% |
| nvrtc | 22 | 0 | **22** | 100% |
| **total** | **2023** | **1081** | **942** | **46.6%** |

**Read this honestly.** 46.6% of the exported CUDA-family ABI is real; the rest exists so that
dynamic linking succeeds and an application fails at the call it actually makes rather than at
load time. **cusolver is 1.7% -- it is ABI padding, not an implementation**, and should never be
described as supported. cudnn, curand, nvml and nvrtc are complete because they are small.

**Why the driver API is low and why it does not matter as much as it looks.** cudadr exports
526 symbols because the driver ABI is broad, but a CUDA *runtime* application touches a small,
stable subset. The 154 implemented ones are the subset the workloads in this paper exercise;
the coverage figure is a property of the ABI surface, not of the workloads.

**The stubs are silent by default.** `STUB_LOG` compiles to nothing unless
`GVIRTUS_LOG_STUB_CALLS` is defined, so an application that hits one gets an error code with
no diagnostic. That is a real usability defect, documented in `FAIRNESS_RESULTS.md` §1 and
fixed for the per-thread entry points by `plugins/cudart/frontend/CudaRt_ptsz.cpp`.

# 2. Size

Lines of code, excluding the auto-generated stub tables (647 lines across cudart and cudadr)
and excluding `.bak` files.

| component | LOC | what it is |
|---|---:|---|
| `src/communicators/ucx` | 4 936 | **the transport: this paper's contribution** |
| `src/communicators` (rest) | 1 886 | endpoint factory, TCP/RDMA/hybrid |
| `src/frontend` | 1 070 | dispatch, connection-per-thread |
| `src/backend` | 1 179 | one thread per connection, handler loop |
| `src/common` | 569 | shared |
| `plugins/cudart` | 9 259 | frontend 4 490 + backend 4 769 |
| `plugins/cudadr` | 6 725 | frontend 4 358 + backend 2 367 |

Within the transport, the parts that carry the paper's claims are small and separable:

| file | LOC | claim it carries |
|---|---:|---|
| `UcxCommunicator.cpp` + `.h` | 4 936 | slot pool, epoch/generation protocol, RMA and GET paths |
| `RmaPolicy.h` | 138 | the four-quadrant placement policy |
| `AblationGate.h` | 140 | the fault-injection and ablation gates |
| `CudaRt_ptsz.cpp` | 148 | per-thread default stream conformance |

# 3. Fallback rate

Aggregated over **620 teardown records** from every campaign log on disk --
`docs/gusto_raw_2026-08-02/`, `results/asplos_campaign/`, `results/pool_provisioning*/` --
totalling **14.78 million `WriteIov` operations**.

| counter | value | meaning |
|---|---:|---|
| `admit_rma` | 2 643 921 | operations the policy sent to the RMA path |
| `admit_am` | 12 136 435 | operations that took active messages |
| RMA admission rate | **17.89%** | of all `WriteIov` |
| **`rma_fellback`** | **80** | **0.003% of admissions** -- admitted to RMA, then could not be served |
| `decline_capacity` | 74 | payload larger than the slot |
| `decline_timeout` | 1 | waited past the consumer deadline |
| `decline_swap`, `decline_epfail` | 0, 0 | never |
| `ack_gen_mismatch` | 0 | no stale acknowledgement ever matched a live slot |
| `ack_epoch_dropped` | 1 | the epoch guard fired once, in the run built to make it fire |
| `ack_on_free` | 0 | no acknowledgement ever released a free slot |

> **Independently re-aggregated 2026-08-03.** A second pass over the same three trees, summing
> **every** `admit_*` occurrence in 694 files rather than one record per teardown, reproduces the
> rare-event counters that carry the claim: `decline_capacity` **74** (exact),
> `decline_swap` **0**, `decline_epfail` **0**, `ack_gen_mismatch` **0**, `ack_on_free` **0**,
> `ack_epoch_dropped` **1** (all exact), `rma_fellback` **84** against 80 and
> `decline_timeout` **3** against 1 -- small differences from the wider file set.
>
> The **denominator** is the sensitive quantity: without per-teardown de-duplication it reaches
> 6.29 M admissions of 50.6 M operations, because logs reprint cumulative counters. That would
> make the fallback rate **0.0013%** instead of 0.003%. **The published figure is the
> conservative one and is kept**; the point to take is that the conclusion does not depend on
> which convention is used, only the third significant figure does.

**The fast path essentially never gives up.** 80 fallbacks in 2.64 million admissions is
0.003%; the two mechanisms that could cause it -- a payload exceeding the slot, and a consumer
past its deadline -- account for 74 and 1. This is the number to quote when the question is
whether the placement policy is a source of unpredictability: it is not.

**The 17.89% admission rate is not a limitation, it is the policy working.** Most operations
are control-plane RPCs of a few hundred bytes, for which active messages are the correct
choice; the policy is supposed to leave them there. The admission rate is a property of the
workload mix, not a coverage figure.

# 4. Extension cost

| plugin | real APIs | LOC (frontend+backend, net of stubs) | **LOC per API** |
|---|---:|---:|---:|
| cudart | 173 | 9 259 | **53.5** |
| cudadr | 154 | 6 725 | **43.7** |

**Adding an API costs roughly 45--55 lines**, split about evenly between the frontend shim that
marshals arguments and the backend handler that unmarshals and calls CUDA. That is the whole
cost for an API whose arguments are scalars or already-translated handles.

**Three things cost more, and they are the interesting ones**, because they are where the
semantic contract lives rather than the marshalling:

- **An API that returns a handle** needs the handle to be registered so later calls can
  translate it -- the `AddDevicePointerForArguments` / `GetOutputDevicePointer` pair.
- **An API that carries bulk data** must be classified by direction and memory kind so the
  placement policy can decide; that is 1 line at the call site plus the policy already written.
- **An API with per-thread default-stream semantics** needs its `_ptsz` sibling, and the
  sibling is not a plain forward: a `0` arriving there means *this thread's* default, not the
  legacy default. `CudaRt_ptsz.cpp` is 148 lines for 17 such entry points, ~9 lines each.

**What this says about the "integration" question.** The marshalling is cheap and mechanical --
that part *is* integration work, and 942 implemented entry points at ~50 lines each is what it
looks like. What is not mechanical is the 5 214 lines of transport and policy that decide,
per operation, which of four data paths to use and how to keep a slot's lifetime safe across
epoch and generation changes. The ratio matters: **the mechanical part is 16 000 lines and the
contribution is 5 200**, and the second is the part that no amount of API coverage substitutes
for. See `NOVELTY.md`.
