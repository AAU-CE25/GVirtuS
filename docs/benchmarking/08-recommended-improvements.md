# 08 — Recommended improvements (GVirtuS, from the benchmark campaign)

Status: **one improvement prototyped and validated (kept in the tree); the rest are prioritized
recommendations.** Last updated: 2026-07-18.

This doc collects concrete engineering improvements the benchmark campaign surfaced, ordered by
**bang-for-buck**. The headline finding across the suite: GVirtuS is excellent for
compute-/bandwidth-bound work (miniBUDE ~0 % overhead, BabelStream ~93–95 % of native at large
sizes) but collapses on **launch-/RPC-bound** work (LLM decode ~79× slower) because **every CUDA
call is a synchronous, blocking network round-trip and each kernel launch is amplified into
~6 RPCs**. The improvements below attack that amplification.

## The core problem: RPC amplification per kernel launch

Measured from the backend dispatch log (doc 07): each `kernel<<<grid,block>>>(...)` becomes
**~6.2 blocking RPCs** over GVirtuS:

| routine | RPCs per launch | should be |
|---------|----------------:|-----------|
| `cudaGetDevice`             | ~1.8 | cached frontend-side (rarely changes) |
| `cudaGetLastError`          | ~1.3 | cached / coalesced |
| `__cudaPushCallConfiguration` | 1.0 | **local (thread-local stack)** — never an RPC |
| `cudaLaunchKernel`          | 1.0 | the only unavoidable one |
| `__cudaPopCallConfiguration`  | 1.0 | **local (thread-local stack)** — never an RPC |

LLM decode issues ~500 launches per token ⇒ ~3,100 serial RPCs per token ⇒ ~124 ms/token over
RDMA (~79× native). The transport is *not* the bottleneck (RDMA is only ~2.2× faster than TCP);
the **count** of serial round-trips is.

---

## Recommended improvement #1 — Local `__cudaPushCallConfiguration` / `__cudaPopCallConfiguration` (PROTOTYPED, KEPT, VALIDATED)

**What:** `__cudaPushCallConfiguration`/`__cudaPopCallConfiguration` only carry the launch config
from the `<<<>>>` syntax to `cudaLaunchKernel` via a **thread-local stack** — exactly how native
CUDA implements them. GVirtuS was remoting both as blocking RPCs even though the frontend
`cudaLaunchKernel` already re-serializes grid/block/shmem/stream to the backend. So those 2 RPCs
per launch were pure overhead.

**Change (kept):** `plugins/cudart/frontend/CudaRt_internal.cpp` — both functions now push/pop a
`thread_local std::stack<GvirtusCallConfig>` and return locally; **no RPC**. Fully
behaviour-preserving (identical semantics to native CUDA). Removes 2 of ~6.2 RPCs per launch
(→ ~4.2), predicting 6.2/4.2 = **1.48×**.

**Measured effect (matches the prediction almost exactly), proportional to how launch-bound the
workload is** — data in `data/prototype_pushpop_summary.csv`:

| workload | regime | metric | baseline | prototype | **speedup** |
|----------|--------|--------|---------:|----------:|------------:|
| llama TinyLlama-1.1B (RDMA)      | launch-bound          | token gen tg16 | 8.04  | 11.64 | **1.45×** |
| llama TinyLlama-1.1B (GPUDirect) | launch-bound          | token gen tg16 | 7.96  | 11.53 | **1.45×** |
| llama TinyLlama-1.1B (TCP)       | launch-bound          | token gen tg16 | 3.58  | 5.35  | **1.49×** |
| llama TinyLlama-1.1B (RDMA)      | launch-bound          | prompt eval pp8| 50.62 | 69.48 | **1.37×** |
| BabelStream 262 K (RDMA)         | launch-overhead-bound | Triad MB/s     | 30217 | 49404 | **1.64×** |
| BabelStream 33 M (RDMA)          | bandwidth-bound       | Triad MB/s     | 614358| 648637| 1.06× |
| miniBUDE bm1 (RDMA)              | compute-bound         | GFLOP/s        | 216.0 | 216.5 | 1.00× |

**Correctness verified:** llama.cpp still produces coherent output ("…the capital of France is
Paris"); miniBUDE `valid: true`. **No regression anywhere; large wins where launches dominate.**
The speedup tracks the workload's launch-boundedness perfectly: compute-bound (miniBUDE) 1.00×,
bandwidth-bound (large BabelStream) 1.06×, launch-overhead-bound (small BabelStream) 1.64×,
LLM decode (llama) 1.45×.

---

## Recommended improvement #2 — Cache/coalesce `cudaGetDevice` and `cudaGetLastError` (IMPLEMENTED, KEPT, VALIDATED)

The next ~3 of the remaining ~4.2 RPCs per launch are `cudaGetDevice` (~1.8) and `cudaGetLastError`
(~1.3), both cheap queries the app fires defensively around launches. The latency trace (doc 10)
showed these two alone were **58 % of all control-plane RPCs and 48 % of control-path time** during
llama decode.

**What these functions actually do (why they should never round-trip):**
- `cudaGetDevice(int*)` returns the calling host thread's *current device ordinal*. It is
  thread-local runtime state that only ever changes via `cudaSetDevice` — it does **not** touch the
  GPU. The frontend already knows the current device, so asking the backend is pure waste.
- `cudaGetLastError()` returns the *sticky last error* for the calling thread **and resets it to
  `cudaSuccess`**. `cudaPeekAtLastError()` returns it without clearing. The "last error" is simply
  the most recent non-success code from any runtime call — which the frontend already sees as the
  exit code of every remoted call.

**Change (kept):**
- `plugins/cudart/frontend/CudaRt_error.cpp` — added thread-local `cudart_state` (current device +
  sticky last error); `cudaGetLastError`/`cudaPeekAtLastError` now answer **locally, no RPC**.
- `plugins/cudart/frontend/CudaRtFrontend.h` — `Execute()` records every remoted call's exit code
  into the sticky last-error (`cudart_state::note_exit_code`), so the local error state is exactly
  what CUDA would report (behaviourally identical, sticky-until-cleared semantics preserved).
- `plugins/cudart/frontend/CudaRt_device.cpp` — `cudaGetDevice` returns the cached current device
  **locally, no RPC**; `cudaSetDevice` updates the cache on success.
- `plugins/cudart/cuda_internals/CudaRt_internal.h` — `cudart_state` accessor declarations.

**Measured effect (KEPT), on top of rec #1 — data in `data/llama/llama_optimizations_progression.csv`:**

| transport | baseline tg16 | + rec #1 (Push/Pop) | + rec #2 (this) | **total speedup** |
|-----------|--------------:|--------------------:|----------------:|------------------:|
| RDMA           | 8.04  | 11.64 | **40.42** | **5.03×** |
| RDMA+GPUDirect | 7.96  | 11.53 | **42.77** | **5.37×** |
| TCP            | 3.58  |  5.35 | **10.63** | **2.97×** |

Prompt-eval pp8 (RDMA): 50.6 → 69.5 → **250.0** t/s; (GPUDirect) → **301.0**. **Correctness verified
on all three transports** (llama.cpp generates "…the capital of France is Paris"). The latency trace
confirms the mechanism: `cudaGetDevice` **14,624 → 0** RPCs and `cudaGetLastError` **11,163 → 0**
RPCs. Per-launch RPCs dropped 6.2 → 4.2 (rec #1) → **~1.1** (rec #2, just `cudaLaunchKernel`),
matching the ~5× speedup. This is **~5× faster LLM inference from two cheap, fully
behaviour-preserving frontend changes and zero async machinery** — strong evidence that the GVirtuS
bottleneck is needless control-plane RPC count, not the transport.

> **GPUDirect honesty note:** the GPUDirect column is a *backend-`GPUDirect=enabled` config*, not a
> demonstration of NIC→GPU DMA for llama. The backend log shows the peer advertised **0 GPU-shadow
> slots** (frontend host-staged) and **zero** GPU-routing events — llama's control-plane-bound
> transfers don't exercise the data path, which is exactly why GPUDirect ≡ RDMA here. NIC→GPU DMA is
> demonstrated on bulk transfers (doc 04), not llama.

> **Correctness refinement (kept):** `note_exit_code` excludes `cudaErrorNotReady` from the sticky
> last error — `cudaStreamQuery`/`cudaEventQuery` return it while work is in flight and native CUDA
> does *not* latch it. (The GPUDirect crash that first surfaced this was actually the separate
> frontend-`GVIRTUS_GPUDIRECT` probe-poisoning bug, doc 05; GPUDirect works once that flag is unset
> on the frontend.)

Note the RDMA-vs-TCP gap *widens* after rec #2 (RDMA now 3.8× TCP vs 2.2× before): with fewer RPCs
per token, per-RPC tail latency matters more, so TCP's heavy tail (doc 10) hurts proportionally more.

---

## Recommended improvement #3 — Asynchronous dispatch (IMPLEMENTED, KEPT, VALIDATED)

`cudaLaunchKernel`, `cudaMemsetAsync`, and other stream-ordered calls are semantically
*fire-and-forget* until a sync point. GVirtuS blocked on each. A **non-blocking dispatch path**
(frontend enqueues, returns immediately, and only stalls at `cudaStreamSynchronize`/
`cudaDeviceSynchronize`/`cudaMemcpy`) overlaps RPCs and hides per-call latency. This was the single
biggest remaining structural lever for RPC-bound workloads and the one the project report lists as
future work.

**Design (dual-path async dispatcher, gated by `GVIRTUS_ASYNC_DISPATCH=1`):**
- The UCX AM envelope gained a **`kEnvelopeFlagNoResponse`** bit (reserved0 bit 0). A frontend
  `ExecuteAsync()` sends the request via `WriteIov` — which already waits for *local* send
  completion, so the request is safely on the wire and in-order — then **skips `Sync()` and the
  entire response read**, eliminating the backend-exec + response round-trip. The backend executes
  the routine and, seeing the flag, **sends no response**, keeping the strictly in-order
  request/response stream aligned.
- **Deferred-error reconciliation (correctness):** a failed async op is latched per-connection on the
  backend and folded into the `status_code` of the **next** response-bearing (sync) call, then
  cleared. On the frontend that flows through the existing sticky-last-error cache, so a later
  **local** `cudaGetLastError` still surfaces it — exactly CUDA's "async errors report at the next
  synchronization point" semantics. No error is silently dropped.
- **Curated allowlist only** (output-less, stream-ordered, small AM-path payloads):
  `cudaLaunchKernel`, `cudaLaunchKernelExC`, `cudaMemsetAsync`, `cudaEventRecord`,
  `cudaEventRecordWithFlags`, `cudaStreamWaitEvent`. `cudaMemcpyAsync` is deliberately **excluded**
  in v1 (its large H2D copies use the RMA slot-reuse path, whose round-robin slots assume synchronous
  consumption — async reuse could overwrite an unconsumed remote slot; it is also only ~374 calls/run
  vs ~10,960 launches, so negligible). D2H, malloc/free-async, and all queries stay synchronous.

**Change (kept):** `include/gvirtus/communicators/UcxAmProtocol.h` (flag),
`src/frontend/Frontend.cpp` (`ExecuteInternal`/`ExecuteAsync` split, fire-and-forget send path),
`include/gvirtus/frontend/Frontend.h`, `plugins/cudart/frontend/CudaRtFrontend.h`
(`ExecuteMaybeAsync` + `GVIRTUS_ASYNC_DISPATCH` gate), the allowlisted stubs, and
`src/backend/Process.cpp` (no-response skip + deferred-error fold).

**Measured effect (KEPT) — TinyLlama-1.1B Q4, `llama-bench -p 8 -n 16 -r 2`, backend at ERROR log
level (see the log-level caveat below), data in `data/llama/llama_async_dispatch.csv`:**

| transport | metric | sync (off) | async (on) | **speedup** |
|-----------|--------|-----------:|-----------:|------------:|
| RDMA            | token gen tg16 | 87.35 | **189.86** | **2.17×** |
| RDMA            | prompt eval pp8 | 558.85 | **1216.84** | **2.18×** |
| RDMA+GPUDirect  | token gen tg16 | 86.71 | **186.51** | **2.15×** |
| RDMA+GPUDirect  | prompt eval pp8 | 558.02 | **1214.67** | **2.18×** |

**Three findings from the sweep (all cross-checked):**
1. **The async speedup (~2.17×) is transport-independent** — identical on RDMA and RDMA+GPUDirect —
   because it removes per-launch RPC *round-trips* (count × latency), which is orthogonal to
   bandwidth or the data path.
2. **GPUDirect ≡ plain RDMA for llama**, with *and* without async (86.71≈87.35 off; 186.51≈189.86 on,
   within ~1%). The backend was `GPUDirect=enabled` with 2/2 GPU-shadow slots advertised, but llama's
   per-token transfers are tiny and host-sourced, so they never trip the ≥4 MB RMA GPU path — NIC→GPU
   peer-DMA has nothing to accelerate. (Confirms doc 07; GPUDirect's win is *bulk* transfer, doc 04.)
3. **Backend log level is a confound to watch:** DEBUG per-RPC logging on the hot path ~**halves**
   throughput. The async *ratio* is robust (2.16× at DEBUG, 2.17× at ERROR), but absolute numbers must
   be read at ERROR. An earlier DEBUG measurement (40.65→87.62) understated the absolute figures ~2×.

**Wire-level engagement verified:** with async ON the backend logged **14,139 `cudaLaunchKernel` as
fire-and-forget** ("no response sent") and **0** with async OFF; `cudaMemcpyAsync` stayed **synchronous**
(901 responses) exactly as the v1 allowlist intends — so the launches genuinely go async and nothing
silently falls back.

**Correctness verified (not assumed):** `llama-cli` still generates coherent text ("…the capital of
France is Paris. It is located in the Île-de-France"); the unit test
`examples/testing/test_async_dispatch.cu` (async kernel launches + memset + event + stream-wait, then
`cudaStreamSynchronize` + D2H readback) produces the **identical** result with the gate on and off.
No hang, no assert, exit 0. **vs bare metal (634.9 tg16): the gap closes from ~79× (pre-opts) to
~3.3× (189.86).**

> **Challenge / rework of the earlier `fix-async-calls` branch:** that branch proved the wire-level
> fire-and-forget mechanism but (a) only wired the trivial `cudaMemsetAsync` — never the actual
> `cudaLaunchKernel` bottleneck; (b) **dropped async errors silently** (returned exit 0 with no
> reconciliation); and (c) predated the current zero-copy AM path (WriteIov / `TryAcquireFrame` /
> direct buffers) and could not be merged. This implementation re-does it on the current path, adds
> the launch-kernel win and the deferred-error reconciliation, and gates it for safe A/B + fallback.

**Remaining gap (~7×) needs #4** (launch batching / CUDA-graph capture): fold a whole decode step's
launches into one RPC. v2 of the dispatcher (drop the per-send local `Sync()` via a header pool +
batched flush at sync points) is a smaller incremental lever on top of this.

## Recommended improvement #4 — Launch batching / CUDA-graph capture

Coalesce a whole decode step's launches into **one** RPC: either batch consecutive
`cudaLaunchKernel` calls into a single request, or fully support CUDA-graph capture/replay so a
repeated decode graph is captured once and replayed with a single round-trip per token. Combined
with #3 this is what would bring LLM inference from ~79× toward single-digit overhead.

## Recommended improvement #5 — Fix RMA registration cache so zero-copy is usable

`GVIRTUS_RMA_ZEROCOPY=1` currently OOMs because UCX can't create a registration cache in the
container ("could not create UCP registration cache: Unsupported operation") → every zero-copy
`ucp_put` re-registers and leaks NIC registrations (doc 05). Provide a working rcache (or a
GVirtuS-managed registration pool) so large transfers avoid the staging memcpy safely.

---

## Also worth hardening (from doc 05)
- **Backend listener non-recovery:** after a client crash or an unclean restart, stray
  `gvirtus-backend` processes keep the RoCE port bound ("Address already in use" /
  "ucp_listener_create failed: Device is busy"). Needs clean listener teardown / SO_REUSEADDR /
  process reaping so the backend recovers without a manual container rebuild.
- **GPUDirect first-connection cold-start stall:** the first llama-bench over a freshly-started
  GPUDirect backend occasionally stalls for minutes (cold NIC registration of the large rx_pool
  slots); a warm-up or lazy slot registration would smooth this.

## Fixes already landed (kept in tree)
- `cudaDeviceGetPCIBusId` missing backend handler (doc 05, doc 07).
- `cudaHostAlloc`/`cudaMallocHost` under-alignment → 256-byte `posix_memalign` (doc 05, doc 07).
- **Improvement #1** (local Push/Pop call configuration).
- **Improvement #2** (frontend-local `cudaGetDevice` / `cudaGetLastError` / `cudaPeekAtLastError`).
- **Improvement #3** (async dispatch: fire-and-forget `cudaLaunchKernel` & friends,
  `GVIRTUS_ASYNC_DISPATCH=1`, deferred-error reconciliation).
- Latency-trace instrumentation `GVIRTUS_LATENCY_TRACE` (doc 10).

**Combined result: at matched (ERROR) backend log level, the async dispatcher gives a further
**2.17×** over the rec#1+#2 synchronous path (RDMA token gen **87.35 → 189.86 t/s**), correctness
preserved. vs native (634.9 tg16) llama decode is now **~3.3× slower** (was ~79× before any opts).**
The remaining gap now needs the structural batching work (#4 launch batching / graph capture).
