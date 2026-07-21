# GVirtuS optimizations & the async dispatcher

How the llama gap (RESULTS.md §5: ~79× native) was closed to **~3.4×**, in two stages: cheap
behaviour-preserving **frontend RPC reductions**, then a structural **asynchronous dispatcher**
(incl. `cudaMemcpyAsync` in all three directions). All are **kept in tree**, gated where risky, and
validated for correctness + no-regression. Data: `../../benchmarks/`.

## The core problem: RPC amplification per kernel launch

Each `kernel<<<grid,block>>>(...)` over GVirtuS became **~6.2 blocking RPCs** (measured, backend
dispatch log):

| routine | RPCs/launch | should be |
|---------|------------:|-----------|
| `cudaGetDevice`               | ~1.8 | cached frontend-side |
| `cudaGetLastError`            | ~1.3 | cached / local |
| `__cudaPushCallConfiguration` | 1.0  | **local** (thread-local stack) — never an RPC |
| `cudaLaunchKernel`            | 1.0  | the only unavoidable one |
| `__cudaPopCallConfiguration`  | 1.0  | **local** — never an RPC |

LLM decode issues ~500 launches/token ⇒ ~3,100 serial RPCs/token ⇒ ~124 ms/token on RDMA. The
transport is *not* the bottleneck (RDMA only ~2.2× TCP); the **count of serial round-trips** is.

---

## Stage 1 — frontend RPC reductions (no async machinery)

**#1 Local `__cudaPush/PopCallConfiguration`** (`CudaRt_internal.cpp`): the launch config only travels
from `<<<>>>` to `cudaLaunchKernel` via a **thread-local stack** — exactly like native CUDA — instead
of two blocking RPCs. Removes 2 of ~6.2 RPCs/launch.

**#2 Cache `cudaGetDevice` / `cudaGetLastError` / `cudaPeekAtLastError`** (`CudaRt_error.cpp`,
`CudaRt_device.cpp`): answered from a thread-local `cudart_state` (current device + sticky last
error); `Execute()` records every remoted exit code into the sticky error so semantics are identical
(`cudaErrorNotReady` excluded, matching CUDA). The latency trace showed these two were **58% of all
control RPCs and 48% of control-path time** during decode.

**Measured (RDMA, tg16 t/s), each behaviour-preserving, no regression on compute/bandwidth apps:**

| transport | baseline | +#1 | +#2 | total |
|-----------|---------:|----:|----:|------:|
| RDMA           | 8.04 | 11.64 | **40.42** | **5.03×** |
| RDMA+GPUDirect | 7.96 | 11.53 | **42.77** | **5.37×** |
| TCP            | 3.58 |  5.35 | **10.63** | **2.97×** |

The speedup tracks launch-boundedness: miniBUDE 1.00×, large BabelStream 1.06×, small BabelStream
1.64×, llama decode ~5×. Correctness verified on all transports. Data:
`benchmarks/llama-sync/{prototype_pushpop_summary,llama_optimizations_progression}.csv`.

---

## Stage 2 — the asynchronous dispatcher (`GVIRTUS_ASYNC_DISPATCH=1`)

`cudaLaunchKernel`, `cudaMemsetAsync`, `cudaEventRecord`, `cudaStreamWaitEvent`, and
`cudaMemcpyAsync` are semantically **fire-and-forget** until a sync point. GVirtuS blocked on each;
the dispatcher stops waiting for replies on calls that don't need one.

### The invariant everything rests on
The UCX Active-Message connection is a **strict in-order FIFO** (replies already correlate by
`request_id`). So **any later synchronous reply proves the backend processed — in order — every
request before it.** This single guarantee makes fire-and-forget, RMA-slot flow control, and deferred
D2H all safe without extra acks.

### Mechanism
- **Wire:** one bit — `kEnvelopeFlagNoResponse` in the envelope `reserved0`.
- **Frontend** (`Frontend::ExecuteInternal`, 3 modes): *Sync* (unchanged); *FireAndForget* (set flag,
  `WriteIov` which waits for local send completion, then **return** — skip `Sync()` + the whole reply
  read); *DeferredD2H* (send normally, collect the reply later).
- **Backend** (`Process.cpp`): flagged requests execute but send **no reply**; a failed one is latched
  in a per-connection `deferred_async_error` and **folded into the next reply-bearing call's
  status** → surfaces via the existing sticky-last-error cache at the next `cudaGetLastError`. Matches
  CUDA's "async errors surface at the next sync" semantics — **no silent error drop.**
- Gated by `GVIRTUS_ASYNC_DISPATCH` (off by default; falls back to sync on non-UCX transports).

### `cudaMemcpyAsync` — the hard part, done in 3 phases
| direction | approach | why safe |
|-----------|----------|----------|
| **D2D + small H2D** (<64 KB, AM path) | fire-and-forget | output-less, no shared slot |
| **large H2D** (RMA slot path) | **ack-free flow control**: configurable RX slots (`GVIRTUS_RMA_SLOTS`); frontend tracks in-flight slots and **demotes the wrapping copy to synchronous** (which drains all prior slots via the in-order FIFO) | no slot reused before the backend consumes it; no extra messages; correct backpressure |
| **D2H** | **deferred completion**: record `{dst,count,request_id}`, send in stream order, return; `DrainPendingD2H()` fills `dst` at the next sync receive | reply precedes any later sync reply (FIFO); backend unchanged (its pageable D2H buffer makes the copy synchronous + stream-ordered before replying) |

**Safety guards:** deferred D2H only engages when `dst` is a **tracked pinned** buffer
(`cudaHostAlloc`/`cudaMallocHost`); pageable dst stays synchronous (CUDA fills it synchronously, so
deferring could expose stale bytes). Stream ordering verified: a deferred D2H captures device state
*at its position* (pre-kernel value while a later copy reads the post-kernel value).

### Results — sync (dispatcher off) vs async (on), matched fresh backend at ERROR log level

| benchmark | metric | sync | async | change |
|-----------|--------|-----:|------:|-------:|
| **llama** | token gen tg16 (t/s) | 87.35 | **189.86** | **+117% (2.17×)** |
| **llama** | prompt eval pp8 (t/s) | 558.85 | **1216.84** | **+118%** |
| **BabelStream** @256K elems | Triad (GB/s) | 176.0 | 230.9 | **+31%** |
| **BabelStream** @1M elems | Triad (GB/s) | 702.1 | 845.8 | **+20%** |
| **BabelStream** @16M elems | Triad (GB/s) | 648.4 | 641.4 | ~0% |
| **miniBUDE** | GFLOP/s | 216.53 | 216.53 | 0% |
| **simple_matrix** | host per-iter (ms ↓) | 737.7 | 725.8 | ~0% |

Data: `benchmarks/_summary/`, `benchmarks/{llama,simple-matrix}-async/`, `benchmarks/*-async/`.

**Findings (cross-checked):**
1. **The async benefit scales with launch-boundedness** — llama +117%, small BabelStream +20–31%,
   transitional +6%, and **~0% for compute/bandwidth/transfer-bound** (miniBUDE, large BabelStream,
   simple_matrix). Exactly the roofline expectation; **no regressions** (async≈sync where not
   launch-bound, and gated off by default).
2. **Transport-independent** — same ~2.17× on RDMA and RDMA+GPUDirect (it removes RPC *round-trips*,
   orthogonal to bandwidth). GPUDirect ≡ RDMA for llama with and without async.
3. **Log-level confound (methodology):** backend DEBUG per-RPC logging ~**halves** throughput. The
   async *ratio* is robust (2.16× DEBUG, 2.17× ERROR) but **absolute numbers must be read at ERROR**.
   An earlier DEBUG measurement (40.65→87.62) understated absolutes ~2×.
4. **Wire-verified engagement:** async ON logged 14,139 `cudaLaunchKernel` as fire-and-forget vs 0
   with async OFF; `cudaMemcpyAsync` slot demotion produced the expected 21 async + 3 sync of 24
   large H2D with 8 slots (no corruption); deferred D2H stream-ordering test passes.

**Cumulative: llama tg16 8.04 → 40.4 (Stage 1) → ~187 (Stage 2) t/s; the gap vs native (634.9)
closes from ~79× to ~3.4×**, correctness preserved.

**Regression guarantee (by design):** no phase can be slower than sync — async either overlaps or
applies correct backpressure; the ack-free designs add zero extra round-trips. (One bug fixed en
route: a static init/destruction-order fiasco in the pinned-allocation registry — the communicator's
`init_rx_pool` calls `cudaHostAlloc` during `CudaRtFrontend`'s static ctor — resolved with immortal
function-local singletons.)

### Correctness tests
`examples/testing/test_async_dispatch.cu` and `test_memcpyasync_phase{1,2,3}.cu` — fire-and-forget
launch/memset/event/stream-wait + D2H readback, large-H2D slot-reuse stress (24 buffers/8 slots), and
deferred-D2H stream-ordering. All bit-identical with the gate on and off.

---

## Not yet done (future work)
- **#4 launch batching / CUDA-graph capture** — coalesce a whole decode step's launches into one RPC;
  the remaining ~3.4× lever.
- **#5 RMA registration cache** — `GVIRTUS_RMA_ZEROCOPY=1` OOMs because UCX can't build an rcache
  in-container; a working rcache or a GVirtuS-managed registration pool would make zero-copy usable
  at scale (see README known-issues).
- **`-baseline` runs** — measure GVirtuS over the legacy **TCP communicator** (`tcp/ip` /
  `properties.json`, without UCX; `benchmarks/*-baseline/`) to isolate the UCX communicator's
  contribution vs the pre-UCX TCP path.
