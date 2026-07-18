# 07 — Llama LLM inference (llama.cpp)

Status: **runs over GVirtuS; synchronous-dispatch bottleneck fixed by rec#1/#2 (frontend RPC opts)
then rec#3 (async dispatch). At matched ERROR log level, async gives 2.17× (RDMA token gen
87.35 → 189.86 t/s); GPUDirect ≡ RDMA here; now ~3.3× off native.**
Last updated: 2026-07-18.

## Goal
LLM inference is the most **transport-/RPC-revealing** real application: token generation is
thousands of tiny **sequential** kernel launches, each a synchronous RPC over GVirtuS. This is
where the report's "synchronous frontend dispatch is the dominant bottleneck for RPC-heavy
workloads" should show most strongly.

Model: **TinyLlama-1.1B-Chat Q4_K_M** (668 MB GGUF). Engine: **llama.cpp** (CUDA backend,
`GGML_CUDA=ON`, `sm_89`), built with `--cudart shared` + `BUILD_SHARED_LIBS` so it links the
GVirtuS frontend stubs. `GGML_CUDA_DISABLE_GRAPHS=1` (GVirtuS has only partial CUDA-graph
support). CUDA libs used: cudart, cublas, cublasLt, cuda-driver — all covered by GVirtuS
plugins.

## Two GVirtuS issues found (one FIXED, one documented)

### FIXED — `cudaDeviceGetPCIBusId` missing backend handler
llama.cpp aborted at CUDA init (`ggml_backend_cuda_reg` → `cudaDeviceGetPCIBusId` → "CUDA
error: unrecognized error code"). The frontend stub existed but there was **no backend
handler**, so the backend rejected the routine. **Fix added and kept** (see
`05-gvirtus-bugs-and-knobs.md`): handler in `plugins/cudart/backend/`. After the fix,
llama.cpp initializes CUDA and **generates coherent text over GVirtuS**
("The capital of France is → Paris").

### FIXED — intermittent ggml buffer-alignment assert (host-pinned allocation under-aligned)
`llama-bench` intermittently (~50 % of runs) aborted with
`GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned")` (exit
134) in `ggml_backend_cpu_buffer_from_ptr`. **Root cause:** ggml's `TENSOR_ALIGNMENT` is 32, and
the asserted pointer is a **host** pointer from `cudaHostAlloc`/`cudaMallocHost`. GVirtuS
implements both as a plain `malloc()` (host memory — the client is GPU-less), and glibc `malloc`
only guarantees **16-byte** alignment, so `ptr % 32 == 0` failed about half the time →
intermittent crash. **Fix added and kept:** `plugins/cudart/frontend/CudaRt_memory.cpp` now uses
`posix_memalign(ptr, 256, size)` for `cudaHostAlloc` and `cudaMallocHost` (256 bytes matches real
CUDA's pinned-allocation alignment). `cudaFreeHost` already uses `free()`, which is
`posix_memalign`-compatible. After the fix, `llama-bench` runs **5/5 with no alignment crash** and
exits cleanly (EXIT=0), so the full transport sweep below is now valid. See
`05-gvirtus-bugs-and-knobs.md`.

> Note: a separate apparent "hang" was **not** a GVirtuS bug — `llama-cli` blocks on stdin in its
> interactive prompt after generation. Run it with `< /dev/null`, or prefer `llama-bench` (exits
> on its own). See the corrected §"the ~8-minute wall time" below.

## Results (verified clean `llama-bench` runs, TinyLlama-1.1B Q4, `-p 8 -n 16 -r 3`)

Data: `data/llama/llama_bench_transports.csv`. GPUDirect state **verified on the backend log**
(doc-00 rule): RDMA run logged `GPUDirect=disabled`, GPUDirect run logged `GPUDirect=enabled`.

| config | prompt eval pp8 (t/s) | token gen tg16 (t/s) | tg16 vs bare metal |
|--------|----------------------:|---------------------:|-------------------:|
| bare metal (native cudart, local L40S) | 2773 ± 1066 | **634.9 ± 26.9** | 1× |
| GVirtuS RDMA (GD off)                  |   50.6 ± 1.8 | **8.04 ± 0.07**  | **79× slower** |
| GVirtuS RDMA + GPUDirect               |   49.1 ± 0.1 | **7.96 ± 0.02**  | **80× slower** |
| GVirtuS TCP                            |   25.4 ± 10.2| **3.58 ± 1.16**  | **177× slower** |

### After the cheap frontend optimizations (docs 08, 10)
Two behaviour-preserving frontend changes — local `__cudaPush/PopCallConfiguration` (#1) and
frontend-local `cudaGetDevice`/`cudaGetLastError` (#2) — cut per-launch RPCs from ~6.2 to ~1.1 and
gave a **~5× token-gen speedup on RDMA with correctness preserved** ("…is Paris"):

| config | baseline tg16 | + rec #1 | + rec #2 | total | vs bare metal |
|--------|--------------:|---------:|---------:|------:|--------------:|
| RDMA           | 8.04 | 11.64 | **40.42** | **5.03×** | 16× slower (was 79×) |
| RDMA+GPUDirect | 7.96 | 11.53 | **42.77** | **5.37×** | 15× slower (was 80×) |
| TCP            | 3.58 |  5.35 | **10.63** | **2.97×** | 60× slower (was 177×) |

RDMA pp8: 50.6 → 69.5 → **250.0** t/s; GPUDirect pp8 → **301.0**. Correctness verified on all three
transports. **GPUDirect caveat (verified, not assumed):** for llama the backend was
`GPUDirect=enabled`, but the **data path was not meaningfully exercised** — the frontend was
host-staged (`GVIRTUS_GPUDIRECT` unset there to avoid the probe-poisoning crash, doc 05), the
backend log shows the peer advertised **0 GPU-shadow slots** (`rma_setup: received 2 remote slots
(0 with gpu shadow)`) and **zero** GPU-routing events during the run. That is *why* GPUDirect ≡ RDMA
here: llama is control-plane-dominated and its per-token transfers are too small/few to trigger the
GPU data path. Actual NIC→GPU DMA is demonstrated on **bulk** transfers (doc 04, ~2.3× D2H), not on
llama. **Operational note:** set `GVIRTUS_GPUDIRECT=1` on the **backend only** — setting it on the
GPU-less frontend triggers a failing probe that can poison the CUDA context and crash the app
(doc 05). Data: `data/llama/llama_optimizations_progression.csv`. The remaining ~15–16×
(RDMA/GPUDirect) motivated the structural work below.

### After async dispatch (rec #3) — token gen doubles again (transport-independent)
Enabling the gated async dispatcher (`GVIRTUS_ASYNC_DISPATCH=1`) makes `cudaLaunchKernel`,
`cudaMemsetAsync`, `cudaEventRecord`/`WithFlags` and `cudaStreamWaitEvent` **fire-and-forget** (send
without waiting for a response; errors reconciled at the next sync). Since llama decode is dominated
by ~500 serial `cudaLaunchKernel` RPCs/token, removing that per-launch round-trip roughly **doubles**
throughput. Measured at matched **ERROR** backend log level (data `data/llama/llama_async_dispatch.csv`):

| transport | metric | sync | async | speedup |
|-----------|--------|-----:|------:|--------:|
| RDMA            | tg16 | 87.35 | **189.86** | **2.17×** |
| RDMA            | pp8  | 558.9 | **1216.8** | **2.18×** |
| RDMA+GPUDirect  | tg16 | 86.71 | **186.51** | **2.15×** |
| RDMA+GPUDirect  | pp8  | 558.0 | **1214.7** | **2.18×** |

**Findings (cross-checked):**
- **Transport-independent:** async gives the same ~2.17× on RDMA and GPUDirect — it removes RPC
  *round-trips* (count × latency), not bandwidth.
- **GPUDirect ≡ RDMA for llama** (off *and* on, within ~1%): backend was `GPUDirect=enabled` with 2/2
  GPU-shadow slots, but llama's per-token transfers are tiny/host-sourced and never trip the ≥4 MB RMA
  GPU path — so peer-DMA has nothing to accelerate (consistent with the 0-GPU-shadow finding below).
- **Log-level confound:** running the backend at DEBUG (per-RPC log line on the hot path) ~halves
  throughput. The async *ratio* is stable (2.16× at DEBUG, 2.17× at ERROR) but absolute numbers must be
  read at ERROR — an earlier DEBUG run (40.65→87.62) understated absolutes ~2×.
- **Wire-verified:** async ON logged **14,139 `cudaLaunchKernel` as fire-and-forget** vs **0** with
  async OFF; `cudaMemcpyAsync` stayed synchronous (901 responses) — launches genuinely go async,
  nothing falls back.

Correctness verified: `llama-cli` still answers "…the capital of France is Paris. It is located in
the Île-de-France"; unit test `examples/testing/test_async_dispatch.cu` is bit-identical with the
gate on/off. **vs bare metal (634.9 tg16): gap narrows from ~79× to ~3.3×.** Full design + the
challenge/rework of the `fix-async-calls` branch:
[`08-recommended-improvements.md`](08-recommended-improvements.md) #3. The remaining ~3.3× needs
launch batching / CUDA-graph capture (#4).

## Findings

### 1. LLM inference is catastrophically RPC-bound over GVirtuS (the headline)
Token generation drops from ~635 t/s native to **~8 t/s** over the best transport (RDMA) — a
**~79× slowdown**, far worse than any other workload in this suite (miniBUDE <1.003×, BabelStream
~1.05× at large sizes). Reason: each generated token runs the full model as **hundreds of small
sequential kernel launches**, and GVirtuS turns every launch into several synchronous
request/response RPCs (see profile below), with no overlap. This is the concrete, dramatic
confirmation of the report's synchronous-dispatch limitation — and exactly why LLM inference is
the workload that most needs the future asynchronous-dispatch work.

### 2. Transport DOES matter here — RDMA ~2.2× faster than TCP (unlike compute-bound apps)
This is the opposite of miniBUDE. Because llama is **dispatch-latency-bound** (thousands of tiny
serial round-trips per token, each dominated by network round-trip time, not payload size),
lowering per-RPC latency directly speeds up generation:

- token gen: **TCP 3.58 → RDMA 8.04 t/s (2.25× faster)**; prompt eval TCP 25.4 → RDMA 50.6 (2×).
- TCP is also much noisier (tg16 ±1.16 vs RDMA ±0.07) — TCP round-trips contend and jitter.

**GPUDirect gives no additional benefit over plain RDMA** (tg16 7.96 vs 8.04, within noise): a
token's transfers are tiny and **host-sourced** (weights already resident on the GPU; per-step
H2D/D2H are small logits/activations from host memory), so GPU↔NIC peer-DMA has nothing to
accelerate. GPUDirect's win shows up in *bulk device-resident transfers* (doc 04), not here.

**Net:** RDMA is the right transport for RPC-heavy inference (2.2× over TCP), but no transport
rescues an architecturally synchronous RPC loop — even RDMA is ~79× off native. Only asynchronous
dispatch / batching / CUDA-graph capture closes that gap.

### 3. Correctness preserved
Generated text is coherent and on-topic ("… the capital of France is Paris") — remote
execution is functionally faithful; the cost is purely performance.

## Which CUDA call is the killer? (evidence-based backend profile)

Captured the backend's **timestamped** dispatch log for one run (TinyLlama-1.1B, RDMA+GPUDirect,
`-n 24 --ignore-eos`), 77,345 dispatches, and analyzed inter-dispatch timing offline
(`data/llama/profile_backend_log.py`, `data/llama/call_profile_summary.txt`). This overturned
two of my earlier assumptions — **init is NOT the bottleneck, and the "8-minute" wall time was
mostly a hang, not steady slowness.**

### Phase split (measured, excluding a terminal hang — see below)
| phase | calls | wall time |
|-------|------:|----------:|
| registration (`cudaRegisterFunction`/`Var`/`FatBinary`) | 8,736 | **0.96 s** |
| compute (prompt-eval + ~22 token forward passes) | 68,458 | **5.51 s** |

**Registration is not the killer** (<1 s), despite being 93 % of the *call count*. The cost is
**token generation**: ~22 forward passes in 5.51 s ≈ **250 ms/token ≈ 4 tok/s** vs ~570 tok/s
native → ~140× slower in steady state. (The earlier "72×" came from a shorter `llama-bench`
tg8 run; ~140× is the fuller-context figure.)

### The real killer: ~6.2 RPCs per kernel launch × ~500 launches per token
Every `<<<grid,block>>>(...)` kernel launch expands, over GVirtuS, into **6.2 blocking RPCs**:

| routine | count | per launch | note |
|---------|------:|-----------:|------|
| `cudaGetDevice`            | 19,358 | ~1.8 | cheap query — **cacheable/local** |
| `cudaGetLastError`         | 14,527 | ~1.3 | cheap query — **cacheable/local** |
| `cudaPushCallConfiguration`| 10,960 | 1.0 | **thread-local stack push — should NEVER be an RPC** |
| `cudaLaunchKernel`         | 10,960 | 1.0 | the actual launch |
| `cudaPopCallConfiguration` | 10,960 | 1.0 | **thread-local stack pop — should NEVER be an RPC** |

~10,960 launches for ~22 forward passes ⇒ **~500 kernel launches per token ⇒ ~3,100 synchronous
RPCs per token.** Per-RPC latency is actually *fast* (~9 µs median — small Active-Messages on
RDMA), but ~3,100 of them **in series, with zero overlap**, is ~250 ms/token.

**So the killer is not one exotic call and not the transport — it is the *number* of blocking
RPCs per token**, and critically **~4 of the 6.2 RPCs per launch are pure overhead that should
not be remoted at all:**
- `cudaPushCallConfiguration` / `cudaPopCallConfiguration` just push/pop the launch config on a
  **thread-local** stack — they should be **local no-ops**, never round-trips.
- `cudaGetDevice` / `cudaGetLastError` are trivial queries that could be **cached** frontend-side.

### Two independent fixes (biggest bang first)
1. **Cheap, no async needed:** make `cudaPushCallConfiguration`/`cudaPopCallConfiguration` local
   and cache `cudaGetDevice`/`cudaGetLastError`. That removes ~5 of 6.2 RPCs per launch →
   ~5× fewer RPCs → roughly ~5× faster generation on its own.
2. **Structural:** **asynchronous dispatch** for `cudaLaunchKernel`/`cudaMemcpyAsync` (queue
   without waiting), ideally + launch batching or CUDA-graph capture so a whole decode step is
   one RPC. Transport speed cannot help — cost is round-trip *count* × latency, not bandwidth.

> **PROTOTYPED + VALIDATED (kept in tree):** the first half of fix #1 —
> `__cudaPushCallConfiguration`/`__cudaPopCallConfiguration` are now **local thread-local stack**
> operations (no RPC), exactly like native CUDA. This removed 2 of ~6.2 RPCs/launch and delivered
> a measured **1.45× token-gen speedup** (RDMA 8.04 → 11.64 t/s; GPUDirect 7.96 → 11.53; TCP
> 3.58 → 5.35), matching the 6.2/4.2 = 1.48× prediction almost exactly, with **correctness
> preserved** and **no regression** on compute-/bandwidth-bound apps (miniBUDE 1.00×, large
> BabelStream 1.06×, small BabelStream 1.64×). Full write-up + data:
> [`08-recommended-improvements.md`](08-recommended-improvements.md),
> `data/llama/llama_bench_proto.csv`, `data/prototype_pushpop_summary.csv`. Caching
> `cudaGetDevice`/`cudaGetLastError` (the rest of fix #1) is the recommended next cheap step.

### Separately: the earlier multi-minute wall times were a HANG, now RESOLVED
Earlier `llama-cli` runs showed ~8-minute wall clocks. Investigation found **two separate
non-steady-state causes, both now addressed:**
1. **`llama-cli` interactive stdin wait** (harness, not GVirtuS): after generating `-n` tokens,
   `llama-cli` drops into its interactive `>` prompt and blocks on stdin until EOF/timeout. Fix:
   run with `< /dev/null` (or use `llama-bench`, which exits on its own). Generation itself
   finishes in seconds.
2. **The alignment assert** (GVirtuS, now FIXED — see §issues): the intermittent EXIT=134 crash
   that invalidated some prior end-to-end wall times. With `posix_memalign` host allocations,
   `llama-bench` completes 5/5 cleanly.

**Corrected takeaway:** steady generation over RDMA is ~79× slower than native (RPC-count bound),
a stable and now-reproducible figure — not the earlier hang-inflated multi-minute numbers.

## Takeaways for the paper
- **miniBUDE (compute-bound) and Llama (call-bound) are the two extremes** of the roofline:
  overhead ~0 % vs ~72× slowdown. Together they bracket where GVirtuS is / isn't viable today.
- Llama is the strongest evidence for the report's **asynchronous-execution future work**:
  RPC-heavy generation is dispatch-latency-bound, and no transport optimization alone rescues
  it. This motivates async dispatch / kernel-launch batching / graph capture as the key next
  step for remote LLM serving.
- Two concrete GVirtuS engineering outcomes: **fixed** `cudaDeviceGetPCIBusId` (missing backend
  handler) **and** the intermittent ggml host-buffer alignment crash (`cudaHostAlloc`/
  `cudaMallocHost` now 256-byte aligned via `posix_memalign`). Both fixes are kept.

## Reproduce
Backend: `GVIRTUS_GPUDIRECT={0|1} bash /tmp/gvirtus-backend-run.sh` (restart between GD phases;
verify the `GPUDirect=enabled/disabled` line in `docker logs gvirtus-kz08ey`). Frontend
(source-built `gvirtus-fe-kz08ey` container): model at `/benchmarks/models/tinyllama-1.1b-q4.gguf`,
binaries at `/benchmarks/llama.cpp/build_cuda/bin/`. **Use `llama-bench` for clean, self-exiting
numbers:** `llama-bench -m <model> -ngl 99 -p 8 -n 16 -r 3`. Force TCP by setting
`UCX_TLS=tcp,self` on the frontend. Bare metal: drop the GVirtuS frontend from `LD_LIBRARY_PATH`
so it links the real `/usr/local/cuda/lib64/libcudart.so.12`. If using `llama-cli`, append
`< /dev/null` to avoid the interactive stdin wait.
