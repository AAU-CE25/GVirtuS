# 02 — BabelStream (memory-bandwidth proxy)

Status: **bare-metal baseline DONE; GVirtuS path WORKS** (the earlier communicator block is resolved).
Current-build GVirtuS is **~5.6× faster on small sizes** than the pre-optimization tables below, and
the async dispatcher adds a further **+20…31% at small (launch-bound) sizes** — see
[`12-async-dispatch-suite.md`](12-async-dispatch-suite.md) for the async A/B and updated numbers
(`data/_async_suite/babel_async_raw.csv`). The multi-transport GB/s tables in this doc and
`data/babelstream/babelstream_summary_gbps.csv` predate the optimization stack and are **stale for the
GVirtuS columns** (baremetal unchanged).
Last updated: 2026-07-18.

## 1. What BabelStream measures
GPU sustained **memory bandwidth** (STREAM for accelerators). 5 kernels — Copy, Mul,
Add, Triad, Dot — over arrays far larger than cache; reports **GB/s**. Under GVirtuS it
primarily stresses the **control path** (many tiny kernel launches) + the initial/final
large H2D/D2H transfers, and establishes the device-bandwidth ceiling.

## 2. Bare-metal baseline (native CUDA, no GVirtuS)
- Node es-dpu-01, L40S, CUDA 12.6 container, `sm_89`, 33.5M doubles (805 MB working set).
- Build: `nvcc -O3 -std=c++17 --extended-lambda -DCUDA -DGRID_STRIDE -arch=sm_89 -Isrc -Isrc/cuda src/main.cpp src/cuda/CUDAStream.cu -o cuda-stream -lnvidia-ml`

| Kernel | GB/s | % of 864 GB/s peak |
|--------|------|--------------------|
| Copy | 743.7 | 86% |
| Mul | 659.0 | 76% |
| Add | 668.6 | 77% |
| Triad | 683.0 | 79% |
| Dot | 669.6 | 78% |

## 3. Corrected diagnosis (2026-07-17, updated)

### 3a. The "invalid response header" was a STALE-IMAGE artifact — NOT a real bug
Running through the **prebuilt** `simple_matrix_gvirtus:cuda12.6` image failed at
`__cudaRegisterFatBinary` with "invalid response header". Root cause: that image's
**frontend stubs are older than the current backend source** (commit `88fac1f`) — a
frontend/backend **version mismatch**, not a framing bug. Backend instrumentation
(`[FATBINDBG]`) confirms the 52 KB fatbinary and all control messages parse correctly
(`want == frame` every time) once frontend and backend are built from the SAME source.

**Lesson / harness rule:** always run the frontend from a **source-built** GVirtuS
(container `gvirtus-fe-kz08ey`, built from mounted `~/GVirtuS` via the `gvirtus-dev`
image), never the stale prebuilt app image.

### 3b. REAL blocker (RESOLVED): zero-copy pinned-host memory used in a kernel
With matched source builds, BabelStream registered kernels, launched them, but faulted
with **CUDA 700 (illegal memory access)** in `dot_kernel`.

**Root cause (confirmed):** BabelStream allocates its dot-reduction partial-sums buffer
`sums` with `cudaHostAlloc` (pinned host) and the `dot_kernel` writes to it **directly
from the device** (zero-copy), then the host reads `sums[i]`. GVirtuS implements
`cudaHostAlloc` as a **frontend-local `malloc`** (`CudaRt_memory.cpp:129`) — it allocates
nothing on the backend. So the backend kernel receives a **frontend host address** that is
not valid device-accessible memory on the backend → illegal access. This is a fundamental
remoting limitation: host memory is not shared across the frontend/backend address spaces,
so **zero-copy pinned-host buffers passed into kernels cannot work transparently**.
(Kernel-argument *marshaling* was verified CORRECT — not the cause.)

**Fix (app adaptation, functionally identical):** allocate `sums` as **device memory**
(`cudaMalloc`) and copy it back to a host vector with an explicit `cudaMemcpy` before the
host-side reduction. Negligible cost (568 doubles ≈ 4.5 KB per dot iteration) and does not
change what BabelStream measures (main-memory bandwidth of the big arrays). Applied to
`~/benchmarks/BabelStream/src/cuda/CUDAStream.cu` on the nodes (marked `// GVirtuS`).

**Result: BabelStream now runs to completion over GVirtuS and validates.** Example
(1M doubles, small size, UCX/TCP, clean build): Copy ~47 GB/s, Triad ~67 GB/s (small-size
numbers are launch-overhead-bound and indicative only — real measurements use the 33.5M
size across all transports).

**GVirtuS limitation to note in the paper:** `cudaHostAlloc`/`cudaMallocHost` zero-copy
device access is unsupported (implemented as plain host malloc). Apps relying on it need a
device-buffer + explicit-copy adaptation.

## 3-OLD (superseded) — the "invalid response header" theory


## 4. Harness (on es-dpu-02, outside git)
- `~/benchmarks/BabelStream` — upstream clone.
- `~/benchmarks/harness/nvml_shim.cpp` — no-op NVML (cosmetic clock printout only).
- `~/benchmarks/harness/build_run_babelstream.sh` — compiles against GVirtuS frontend
  stubs (`-L$GVIRTUS_HOME/lib/frontend -lcudart -lcuda` + shim), runs with `SIZE`/`ITERS`/
  `TIMEOUT` env; persists binary to `/harness/cuda-stream-gvirtus`.
