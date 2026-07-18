# 05 ? GVirtuS bugs & knobs found during benchmarking

Living note of GVirtuS-internal issues surfaced by the benchmark campaign. Not blockers for
the current results (workarounds noted), but candidates for hardening / paper "future work".

## Frontend `GVIRTUS_GPUDIRECT=1` triggers a failing probe that can poison the app's CUDA context (WORKAROUND: don't set it on the frontend)

**Symptom:** with `GVIRTUS_GPUDIRECT=1` set on the **frontend** (GPU-less client), llama.cpp
crashes at the first kernel launch — `CUDA error: initialization error` in
`ggml_cuda_kernel_launch`, `ggml_abort` at `ggml-cuda.cu:106` (exit 134 / core dump). Intermittent:
only when the frontend's local GPU/driver is in a degraded state (heavy churn), otherwise the probe
"fails gracefully".
**Root cause:** the frontend's `UcxCommunicator` init runs a GPUDirect probe when
`GVIRTUS_GPUDIRECT=1` — it does a real `cudaMalloc(4K)` via the libcuda passthrough
(`/usr/local/cuda/compat/libcuda.so.1`). On a degraded frontend GPU this probe fails
(`GPUDirect=disabled ... cudaMalloc(4K) failed (no GPU? OOM?)`), and the failure leaves the
process's CUDA driver state poisoned, so ggml's later CUDA init inherits an
`initialization error`. The frontend **does not need** `GVIRTUS_GPUDIRECT` at all — the frontend is
GPU-less and GPUDirect is negotiated from the **backend** side (backend `GVIRTUS_GPUDIRECT=1` +
RDMA transport).
**Workaround (used for all GPUDirect results):** do **not** set `GVIRTUS_GPUDIRECT` on the frontend;
set it only on the backend launcher. With that, GPUDirect works end-to-end and llama runs cleanly
(tg16 42.8 t/s with rec #1+#2, correctness "…is Paris"). **Hardening:** the frontend should skip the
GPUDirect probe entirely (it's the backend's concern), or a probe failure must not poison the
process CUDA context.
**Related:** the persistent GPUDirect "cold-start stall" observed this session was this crash in
disguise — the frontend was launched with `GVIRTUS_GPUDIRECT=1` and the slow core dumps looked like
stalls. Also compounded by orphaned backend containers leaking GPU memory (kill only reaps
container procs via `docker rm -f`, not host `kill` as non-root).

## FIXED BUGS (changes kept in the tree)

### cudaDeviceGetPCIBusId — missing backend handler (FIXED)
**Symptom:** llama.cpp aborts at CUDA init (`ggml_backend_cuda_reg`,
`cudaDeviceGetPCIBusId(...)` -> "CUDA error: unrecognized error code"). Any app that calls
`cudaDeviceGetPCIBusId` crashes over GVirtuS.
**Root cause:** the frontend stub existed (`plugins/cudart/frontend/CudaRt_device.cpp`) and
sent the routine, but there was **no backend handler** — the backend rejected `cudaDeviceGetPCIBusId`
as an unknown routine, returning an error the caller surfaced as "unrecognized error code".
**Fix (kept):** added the backend handler:
- `plugins/cudart/backend/CudaRtHandler.h` — declaration `CUDA_ROUTINE_HANDLER(DeviceGetPCIBusId);`
- `plugins/cudart/backend/CudaRtHandler.cpp` — registration
  `mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetPCIBusId));`
- `plugins/cudart/backend/CudaRtHandler_device.cpp` — impl: `AssignAll<char>()` for the
  buffer, `Get<int>()` len + device, call `cudaDeviceGetPCIBusId`, return the filled buffer.
**Verified:** llama.cpp (TinyLlama-1.1B) now initializes CUDA and generates coherent tokens
over GVirtuS ("The capital of France is -> Paris"). Backend stays healthy.

### cudaHostAlloc / cudaMallocHost under-aligned host memory (FIXED)
**Symptom:** llama.cpp / `llama-bench` intermittently (~50% of runs) aborts with
`GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned")`
(exit 134), in `ggml_backend_cpu_buffer_from_ptr`. Non-deterministic — failed about half the
time, invalidating end-to-end wall times.
**Root cause:** ggml's `TENSOR_ALIGNMENT` is 32 and it asserts alignment on a **host** pointer
returned by `cudaHostAlloc`/`cudaMallocHost`. GVirtuS implements both as a plain `malloc()`
(the client is GPU-less, so "pinned host" memory is just host memory), and glibc `malloc` only
guarantees **16-byte** alignment. Real CUDA returns pinned memory aligned to >=256 bytes, so
apps that rely on that alignment break under GVirtuS ~half the time (16 % 32 != 0).
**Fix (kept):** `plugins/cudart/frontend/CudaRt_memory.cpp` — `cudaHostAlloc` and
`cudaMallocHost` now use `posix_memalign(ptr, 256, size ? size : 1)` (returns
`cudaErrorMemoryAllocation` on failure) instead of `malloc(size)`. `cudaFreeHost` already uses
`free()`, which is `posix_memalign`-compatible. 256 bytes matches real CUDA's pinned alignment.
**Verified:** `llama-bench` ran 5/5 with **no alignment crash** (was ~50% failure), exiting
cleanly EXIT=0, enabling the full TCP/RDMA/GPUDirect llama sweep (doc 07).

## RMA_ZEROCOPY (GVIRTUS_RMA_ZEROCOPY) ? what/why/failure

**What it controls** (src/communicators/ucx/UcxCommunicator.cpp, WriteIovRma ~L1613-1740):
the RDMA one-sided send (`ucp_put`) strategy for large payloads.
- `=0` STAGED (default): copy all iov fragments (incl. big payload) into a pre-registered
  scratch buffer, one ucp_put. Extra host memcpy (~2.5ms/64MB) but registration is done ONCE
  and reused -> stable/warm.
- `=1` ZEROCOPY: ucp_put the big payload DIRECTLY from the caller buffer (skip the memcpy).
  Faster (~2.5ms/call) BUT requires per-call NIC registration of the caller buffer, relying on
  UCX's registration cache (rcache) to amortize.

**Why it exists:** perf optimization ? remove the staging memcpy on the hot path for big
transfers. Also: GPUDirect FORCES zerocopy for GPU-resident buffers (big_is_gpu overrides the
flag, ~L1706) because the staged path can't memcpy device memory through the CPU.

**Why it fails (OOM, backend exit 137 at >=32MiB):** in this container UCX logs
"could not create UCP registration cache: Unsupported operation" (noted in code comment
~L1650-1658). With no working rcache, every zerocopy ucp_put RE-REGISTERS the source buffer
with the NIC and nothing evicts/reuses them -> pinned/locked NIC registrations accumulate ->
at large payloads pinned memory is exhausted -> backend OOM-killed.

**Status/workaround:** keep default `=0` (staged). All sweep data + the validated GPUDirect
~2.3x D2H result were obtained with zerocopy OFF, so unaffected.
**Proper fix (future):** make UCX rcache work in the container (nvidia-peermem + UCM event
handling), or add a persistent registration pool for reused buffers instead of relying on rcache.

## Buffer-reuse RMA edge case (narrow)
Transferring growing SUB-RANGES of one large, registered, REUSED device buffer (my first
transfer_bw.cu, 256MB buffer) resets the RDMA connection at >=8-16MiB. Normal exactly-sized
allocations (simple_matrix, transfer_bw2.cu) are fine to >=976MB. See doc 04 section 5.

## Backend listener non-recovery
After an RDMA connection reset/crash, the backend's next `ucp_listener_create` fails with
"Device is busy" (port 32223 stuck) ? backend wedged (shows "Up" but unusable) until full
`docker rm -f` + relaunch. Should release the listener/port on connection teardown.

## Frontend GPUDirect probe fails (cosmetic for current workloads)
Frontend `cudaMalloc(4K)` probe fails because it dlopens the GVirtuS STUB libcudart (first on
LD_LIBRARY_PATH) instead of the real one; even forcing the real path fails in-process because
the stub libcuda captures the driver. Harmless for host-source transfers (GVirtuS clients are
GPU-less by design); backend-side GPUDirect (the meaningful one) works. See doc 04 / plan.

## cudaHostAlloc zero-copy in kernels unsupported
GVirtuS cudaHostAlloc = frontend-local malloc; a kernel writing that pointer faults (CUDA 700)
on the backend. Needs device-buffer + explicit copy adaptation (BabelStream sums fix, doc 02).
