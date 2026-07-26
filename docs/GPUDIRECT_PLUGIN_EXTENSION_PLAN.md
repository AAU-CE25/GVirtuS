# GPUDirect Plugin Extension Plan — Zero-Copy for cudadr / cublas / cudnn

Branch: `feature/clean-sheet-comms` (compared against local `master`).
Verified against the live repo on 2026-07-06. All file paths and line
numbers below were confirmed by reading the actual sources; where the
starting summary was slightly stale it has been corrected inline.

This document assumes familiarity with `docs/GPUDIRECT.md`,
`docs/IOV_REFACTOR.md`, and `docs/UCX_OPTIMIZATIONS.md`. It uses the same
terminology: **Fase / Phase 1+2** (backend page-fault elimination),
**Phase/Fase 4** (frontend D2H zero-copy), **Phase/Fase 5** (frontend H2D
iov split), **Variant A** (backend D2H → GPU scratch → peer-DMA to
frontend), **Variant B / Step B3/B4** (frontend H2D → remote GPU shadow →
D2D on backend). This plan introduces no new phase numbers; it is a
*refactor + generalisation* of the machinery those phases already built,
so it is described in terms of the components it touches.

---

## 1. Current State (verified)

### 1.1 Which plugins differ from master

`git diff master...feature/clean-sheet-comms --stat -- plugins/` shows
**exactly two** plugin directories changed:

| Plugin | Files changed |
|---|---|
| `cudadr` | `CMakeLists.txt` (+39/-…), `frontend/CudaDr_initialization.cpp` (+70/-…) |
| `cudart` | `backend/CudaRtHandler_memory.cpp` (+184), `frontend/CudaRtFrontend.h` (+29), `frontend/CudaRt_memory.cpp` (+13) |

The other **8** plugins are byte-identical to master — verified with a
per-plugin `git diff … -- plugins/<name>` producing **0 diff lines** for
each of: `cublas`, `cudnn`, `cufft`, `curand`, `cusolver`, `cusparse`,
`nvml`, `nvrtc`. The starting summary is correct.

### 1.2 cudadr changes (not part of the GPUDirect data path)

- `plugins/cudadr/CMakeLists.txt`: adds `<toolkit>/lib64/stubs` to the
  `libcuda.so` search `PATHS`, guards the `CUDADR_VERSION` regex so an
  unversioned stub `libcuda.so` falls back to `"1"`, forces
  `VERSION "${CUDADR_VERSION}.0"` / `SOVERSION "${CUDADR_VERSION}"` so
  the `libcuda.so → libcuda.so.1 → libcuda.so.1.0` symlink chain is
  built, and links `dl`.
- `plugins/cudadr/frontend/CudaDr_initialization.cpp`: `cuInit()` now
  dlopen/dlsym's the **real local** `libcuda.so.1` (candidate list at
  lines 55-59) and passes through, instead of routing over RPC. Comment
  (lines 38-46) explains the reentrancy crash: UCX's `libuct_cuda.so`
  calls `cuInit(0)` during `ucp_init()` while `Frontend::Init()` is still
  constructing; the old RPC `cuInit` recursed into `Frontend::Prepare()`
  on a not-yet-initialised buffer and segfaulted. Log prefix `[GVIRTUS]`.

These are build/reentrancy fixes and are **out of scope** for the
GPUDirect extension, but must not be regressed by later phases.

### 1.3 cudart changes — the actual GPUDirect data path

`plugins/cudart/backend/CudaRtHandler_memory.cpp`:
- Env + TLS gate `gvirtus_gpudirect_enabled()` at **lines 63-70**: ANDs
  the process-wide `GVIRTUS_GPUDIRECT_ACTIVE` env var with the
  thread-local `gvirtus::communicators::tls_connection_supports_cuda`.
- TLS GPU scratch (`tls_gpu_scratch`, `get_tls_gpu_scratch`) at **lines
  76-95**; TLS D2H host slot (`tls_d2h_slot`) at **lines 109-110+**.
- **D2H fast path** (`cudaMemcpyDeviceToHost` case, entered ~line 367):
  threshold `constexpr size_t kGpuDirectD2HThreshold = 4u*1024u*1024u;`
  at **line 391**; when `gvirtus_gpudirect_enabled() && count >= 4 MB`
  it does `cudaMemcpy(gpu_scratch, src, count, D2D)`, builds an `out`
  Buffer holding only `Add<size_t>(count)`, and calls
  **`result->SetGpuPayload(gpu_scratch, count)`** at **line 409** — this
  is the plugin-code call into the Result GPU side-channel that Phase 3
  of this plan removes.
- **H2D fast path** (`cudaMemcpyHostToDevice` case): reads
  `input_buffer->GetGpuPayload()` / `GetGpuPayloadSize()` at **lines
  348-349** and, if the peer routed the payload into the RX slot's GPU
  shadow, does `cudaMemcpy(dst, gpu_src, count, D2D)` (lines 350-355)
  instead of the host `AssignAll<char>()` + H2D bounce.
- The other, unchanged D2H sites still use the plain copy:
  `out->Add<char>((char*)dst, …)` at lines **498, 665, 743, 904**.

`plugins/cudart/frontend/CudaRtFrontend.h` (lines 110-181) adds thin
static forwarders `AddHostPointerForArgumentsDirect<T>`,
`SetOutputDestination`, `ClearOutputDestination`, `DirectOutputConsumed`
onto `Frontend`. `plugins/cudart/frontend/CudaRt_memory.cpp` uses them at
lines **312, 326, 328, 332**.

### 1.4 Supporting infrastructure OUTSIDE plugins/

**`include/gvirtus/communicators/Buffer.h`** (verified):
- `Add<T>(T *item, size_t n = 1)` — the pointer+count overload plugins
  call for bulk payloads — is at **lines 115-131**; it always
  `memmove`s into `mpBuffer`.
- IoV segment model: `AddRef<T>` at **lines 159-173**, `HasSegments()`
  line 176, `GetLogicalSize()` line 180, `GetIov(std::vector<struct
  iovec>&)` **declared** line 184.
- GPU side-channel `SetGpuPayload` / `GetGpuPayload` /
  `GetGpuPayloadSize` declared **lines 363-365**, backing fields
  `mGpuPayload` / `mGpuPayloadSize` at **lines 386-387**.
- **`enum class SegKind { Inline, HostRef };`** at **line 371**, with the
  extensibility comment at **lines 369-370**: *"Kind is intentionally
  extensible: a future GpuRef can fold the GPUDirect payload (mGpuPayload)
  into this same ordered model."* — this plan realises exactly that.
- `struct Segment { SegKind kind; size_t offset; const void *ptr; size_t
  len; };` at **lines 372-377**.

**`src/communicators/Buffer.cpp`** (174 lines): `GetIov()` at **lines
127-147** (emits `struct iovec` for Inline arena slices and HostRef
borrowed pointers, plus a trailing inline run); `SetGpuPayload` /
`GetGpuPayload` / `GetGpuPayloadSize` at **lines 149-156**; `Dump()` at
**lines 159-174** frames with `GetLogicalSize()` and calls `WriteIov`
when segments exist.

**`include/gvirtus/communicators/Result.h`** — `SetGpuPayload` /
`GetGpuPayload` / `GetGpuPayloadSize` declared **lines 65-67**, fields
lines 73-74. **`src/communicators/Result.cpp`** defines them **lines
37-44**. `GetOutputBuffer()` at line 17.

**`include/gvirtus/communicators/Communicator.h`**:
- Default `WriteIov(const struct iovec*, size_t)` at **lines 93-104** —
  the **blind per-fragment `std::memcpy`** fallback used by every
  non-UCX transport. *Unsafe if ever handed a device pointer.* This is
  the single most important correctness fix in this plan (Phase 2).
- `current_frame_gpu(void*&, size_t&)` default at **lines 143-146**;
  `current_connection_supports_cuda()` default `false` at **line 159**.
- `extern thread_local bool tls_connection_supports_cuda;` at **line
  198** (definition in `src/communicators/CommunicatorFactory.cpp` line
  29).

**`src/backend/Process.cpp`**: dispatch loop calls `am::ReadRequest`
(line 119); sets `tls_connection_supports_cuda =
client_comm->current_connection_supports_cuda()` at **lines 153-154**
(reset to `false` line 159); then `am::WriteResponse(… ,
result->GetOutputBuffer(), result->GetGpuPayload(),
result->GetGpuPayloadSize(), …)` at **lines 163-166**.

**`src/communicators/RpcCodec.cpp`**:
- `ReadRequest` (lines 31-82) threads `void*& gpu_payload` /
  `size_t& gpu_payload_size` out-params, filled from
  `c->current_frame_gpu(...)` at line 79.
- `WriteResponse` (lines 84-135) takes `void* gpu_payload,
  size_t gpu_payload_size` params and **hand-builds `struct iovec
  iov[4]`** at **lines 107-124** (header, exec_sec, host bytes, optional
  GPU tail) rather than calling `Buffer::GetIov()`.
- `WriteRequest` (lines 137-180) already builds its iov via a
  `std::vector<struct iovec>` and appends caller payload fragments.

**`src/communicators/ucx/UcxGpu.cpp`** (223 lines): dlsym-based CUDA
resolver, **no static libcudart link**. `is_gpu_pointer(const void*)` at
**lines 168-177** short-circuits on the atomic `g_gpudirect_enabled`
(line 170) then calls a dlsym'd `cudaPointerGetAttributes`.
`alloc_gpu_slot`/`free_gpu_slot` (lines 145-158) via dlsym'd
`cudaMalloc`/`cudaFree`; `probe_gpudirect` (lines 179-218);
`set_gpudirect_enabled`/`gpudirect_enabled` (lines 220-221). Declared in
`src/communicators/ucx/UcxInternal.h` lines 40-68. Note: there is **no**
existing D2H device→host memcpy helper here — `DeviceMemcpyD2H` is net
new (Phase 0).

**`src/communicators/ucx/UcxRma.cpp`**: `WriteIovRma` at **line 330**;
finds the biggest fragment (loop lines 383-390) and calls
**`const bool big_is_gpu = is_gpu_pointer(iov[biggest_idx].iov_base);`**
at **line 402** — probing only the biggest fragment. `use_zerocopy`
forced when `big_is_gpu` (lines 407-410); `UCS_MEMORY_TYPE_CUDA`
registration at line 491.

### 1.5 The un-migrated call sites (the target of this plan)

The **same** "copy a big device-resident buffer back to the caller via
`out->Add<char>(ptr, n)` with no GPUDirect awareness" pattern exists,
**unchanged from master**, in three untouched backends:

| File | Line | Code |
|---|---|---|
| `plugins/cudadr/backend/CudaDrHandler_memory.cpp` | 74-81 | `MemcpyDtoH`: `void *dstHost = new char[ByteCount];` → `cuMemcpyDtoH(dstHost, srcDevice, ByteCount);` → `out->Add<char>((char*)dstHost, ByteCount);` → `delete[] (char*)dstHost;` |
| `plugins/cublas/backend/CublasHandler_Helper.cpp` | 132 | `out->Add<char>((char*)y, n * elemSize);` |
| `plugins/cublas/backend/CublasHandler_Helper.cpp` | 155 | `out->Add<char>((char*)B, rows * cols * elemSize);` |
| `plugins/cudnn/backend/CudnnHandler.cpp` | 4038 | `out->Add<char>((char*)arrayOfElements, elementsToWrite * getCudnnTypeSize(attrType));` |

Note on cudadr vs cublas/cudnn (this differs and matters — see Risks):
- **cudadr `MemcpyDtoH`** copies **device** memory (`srcDevice`) to a
  freshly `new`'d host buffer, then copies that host buffer into the
  Buffer. Both the `new[]` bounce *and* the `cuMemcpyDtoH` become
  redundant once `Add()` can borrow `srcDevice` directly.
- **cublas `y` / `B`** and **cudnn `arrayOfElements`**: here the pointer
  handed to `Add<char>` may already be **host** memory (e.g. cudnn's
  `arrayOfElements` comes from `in->Assign<char>(...)`, i.e. it points
  into the *input Buffer's host arena* — line 4013). For those, the
  device-detect in `Add()` will correctly return false and fall through
  to today's copy. This is the key reason the design auto-detects inside
  `Add()` rather than adding a new device-specific method: the same call
  site is correct whether the pointer is host or device.

### 1.6 What the docs already say about extending GPUDirect

`docs/GPUDIRECT.md` §10 "Known limitations → Handler GPU-awareness
coverage" (**lines 1073-1082**) already flags this exact gap:

> Only `CudaRtHandler_memory::Memcpy` (`HostToDevice` case) reads the
> Buffer's GpuPayload. Other handlers (cuBLAS GEMM with > 4 MB args,
> cuDNN conv with large tensors) would receive a Buffer with a "hole"
> … extending GPUDirect to cover cuBLAS/cuDNN handlers needs each
> handler to call `GetGpuPayload` and route accordingly. ~10–20 LOC per
> handler.

**This plan supersedes that suggested approach.** The doc envisioned each
handler calling `GetGpuPayload` explicitly (which would grow the
plugin-facing surface). The chosen design instead makes the **existing**
`Add()` overload auto-detect device memory, so plugin code shrinks rather
than grows and **no new Buffer method is exposed to plugins**. Phase 5's
migration reconciles the doc: after this work, GPUDIRECT.md §10 should be
updated to state the coverage gap is closed via `Add()` auto-detection,
not via per-handler `GetGpuPayload` calls.

`docs/IOV_REFACTOR.md` documents the `SegKind { Inline, HostRef }` model
and the "byte-identical wire output" invariant that Phase 1 must
preserve. Its recurring rule ("extends, doesn't rewrite"; "Plugin-facing
APIs unchanged") is the governing constraint for this plan.

---

## 2. Goal

Extend zero-copy GPUDirect from **cudart-only** to **cudadr, cublas, and
cudnn**, so that a large device-resident buffer returned to the caller is
peer-DMA'd straight from the GPU (on RDMA-capable UCX connections)
instead of being bounced through host memory, **with these hard
constraints**:

1. **Zero new Buffer methods exposed to plugin code.** Plugins keep
   calling exactly the methods they call today — chiefly
   `Buffer::Add<T>(T*, size_t)`. The device-vs-host decision moves
   *inside* that existing overload.
2. **Identical behaviour when GPUDirect is inactive** (the common case):
   gated behind the existing cheap atomic `g_gpudirect_enabled` check
   first, then a 4 MB size threshold, so host-only builds and
   sub-threshold buffers pay essentially nothing and produce
   byte-identical wire output (the IOV_REFACTOR invariant).
3. **Correctness on every transport, including plain TCP**, by making the
   `Communicator::WriteIov` default fallback GPU-safe — today it would
   corrupt/crash on a device pointer.

The mechanism: fold the GPUDirect payload into the Buffer's existing
ordered segment model as a new **private** `SegKind::GpuRef` (the
"future GpuRef" the Buffer.h comment already anticipates), retire the
parallel `mGpuPayload` side-channel as plugin-facing, and have the
Communicator layer (not the plugins) decide per-fragment how each device
fragment travels.

---

## 3. Phased Implementation Plan

Ordering is bottom-up: shared primitive → Buffer → safe fallback →
codec/Result → UCX fast path → plugin migration → validation. Each phase
leaves the tree building and the existing cudart GPUDirect path working,
so phases can land as separate commits.

### Phase 0 — Extract device-memory primitives into shared core code

**New files:** `include/gvirtus/communicators/DeviceMemory.h`,
`src/communicators/DeviceMemory.cpp`.

**What:** Move the dlsym-based CUDA-runtime detection/copy logic out of
the UCX-specific `UcxGpu.cpp` into transport-agnostic communicator code,
so Buffer and the base Communicator (neither of which may depend on UCX)
can use it. Expose:

```cpp
namespace gvirtus::communicators {
  bool   IsDevicePointer(const void *p);         // dlsym'd cudaPointerGetAttributes
  void   SetDeviceProbeEnabled(bool on);         // wraps the atomic gate
  bool   DeviceProbeEnabled();
  bool   DeviceMemcpyD2H(void *dst_host,         // dlsym'd cudaMemcpy(...,D2H)
                         const void *src_gpu, std::size_t n);
}
```

- `IsDevicePointer` is the exact logic of `UcxGpu.cpp::is_gpu_pointer`
  (lines 168-177): NULL check → `!DeviceProbeEnabled()` short-circuit →
  `std::call_once` load of `cudaPointerGetAttributes` → `type == 2||3`.
- `DeviceMemcpyD2H` is net new: `std::call_once` dlsym of `cudaMemcpy`
  (candidates `libcudart.so.12/.11/.so`, mirroring `load_cuda_device_funcs`
  at `UcxGpu.cpp` lines 83-106), then `fn(dst, src, n, /*D2H=*/2)`.
- The atomic gate: **reuse the existing `g_gpudirect_enabled` plumbing**
  rather than adding a second flag. Cleanest: move the atomic and its
  `set_gpudirect_enabled` / `gpudirect_enabled` accessors
  (`UcxGpu.cpp` lines 109, 220-221) into `DeviceMemory.cpp` and have
  `ucx_internal::set_gpudirect_enabled` / `gpudirect_enabled` /
  `is_gpu_pointer` become **thin forwarders** to the new
  `SetDeviceProbeEnabled` / `DeviceProbeEnabled` / `IsDevicePointer`. This
  keeps a single source of truth for "is GPUDirect active" across UCX and
  core, and keeps `UcxRma.cpp`'s `is_gpu_pointer` call site (line 402)
  working unchanged in Phase 0.

**Where the flag gets set (unchanged):** UCX's `probe_gpudirect`
(`UcxGpu.cpp` line 179) still runs at startup and still calls
`set_gpudirect_enabled(...)` — now forwarding into the shared atomic. No
new probe is introduced.

**Build:** add `DeviceMemory.cpp` to the `libgvirtus-communicators`
target (same library that Buffer/Result/CommunicatorFactory live in, and
that both backend and plugins already link — see Communicator.h line
196-197 note). No new libcudart link dependency (dlopen only), so
host-only / no-driver dev-container builds are unaffected — this is the
same property that lets `UcxGpu.cpp` compile today without a GPU.

**Why:** Buffer.h/.cpp cannot include UCX headers; the detection logic
must live somewhere both the marshaller and the base transport can reach.
This is the enabling refactor for everything downstream.

### Phase 1 — Buffer: `SegKind::GpuRef`, `Add()` auto-detect, tagged `GetIov`, `HasGpuSegments`

**Files:** `include/gvirtus/communicators/Buffer.h`,
`src/communicators/Buffer.cpp`.

1. **New private segment kind.** Extend the enum at `Buffer.h` line 371
   to `enum class SegKind { Inline, HostRef, GpuRef };`. This is
   **private** (inside `class Buffer`'s private section) — not part of
   any public signature, so it does not touch plugin API. This is exactly
   the "future GpuRef" the existing comment (lines 369-370) predicted.

2. **`Add<T>(T *item, size_t n)` auto-detect** (Buffer.h lines 115-131).
   Insert, before the `memmove` at line 128, a guarded probe:

   ```cpp
   size_t size = safe_sizeof<T>() * n;
   Add(size);                                  // length prefix (unchanged)
   // GpuRef fast path: borrow a device pointer instead of copying.
   if (DeviceProbeEnabled() &&                 // cheap atomic, ~0 cost when off
       size >= kGpuRefThreshold &&             // 4 MB, mirrors CudaRtHandler
       IsDevicePointer(item)) {                // real driver call, gated above
       mSegments.push_back(Segment{SegKind::Inline, mInlineConsumed,
                                   nullptr, mLength - mInlineConsumed});
       mInlineConsumed = mLength;
       mSegments.push_back(Segment{SegKind::GpuRef, 0,
                                   static_cast<const void*>(item), size});
       mExternalBytes += size;
       return;                                 // NO memmove — bytes stay on GPU
   }
   // …unchanged copy-in path (realloc + memmove) below…
   ```

   Bookkeeping mirrors `AddRef()` (lines 159-173) exactly — same
   Inline-flush-then-external-segment pattern — the only difference being
   `SegKind::GpuRef` instead of `HostRef`. `kGpuRefThreshold =
   4u*1024u*1024u` matches `kGpuDirectD2HThreshold` (CudaRtHandler_memory
   line 391). Because `Add<T>` is a header template, `Buffer.h` must
   `#include "gvirtus/communicators/DeviceMemory.h"` (both are in the
   public include tree — no new plugin include is *required* of plugins;
   they already include Buffer.h transitively).

   **Ordering of the gate matters** (a documented cost, §5): the atomic
   `DeviceProbeEnabled()` is checked *first* so that when GPUDirect is off
   (host-only builds, TCP-only backends, frontends) not a single
   `cudaPointerGetAttributes` call is issued; the size threshold is
   checked *before* the real probe so tiny buffers never pay for it
   either.

3. **Tagged `GetIov`.** A plain `struct iovec` has no room for a
   memory-kind flag, so introduce (in `Buffer.h`, communicator-layer
   only):

   ```cpp
   struct IovFrag { void *base; std::size_t len; bool is_device; };
   void GetIov(std::vector<IovFrag> &out) const;
   ```

   `GetIov` (Buffer.cpp lines 127-147) tags each fragment: Inline / HostRef
   → `is_device = false`, GpuRef → `is_device = true`. **Signature
   change:** `GetIov(std::vector<struct iovec>&)` becomes
   `GetIov(std::vector<IovFrag>&)`. This is safe under the "no new plugin
   API" constraint because `GetIov` is called **only** by
   communicator-layer code — `Buffer::Dump()` (Buffer.cpp line 171),
   `RpcCodec.cpp`, `UcxRma.cpp` — **never by any plugin** (verified: the
   only `GetIov` references in the repo are in those communicator files).
   All call sites are updated in their respective phases. Keep an internal
   overload or an inline `iovec`-emitting adapter where a caller genuinely
   only needs host fragments, to minimise churn.

4. **`HasGpuSegments()`** — new small query, communicator-layer only:

   ```cpp
   bool HasGpuSegments() const;  // true iff any Segment.kind == GpuRef
   ```

   Used by Result (Phase 3) to derive the GPU-payload existence without
   the `SetGpuPayload` side-channel. Not called by plugins.

5. **Retire `mGpuPayload` (internal only).** `SetGpuPayload` /
   `GetGpuPayload` / `GetGpuPayloadSize` (Buffer.h 363-365, Buffer.cpp
   149-156) can be **removed** once nothing calls them — the H2D reader in
   CudaRtHandler_memory (lines 348-349) is migrated in Phase 5, and the
   D2H writer's `result->SetGpuPayload` in Phase 3/5. If a staged rollout
   is preferred, keep the fields but have them derived from the GpuRef
   segment; the end state is a single representation (the segment list).

**Why:** This is the crux — it satisfies constraint (1). Every plugin
that already calls `Add<char>(ptr, n)` for a bulk payload now transparently
gets zero-copy when `ptr` is device memory and GPUDirect is active, and
byte-identical copy behaviour otherwise. Update `tests/test_buffer_iov.cpp`
(§Phase 6).

### Phase 2 — `Communicator::WriteIov` default fallback becomes GPU-safe

**File:** `include/gvirtus/communicators/Communicator.h` (lines 93-104).

Change the default `WriteIov` to take the tagged fragment list and branch
per-fragment:

```cpp
virtual size_t WriteIov(const IovFrag *iov, size_t iov_count) {
    // total = Σ len; allocate host staging buffer
    for each fragment:
        if (frag.is_device)
            DeviceMemcpyD2H(buf + off, frag.base, frag.len);  // safe bounce
        else
            std::memcpy(buf + off, frag.base, frag.len);      // as today
    return Write(buf, total);
}
```

`WriteFrame` (lines 111-116), which forwards to `WriteIov`, updates its
signature accordingly.

**Why:** This is the single correctness-critical fix of the whole plan.
Today the default fallback (lines 99-101) does a **blind `std::memcpy`
from `iov[i].iov_base`** — if any transport other than UCX (plain
`TcpCommunicator`, or the loopback test transport) is ever handed a
GpuRef fragment, it dereferences a device pointer from host code and
segfaults/corrupts. Routing device fragments through `DeviceMemcpyD2H`
makes this **the one place** a non-RDMA-capable connection safely bounces
GPU data through host memory — so no plugin ever needs its own bounce
logic, and mixed-transport backends (UCX-RDMA + UCX-TCP + plain TCP on
one backend, per `current_connection_supports_cuda`) stay correct. This
directly addresses the "Phase 5 iov-split breaks plain TCP" class of bug
documented in `docs/GPUDIRECT.md` §5.2 / `docs/UCX_OPTIMIZATIONS.md` §7.

### Phase 3 — RpcCodec simplification + Result cleanup

**Files:** `src/communicators/RpcCodec.cpp`,
`include/gvirtus/communicators/RpcCodec.h`, `Result.{h,cpp}`,
`src/backend/Process.cpp`.

1. **`WriteResponse`** (RpcCodec.cpp 84-135): drop the `void* gpu_payload,
   size_t gpu_payload_size` parameters and the hand-rolled `struct iovec
   iov[4]` (lines 107-124). Instead:

   ```cpp
   std::vector<IovFrag> frags;
   frags.push_back({&rh, sizeof(rh), false});
   frags.push_back({&server_exec_sec, sizeof(double), false});
   if (output_buffer) output_buffer->GetIov(body);   // tagged, incl. GpuRef
   // append body frags…
   c->WriteFrame(frags.data(), frags.size());
   ```

   The GpuRef tail is now just another tagged fragment in the Buffer's
   own iov — UCX peer-DMAs it, TCP bounces it (Phase 2). This removes the
   parallel GPU-payload channel from the codec entirely.

2. **`ReadRequest`** (RpcCodec.cpp 31-82): drop the `void*& gpu_payload,
   size_t& gpu_payload_size` out-params (lines 33, 38-39, 78-79).
   Reconstruct the incoming Buffer so any GPU-resident tail is represented
   as a `GpuRef` segment the Buffer already understands (the receive side
   in UCX lands it in the slot's GPU shadow; the codec records it as
   GpuRef pointing at that shadow), rather than a side-channel out-param.

3. **`Result`** (Result.h 65-67 / Result.cpp 37-44): retire
   `SetGpuPayload` / `GetGpuPayload` / `GetGpuPayloadSize` as
   plugin-facing. `Result` can keep the values internally but should
   **derive** them from `GetOutputBuffer()->HasGpuSegments()` (Phase 1) —
   `HasGpuSegments()` is communicator-layer, so this is compliant. Plugin
   code stops calling `result->SetGpuPayload(...)`.

4. **`Process.cpp`** (lines 163-166): change the `WriteResponse` call to
   drop the `result->GetGpuPayload(), result->GetGpuPayloadSize()`
   arguments. The `tls_connection_supports_cuda` set/reset (lines 153-159)
   stays exactly as is — it still gates whether the active connection may
   use GPUDirect, and it is what drives `DeviceProbeEnabled()` behaviour
   per-call.

**Why:** collapses two representations of "there is a GPU tail" (the
segment model vs the `gpu_payload` out-param) into one — the segment
model — which is a prerequisite for the plugin migration in Phase 5 to be
uniform. Nothing here touches plugin API.

### Phase 4 — UCX `WriteIovRma` reads the tag instead of re-probing

**File:** `src/communicators/ucx/UcxRma.cpp`.

`WriteIovRma` (line 330) currently takes `const struct iovec*` and
recomputes `is_gpu_pointer(iov[biggest_idx].iov_base)` at **line 402**.
Change its signature to accept `const IovFrag*` (the tagged list plumbed
through `WriteFrame`/`WriteIov`) and:

- Replace line 402's probe with `const bool big_is_gpu =
  iov[biggest_idx].is_device;` — **no driver call**, the Buffer already
  decided this in Phase 1's `Add()`.
- Generalise: instead of only trusting the biggest fragment, register
  **any** fragment whose `is_device` is true with `UCS_MEMORY_TYPE_CUDA`
  (line 491) and force zerocopy if *any* device fragment is present, not
  only when the biggest happens to be device. In practice today's payloads
  put the device data in the biggest fragment, but this removes a latent
  correctness cliff (a smaller GpuRef fragment ahead of a big host one
  would currently be mis-registered).

**Why:** saves one redundant `cudaPointerGetAttributes` per large RMA
send, and makes GPU-memory-type registration correct for arbitrary
fragment layouts rather than the biggest-only heuristic. The
`use_zerocopy` forcing logic (lines 407-410) stays; it now keys off the
tag. Preserves the `big_is_gpu`-forces-zerocopy invariant the comment at
lines 398-406 relies on (staged path can't memcpy device memory through
the CPU).

### Phase 5 — Migrate the three plugin call sites

**Files:** `plugins/cudadr/backend/CudaDrHandler_memory.cpp`,
`plugins/cublas/backend/CublasHandler_Helper.cpp`,
`plugins/cudnn/backend/CudnnHandler.cpp`, and the D2H writer in
`plugins/cudart/backend/CudaRtHandler_memory.cpp`.

1. **cudadr `MemcpyDtoH`** (lines 74-81) becomes:

   ```cpp
   CUDA_DRIVER_HANDLER(MemcpyDtoH) {
       CUdeviceptr srcDevice = input_buffer->Get<CUdeviceptr>();
       size_t ByteCount = input_buffer->Get<size_t>();
       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
       out->Add<char>((char *)srcDevice, ByteCount);  // same call — no cuMemcpyDtoH, no new/delete
       return std::make_shared<Result>(CUDA_SUCCESS, out);
   }
   ```

   `Add()` (Phase 1) detects `srcDevice` is device memory (when GPUDirect
   active + ≥4 MB) and borrows it as GpuRef; otherwise it copies. The
   `new char[ByteCount]` / `cuMemcpyDtoH` / `delete[]` all disappear. Note
   the exit-code semantics: today the handler returns `cuMemcpyDtoH`'s
   result; the zero-copy form returns `CUDA_SUCCESS` because the actual DMA
   happens in the transport. If callers rely on a DtoH error being
   surfaced, keep a lightweight validity check on `srcDevice`, or retain
   `cuMemcpyDtoH` only on the copy fallback. **Open question — see §5.**

2. **cublas** (`CublasHandler_Helper.cpp` lines 132, 155): the
   `out->Add<char>((char*)y, …)` / `((char*)B, …)` calls are **left
   textually identical** — no code change is strictly required. Verified
   (§5.2) that `y`/`B` are host pointers (`in->Assign<void>()`), so this
   is purely a verification step, not an edit.

3. **cudnn** (`CudnnHandler.cpp` line 4038): same — `arrayOfElements` at
   this site derives from `in->Assign<char>(...)` (line 4013), i.e. it
   points into the **host** input arena, so `IsDevicePointer` returns
   false and the copy path is taken unchanged. This call site is therefore
   a **no-op migration**: it is safe precisely because `Add()`
   auto-detects. (This is worth an explicit note in the code review so a
   reader doesn't expect a behaviour change here.)

4. **cudart D2H writer** (`CudaRtHandler_memory.cpp` lines 392-415):
   simplify to stop calling `result->SetGpuPayload(gpu_scratch, count)`
   (line 409). Instead build `out` and call `out->Add<char>(gpu_scratch,
   count)` like every other plugin — `gpu_scratch` is device memory, so
   Phase 1 records it as GpuRef and Phase 3's `HasGpuSegments()` lets
   Result/Process/RpcCodec route it. The TLS scratch pattern
   (`get_tls_gpu_scratch`, lines 76-95) **stays** — it is exactly the
   lifetime guarantee Phase 1 requires (see §5). The H2D reader (lines
   342-355) can likewise be simplified to consume the GpuRef segment
   rather than `GetGpuPayload()`, once ReadRequest (Phase 3) reconstructs
   the input Buffer with a GpuRef tail.

**Why:** this is the payoff — three plugins gain GPUDirect for free, the
cudart plugin drops its bespoke `SetGpuPayload` call, and the
`docs/GPUDIRECT.md` §10 coverage gap closes. **No plugin gained a new
Buffer/Result method; cudadr lost `new`/`delete`/`cuMemcpyDtoH`; cudart
lost `SetGpuPayload`.** Net plugin API surface *shrinks*.

### Phase 6 — Testing & validation

Detailed in §4.

---

## 4. Testing & Validation Strategy

Two framework-free unit tests already exist and are the primary
regression gate (they build without CUDA/UCX/log4cplus):
`tests/test_buffer_iov.cpp` (143 lines; build recipe in its header,
target at `tests/CMakeLists.txt` lines 75-86) and
`tests/test_protocol_loopback.cpp` (125 lines; target lines 91+).

### Per-phase validation

- **Phase 0.** Add a `DeviceMemory` unit check: with the probe disabled,
  `IsDevicePointer(anything)` returns false and issues **no** driver call
  (assert via a dlsym-injected counter, or simply that it returns false
  on a host pointer with the gate off). Confirm host-only build still
  links `libgvirtus-communicators` with no libcudart dependency
  (`ldd`/`readelf -d` shows no `libcudart` NEEDED entry).

- **Phase 1.** Extend `tests/test_buffer_iov.cpp`. The existing test
  asserts `AddRef` produces **byte-identical** wire output to
  `Add<T>(ptr, n)` and correct IoV ordering — preserve those. Add cases:
  (a) with the probe **disabled**, `Add<char>(host_ptr, big_n)` produces
  **exactly** the legacy contiguous fragment and `!HasGpuSegments()`
  (proves zero behaviour change in the common case — the IOV_REFACTOR
  invariant); (b) with a **mocked** `IsDevicePointer` returning true (the
  test can toggle it via `SetDeviceProbeEnabled` + a dlsym seam, since no
  real GPU is present in CI), `Add<char>(fake_dev_ptr, ≥4MB)` records a
  `GpuRef` segment, `HasGpuSegments()` is true, `GetIov` emits the
  fragment with `is_device == true`, and the length-prefix framing is
  byte-identical to the copy path; (c) sub-threshold device pointer →
  copy path, `!HasGpuSegments()`.

- **Phase 2.** In `test_buffer_iov.cpp`'s `MemComm`-style transport,
  drive the tagged `WriteIov` with a mix of host and (mocked) device
  fragments and assert the concatenated output equals the logical
  message — i.e. the device fragment went through `DeviceMemcpyD2H`
  (mockable) rather than a raw memcpy. This is the plain-TCP-safety
  regression test.

- **Phase 3.** Update `tests/test_protocol_loopback.cpp` — it already
  round-trips `WriteRequest → ReadRequest → WriteResponse → ReadResponse`
  over the base-class framing with an `AddRef` payload. After the codec
  signature change it must compile and pass unchanged for the host path,
  and a new case should confirm a Buffer carrying a (mocked) GpuRef tail
  survives the round trip with the GPU bytes landing where a device
  fragment is expected. Also assert `WriteResponse` no longer takes
  `gpu_payload` params (compile-time).

- **Phase 4.** Requires real UCX + GPU; covered by the smoke test below.
  Add an assert/log confirming `big_is_gpu` is read from the tag, e.g. a
  `ucx_debug_log("WriteIovRma: is_device from tag=%d", ...)`.

- **Phase 5.** Per-plugin: build each plugin and confirm the migrated
  call sites compile; confirm cudadr `MemcpyDtoH` no longer references
  `cuMemcpyDtoH`/`new`/`delete` (grep), and cudart no longer calls
  `result->SetGpuPayload` (grep returns 0 hits across `plugins/`).

### End-to-end smoke test

Use `examples/simple_matrix/simple_matrix.cu` — the canonical GPUDirect
benchmark in `docs/GPUDIRECT.md` §7 (2× cudaMemcpy H2D + cublasSgemm + 1×
cudaMemcpy D2H, wrapped in cudaEvents). Run with a large N (≥ 4 MB
payloads) on the RDMA path:

1. Backend with `GVIRTUS_GPUDIRECT=1` and `UCX_TLS` including an RDMA
   lane; frontend without the env var.
2. Enable debug logging (`GVIRTUS_LOGLEVEL=DEBUG`) to surface the
   established `ucx_debug_log` lines — `WriteIovRma(zerocopy) …
   biggest_idx=…` (UcxRma.cpp line 428), `WriteIovRma(B3 gpu-split) …`
   (line 564). Confirm the **cublas** GEMM-result D2H (`y`/`B`) now trips
   the zerocopy path where before it took the staged/host path — the log
   should show a device-registered fragment for a call that previously
   didn't. cudart backend log prefixes are `[GVIRTUS]` /
   `[GVS]`-style; UCX uses `ucx_debug_log`.
3. Compare wall-clock D2H time against the pre-migration baseline in
   GPUDIRECT.md §7 tables — the extended plugins should match cudart's
   D2H speedup profile at large N.
4. **Negative/fallback test:** re-run the *same* binary over **plain
   TCP** (`TcpCommunicator`, no RDMA lane, GPUDirect probe fails →
   `g_gpudirect_enabled` false). Assert: correct results, no crash, and
   `HasGpuSegments()` never true (so Phase 2's device-bounce path is not
   even exercised — the probe short-circuits at `Add()`). Then force the
   device-fragment bounce by running UCX-**TCP** with GPUDirect probed on
   but no RDMA lane, exercising Phase 2's `DeviceMemcpyD2H` fallback, and
   confirm correctness. This reproduces and guards against the
   `docs/GPUDIRECT.md` §5.2 "Fase 5 breaks plain TCP" regression class.

### Coverage of the untouched plugins

Run the existing `tests/test_cublas.cu`, `tests/test_cudnn.cu`,
`tests/test_cudadr.cu` (GPU required) before and after to confirm no
functional regression on non-GPUDirect builds.

---

## 5. Risks & Open Questions

### 5.1 Pointer lifetime contract (newly load-bearing)

Once `Add()` can skip the copy for a device pointer, that pointer **must
remain valid until the Communicator has actually transmitted the
response** — not merely until the plugin handler returns. `AddRef`'s
existing contract (Buffer.h lines 155-157: *"The caller MUST keep `ptr`
valid until the buffer has been sent"*) now silently applies to any
`Add<T>(ptr, n)` that hits the GpuRef path.

- **Safe by construction — cudadr `MemcpyDtoH`:** `srcDevice` is a
  *caller-owned, long-lived* device allocation the frontend won't free
  mid-call. Borrowing it is safe.
- **Requires care — any handler using a *scratch* device buffer:** the
  cudart D2H writer already gets this right by staging into a
  **thread-local** `tls_gpu_scratch` (CudaRtHandler_memory lines 76-95)
  that outlives the handler frame. Any future handler that allocates a
  temporary device buffer to stage a copy **must** use a TLS/longer-lived
  allocation, never a function-scope buffer freed at handler return.
  Document this next to the migrated call sites.

### 5.2 Are cublas `y`/`B` and cudnn `arrayOfElements` long-lived?

**Resolved during review** (checked directly against
`plugins/cublas/backend/CublasHandler_Helper.cpp`): both cublas sites are
the **same shape as cudnn**, not an open question.

- **cudnn line 4038:** `arrayOfElements` points into the input Buffer's
  **host** arena (`in->Assign<char>`, line 4013) — `IsDevicePointer`
  returns false, copy path taken, **no lifetime issue**. No-op migration.
- **cublas line 132** (`GetVector` handler, ~line 125): `void* y =
  in->Assign<void>();` — also the **host** input arena, not device
  memory. `cublasGetVector(n, elemSize, x, incx, y, incy)` copies
  device source `x` into this host-resident `y` via cuBLAS's own D2H
  copy; `y` is just the frontend-reserved landing spot.
- **cublas line 155** (`GetMatrix` handler): `void* B = in->Assign<void>();`
  — identical pattern to `y` above.

So all three of cublas-`y`, cublas-`B`, and cudnn-`arrayOfElements` are
host pointers today, and `Add()`'s auto-detect correctly falls through to
the unchanged copy path for all of them — **no lifetime risk, no
provenance audit needed, migration is a true no-op for these three call
sites.** The only call site in this batch where `Add()`'s GpuRef path
actually activates is **cudadr's `MemcpyDtoH`** (`srcDevice`, a genuine
`CUdeviceptr`) and, after Phase 5.4, **cudart's D2H writer**
(`gpu_scratch`, already a `cudaMalloc`'d TLS buffer). This narrows the
lifetime-contract concern in §5.1 to exactly those two sites — the design
still generalizes correctly to cublas/cudnn, it just doesn't have
anything to bite on there yet. If a *future* cublas/cudnn routine hands
`Add()` a genuine device pointer (e.g. a raw `cudaMemcpy`-style GEMM
result path added later), the same TLS-scratch discipline from §5.1
applies.

### 5.3 cudadr exit-code semantics

The zero-copy `MemcpyDtoH` returns `CUDA_SUCCESS` without calling
`cuMemcpyDtoH`, so a genuine DtoH failure (e.g. invalid `srcDevice`) is
no longer surfaced by that call. Mitigations: keep a cheap validity
guard, or perform `cuMemcpyDtoH` only on the copy fallback and rely on the
transport's DMA to fault on a bad device pointer (which then surfaces as a
transport error, not a CUDA error). Decide and document.

### 5.4 Probe cost

Bounded by the atomic-bool gate (near-zero when GPUDirect inactive — the
overwhelming common case: all frontends, all host-only backends, all
sub-4 MB buffers) plus the 4 MB size threshold. When GPUDirect **is**
active and the buffer is large, `Add()` pays one real
`cudaPointerGetAttributes` — acceptable, because that is exactly the case
where skipping a multi-MB host copy is worth far more than one driver
call. This matches the reasoning already in CudaRtHandler_memory lines
384-390.

### 5.5 Thread-safety of shared state

The moved `g_gpudirect_enabled` atomic and the `std::once_flag`-guarded
dlsym loaders are already thread-safe in `UcxGpu.cpp`; keep them so in
`DeviceMemory.cpp`. `Add()` runs on the plugin/handler thread; the
per-connection `tls_connection_supports_cuda` (set in Process.cpp lines
153-159) is thread-local and drives the same flag semantics per worker.
No new shared mutable state is introduced beyond relocating existing
atomics. The Buffer's `mSegments`/`mExternalBytes` are per-Buffer,
per-call — no cross-thread sharing.

### 5.6 Backward compatibility with non-UCX transports

Phase 2 is precisely what makes this safe: plain `TcpCommunicator` and
the loopback test transport inherit the GPU-safe default `WriteIov`, so a
GpuRef fragment is transparently bounced via `DeviceMemcpyD2H`. Without
Phase 2, Phase 1 would reintroduce the `docs/GPUDIRECT.md` §5.2 crash on
those transports. Phase 2 must therefore land **with or before** any
phase that can produce a GpuRef fragment on a non-UCX connection. In
practice the probe won't fire on such connections (their
`current_connection_supports_cuda()` is false → `tls_...` false → gate
off), but Phase 2 is the belt-and-braces guarantee for mixed-transport
backends.

### 5.7 Signature-change blast radius for `GetIov` / `WriteIov` / `WriteIovRma`

Changing `GetIov`, `WriteIov`, `WriteFrame`, and `WriteIovRma` to the
`IovFrag` list touches every override in the transport layer
(`UcxCommunicator`, `TcpCommunicator`) and the base class. This is
mechanical but must be done atomically per phase to keep the tree
building. Verified the only `GetIov` callers are `Buffer::Dump`,
`RpcCodec.cpp`, and `UcxRma.cpp` — **no plugin** — so the constraint holds.

### 5.8 Reconciling the docs

After Phase 5, update `docs/GPUDIRECT.md` §10 "Handler GPU-awareness
coverage" (lines 1073-1082): the gap is closed not by per-handler
`GetGpuPayload` calls (as the doc currently proposes) but by `Add()`
auto-detection folding the payload into `SegKind::GpuRef`. Add a short
note to `docs/IOV_REFACTOR.md`'s segment-model section recording that the
predicted "future GpuRef" (Buffer.h lines 369-370) is now implemented.
Do **not** contradict the existing Fase/Variant naming.

---

## 6. Summary of files touched

| Phase | Files | Nature |
|---|---|---|
| 0 | `include/gvirtus/communicators/DeviceMemory.h` **(new)**, `src/communicators/DeviceMemory.cpp` **(new)**, `src/communicators/ucx/UcxGpu.cpp` + `UcxInternal.h` (forwarders), build files | shared primitive |
| 1 | `include/gvirtus/communicators/Buffer.h`, `src/communicators/Buffer.cpp` | GpuRef, Add() auto-detect, tagged GetIov, HasGpuSegments |
| 2 | `include/gvirtus/communicators/Communicator.h` | GPU-safe WriteIov fallback |
| 3 | `src/communicators/RpcCodec.cpp` + `.h`, `Result.{h,cpp}`, `src/backend/Process.cpp` | codec/Result cleanup |
| 4 | `src/communicators/ucx/UcxRma.cpp` (+ `UcxCommunicator.{h,cpp}` sig) | read tag, not re-probe |
| 5 | `plugins/cudadr/backend/CudaDrHandler_memory.cpp`, `plugins/cublas/backend/CublasHandler_Helper.cpp`, `plugins/cudnn/backend/CudnnHandler.cpp`, `plugins/cudart/backend/CudaRtHandler_memory.cpp` | migrate call sites, drop SetGpuPayload |
| 6 | `tests/test_buffer_iov.cpp`, `tests/test_protocol_loopback.cpp`, docs | validation |

**Constraint check:** no phase adds a method that plugin code must call.
Plugins keep calling `Buffer::Add<T>(T*, size_t)`; everything else
(`SegKind::GpuRef`, `IovFrag`, `GetIov`, `HasGpuSegments`,
`DeviceMemory`, the codec/Result changes) is confined to the
communicator layer. Net plugin API surface **shrinks** (cudadr loses
`cuMemcpyDtoH`/`new`/`delete`; cudart loses `SetGpuPayload`).
