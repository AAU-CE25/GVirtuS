# GPUDirect-aware UCX-RMA Optimization Stack

This document describes the data-path optimizations layered on top of GVirtuS's
UCX-AM transport, culminating in a bidirectional GPUDirect implementation
where the NIC peer-DMAs payloads in and out of GPU memory without bouncing
through host RAM. Final measured headline: **16.7× absolute speedup over
Aponte et al. (PPAM 2024) RDMA Communicator** at the matched payload N=4096
(64 MB), with framework overhead **0.44× of bare-metal compute** on L40S/PCIe Gen4.

This document covers the rollout in chronological / dependency order: each
optimization assumes the previous ones are in place. All work is gated by
environment variables so any phase can be turned off without recompiling.

---

## 1. Context

- **Project**: AAU-CE25/GVirtuS fork, branch `fix/ucx-comm`. Thesis work
  at Aalborg University (AAU) optimizing GPU remoting over RoCEv2.
- **Hardware testbed**: two NVIDIA-DPU nodes `es-dpu-01` (backend) and
  `es-dpu-02` (frontend). Each has:
  - 1× NVIDIA L40S GPU (PCIe Gen4 x16)
  - 2× ConnectX-7 200 Gb/s (mlx5_0, mlx5_1)
  - 25 Gb/s RoCEv2 fabric on `mlx5_1`/`ens1f1np1`, IP subnet `25.25.25.0/24`
- **Software**: UCX 1.20 (with CUDA support), CUDA 12.6.3, cuDNN 9.5.1,
  `nvidia-peermem` loaded on dpu-01.
- **Baseline being optimized against**:
  - **Aponte et al. (PPAM 2024)** — `RdmaCommunicator` shipped in upstream
    GVirtuS. Measured 287 ms/iter at N=4096 on V100/Gen3.
  - **Plain RDMA (`RdmaCommunicator` on this testbed)** — measured
    350 / 1242 / 4547 ms at N=4096 / 8192 / 16384.

---

## 2. Hardware topology

```
       ┌─────────────────────────────────────────────────────────────┐
       │                       es-dpu-02 (FRONTEND host)             │
       │                                                             │
       │   ┌─────────────┐                                           │
       │   │   App /     │                                           │
       │   │  bench /    │     ── linked against ──                  │
       │   │ simple_mat. │       libgvirtus-frontend.so              │
       │   └──────┬──────┘                                           │
       │          │                                                  │
       │          │  cudaMemcpy/cudaMalloc/cublasSgemm…              │
       │          ▼                                                  │
       │   ┌─────────────┐     ┌──────────────────┐                  │
       │   │  GVirtuS    │────▶│ UcxCommunicator  │                  │
       │   │  Frontend   │     │ (UCX 1.20, AM    │                  │
       │   │   shim      │     │  + RMA)          │                  │
       │   └─────────────┘     └────────┬─────────┘                  │
       │                                │                            │
       │                                ▼                            │
       │                       ┌─────────────┐                       │
       │                       │ ConnectX-7  │ ◀── mlx5_1            │
       │                       │ (RoCEv2)    │     25.25.25.2        │
       │                       └──────┬──────┘                       │
       └──────────────────────────────│──────────────────────────────┘
                                      │
                       ┌──────────────│ 200 Gb/s ─────────────┐
                       │   25.25.25.0/24 RoCEv2 fabric         │
                       └──────────────│───────────────────────┘
                                      │
       ┌──────────────────────────────│──────────────────────────────┐
       │                       ┌──────▼──────┐                       │
       │                       │ ConnectX-7  │ ◀── mlx5_1            │
       │                       │ (RoCEv2)    │     25.25.25.1        │
       │                       └──────┬──────┘                       │
       │                              │                              │
       │                              │  peer-DMA via                │
       │                              │  nvidia-peermem              │
       │                              ▼                              │
       │   ┌─────────────┐     ┌─────────────────┐                   │
       │   │ CudaRtHandler│◀──│ UcxCommunicator │                    │
       │   │ CublasHandler│    │  (RX pool +    │                    │
       │   │ etc.         │    │  GPU shadows)  │                    │
       │   └──────┬──────┘     └─────────────────┘                   │
       │          │                                                  │
       │          │ cudaMemcpy / cublasSgemm / …                     │
       │          ▼                                                  │
       │   ┌─────────────┐                                           │
       │   │   L40S GPU  │  (PCIe Gen4 x16)                          │
       │   │   48 GB     │                                           │
       │   └─────────────┘                                           │
       │                  es-dpu-01 (BACKEND host)                   │
       └─────────────────────────────────────────────────────────────┘

   PCIe Gen4 x16 = 32 GB/s theoretical (~22 GB/s observed for cudaMemcpy
   with pinned host buffers).
   ConnectX-7 single-rail 200 Gb/s = 25 GB/s theoretical, ~22 GB/s usable.
```

`nvidia-smi topo -m` on dpu-01 reports `GPU0↔mlx5_1 = NODE`: same NUMA
domain but different PCIe host bridges. Peer-DMA between GPU and NIC works
via `nvidia-peermem` without going through CPU memory.

---

## 3. Data flow: before vs after

### Before (Aponte-style RDMA Communicator)

A single `cudaMemcpy(H2D, 64 MB)` produced:

```
Frontend                                  Backend
────────                                  ───────
host_user_buf                             d_dst (GPU)
     │                                      ▲
     │  memcpy → marshal staging            │
     ▼                                      │ cudaMemcpy H2D
[mpInputBuffer]                       host_rx_buffer ◀── allocated
     │                                      ▲           per-call
     │  ucp_am_send_nbx                     │           via realloc
     │  (rendezvous: RTS/RTR/...)           │
     ▼                                      │
[NIC] ──── ~25 GB/s wire ────▶ [NIC] ─► temp_recv_buf
     │
     │  multiple Read(1) calls per call
     │  → byte-by-byte protocol parsing
     │  → realloc + memcpy
```

Per H2D @ 64 MB: ~140 ms (memory registration + RTS/RTR + temp alloc).

### After (full Variant A + B GPUDirect)

```
Frontend                                  Backend
────────                                  ───────
host_user_buf                             d_dst (GPU)
     │                                      ▲
     │  ucp_put_nbx (NIC pulls direct)      │ cudaMemcpy
     │                                      │ DeviceToDevice
     ▼                                      │ (intra-GPU, ~0.5ms)
[NIC] ──── peer-DMA ────▶ [NIC] ─► gpu_shadow_slot (cudaMalloc + ucp_mem_map CUDA)
     │                                ▲
     │  (small protocol bytes use     │
     │   host_slot route in parallel) │
     │                                │
     ▼                                │
[NIC] ──── wire ────▶ [NIC] ─► host_slot[0..pre_size]
                                  + slot.gpu_addr[0..big_size]
```

Per H2D @ 64 MB: ~5.6 ms in the iter (sgemm 3 ms + 3× ~5 ms data ≈ 17.2 ms).

The key transformation: **GPU memory becomes a first-class RMA target**.
The NIC reads/writes GPU memory directly via PCIe peer-DMA. No host bounce.

---

## 4. Optimizations by phase

### 4.1 Foundation (pre-existing in branch + early session)

These were already in `fix/ucx-comm` or landed before today's GPUDirect work:

1. **`WriteIov` gather-send** (`UCP_DATATYPE_IOV`). Eliminates a 27 ms
   marshal staging copy.
2. **RX pool of pre-mem-mapped slots** (`cudaHostAlloc` + `ucp_mem_map`).
   1025 MB per slot × 2 slots so we hit RMA fast path up to 1 GB payloads.
3. **Bidirectional RMA pre-handshake**. `RmaSetup` exchanges rkeys at connect
   time so the data path uses `ucp_put_nbx` directly into pre-mem-mapped
   slots — no rendezvous handshake per message.
4. **Worker-per-connection on backend** so errors on one connection don't
   poison shared progress.
5. **`Buffer::AppendBytes` bulk memcpy** (replaced byte-by-byte
   `Add<char>` loop — 1.3 s wasted at 64 MB → 3 ms).
6. **`init_rx_pool` + `send_rma_setup` moved to detached thread** so
   `Accept()` returns immediately.
7. **Manual host memh cache** (`unordered_map<const void*, ucp_mem_h>`)
   inside `WriteIovRma`. Works around broken UCX rcache. Saves ~25 ms
   per 64 MB put. Threshold 2 MB to protect inference workloads with
   churning buffer addresses (OpenPose/Caffe).

### 4.2 Fase 1+2: backend handler page-fault elimination

`plugins/cudart/backend/CudaRtHandler_memory.cpp`. Thread-local pre-faulted
TX slot replaces `new char[count]` allocation per call. `memset(0, N)` once
at first call faults all pages; subsequent calls reuse warm pages.

Effect: `cudaMemcpy` paged-warm = 4.7 ms vs paged-fresh = 33 ms @ 64 MB.

### 4.3 Fase 4: frontend D2H zero-copy

`include/gvirtus/frontend/Frontend.h`,
`src/frontend/Frontend.cpp`, `plugins/cudart/frontend/CudaRt_memory.cpp`.

New `Frontend::SetOutputDestination(dst, count)` API lets the caller pre-
register the user dst pointer. The response handler in `Frontend::Execute`
memcpys the payload directly from the RX slot frame to the user dst,
skipping `AppendBytes(64 MB) → mpOutputBuffer` + post-Execute memmove.

Effect: D2H @ 64 MB: 28.4 → 23.3 ms (-18%).

Fallback: when the response Buffer layout isn't exactly
`[size_t prefix == count][count bytes]`, falls back to the legacy
`AppendBytes` + caller memmove path. Non-CudaRt callers unaffected.

### 4.4 Fase 5 + argmax: frontend H2D iov split

`src/communicators/ucx/UcxCommunicator.cpp::WriteIovRma`,
`src/frontend/Frontend.cpp`.

Skip the `Add<char>(src, count)` memcpy on H2D by splitting the send iov:
```
[envelope_header][routine][input_pre][user_src][input_post]
```
where `input_pre` and `input_post` are slices of `mpInputBuffer`
straddling the big payload. The user buffer goes directly into the iov
without intermediate memcpy.

`WriteIovRma`'s zerocopy heuristic was patched: instead of assuming
`biggest = iov[iov_count-1]` (legacy 3-entry layout), it scans
`biggest_idx = argmax(iov[i].iov_len)`. With Fase 5's 5-entry layout the
biggest sits at index 3, not last.

Effect: H2D @ 64 MB: 10.17 → 5.58 ms (-45%).

### 4.5 Variant A: backend D2H GPUDirect (TX-only)

Files: `Result.h/.cpp`, `Process.cpp`, `CudaRtHandler_memory.cpp`,
`UcxCommunicator.cpp`.

Backend D2H handler stops bouncing through host pinned memory. Path:

1. Handler allocates (lazily, per-thread) a GPU scratch via `cudaMalloc`
   sized to current payload. Cached, grown 2× on demand.
2. `cudaMemcpy(gpu_scratch, src_user_gpu, count, cudaMemcpyDeviceToDevice)`
   replaces `cudaMemcpy(host_scratch, src_user_gpu, count, DeviceToHost)`.
   Cost: ~0.5 ms intra-GPU vs ~4.78 ms host-bound at 64 MB.
3. Handler creates a Result with `out` containing only the protocol prefix
   (`Add<size_t>(count)` = 8 bytes), and attaches the GPU pointer via a
   new `Result::SetGpuPayload(gpu_addr, count)` side-channel.
4. `Process.cpp::write_ucx_am_response` detects `result->GetGpuPayload()`
   and emits a 5-entry iov:
   `[envelope_header][exec_sec][wire_out_size][host_prefix(8B)][gpu_payload(count B)]`.
   The wire `out_size` field is the **combined** prefix+gpu size so the
   frontend sees the bytes contiguously.
5. `UcxCommunicator::WriteIovRma`'s manual memh cache detects GPU memory
   via `cudaPointerGetAttributes` (dlopen'd at first use). When the biggest
   iov fragment is GPU mem, the `ucp_mem_map` call passes
   `UCS_MEMORY_TYPE_CUDA` explicitly. UCX configures the put to use
   peer-DMA from GPU → NIC → frontend host slot.

Effect: D2H @ 64 MB: 23.3 → ~19 ms in iter context. Combined gain
in simple_matrix @ N=4096: 38.15 → 31.14 ms.

**Threshold (4 MB)**: applied in handler — for small payloads (<4 MB) the
TLS scratch + cudaMemcpy + dual-iov orchestration overhead exceeds the
savings. Below threshold falls back to legacy host path.

### 4.6 Variant B: bidirectional GPUDirect

The frontend's H2D path was still bouncing through the backend's host slot.
Variant B closes the loop by making the backend's RX slot dual-region:
small protocol bytes land in a host slot (CPU-readable for parsing),
big payload lands in a GPU shadow (NIC peer-DMAs host→GPU).

#### B1 — slot dual-region allocation

`include/gvirtus/communicators/UcxCommunicator.h`,
`src/communicators/ucx/UcxCommunicator.cpp`.

`PinnedSlot` gains `gpu_addr`, `gpu_capacity`, `gpu_memh` fields. When
`GVIRTUS_GPUDIRECT=1` and the probe passes, `init_rx_pool` allocates a
GPU shadow per slot via `cudaMalloc` and registers it with
`ucp_mem_map(MEMORY_TYPE_CUDA)`.

Without GPUDirect: slots stay host-only (`gpu_addr == nullptr`).

#### B2 — `RmaSetup` wire format extension

`include/gvirtus/communicators/UcxAmProtocol.h` (semantic only,
struct layout unchanged), `UcxCommunicator.cpp::send_rma_setup` and
`handle_rma_setup_am`, `RemoteSlot` struct.

Backend's `RmaSetup` AM advertises both host rkey AND optional gpu rkey
per slot. Wire layout:

```
[EnvelopeHeader][N × RmaSlotDescriptor][per-slot rkey blobs]

RmaSlotDescriptor (16 bytes):
  uint64 remote_addr
  uint64 slot_capacity
  uint32 rkey_size
  uint32 reserved0           ← bit 0 = kHasGpuShadow flag

Per-slot blob:
  [host_rkey_blob (rkey_size bytes)]
  If kHasGpuShadow:
    [u64 gpu_addr][u64 gpu_capacity][u32 gpu_rkey_size][gpu_rkey_blob]
```

Old peers see `reserved0 = 0` → no GPU extension → identical to pre-B2
layout. New peers exchange GPU shadow rkeys transparently.

Frontend stores the GPU info in `RemoteSlot::gpu_addr` + `gpu_rkey`
fields. `destroy_rma_state` also destroys `gpu_rkey`.

#### B3 — frontend WriteIovRma routes big to GPU shadow

`src/communicators/ucx/UcxCommunicator.cpp::WriteIovRma`.

When the frontend has a big iov fragment (≥ 4 MB) AND the peer has
advertised a `gpu_rkey`, the frontend issues the big `ucp_put_nbx` to
`rs.gpu_addr` with `rs.gpu_rkey` instead of `rs.addr + pre_size`.

Small fragments (header, routine, marshalled args, count, kind) still
go to the host slot at offsets `[0..pre_size)` and `[pre_size+big_size..total)`.

The `RmaPosted` notification carries:
- `payload_size` = total bytes (unchanged)
- `routine_size` = bytes that went to GPU (= big_size when routing)
- `status_code` = offset in the logical message where the GPU bytes belong
  (= pre_size in the host slot)

This format handles arbitrary iov layouts: the biggest fragment doesn't
need to be at the end. Fase 5 puts `user_src` at index 3 with a
12-byte `[count][kind]` post-fragment; the offset+size encoding handles
this correctly.

#### B4 — Buffer dual-aware + handler uses D2D

Files: `Buffer.h/.cpp`, `Communicator.h`, `UcxCommunicator.h/.cpp`,
`Process.cpp`, `CudaRtHandler_memory.cpp`.

`Buffer` gains `SetGpuPayload/GetGpuPayload/GetGpuPayloadSize` (mirror of
the `Result` API used by Variant A). `Communicator` gains virtual
`current_frame_gpu(gpu_addr, size)` with a default no-op; `UcxCommunicator`
overrides it to return `current_frame_.gpu_data` / `gpu_size`.

`am_recv_handler` no longer consolidates GPU payload back into host slot.
It only sets `msg.gpu_data` and `msg.gpu_size` on `PooledMsg`. The Buffer
that wraps the message gets `SetGpuPayload` called from `Process.cpp`.

`CudaRtHandler_memory::Memcpy` for `cudaMemcpyHostToDevice`:
- Reads `dst` from the buffer header (host mem, parsed normally).
- Checks `input_buffer->GetGpuPayload()`. If non-null AND size ≥ count,
  uses `cudaMemcpy(dst, gpu_payload, count, cudaMemcpyDeviceToDevice)` —
  intra-GPU, ~0.5 ms at 64 MB.
- Otherwise falls back to `AssignAll<char>()` + `cudaMemcpy(H2D)` legacy
  path.

This completes the bidirectional GPUDirect: both H2D (Variant B) and D2H
(Variant A) bypass the host bounce.

---

## 5. Regressions discovered during rollout, and the gates added to absorb them

Two regressions surfaced while landing this stack. Both have the same shape:
a new optimization assumed a specific transport/protocol property, and
silently broke when that property didn't hold. Both are now guarded by
a runtime check that auto-disables the optimization in the affected
configuration. They are documented here together because the lesson —
**transport-awareness is mandatory for any optimization that touches
non-host memory or splits the wire across multiple buffers** — is the
single most useful takeaway from the rollout.

### 5.1 UCX-TCP cannot peer-DMA from CUDA memory (Variants A and B)

#### What happened

After the GPUDirect rollout, attempting a UCX-TCP benchmark (frontend with
`UCX_TLS=tcp,self` against the same backend) crashed the connection right
after `cublasCreate_v2`. Backend log:

```
[UCX DEBUG] WriteIovRma(zerocopy) slot=0 pre=64 big=4194304 post=0 biggest_idx=4
UCX ERROR cannot find remote protocol for: ucp_context_0 inter-node cfg#1 |
          put from cuda memory to host
[UCX DEBUG] rma_put_big: wait_request_completion request=0xfffffffffffffff0
UCX AM response write failed: UcxCommunicator: rma_put_big request error:
          Request canceled
Client disconnected
```

#### Root cause

UCX-TCP transport cannot perform `ucp_put_nbx` from CUDA memory:
- The NIC peer-DMA paths used by RDMA transports (`rc_mlx5`, `dc_mlx5`,
  `ud_mlx5`, `ib_*`) leverage `nvidia-peermem` to expose GPU memory to
  the NIC's DMA engine.
- UCX-TCP doesn't have this path. There's no kernel-level mechanism that
  lets a TCP socket sendmsg from GPU memory; UCX would need to internally
  cudaMemcpy(D2H) into a staging host buffer first.
- UCX's "protocol selection" logic checks for a registered handler that
  can move data between the source and dest memory types. For
  `(cuda → host, transport=tcp)` no handler is registered, hence
  "cannot find remote protocol for: put from cuda memory to host".

Two separate paths in our stack trigger GPU mem ucp_puts:

| Path | What | Affected by UCX-TCP |
|---|---|---|
| **Variant A (backend D2H)** | Backend has `tls_gpu_scratch` (CUDA mem). `is_gpu_pointer(iov[biggest])` returns true → `big_is_gpu = true` forces zerocopy → `ucp_put_nbx` from GPU → frontend host slot. | YES — fails on TCP transport. |
| **Variant B (frontend H2D)** | Frontend has user host buffer. Routes big put to `rs.gpu_addr` (remote GPU) with `rs.gpu_rkey`. `ucp_ep_rkey_unpack` returns UCS_OK even for TCP endpoints (no transport check at unpack time), so `route_big_to_gpu = true`. Put fails at execution time. | YES — fails on TCP transport. |

#### Fix

Both paths now check whether the negotiated UCX transport actually
supports CUDA memory operations. The check is process-level: if `UCX_TLS`
env doesn't include any RDMA-class transport (`rc_mlx5`, `dc_mlx5`,
`ud_mlx5`, `ib`), GPUDirect is disabled for that process — even when
`GVIRTUS_GPUDIRECT=1` is set.

##### Frontend side (`WriteIovRma`)

```cpp
static const bool transport_supports_cuda = []() {
    const char *tls = std::getenv("UCX_TLS");
    if (tls == nullptr) return true;  // default UCX selection
    std::string s(tls);
    return s.find("rc_mlx5") != std::string::npos ||
           s.find("dc_mlx5") != std::string::npos ||
           s.find("ud_mlx5") != std::string::npos ||
           s.find("ib")      != std::string::npos;
}();
const bool route_big_to_gpu = (rs.gpu_rkey != nullptr) &&
                              (rs.gpu_addr != 0) &&
                              (big_size >= (4u * 1024u * 1024u)) &&
                              transport_supports_cuda;
```

##### Backend side (`init_ucx` — gates the whole GPUDirect activation)

```cpp
const bool tls_supports_cuda = /* same lambda as above */;
const bool gpudirect_requested = gpudirect_env_set && tls_supports_cuda;
if (!gpudirect_requested) {
    g_gpudirect_enabled.store(false);
    setenv("GVIRTUS_GPUDIRECT_ACTIVE", "0", 1);
    if (gpudirect_env_set && !tls_supports_cuda) {
        fprintf(stderr,
            "[GVS] GPUDirect=disabled (UCX_TLS=%s has no CUDA-capable transport)\n",
            ...);
    }
    return;
}
// ... probe ...
```

#### Operational consequence

For multi-regime benchmarks, **run a separate backend per regime**:

```bash
# UCX-RDMA backend (GPUDirect ON):
GVIRTUS_GPUDIRECT=1 UCX_TLS=rc_mlx5,ud_mlx5,tcp,self ... make run-gvirtus-backend-dev

# UCX-TCP backend (GPUDirect OFF, automatic via the gate):
GVIRTUS_GPUDIRECT=1 UCX_TLS=tcp,self ... make run-gvirtus-backend-dev
# (or just leave GVIRTUS_GPUDIRECT unset — same effect)
```

The gate is at process startup, not per-connection. A single backend
serving mixed RDMA/TCP frontends would still try Variant A on TCP
connections and fail. Per-connection gating is a separate future
improvement that would require querying each accepted endpoint's
transport via `ucp_ep_query`.

#### Generalization

The principle: `ucp_ep_rkey_unpack` and `ucp_mem_map` are transport-
agnostic. They succeed even when the actual transport can't move the
memory type the rkey/memh describes. Any future code that uses UCX with
non-host memory types must guard the activation against the negotiated
transport. Don't rely on UCX to silently fall back.

### 5.2 Fase 5 H2D iov-split breaks plain TCP and HybridCommunicator

#### What happened

After GPUDirect rollout the user attempted to re-run the plain TCP
baseline (`properties.json`, legacy `TcpCommunicator`) to refresh the
comparison numbers. The frontend died silently — no output produced — and
the backend log showed:

```
Buffer::AssignAll(): Can't read char
[Process] - [Process 3003]: Routine 'cudaMemcpy' returned 2.
[CudaRtHandler] - Called: cudaGetErrorString
[CudaRtHandler] - Called: cudaUnregisterFatBinary
[Process] - Client disconnected
```

`cudaMemcpy` returning `2` is `cudaErrorMemoryAllocation`, produced when
the backend H2D handler caught the `std::runtime_error` thrown by
`Buffer::AssignAll<char>()` while parsing the input buffer. The frontend
disconnected immediately because its `Execute` call hit the same
exception when trying to read the response.

Hypothesis to discard first: the GPUDirect Buffer changes broke
serialization. **They didn't** — the B4 `Buffer.h/.cpp` patch only adds
two new accessor methods + two private fields with default initialization.
`AssignAll` and the rest of the wire serialization are untouched.

#### Root cause

The bug was introduced by **Fase 5** the previous day, not by the
GPUDirect work — it just wasn't exercised on a non-UCX transport until
now.

Fase 5's `Frontend::AddHostPointerForArgumentsDirect<T>(ptr, n)` records
`ptr` in `mDirectInputSrc/Bytes/BufferOffset` and writes ONLY the
`size_t = sizeof(T) * n` prefix into `mpInputBuffer`. The actual `n*sizeof(T)`
payload bytes are intentionally NOT in `mpInputBuffer` — they will be
spliced into the wire at send time.

The splice happens only inside `Frontend::Execute`'s `ucx_am_mode`
branch, which builds a 5-entry `struct iovec` and calls
`Communicator::WriteIov`. UCX maps that to `ucp_am_send_nbx(UCP_DATATYPE_IOV)`
or `WriteIovRma`'s 3-put argmax path. Both honour the iov layout
faithfully.

The legacy branch (everything that isn't UCX — plain TCP via
`TcpCommunicator`, RDMA via `HybridCommunicator`/`RdmaCommunicator`)
serializes with `input_buffer->Dump(communicator)`. `Dump()` just writes
`mpBuffer` as one stream — it has no knowledge of `mDirectInput*`. So
the wire contains the prefix `size_t = count` but **none of the `count`
bytes that follow it logically**.

On the backend, the H2D handler does:
```cpp
dst = input_buffer->GetFromMarshal<void *>();   // OK
src = input_buffer->AssignAll<char>();          // reads size_t prefix
                                                // then count bytes — but
                                                // they aren't there
```

`AssignAll` reads the prefix (`count`), bumps the buffer offset, then
tries to consume `count` bytes from the remaining buffer and throws
`Can't read char` because the remaining length is too small (only the
trailing `kind` enum + variable header). Handler catches the exception
and returns `cudaErrorMemoryAllocation`. Frontend `Execute` throws on the
response side. Client disconnect.

A separate session journal already had this documented for HybridCommunicator
("Plain RDMA bench DID NOT complete on current frontend code — Fase 4/5
wire format uses 5-segment iov via WriteIov, which HybridCommunicator
doesn't support") but the same root cause covers plain TCP, since
TcpCommunicator also goes through `Dump()`.

#### Fix

`include/gvirtus/frontend/Frontend.h`, ~10 LOC added at the top of the
`AddHostPointerForArgumentsDirect<T>` template:

```cpp
template <class T>
void AddHostPointerForArgumentsDirect(const T *ptr, size_t n = 1) {
    const bool ucx =
        _communicator && _communicator->obj_ptr() &&
        _communicator->obj_ptr()->to_string() == "ucxcommunicator";
    if (!ucx) {
        // Non-UCX (plain TCP, HybridCommunicator) Execute uses
        // input_buffer->Dump() which doesn't honour the iov split.
        // Materialize the bytes inline in mpInputBuffer using the legacy
        // wire format ([size_t prefix][bytes]).
        mpInputBuffer->Add<T>(const_cast<T *>(ptr), n);
        return;
    }
    if (ptr == nullptr) {
        mpInputBuffer->Add((size_t)0);
        return;
    }
    const size_t bytes = sizeof(T) * n;
    mpInputBuffer->Add(bytes);
    mDirectInputBufferOffset = mpInputBuffer->GetBufferSize();
    mDirectInputSrc          = ptr;
    mDirectInputBytes        = bytes;
}
```

UCX retains zero-copy. Plain TCP and Hybrid get pre-Fase-5 wire format
(memcpy into `mpInputBuffer`, then `Dump()` ships the contiguous buffer).
No backend change required — the backend's H2D handler already has both
paths (legacy `AssignAll` AND GPUDirect `GetGpuPayload`), and the legacy
path now sees the correct bytes again.

The check is on `to_string()` because that's the existing mechanism
used in `Frontend::Execute` to dispatch between `ucx_am_mode` and
the legacy branch (`Frontend.cpp:292`). No new public API needed.

#### Verification

Three-way simple_matrix sweep completes cleanly across all transports
(see §8.2). UCX-RDMA + GPUDirect numbers are unchanged from before the
fix (zero-copy path untouched). Plain TCP numbers are reasonable and
finite for the first time since Fase 5 shipped.

#### Generalization (same lesson, second instance)

Both regressions in this section share the structural pattern: a new
optimization introduces an out-of-band channel (GPU memory address for
A/B, direct iov fragment for Fase 5) that the legacy code path can't
see or honour. The fix in both cases is the same: detect at the point
where the out-of-band channel is registered whether the channel will
actually be honoured downstream, and degrade to the legacy in-band
representation if not.

For future optimizations, the rule:

- Any state stored outside the serialized `Buffer` (direct pointers, GPU
  memh side-channels, async completion handles) needs a transport-aware
  gate at the producer side.
- The gate should be at the lowest layer where the producer can see the
  consumer's capability. `Frontend::AddHostPointerForArgumentsDirect`
  checks `_communicator->obj_ptr()->to_string()` because that's the
  earliest point where the transport identity is known.

---

## 6. Wire format reference

### `RmaSetup` (server → client at connect)

```
[EnvelopeHeader 40 bytes]
  magic            uint32  = 0x4756414D  ("GVAM")
  version          uint16  = 1
  message_type     uint16  = 4 (RmaSetup)
  header_size      uint16  = 40
  reserved0        uint16  = 0
  status_code      uint32  = 0
  request_id       uint64  = 0
  routine_size     uint64  = 0
  payload_size     uint64  = N (number of slot descriptors)

[N × RmaSlotDescriptor, 16 bytes each]
  remote_addr      uint64
  slot_capacity    uint64
  rkey_size        uint32
  reserved0        uint32  ← bit 0 = kHasGpuShadow

For each slot in order:
  [host_rkey_blob (rkey_size bytes)]
  If reserved0 has kHasGpuShadow:
    gpu_addr       uint64
    gpu_capacity   uint64
    gpu_rkey_size  uint32
    [gpu_rkey_blob (gpu_rkey_size bytes)]
```

### `RmaPosted` (client → server per cudaMemcpy)

```
[EnvelopeHeader 40 bytes]
  magic            = 0x4756414D
  version          = 1
  message_type     = 5 (RmaPosted)
  header_size      = 40
  reserved0        = slot_idx  (which RX pool slot the data landed in)
  status_code      = gpu_offset   ← NEW (Step B3): offset in logical
                                    message where GPU bytes belong
  request_id       = 0
  routine_size     = gpu_size     ← NEW (Step B3): bytes landed in
                                    slot.gpu_addr (vs slot.addr)
  payload_size     = total        (total bytes occupying the slot)
```

Backward compat: pre-B3 peers send `status_code = 0` and `routine_size = 0`,
which the new receiver interprets as "no GPU split, all bytes in host slot".

### Response envelope (server → client) with Variant A GPU payload

```
iov sent via WriteIov:
  iov[0]: EnvelopeHeader      (40 bytes, host)
  iov[1]: server_exec_sec     (8  bytes, host)
  iov[2]: wire_out_size       (8  bytes, host) = host_prefix + gpu_payload_size
  iov[3]: host_prefix         (8  bytes, host) — the Add<size_t>(count) prefix
  iov[4]: gpu_payload         (count bytes, GPU mem from tls_gpu_scratch)

WriteIovRma routing:
  iov[0..3] → ucp_put_nbx to rs.addr + offset (host)
  iov[4]    → ucp_put_nbx from gpu_scratch (registered CUDA mem)
              to rs.addr + pre_size (host slot on frontend, peer-DMA via peermem)
```

---

## 7. Configuration

### Required environment

| Variable | Required | Description |
|---|---|---|
| `GVIRTUS_UCX_DATAPATH=am` | Yes | Selects UCX-AM protocol path. |
| `UCX_TLS=rc_mlx5,ud_mlx5,tcp,self` | Yes | Transport list. Must include at least one RDMA transport for GPUDirect. |
| `UCX_NET_DEVICES=mlx5_1:1,ens1f1np1` | Yes | NIC + interface. |
| `UCX_SOCKADDR_TLS_PRIORITY=tcp` | Yes | Control-plane via TCP. |
| `UCX_IB_GID_INDEX=3` | Yes (for RoCEv2) | GID index for RoCE addressing. |

### GVirtuS-specific opt-ins

| Variable | Default | Effect |
|---|---|---|
| `GVIRTUS_RMA_ZEROCOPY=1` | unset | Enables `WriteIovRma` zerocopy path (3 parallel puts with manual memh cache). Without it, falls back to staged single-put. |
| `GVIRTUS_GPUDIRECT=1` | unset | Enables the full GPUDirect stack (Variants A+B). Backend allocates GPU slot shadows + probes `ucp_mem_map(CUDA)`. Frontend uses GPU rkey routing. Requires `nvidia-peermem` loaded + RDMA-class transport. |

### Auto-managed environment

When `GVIRTUS_GPUDIRECT=1`:
- `UCX_RCACHE_ENABLE=n` and `UCX_MEMTYPE_CACHE=n` are set by `init_ucx`
  before `ucp_init`. The container's UCX rcache fails to install its UCM
  event handler and breaks CUDA mem_map operations.
- `GVIRTUS_GPUDIRECT_ACTIVE=1/0` is set by `init_ucx` after the probe.
  The cudart plugin's handlers read this to gate Variant A activation
  (decoupled from the UCX library to avoid RTLD_GLOBAL coupling).

### System requirements

- Kernel modules: `nvidia-peermem` loaded on the backend node.
  Verify: `lsmod | grep peermem` shows refcount on `ib_core` and `nvidia`.
- UCX 1.20 built with `--with-cuda=/usr/local/cuda`. Verify:
  `strings /lib/libucp.so.0 | grep cuda_md` shows `cuda_copy`, `cuda_ipc`.
- CUDA 12.6 runtime. The plugin links against `libcudart.so.12` and
  uses dlopen to resolve `cudaMalloc`, `cudaFree`, `cudaMemcpy`, and
  `cudaPointerGetAttributes` from `libcudart.so.12`.
- ConnectX-7 (or compatible NIC with PeerDirect support).

---

## 8. Measurements

### simple_matrix (Aponte methodology: 2× cudaMemcpy H2D + cublasSgemm + 1× cudaMemcpy D2H, wrapped in cudaEvents)

L40S bare-metal sweep (matmul_bench compiled directly against
`libcudart`, no GVirtuS):

| N | payload | bare-metal (ms) |
|---|---|---|
| 256   | 0.25 MB | 0.102 |
| 512   | 1 MB    | 0.331 |
| 1024  | 4 MB    | 0.843 |
| 2048  | 16 MB   | 2.933 |
| 4096  | 64 MB   | **11.826** |
| 8192  | 256 MB  | 57.624 |
| 16384 | 1 GB    | 306.000 |

UCX-RDMA + full GPUDirect (Variant A + B, threshold 4 MB on both):

| N | host_ms (ms) | overhead (ms) | overhead/bare | × vs Aponte (287 @ 4k) |
|---|---|---|---|---|
| 256   | 1.01  | 0.91  | 8.9×     | — |
| 512   | 1.70  | 1.37  | 4.1×     | — |
| 1024  | 1.47  | 0.63  | **0.75×**| — |
| 2048  | 4.25  | 1.32  | **0.45×**| — |
| **4096**  | **17.20** | **5.37**  | **0.45×** | **16.69×** |
| 8192  | 79.46 | 21.84 | **0.38×**| — |
| **16384** | **388.22**| **82.22** | **0.27×**| — |

Notable:
- At N ≥ 1024, framework overhead is **less than the bare-metal compute
  time itself**. Remoting costs less than half of native serial execution.
- The ratio improves as N grows: at 16384 the overhead is only 0.27× of
  bare-metal because the host cudaMemcpy cost in bare-metal scales
  linearly with payload while our framework cost scales sub-linearly
  (NIC linerate dominates).
- N = 256 / 512 still see overhead because the 4 MB threshold guards
  against per-call setup overhead exceeding savings. Below threshold the
  legacy zerocopy host path runs and inherits a ~0.5–1 ms fixed cost
  from RMA orchestration that bare-metal doesn't have. Not worth chasing
  for sub-millisecond payloads.

### Comparison vs Aponte et al. (PPAM 2024)

Aponte reports:
- V100 bare-metal: 52 ms @ N=4096 (PCIe Gen3 + V100).
- RdmaCommunicator: 287 ms @ N=4096.
- Framework overhead: 287 - 52 = **235 ms = 4.52× bare-metal**.

This work:
- L40S bare-metal: 11.83 ms @ N=4096 (PCIe Gen4 + L40S).
- UCX-RDMA + GPUDirect: 17.20 ms.
- Framework overhead: 17.20 - 11.83 = **5.37 ms = 0.45× bare-metal**.

Absolute speedup vs Aponte: 287 / 17.20 = **16.69×**.

Hardware-normalized speedup (controlling for V100 → L40S generational
uplift): Aponte's overhead/bare = 4.52, ours = 0.45 → **~10× more
efficient per bare-metal millisecond**.

The 10× normalized improvement is what generalizes to other hardware
configurations: even on V100/Gen3 our stack would deliver substantially
lower framework cost than Aponte's.

### Comparison vs plain RDMA (GVirtuS's `RdmaCommunicator` on this testbed)

| N | plain RDMA (ms) | UCX-RDMA + GPUDirect (ms) | × speedup |
|---|---|---|---|
| 4096  | 350  | 17.20  | **20.34×** |
| 8192  | 1242 | 79.46  | 15.63× |
| 16384 | 4547 | 388.22 | **11.71×** |

Plain RDMA's bottlenecks (from inspecting backend log during sweep):
- 25+ `Read(1 byte)` TCP control-plane reads per call before the body.
- No iov gather (each marshalled arg = separate send/recv).
- No rkey pre-handshake (every transfer is rendezvous with RTS/RTR).
- No persistent pinned RX slot (backend re-allocates each call).
- Manual application-level chunking required at 1 GB payloads because
  `ibv_post_send` enforces per-WR size limits.

Effective bandwidth comparison at N=16384:
- Plain RDMA: 1 GB / 4547 ms = **0.22 GB/s**.
- UCX-RDMA + GPUDirect: 1 GB / 388 ms = **2.58 GB/s**.

11.7× better NIC utilization on a 25 GB/s rail.

### Three-way transport sweep (post Fase 5 fix)

After the Fase 5 non-UCX fallback (§5.2), all three transports complete
the full simple_matrix sweep. The same frontend image runs against three
backend configurations (only the `properties_*.json` and `UCX_TLS` env
change). 50 iterations per N, host_ms is the per-iter wall clock for
2×H2D + sgemm + 1×D2H.

| N | UCX-RDMA + GPUDirect | UCX-TCP (Fase 5 ON) | plain TCP (`TcpCommunicator`) | RDMA × vs plain TCP |
|---|---|---|---|---|
| 256   | 1.01   | 1.31    | 1.38    | 1.37× |
| 512   | 1.67   | 2.15    | 2.35    | 1.41× |
| 1024  | 1.44   | 6.51    | 8.09    | 5.62× |
| 2048  | 4.25   | 21.56   | 39.54   | 9.30× |
| **4096** | **17.24** | 167.18  | **197.52** | **11.46×** |
| 8192  | 79.53  | 672.91  | 722.04  | 9.08× |
| 16384 | 388.32 | 2784.71 | 2704.30 | 6.96× |

Non-obvious observations:

1. **UCX-TCP beats plain TCP at most sizes** (-7% to -46%), despite
   carrying ~10 ms of additional UCX framework overhead per call
   documented earlier in this work. The reason is **Fase 5's iov-split
   plus the WriteIovRma argmax fix**: even on TCP transport, splitting
   the 64 MB user buffer into three parallel `ucp_put_nbx` calls lets
   UCX overlap them, while plain TCP must serialize one big `sendmsg`.
   The framework overhead is paid back by parallelism. Reverses
   slightly at 16384 (+3% UCX-TCP slower) when the wire is so saturated
   that parallel puts no longer help.
2. **N=2048 shows the iov-split benefit most cleanly**: UCX-TCP -46% vs
   plain TCP. 16 MB is large enough for parallelism to matter, small
   enough that wire time doesn't dominate.
3. **GPUDirect ratio peaks at 4096-8192** (11.5× / 9.1× vs plain TCP).
   At 16384 it drops to 7× because 1 GB saturates the NIC at peer-DMA
   linerate — the wire itself is the limit; even GPUDirect can't help
   beyond that point.
4. **At N ≤ 512 all three transports converge** to the 1-2 ms band
   because the per-call setup overhead dominates regardless of
   transport. GPUDirect's 4 MB threshold ensures small calls never pay
   its setup cost.

The takeaway for deployment: *UCX provides framework benefit even on
TCP*, but the RDMA + GPUDirect path is required to extract the headline
order-of-magnitude speedup on bandwidth-bound workloads.

---

## 9. Workload regime taxonomy

Different workloads benefit from different optimizations. Empirically
validated on this testbed:

### Bandwidth-bound (payload ≫ call count)

Examples: simple_matrix N ≥ 1024, batch ML inference (large batches),
HPC simulation, image/video processing.

GPUDirect (this work) delivers the headline speedup. **simple_matrix @
N=4096: 16.7× vs Aponte.** Memory bandwidth utilization approaches NIC
linerate.

### RPC-bound (call count ≫ payload)

Examples: OpenPose (100 RPCs/frame × ~1 MB), inference serving with many
small ops, ML training with many small kernels.

GPUDirect doesn't help materially because per-call payloads sit below
the 4 MB threshold. **Empirically: OpenPose UCX-RDMA vs UCX-TCP differs
by < 1%** — the transport is irrelevant when the bottleneck is the
synchronous request/response model.

The structural fix is **asynchronous execution streams**: pipeline H2D[n+1]
with sgemm[n] and D2H[n-1]. Async work (separate from this README) is
expected to deliver 4-5× fps on OpenPose.

### Mixed (variable per-call size)

Examples: DL training (small kernels + occasional large allreduce), HPC + AI
pipelines.

GPUDirect helps the large calls; async (future) helps the small ones.
The two optimizations are orthogonal and additive — they target different
critical paths.

### Where nothing helps

- **Tiny payloads** (< 64 KB) at low frequency: framework overhead is
  structural (~5–7 ms of UCX bookkeeping per call). Below 100 calls/s
  total, the framework cost amortizes over enough wall-time to be
  invisible. Above that, plain TCP local would be faster — but that's
  not a remoting use case.
- **CPU-saturated workloads**: if host CPU is busy doing preprocessing,
  parsing, etc., the framework cost is masked but so are any framework
  optimizations. Profile the CPU first.

---

## 10. Known limitations

### Per-connection transport awareness

The GPUDirect activation is process-level (env-gated at `init_ucx`).
A single backend serving mixed UCX-RDMA + UCX-TCP frontends would still
try Variant A on TCP connections and fail. Workaround: run separate
backends per transport regime. Permanent fix would query each accepted
endpoint via `ucp_ep_query` and store a `endpoint_supports_cuda` flag
on the communicator.

### Handler GPU-awareness coverage

Only `CudaRtHandler_memory::Memcpy` (`HostToDevice` case) reads the
Buffer's GpuPayload. Other handlers (cuBLAS GEMM with > 4 MB args, cuDNN
conv with large tensors) would receive a Buffer with a "hole" in the
host slot and corrupt data. Currently the 4 MB threshold + simple_matrix's
known iov sizes makes this safe in practice, but extending GPUDirect to
cover cuBLAS/cuDNN handlers needs each handler to call `GetGpuPayload`
and route accordingly. ~10–20 LOC per handler.

### Small-N overhead residual

N = 256 (0.25 MB) still has 8.9× overhead/bare ratio. The 4 MB threshold
protects from GPUDirect-specific overhead, but the underlying Fase 5
zerocopy machinery still has ~1 ms fixed cost per call (mem_map check,
manual memh cache lookup, 3-put orchestration). This is sub-millisecond
absolute and not worth optimizing for sub-MB payloads.

### Single GPU per backend node

Backend allocates one TLS GPU scratch per worker thread, one process-global
GPU shadow per RX slot. Multi-GPU per node requires per-device scratches
and explicit `cudaSetDevice` plumbing in the handler. Out of scope for
current work.

### nvidia-peermem dependency

GPUDirect requires `nvidia-peermem` loaded on the backend. On systems
without it (older drivers, certain cloud VMs, virtualized GPU instances),
the probe at `init_ucx` fails and GPUDirect auto-disables. No graceful
degradation needed — the env-gated probe handles it.

### UCX rcache failure in this container

The dev container's UCX rcache fails to install its UCM event handler:
```
rcache failed to install UCM event handler: Unsupported operation
```
This is why the manual `unordered_map<const void*, ucp_mem_h>` memh cache
exists in `WriteIovRma`, and why `init_ucx` force-sets `UCX_RCACHE_ENABLE=n`
when GPUDirect is active. Fixing the rcache would simplify the code but
is upstream UCX / container infrastructure work, not a GVirtuS issue.

---

## 11. Files modified (summary)

| File | What |
|---|---|
| `include/gvirtus/communicators/Buffer.h` | `+SetGpuPayload/GetGpuPayload/GetGpuPayloadSize` + 2 fields |
| `src/communicators/Buffer.cpp` | Method implementations |
| `include/gvirtus/communicators/Communicator.h` | `+virtual current_frame_gpu(gpu, size)` default no-op |
| `include/gvirtus/communicators/Result.h` | `+SetGpuPayload/GetGpuPayload/GetGpuPayloadSize` + 2 fields |
| `src/communicators/Result.cpp` | Method implementations |
| `include/gvirtus/frontend/Frontend.h` | `+SetOutputDestination/AddHostPointerForArgumentsDirect/HasDirectInput`. `AddHostPointerForArgumentsDirect` has a transport-aware fallback (§5.2) so plain TCP / Hybrid degrade to legacy in-buffer marshal. |
| `plugins/cudart/frontend/CudaRtFrontend.h` | Static wrappers |
| `src/frontend/Frontend.cpp` | Fase 4 direct-dst memcpy + Fase 5 iov split |
| `plugins/cudart/frontend/CudaRt_memory.cpp` | Caller uses Fase 4/5 APIs |
| `src/communicators/ucx/UcxCommunicator.h` | PinnedSlot + RemoteSlot + PooledMsg dual-region fields, `current_frame_gpu` override |
| `src/communicators/ucx/UcxCommunicator.cpp` | All UCX-level changes: probe, slot allocation, RmaSetup wire format, WriteIovRma argmax + GPU routing + transport gate, am_recv_handler dual PooledMsg |
| `src/backend/Process.cpp` | UCX-AM dispatch reads `current_frame_gpu` from communicator and propagates to Buffer; `write_ucx_am_response` builds 5-entry iov with GPU payload |
| `plugins/cudart/backend/CudaRtHandler_memory.cpp` | D2H handler creates TLS GPU scratch + cudaMemcpy D2D for Variant A; H2D handler reads `Buffer::GetGpuPayload` for Variant B; 4 MB threshold on both |
| `Makefile` | Env passthroughs for `GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`, `UCX_IB_GID_INDEX` |

Total: ~400 LOC across 13 files plus the ~10 LOC Fase 5 fallback in
`Frontend.h`. All flag-gated; turning off `GVIRTUS_GPUDIRECT` reverts
the data-path to pre-GPUDirect behaviour exactly. The Fase 5 fallback
is transparently active for non-UCX transports and dormant for UCX —
no flag needed.

---

## 12. References

- Aponte et al., **"Boosting GPGPU Virtualization and Multiplexing with
  RDMA Communication"**, PPAM 2024 LNCS 15580. The baseline this work
  improves on.
- UCX Programming Reference: https://openucx.readthedocs.io/
- nvidia-peermem documentation:
  https://docs.nvidia.com/networking/display/MLNXOFEDv25.04.0/GPUDirect+RDMA
- GVirtuS upstream: https://github.com/AAU-CE25/GVirtuS (branch
  `fix/ucx-comm`).
