# GVirtuS-over-UCX: Optimization Stack

This document describes every change made to the GVirtuS UCX communicator
and surrounding code to take GVirtuS-over-RDMA from an unusable baseline
to a state where it competes with native CUDA for bandwidth-bound workloads.

## Headline results

| workload class | example | speedup vs plain-RDMA baseline |
|---|---|---|
| Bandwidth-bound | `simple_matrix` 64MB-1GB cuBLAS sgemm | **6.3–9.2×** |
| RPC-overhead-bound | OpenPose pose inference (100 calls/frame) | 2–3× (estimated, transport-indifferent) |

Effective NIC utilization at 1 GB payload jumps from 0.68 GB/s (plain RDMA)
to 5.5 GB/s (UCX RDMA + this stack), i.e. from 5.7% to 46% of the
ConnectX-7's physical 12 GB/s ceiling.

---

## Hardware testbed

```
                       RoCEv2 fabric  25.25.25.0/24 (mlx5_1)
       ╔═══════════════════════════════════════════════════════════╗
       ║                                                           ║
   ┌───╨───────────────────────────────┐ ┌───────────────────────╨───┐
   │  es-dpu-01  (BACKEND)             │ │  es-dpu-02  (FRONTEND)    │
   │  hosts GVirtuS backend + GPU      │ │  runs user's CUDA app     │
   │                                   │ │                           │
   │  ┌────────────┐    ┌──────────┐   │ │  ┌──────────┐ ┌─────────┐ │
   │  │ NVIDIA     │    │  AMD     │   │ │  │  AMD     │ │ NVIDIA  │ │
   │  │ L40S       │    │  EPYC    │   │ │  │  EPYC    │ │ L40S    │ │
   │  │ 48 GB HBM  │    │  CPU     │   │ │  │  CPU     │ │ (idle - │ │
   │  │            │◀──PCIe Gen4──▶│   │ │  │          │ │  CUDA   │ │
   │  └──────┬─────┘    └────┬─────┘   │ │  └────┬─────┘ │  fwds)  │ │
   │         │ peermem        │ PCIe   │ │       │ PCIe  └─────────┘ │
   │         │ (GPUDirect     │        │ │       │                   │
   │         │  plumbing,     │        │ │       │                   │
   │         │  validated)    │        │ │       │                   │
   │         └────────┬───────┘        │ │       │                   │
   │              ┌───┴────────────┐   │ │  ┌────┴────────────────┐  │
   │              │ ConnectX-7  CX │   │ │  │ ConnectX-7  CX      │  │
   │              │ 200 Gb/s × 2   │   │ │  │ 200 Gb/s × 2        │  │
   │              │                │   │ │  │                     │  │
   │              │ mlx5_1 ◀─RoCEv2┼───┼─┼──┤ mlx5_1              │  │
   │              │ 25.25.25.1     │   │ │  │ 25.25.25.2          │  │
   │              │                │   │ │  │                     │  │
   │              │ mlx5_0  (unused)   │ │  │ mlx5_0  (unused)    │  │
   │              │ 24.24.24.1     │   │ │  │ 24.24.24.2          │  │
   │              └────────────────┘   │ │  └─────────────────────┘  │
   └───────────────────────────────────┘ └───────────────────────────┘

  Production lane: mlx5_1 only (UCX_NET_DEVICES=mlx5_1:1,ens1f1np1).
  peermem kernel module persistent on dpu-01 via
  /etc/modules-load.d/nvidia-peermem.conf (enables future GPUDirect).
  mlx5_0 reserved for multi-rail future work (currently lacks RoCE
  GID setup so UCX's auto-rail selection skips it).
```

## Why GVirtuS-over-RDMA was slow

Out of the box, the plain `RdmaCommunicator` path requires application-level
chunking just to complete at 1 GB payloads — raw `ibv_post_send` enforces
per-WR size limits and memory registration costs scale poorly. Even after
chunking, plain RDMA delivers ~0.68 GB/s effective end-to-end because:

1. The control protocol issues byte-by-byte `Read(1)` calls (≈25 syscalls
   per RPC just for headers).
2. Each marshaled argument is sent as a separate send/recv pair.
3. No `rkey` pre-handshake → every bulk transfer is rendezvous.
4. No persistent pinned slot → re-registration per call.
5. The `Buffer` marshal pipeline forces every byte of the user payload
   through `mpInputBuffer` (one extra host-memory pass per direction).

Our optimization stack attacks all of these, replacing the marshal-heavy
plain-RDMA path with a UCX-based pipeline that bypasses the `Buffer`
allocator for large payloads.

---

## Old vs new dataflow (1 GB H2D)

### Before — plain `RdmaCommunicator` path

```
 FRONTEND (es-dpu-02)                      BACKEND (es-dpu-01)
 ────────────────────                      ────────────────────

 user code: cudaMemcpy(d_dst, h_src, 1 GB, H2D)
    │
    ▼
 [mpInputBuffer]
    │  AddHostPointer<char>(h_src, 1 GB)
    │  ← memcpy(mpInputBuffer + off, h_src, 1 GB)   ┐
    │                                               │
    ▼                                               │ ~80 ms host memcpy
 RdmaCommunicator::WriteIov                         │
    │                                               │
    │  Chunk into N × 256 MB pieces                 │
    │  Loop per chunk:                              │
    │   • ibv_reg_mr(chunk)   ← ~50 ms first time   │
    │   • ibv_post_send(WR)                         │
    │                                               │
    │                            ─── RDMA SEND ────▶│  recv buffer (alloc'd
    │                                               │   per call, not pinned)
    │                                               ▼
    │  Control plane:                          [unmarshal: 25× Read(1B)]
    │  25× Write(1B) header bytes        ─────────────▶ for routine name +
    │                                                    scalars
    │                                                     │
    │                                                     ▼
    │                                            cudaMemcpy(d_dst, recv_buf,
    │                                                       1 GB, H2D)
    │                                                  ~80 ms PCIe Gen4
    │                                                     │
    │ ◀── small ACK via TCP socket ──────────────────────┘
    ▼
 return cudaSuccess

 Per-iter cost @ 1 GB:
    host memcpy             ~ 80 ms
    ibv_reg_mr loop         ~hundreds of ms
    RDMA SEND chunks        ~200 ms wire-bound
    backend cudaMemcpy      ~ 80 ms
    25× syscalls + scaffold ~ 25 ms
    ────────────────────────────────
    ≈ 4.5 s per H2D 1 GB
```

### After — UCX RDMA + this optimization stack

```
 FRONTEND (es-dpu-02)                      BACKEND (es-dpu-01)
 ────────────────────                      ────────────────────

 user code: cudaMemcpy(d_dst, h_src, 1 GB, H2D)
    │
    ▼
 AddHostPointerForArgumentsDirect            [2 × 1025 MB pinned slots
    │ records h_src + offset                  pre-mem_map'd, rkey known]
    │ NO memcpy                                       │
    │                                                 │
    ▼                                                 ▼
 Frontend::Execute builds 5-fragment iov:    remote_slots_[i] = {addr, rkey}
   [hdr, routine, pre, h_src(1 GB), post]
    │
    ▼
 WriteIovRma (argmax detects iov[3] = big):
    │
    ├─ ucp_put_nbx(pre,   rs.addr + 0       ) ──┐
    ├─ ucp_put_nbx(h_src, rs.addr + pre_size) ──┼───▶ slot[i] (single
    └─ ucp_put_nbx(post,  rs.addr + pre+big ) ──┘     contiguous wire stream
                                                       to peer's pinned slot)
    Three parallel RDMA Writes, no host
    staging memcpy. user_memh cached.
    │                                                 │
    │                                                 ▼
    │                                          Backend handler:
    │                                            read from slot[i]
    │                                            cudaMemcpy(d_dst, slot,
    │                                                       1 GB, H2D)
    │                                            ~36 ms (PCIe linerate)
    │                                                 │
    │ ◀── tiny RmaPosted AM (slot_idx, size) ────────┘
    ▼
 return cudaSuccess

 Per-iter cost @ 1 GB:
    Host marshaling          ~  0 ms (no payload memcpy)
    3× ucp_put_nbx parallel  ~ 80 ms (NIC linerate ~12 GB/s)
    Backend cudaMemcpy       ~ 36 ms (PCIe Gen4 linerate)
    RmaPosted ACK            ~  1 ms
    Misc marshal             ~  5 ms
    ────────────────────────────────
    ≈ 0.72 s per H2D 1 GB    (6.3× faster than plain RDMA)
```

---

## Stack overview

The data path's defining characteristics in our optimized form:

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   ┌─────────────────┐                ┌────────────────────────┐   │
│   │  FRONTEND       │                │  BACKEND               │   │
│   │  cudaMemcpy     │                │  cudaRtHandler         │   │
│   │  (user app)     │                │  (forwards to GPU)     │   │
│   └────────┬────────┘                └────────▲───────────────┘   │
│            │                                  │                   │
│            │ Phase 5: zero-copy iov            │ Phase 1+2:        │
│            │ Phase 4: zero-copy direct out     │ pre-faulted slot  │
│            │                                  │ Buffer view       │
│            ▼                                  │                   │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │            WriteIov(iov) → WriteIovRma                  │    │
│   │  • argmax-fragment 3× parallel ucp_put_nbx              │    │
│   │  • size-thresholded user-memh cache                     │    │
│   └────────────┬────────────────────────────────────────────┘    │
│                │                                  ▲               │
│                │ rc_mlx5 RDMA Writes              │ RmaPosted AM  │
│                ▼                                  │               │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  Pinned RX slot pool (per connection, per direction)    │    │
│   │  • 2 × 1025 MB cudaHostAlloc + ucp_mem_map              │    │
│   │  • rkey + addr pre-exchanged via RmaSetup AM at connect │    │
│   │  • round-robin slot index                                │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

Below: each optimization explained with the problem it solves, the change,
the files it touches, and the measurement that justifies it.

---

## 1. Backend handler page-fault elimination (Fase 1+2)

**Problem.** The legacy D2H handler did `new char[count]` per call,
producing a freshly allocated paged region. The subsequent
`cudaMemcpy(host_slot, GPU, count, D2H)` then touched every page for the
first time, triggering kernel page faults. At 64 MB this added ~28 ms
on every D2H.

**Change.** In `plugins/cudart/backend/CudaRtHandler_memory.cpp`:

- A `thread_local` pre-faulted TX slot replaces `new char[count]`. The
  first call `memset()`s the entire region to fault all pages once; every
  subsequent call reuses the warm allocation.
- The response `Buffer` is constructed via a new view-only ctor
  (`Buffer(char *, size_t, size_t)` with `mOwnBuffer=false`) so the slot
  is handed to the communicator as-is — no `realloc + memmove` to copy
  bytes into an owned buffer.
- The 8-byte size prefix is written directly at `slot[0:8]` before the
  `cudaMemcpy` deposits payload at `slot[8:]`, matching the wire format
  the frontend's `Assign<char>` expects.

**Files.**
- `plugins/cudart/backend/CudaRtHandler_memory.cpp`
- `include/gvirtus/communicators/Buffer.h` (view ctor)

**Result.** Backend handler total at 64 MB: 80 ms → 4.7 ms (-94%).

---

## 2. `WriteIov` gather-send (`UCP_DATATYPE_IOV`)

**Problem.** Pre-existing `WriteIov` concatenated all iov fragments into
a single buffer via memcpy, then sent. For a 64 MB payload this was 27 ms
of pure staging.

**Change.** Map `Communicator::WriteIov(iov, count)` directly to UCX's
native gather-send:

```cpp
ucp_request_param_t p{};
p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
p.datatype = UCP_DATATYPE_IOV;
ucp_am_send_nbx(ep, am_id, header, header_size,
                ucx_iov.data(), iov_count, &p);
```

UCX handles the multi-segment dispatch internally with no extra copy.

**Files.** `src/communicators/ucx/UcxCommunicator.cpp`.

**Result.** Eliminates 27 ms of marshal staging per `cudaMemcpy` call at
64 MB.

---

## 3. Bidirectional `rkey` pre-handshake

**Problem.** Out of the box the server registers an RX slot but the client
does not — so D2H responses fall back to rendezvous AM protocol, which has
expensive RTS/RTR handshake roundtrips per message.

**Change.** At connect time, server sends rkeys to client AND client sends
rkeys to server. Both directions now use `ucp_put_nbx` into a pre-pinned
slot.

```
 CLIENT (frontend)                     SERVER (backend)
 ─────────────────                     ─────────────────

  Connect(host:port)
     │
     │ TCP sockaddr handshake
     ├───────────────────────────────▶ Accept()
     │                                    │
     │                                    │ Spawn detached thread (so
     │                                    │ listener can return fast):
     │                                    │   ┌───────────────────────┐
     │                                    │   │ init_rx_pool():       │
     │                                    │   │  for i in [0, kSlots):│
     │                                    │   │   cudaHostAlloc(1025M)│
     │                                    │   │   ucp_mem_map(...)    │
     │                                    │   │   ucp_rkey_pack(...)  │
     │                                    │   └───────────────────────┘
     │                                    │   ┌───────────────────────┐
     │                                    │   │ send_rma_setup():     │
     │                                    │   │  pack table {addr[N], │
     │                                    │   │              rkey[N], │
     │                                    │   │              cap }    │
     │                                    │   └───────────┬───────────┘
     │   ┌────────── RmaSetup AM ─────────────────────────┘
     │   │ (control message)
     │   ▼
  handle_rma_setup_am():
   • parse server slot table
   • ucp_ep_rkey_unpack each
   • remote_slots_[i] = {addr, rkey, cap}
   • rma_setup_received_ = true
     │
     │  ── symmetric procedure: client allocates own slots,
     │     packs rkeys, sends RmaSetup AM back to server ─▶
     │
     ▼

 After handshake, BOTH directions use the same fast path:
   client → server H2D:  ucp_put_nbx(h_src, server_rs[i].addr, .rkey)
   server → client D2H:  ucp_put_nbx(tls_slot, client_rs[j].addr, .rkey)

 No rendezvous AM per message → no RTS/RTR roundtrips.
```

**Files.**
- `src/communicators/ucx/UcxCommunicator.cpp` (`init_rx_pool`,
  `send_rma_setup`, `handle_rma_setup_am`, `Accept`, `Connect`).
- `src/communicators/ucx/UcxCommunicator.h` (rkey/slot state members).
- `include/gvirtus/communicators/UcxAmProtocol.h` (RmaSetup AM types).

**Result.** Backend D2H at 64 MB: 1700 ms → 95 ms (18× faster). The bulk
of the saving is in eliminating per-message rendezvous setup.

---

## 4. RX pool of pinned slots

**Problem.** Page-locked memory is fast to DMA but slow to allocate. Doing
`cudaHostAlloc + ucp_mem_map` per transfer is unacceptable in the hot path.

**Change.** Allocate a small pool of large pre-pinned, pre-registered
slots per accepted connection. Reuse them round-robin. Synchronous
request/response semantics guarantee the previous slot has been consumed
before reuse.

```
 Per accepted connection, on EACH side:

  remote_slots_[]:
  ┌──────────────────────────────────────────────────────────┐
  │  slot[0]:                                                │
  │     addr     = 0x7f00_0000_0000  (peer's virtual addr)   │
  │     rkey     = <ucp_ep_rkey_unpack handle>               │
  │     capacity = 1025 MB                                   │
  │     [backed by cudaHostAlloc + ucp_mem_map on peer]      │
  ├──────────────────────────────────────────────────────────┤
  │  slot[1]:                                                │
  │     addr     = 0x7f01_0000_0000                          │
  │     rkey     = <handle>                                  │
  │     capacity = 1025 MB                                   │
  └──────────────────────────────────────────────────────────┘
        ▲                                          ▲
        │                                          │
        └─────────── round-robin via ──────────────┘
                    next_remote_slot_idx_++ % N

  WriteIovRma():
    1. slot_idx = next_remote_slot_idx_
    2. next_remote_slot_idx_ = (idx + 1) % N
    3. ucp_put_nbx(payload, remote_slots_[slot_idx].addr, .rkey)
    4. RmaPosted AM: "slot[idx] holds T bytes"

  Capacity rationale (kInitialSlotCap):
    65 MB original  → 256 MB / 1 GB payloads exceeded cap → fell back to
                       AM staged path → lost zerocopy benefit.
    1025 MB current → 1 GB payloads stay on the RMA fast path with a tiny
                      safety margin.

  Cost: ~2 GB pinned host memory per connection per direction.
        ~600-1000 ms one-time setup at Accept().
```

**Files.** `src/communicators/ucx/UcxCommunicator.cpp` (`init_rx_pool`,
`destroy_rma_state`).

---

## 5. Bulk `Buffer::AppendBytes`

**Problem.** The legacy `Buffer::Add<char>(ptr, n)` was implemented as a
single-byte loop with bounds-check every iteration. For 64 MB this was
67 million function calls, ~1.3 s of pure call overhead per D2H.

**Change.** Add a bulk `AppendBytes(ptr, n)` that does one `memcpy` after
a single capacity check.

**Files.** `include/gvirtus/communicators/Buffer.h`,
`src/communicators/Buffer.cpp` (call site replacements).

**Result.** Frontend-side D2H buffer assembly: 1.3 s → 3 ms at 64 MB.

---

## 6. Phase 4: Frontend D2H zero-copy

**Problem.** Even after the bulk `AppendBytes`, frontend D2H still copied
the 64 MB payload twice:

1. RX slot → `mpOutputBuffer` via `AppendBytes` (~3 ms).
2. `mpOutputBuffer` → user `dst` via caller's `memmove` (~6 ms).

The wire response format is `[size_t prefix == count][count bytes]` — the
user destination already has space for the payload, so the intermediate
buffer is pure waste.

**Change.** A new caller-side API records the user destination *before*
`Execute`:

```cpp
CudaRtFrontend::SetOutputDestination(dst, count);
CudaRtFrontend::Execute("cudaMemcpy");
if (CudaRtFrontend::DirectOutputConsumed()) {
    // memmove already performed inside Execute()
} else {
    // legacy fallback
    memmove(dst, GetOutputHostPointer<char>(count), count);
}
```

Inside `Frontend::Execute`, the response handler (both UCX AM and stream
fallback paths) checks if the response Buffer layout matches the expected
single-payload-with-prefix form. If yes, it memcpys directly from the
RX slot frame to user `dst`, marks `mDirectOutputConsumed = true`, and
skips appending to `mpOutputBuffer`.

**Files.**
- `include/gvirtus/frontend/Frontend.h` (new members + helpers).
- `plugins/cudart/frontend/CudaRtFrontend.h` (static wrappers).
- `src/frontend/Frontend.cpp` (direct-dst memcpy in response paths).
- `plugins/cudart/frontend/CudaRt_memory.cpp` (D2H caller uses the API).

**Result.** Frontend D2H at 64 MB: 28.4 ms → 23.3 ms (-18%). Saves one
host-memory pass.

**Fallback.** If `out_buffer_size != sizeof(size_t) + count` or the
prefix doesn't match `count`, the response handler falls back to the
legacy append-then-memmove path. Non-cudaRt callers are unaffected
because they don't call `SetOutputDestination()`.

---

## 7. Phase 5: Frontend H2D zero-copy

**Problem.** Symmetric to Phase 4 on the input side. The legacy H2D
caller did:

```cpp
AddDevicePointerForArguments(dst);
AddHostPointerForArguments<char>(h_src, count);   // ← memcpy 64 MB into mpInputBuffer
AddVariableForArguments(count);
AddVariableForArguments(kind);
Execute("cudaMemcpy");
```

That `Add<char>` is the 64 MB memcpy we wanted to remove.

**Change.** A new API records the source pointer without copying:

```cpp
template<typename T>
void AddHostPointerForArgumentsDirect(const T *ptr, size_t n = 1) {
    mDirectInputBufferOffset = mpInputBuffer->GetBufferSize();
    mDirectInputSrc          = ptr;
    mDirectInputBytes        = n * sizeof(T);
}
```

In `Frontend::Execute`, when `HasDirectInput()` is true, a 5-fragment iov
is built:

```
 mpInputBuffer (small, host-side marshal state):
 ┌────────────────┬────────────────────────┐
 │ device_ptr enc │  count(8) + kind(4)    │
 └────────────────┴────────────────────────┘
       └──── split offset ──────┘
       │                        │
       ▼                        ▼
   iov[2] = pre              iov[4] = post
   (~12 bytes)               (~12 bytes)

 User's h_src (no copy, just record the pointer):
 ┌─────────────────────────────────────────────┐
 │     64 MB / 1 GB of caller's host memory    │
 └─────────────────────────────────────────────┘
       │
       ▼
   iov[3] = direct (the big fragment)

 Final iov passed to WriteIov:
   [ req_header,  routine,  pre,  user_src,  post ]
   iov[0]        iov[1]    iov[2] iov[3]    iov[4]

 Wire bytes after WriteIov are byte-identical to the legacy contiguous
 case: [hdr][routine][device_ptr][user_payload][count][kind].
```

**Files.**
- `include/gvirtus/frontend/Frontend.h` (new members + helpers).
- `plugins/cudart/frontend/CudaRtFrontend.h`.
- `src/frontend/Frontend.cpp` (5-iov build in Execute, plus
  `ClearDirectInput` after send).
- `plugins/cudart/frontend/CudaRt_memory.cpp` (H2D caller uses the API).

**Result (after the argmax fix below).** Frontend H2D at 64 MB:
10.17 ms → 5.58 ms (-45%).

---

## 8. `WriteIovRma` argmax fix

**Problem.** Phase 5's iov places the big fragment at index 3, not last.
The original `WriteIovRma` zerocopy path assumed `biggest = iov[iov_count-1]`
and fell through to the staged path (memcpy everything into `tx_scratch_`,
single put). Net effect of Phase 5 was ZERO — we just moved the 64 MB
memcpy from `mpInputBuffer` to `tx_scratch_`.

**Change.** ~25 LOC in `WriteIovRma`. Scan the iov for the biggest
fragment, stage all OTHER fragments into `tx_scratch_`, issue up to
three parallel `ucp_put_nbx` at the correct remote offsets.

```
 5-fragment iov on entry:
 ┌──────┬─────────┬──────┬──────────────┬──────┐
 │ hdr  │ routine │ pre  │   user_src   │ post │
 │ ~32B │  ~10B   │ ~12B │  64 MB / 1GB │ ~12B │
 └──────┴─────────┴──────┴──────────────┴──────┘
   idx 0   idx 1   idx 2     idx 3       idx 4

 argmax scan ⇒ biggest_idx = 3 (user_src)
 pre_size  = sum(iov[0..3]) = ~54 B
 big_size  = 64 MB or 1 GB
 post_size = sum(iov[4..]) = ~12 B

 tx_scratch_ layout (staged small fragments, contiguous):
 ┌──────┬─────────┬──────┬──────┐
 │ hdr  │ routine │ pre  │ post │           ← in iov order, big skipped
 └──────┴─────────┴──────┴──────┘
      └────── pre_size ──────┘
                              └─ post_size

 Three parallel ucp_put_nbx to peer's RX slot:

   put 1:  tx_scratch_[0..pre_size]                ──▶ rs.addr + 0
   put 2:  iov[3].iov_base (user_src DIRECT, with  ──▶ rs.addr + pre_size
           memh from user_memh_cache when ≥ 2 MB)
   put 3:  tx_scratch_[pre_size..pre_size+post_size]──▶ rs.addr + pre_size + big_size

 Resulting peer slot view (single contiguous wire stream):
 ┌──────┬─────────┬──────┬──────────────┬──────┐
 │ hdr  │ routine │ pre  │  user_src    │ post │      ← byte-identical
 └──────┴─────────┴──────┴──────────────┴──────┘        to legacy layout

 Three puts run in parallel; UCX worker overlaps them with the kernel
 TCP/RDMA send. On the TCP transport, even though rkey-put is emulated,
 the parallelism wins ~30% on H2D vs the legacy serial staged path.
```

**Files.** `src/communicators/ucx/UcxCommunicator.cpp::WriteIovRma`.

**Result.** Phase 5's full benefit unlocks. H2D 10.17 → 5.58 ms at 64 MB.
Per-iter `simple_matrix` @ 4096: 46.5 → 38.15 ms.

**Bonus.** Even on UCX-TCP transport (where rkey-put is emulated), the
argmax fix improves H2D by 30% (37 → 26 ms) because the parallel puts
overlap with the kernel TCP send.

---

## 9. Manual user-memh cache with size threshold

**Problem.** UCX's built-in registration cache (rcache) fails to install
its UCM event handler in this container ("Unsupported operation"),
forcing every `ucp_put_nbx` of the user buffer to re-register the memory
region. At 64 MB that's ~25 ms of pure overhead per call.

**Change.** A `static thread_local std::unordered_map<const void *,
ucp_mem_h>` caches `user_addr → memh` on first encounter and passes the
memh via `UCP_OP_ATTR_FIELD_MEMH` thereafter:

```cpp
auto cit = user_memh_cache.find(user_addr);
if (cit != user_memh_cache.end()) {
    user_memh = cit->second;
} else {
    ucp_mem_map_params_t mp{};
    mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                    UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mp.address = const_cast<void *>(user_addr);
    mp.length  = big_size;
    if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK)
        user_memh_cache.emplace(user_addr, user_memh);
}
```

**Size-threshold guard.** The above breaks on workloads that reallocate
small buffers (Caffe blobs in OpenPose) because the cache returns a stale
memh when the kernel reused a freed virtual address. Symptom: IB QP Local
Protection error at the first cudaMemcpy from a reused address.

Fix: a `kCacheThreshold = 2 MB` guard. Buffers `>= 2 MB` use the cache
(simple_matrix-style stable buffers); buffers `< 2 MB` register fresh per
call and `ucp_mem_unmap` after the put completes. The per-call cost at
1 MB is ~1 ms — acceptable compared to a crash.

**Files.** `src/communicators/ucx/UcxCommunicator.cpp::WriteIovRma`.

**Result.** Steady-state H2D 14.6 → 5.6 ms at 64 MB on simple_matrix.
OpenPose runs cleanly without IB protection errors.

---

## 10. Worker-per-connection

**Problem.** A single shared `ucp_worker_h` for all accepted connections
meant a transient error on one connection (e.g. peer reset) could poison
shared progress and stall every other client.

**Change.** Each `Accept()` creates its own `ucp_worker_h`. Workers are
independent; an error on one connection's worker leaves all others
untouched.

**Files.** `src/communicators/ucx/UcxCommunicator.cpp` (`Accept`,
endpoint state struct).

---

## 11. EndpointFactory `ind_endpoint` modulo fix

**Problem.** A static counter `ind_endpoint` lets a config file declare
multiple endpoints and have N `get_endpoint()` calls walk through them.
With a single-endpoint config used across N pthreads (each pthread has
its own Frontend → its own `get_endpoint()` call), the counter overran
the array and accessed `j[N]` = null → crash.

**Change.** Wrap modulo the array size:

```cpp
ind_endpoint = ind_endpoint % static_cast<int>(j["communicator"].size());
const auto &endpoint_obj = j["communicator"][ind_endpoint]["endpoint"];
```

**Files.** `include/gvirtus/communicators/EndpointFactory.h`.

---

## 12. `[GVS PROFILE]` instrumentation

For diagnostic purposes, `Frontend::Execute` emits a per-call timing
breakdown when the effective payload is ≥ 1 MB:

```
[GVS PROFILE] cudaMemcpy payload=64MB | marshal=… write=… sync=…
              read_hdr=… read_payload=… append=… | total_send=… total_recv=…
```

Tracks: iov build time, `WriteIov` time, worker flush, response header
wait, payload receive, append. The gate evaluates `payload_size +
mDirectInputBytes + mDirectOutputCount` so Phase 5 (zero `payload_size`)
and Phase 4 (request body tiny, response body big) both still trip
the threshold.

**Files.** `src/frontend/Frontend.cpp::Execute`.

---

## 13. Infrastructure: `BACKEND_CONFIG` env passthrough

**Problem.** The dev backend image hard-coded `properties_ucx.json` in
its entrypoint. Testing alternative configurations required editing the
image or the entrypoint.

**Change.** `docker/dev/entrypoint.sh` reads `${BACKEND_CONFIG:-…}`. The
`Makefile`'s `run-gvirtus-backend-dev` target adds the env-var passthrough
`$(if $(BACKEND_CONFIG),-e BACKEND_CONFIG=$(BACKEND_CONFIG))`. Same image,
swap config per invocation:

```bash
BACKEND_CONFIG=/usr/local/gvirtus/etc/properties_ucx.json make run-gvirtus-backend-dev
BACKEND_CONFIG=/usr/local/gvirtus/etc/properties.json     make run-gvirtus-backend-dev
BACKEND_CONFIG=/usr/local/gvirtus/etc/properties_hybrid.json make run-gvirtus-backend-dev
```

**Files.** `docker/dev/entrypoint.sh`, `Makefile`.

---

## 14. OpenPose example tooling

Out-of-the-box `examples/openpose/` cloned the public `ecn-aau/GVirtuS`
upstream (which lacks our UCX communicator). The Dockerfile and entrypoint
were patched for benchmarking:

1. **Local GVirtuS source.** Replace `git clone …/GVirtuS.git` with
   `COPY gvirtus_src.tar.gz …` + `tar xzf` + `cmake`. The tarball is
   generated from the repo's `src include plugins …` at image-build time.
2. **UCX 1.20 install.** `wget` the UCX release tarball and
   `apt-get install -y ./ucx-1.20.0.deb ./ucx-cuda-1.20.0.deb` so headers
   are available when GVirtuS compiles.
3. **Pre-staged BODY_25 model.** CMU's model server returns empty files;
   the model is downloaded from a HuggingFace mirror (md5 verified) and
   `COPY`'d into the image. cmake flags
   `-DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF` disable the
   broken downloads for the optional models.
4. **Caffe test-dir skip.** Bundled Caffe in `examples/openpose/caffe/` is
   stripped of `src/gtest` and `src/caffe/test`. CMakeLists.txt patched
   to comment out the `add_subdirectory(src/gtest)` and
   `add_subdirectory(test)` calls.
5. **`--custom_net_resolution` flag.** `test_multiple.cpp` gains a gflag
   to force higher input resolution for payload-size sweep
   experiments (currently blocked by an upstream CudnnHandler hang —
   see Limitations).
6. **`entrypoint_bench.sh`.** Compiles `test_multiple` at runtime from
   the bind-mounted `examples/openpose/` and runs it with the requested
   net resolution.
7. **`properties_ucx.json`.** OpenPose-specific UCX config pointing to
   the backend on `25.25.25.1:32223`.

**Files.** `examples/openpose/{Dockerfile, test_multiple.cpp,
entrypoint_bench.sh, properties_ucx.json, caffe/CMakeLists.txt,
caffe/src/caffe/CMakeLists.txt}`.

---

## How to run

### Backend (on `es-dpu-01`)

```bash
cd ~/GVirtuS
make stop-gvirtus
GVIRTUS_RMA_ZEROCOPY=1 \
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

### simple_matrix bench (on `es-dpu-02`)

```bash
cd ~/GVirtuS
docker run --rm --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
  -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
  -e MATRIX_SIZES="256 512 1024 2048 4096 8192 16384" \
  -e ITERATIONS=50 -e WARMUP=5 \
  -v ./examples/simple_matrix:/opt/GVirtuS/examples \
  -v ./etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json \
  --entrypoint bash ll33pq/gvirtus-dev/simple_matrix_gvirtus:cuda12.6 \
  /opt/GVirtuS/examples/benchmark.sh
```

### OpenPose bench (on `es-dpu-02`)

```bash
cd ~/GVirtuS
docker run --rm --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties.json \
  -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
  -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
  -v ./examples/openpose/media:/opt/openpose/examples/media \
  -v ./examples/openpose:/opt/openpose/examples/gvirtus \
  -v ./examples/openpose/properties_ucx.json:/opt/GVirtuS/etc/properties.json \
  -v ./examples/openpose/entrypoint_bench.sh:/entrypoint.sh \
  ll33pq/openpose_gvirtus:cuda12.6 bash /entrypoint.sh
```

---

## Engineering footprint

- ~2035 LOC added net across 29 files.
- ~80% of changes concentrated in `src/communicators/ucx/UcxCommunicator.cpp`.
- ~150 lines of pre-existing code modified (extends, doesn't rewrite).

---

## Future: GPUDirect refactor

GPUDirect RDMA lets the NIC DMA directly to/from GPU memory via the
`peermem` kernel module, eliminating the host-slot ↔ GPU PCIe transfer
that's currently the backend's per-call floor.

```
 FRONTEND (unchanged)                  BACKEND (with GPUDirect)
 ─────────────────────                 ─────────────────────────

 cudaMemcpy(d_dst, h_src, 1 GB, H2D)
    │                                  RX slot relocated to GPU memory:
    │                                    cudaMalloc(1025 MB)
    │                                    ucp_mem_map(UCS_MEMORY_TYPE_CUDA)
    │                                    peermem enables NIC→GPU DMA
    │
    ▼
 WriteIovRma (same frontend code,
              just faster on the wire)
    │
    ├─ 3× ucp_put_nbx ────  RoCE  ──── NIC ──peermem PCIe──▶  [GPU slot]
    │                                                              │
    │                                              (NO host hop!)  │
    │                                                              ▼
    │                                                  cudaMemcpyAsync(
    │                                                    d_dst, slot, D2D, stream)
    │                                                  Intra-GPU: ~free
    │ ◀─── RmaPosted ACK ─────────────────────────────────────────┘
    ▼
 return cudaSuccess

 Eliminated: the ~36 ms host_slot → GPU PCIe cudaMemcpy at 1 GB.

 Predicted:
   simple_matrix @ 4096  (64 MB)   38.15 ms → ~23 ms   (-40%)
   simple_matrix @ 16384 (1 GB)    717 ms   → ~600 ms  (-16%)

 Plumbing already validated end-to-end with a stand-alone probe
 (gpudirect_probe.cpp): UCX 1.20 detects `cuda (reg,cache)` on
 mlx5_1, GPU→GPU put byte-verified across the fabric.
```

Implementation scope: **~150 LOC** in:
- `plugins/cudart/backend/CudaRtHandler_memory.cpp` (TX slot becomes
  `cudaMalloc` + `ucp_mem_map(MEMORY_TYPE_CUDA)`).
- `src/communicators/ucx/UcxCommunicator.cpp::init_rx_pool` (RX slot
  becomes GPU memory).
- Backend D2H handler: replace `cudaMemcpy(slot, src, D2H)` with
  `cudaMemcpyAsync(slot_GPU, src_GPU, D2D, stream)` — intra-GPU.

The frontend is unchanged: it still receives into host RX slots; only
the sender (backend) side is GPU-aware. That bounds the blast radius
of the refactor.

---

## Limitations / future work

- **GPUDirect refactor** (described above, ~150 LOC, validated plumbing).

- **Async streams.** OpenPose's 75% of frame time is RPC serialization.
  Fire-and-forget `cudaMemcpyAsync` / `cudaLaunchKernel` would unblock
  the CPU to pipeline frames. Predicted ~250 LOC; predicted OpenPose
  throughput 0.48 → ~2 fps regardless of transport.

- **Multi-rail RDMA.** UCX's `UCX_MAX_RMA_RAILS=N` setting requires both
  HCAs (mlx5_0 and mlx5_1) to be configured with RoCE GIDs that route
  to the same peer. mlx5_0's GID setup is currently missing; this is a
  sysadmin task, not a code change.

- **HybridCommunicator iov support.** The plain-RDMA path uses
  `RdmaCommunicator` (via `properties_hybrid.json`), whose `WriteIov`
  doesn't handle the 5-fragment iov layout Phase 5 produces. Plain-RDMA
  benchmarks on the optimized frontend crash with
  `Buffer::AssignAll(): Can't read char`. Either revert Phase 5 in
  `CudaRt_memory.cpp` (see `fase4-fase5-revert-guide.md`) or patch
  `HybridCommunicator::WriteIov` to gather-send correctly.

- **CudnnHandler hang at high net_resolution.** OpenPose with
  `--custom_net_resolution -1x512` or higher wedges the backend during
  cuDNN autotuner's burst of `cudnnSetTensor4dDescriptorEx` calls.
  Backend container becomes unresponsive (kernel module wedge);
  requires `reboot`. Out of scope for the optimization stack — the
  `Process.cpp ↔ CudnnHandler ↔ communicator response write` path has
  a latent bug under burst load. Documented for future engineering.

---

## Inherited / non-thesis changes

The following files contain changes that pre-date this work and were
carried forward in the commit for build consistency:

- `plugins/cudadr/CMakeLists.txt`
- `plugins/cudadr/frontend/CudaDr_initialization.cpp`

These implement CUDA 12 `cuGetProcAddress` survival per a previous
contributor; they are not part of the UCX optimization story.
