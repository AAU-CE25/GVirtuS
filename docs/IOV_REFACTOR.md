# IoV Transport Refactor — Summary

A clean-sheet refactor of the GVirtuS communicator layer. The goal: replace
the old byte-stream-per-Add path with an **IoV (scatter/gather) transport**
that lets every transport — TCP and UCX today, anything else tomorrow —
plug in behind a single, narrow interface, with **zero transport-specific
code in the frontend or backend**.

Branch: `feature/clean-sheet-comms`
Commits: `bfab776c` → `dac31be1` (6 commits + one rename)

---

## Goals

1. **One wire format, one codec, one dispatch loop** — no per-transport
   branching in frontend/backend.
2. **Zero-copy large payloads** — the marshaller can reference borrowed
   host memory in place; transports gather-send the fragments natively.
3. **Modular transports** — TCP keeps a length-prefixed byte stream, UCX
   uses native active-message delimiting; both satisfy the same
   `Communicator` contract.
4. **Readable** — every transport difference lives behind a virtual call,
   not behind `to_string()` string checks or `#ifdef`s.

---

## Result at a glance

| File | Before | After |
|---|---|---|
| `src/backend/Process.cpp` | 560 lines, hand-rolled codec + getstring/read_exact + hybrid dispatch | 233 lines, one loop calling `am::ReadRequest`/`am::WriteResponse` |
| `src/frontend/Frontend.cpp` Execute path | UCX/TCP branch, stream-fallback duplicate, inline envelope build | One path: marshal IoV → `am::WriteRequest` → `am::ReadResponse` |
| Transports | 12 communicators (RDMA, IB, Hybrid, AfUnix, Shm, VMShm, VMSocket, Virtio, Vmci, Zmq, TCP, UCX) | 2 (TCP, UCX) |
| Transport checks in frontend/backend | `to_string()` string compares, env-var gates | 0 — no `"ucx"`/`"tcp"` mentions in dispatch code |
| Zero-copy ceiling | 1 borrowed pointer per call (Frontend fields) | N borrowed segments per Buffer |

---

## The 6 commits

### 1. `bfab776c` — Drop all communicators except TCP and UCX
Removed `RdmaCommunicator`, `IbvCommunicator`, `HybridCommunicator`,
`AfUnixCommunicator`, `ShmCommunicator`, `VMShmCommunicator`,
`VMSocketCommunicator`, `VirtioCommunicator`, `VmciCommunicator`,
`ZmqCommunicator`, plus their endpoints and properties files. Stripped
the dead RDMA/Hybrid gates (`getstring`, the hybrid dispatch block,
`begin_call`/`end_call`) from `Frontend.cpp` and `Process.cpp`.

> **−3734 lines.** Plugin-facing APIs unchanged.

### 2. `9bdc2fb0` — Move endpoint sources beside their communicators
`Endpoint_Tcp.cpp` → `src/communicators/tcp/`,
`Endpoint_Ucx.cpp` → `src/communicators/ucx/`. Headers stay in the public
include tree, so no `#include` changes. Each transport is now a
self-contained directory.

### 3. `c3bbad90` — Buffer: add the IoV segment model
The send-side change that enables everything else. `Buffer` gains:

```cpp
template <class T> void AddRef(const T *ptr, size_t n = 1); // borrowed segment
void   GetIov(std::vector<struct iovec> &out) const;        // ordered fragments
size_t GetLogicalSize() const;                              // arena + borrowed
bool   HasSegments() const;
enum class SegKind { Inline, HostRef };                     // extensible
```

`AddRef<T>` writes the same `[size_t len][bytes]` wire layout as
`Add<T>(ptr, n)` but **without copying** the payload — the pointer and
length are recorded as a `HostRef` segment, and `GetIov()` emits them
interleaved with arena slices in the correct order.

`Buffer::Dump()` is now segment-aware: it frames with `GetLogicalSize()`
and uses `WriteIov` when segments exist, falling back to the single
contiguous `Write` otherwise. **With no `AddRef` segments the behaviour
is byte-for-byte identical to before.**

Added [`tests/test_buffer_iov.cpp`](../tests/test_buffer_iov.cpp) (no
CUDA/UCX/log4cplus required) covering wire-identity, IoV ordering,
round-trip receive, framing parity, and null/empty handling.

### 4. `8a195023` — Frontend uses the Buffer IoV API
Replaced the hand-rolled `iov[5]` splice in `Frontend::Execute` with
`[header][routine] + input_buffer->GetIov()`. Reimplemented
`AddHostPointerForArgumentsDirect` as `mpInputBuffer->AddRef`.

Borrowed payloads now live inside `Buffer`, not as separate `Frontend`
fields:

- The **single-direct-pointer limit is gone** — N segments supported.
- `Buffer`'s internal layout no longer leaks into `Frontend`
  (`mDirectInputBufferOffset` splitting removed).
- The transport gate in `AddHostPointerForArgumentsDirect` is removed —
  `AddRef` works on every transport (UCX scatters via `WriteIov`,
  TCP concatenates once in `Buffer::Dump`).

Plugin API unchanged (`AddHostPointerForArgumentsDirect` signature
preserved), so `cudart`'s call sites are untouched.

### 5. `eec3ce63` — Unify receive: framing in Communicator, codec out of Process
The big architectural step. The `Communicator` base class now owns
**whole-message framing**:

```cpp
class Communicator {
    // Default: length-prefixed [uint64 size][body] byte-stream framing.
    virtual size_t WriteFrame(const struct iovec *iov, size_t iov_count);
    virtual bool   TryAcquireFrame(const unsigned char *&data, size_t &size);
    virtual void   ReleaseFrame();
    // ...
};
```

- The default implementation gives **TCP** a length-prefixed byte stream
  for free.
- `UcxCommunicator` overrides both with native active-message delimiting
  (zero-copy: `TryAcquireFrame` returns a pointer into the pinned
  RX-pool slot).

A new codec — `communicators::am` in `RpcCodec.{h,cpp}` (then named
`AmProtocol`; see the rename below) — owns envelope encode/decode:

```cpp
bool ReadRequest(Communicator*, EnvelopeHeader&, string& routine,
                 const unsigned char*& payload, size_t& size, ...);
bool WriteResponse(Communicator*, const EnvelopeHeader& req, int exit_code,
                   double exec_sec, const shared_ptr<Buffer>& out, ...);
```

`Process::Start` collapses to **one** dispatch loop calling these two
functions. `getstring()`, `read_exact()`, and the two ~80-line codec
functions in `Process.cpp` are gone (560 → 233 lines).

`Frontend::Execute` collapses to **one** path: `WriteFrame` (gather-send
the marshaled IoV) + a single `TryAcquireFrame` receive. The
`ucx_am_mode` branch, the legacy TCP `Write`/`Dump`/`Read` path, and the
stream-fallback duplicate are all removed.

**Zero `to_string()` transport checks remain in frontend/backend.**

Added [`tests/test_protocol_loopback.cpp`](../tests/test_protocol_loopback.cpp):
an in-memory loopback `Communicator` drives a full
request → dispatch → response round trip through the base-class framing
+ codec (i.e. exactly the TCP path), asserting envelope, routine,
`AddRef` payload, and response parsing.

### 6. `dac31be1` — Codec fully transport-agnostic; frontend routes through it
The codec already owned the backend-side wire format. The frontend
still hand-built the envelope inline — two places to keep in sync, and
`Frontend.cpp` was littered with `"UCX AM"` identifiers and error
strings even though the path is identical for every transport.

- Renamed the wire-types header: `UcxAmProtocol.h` → `Protocol.h`,
  namespace `ucxam` → `am`. Header docstring no longer claims it's a
  "UCX Active Message wire protocol" — RMA messages describe RDMA-write
  semantics instead of naming `ucp_put_nbx`/`ucp_init`.
- Extended the codec with the frontend-side mirror:

  ```cpp
  bool WriteRequest(Communicator*, uint64_t request_id, const string& routine,
                    const struct iovec* payload_iov, size_t iov_count,
                    size_t payload_logical_size, string& error);
  bool ReadResponse(Communicator*, uint64_t expected_id, int& exit_code,
                    double& exec_sec, const unsigned char*& out, size_t& size,
                    bool& owns_frame, string& error);
  ```

- `Frontend::Execute` becomes: marshal IoV → `WriteRequest` → `ReadResponse`
  → (zero-copy direct-output fast path **or** bulk `AppendBytes`) →
  `ReleaseFrame`. Profile timing, `mDataSent`/`mDataReceived` accounting,
  and the zero-copy direct-output path are preserved.
- UCX-specific names scrubbed from frontend/backend: `gUcxAmRequestId`
  → `gRequestId`, `[UCX AM]` log prefix dropped, `"Frontend UCX AM:"`
  error strings genericised, every comment that named `ucp_init` /
  `libuct_cuda` reworded as "some transports' init paths" so the
  reentrancy rationale still reads but no longer couples to UCX.
  `grep -i ucx` on `src/frontend/` and `src/backend/` now returns **no
  matches**.

### Post-commit: `AmProtocol` → `RpcCodec` rename
Renamed `AmProtocol.{h,cpp}` → `RpcCodec.{h,cpp}` (history preserved via
`git mv`) to break the last lexical tie to UCX active messages — what's
in those files is a generic RPC codec. The wire-types header keeps the
neutral `Protocol.h` name. The codec namespace stays `am` for symbol
brevity (`am::ReadRequest`, `am::WriteResponse`, …).

---

## Final layered architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend.cpp / Process.cpp        ← single, transport-agnostic │
│  (call codec, never touch wire format)                          │
├─────────────────────────────────────────────────────────────────┤
│  am::WriteRequest  am::ReadRequest                              │
│  am::ReadResponse  am::WriteResponse           [RpcCodec.{h,cpp}]│
│  (encode/decode the EnvelopeHeader)                             │
├─────────────────────────────────────────────────────────────────┤
│  Communicator::WriteFrame / TryAcquireFrame / ReleaseFrame      │
│  Communicator::WriteIov                          [Communicator.h]│
│  (default: length-prefixed byte stream; override for native AM) │
├──────────────────────────┬──────────────────────────────────────┤
│  TcpCommunicator         │  UcxCommunicator                     │
│  (inherits default frame)│  (overrides with ucp_am_send_nbx +   │
│                          │   pinned RX-pool zero-copy)          │
└──────────────────────────┴──────────────────────────────────────┘
```

The contract is **narrow** — `WriteFrame` / `TryAcquireFrame` / `ReleaseFrame`
+ optional `WriteIov` / `current_frame_gpu` / `current_connection_supports_cuda`
— so adding a third transport (e.g. shared-memory, io_uring) means
implementing five functions, not threading transport-specific branches
through frontend, backend, and the codec.

---

## Final source layout

```
include/gvirtus/communicators/
├── Buffer.h            ← IoV segment model (AddRef / GetIov / GetLogicalSize)
├── Communicator.h      ← WriteFrame / TryAcquireFrame contract
├── Endpoint.h / EndpointFactory.h / CommunicatorFactory.h
├── Endpoint_Tcp.h / Endpoint_Ucx.h
├── Protocol.h          ← wire types only (EnvelopeHeader, MessageType, ...)
├── Result.h
└── RpcCodec.h          ← RPC codec (depends on Buffer + Communicator + Protocol)

src/communicators/
├── Buffer.cpp / CommunicatorFactory.cpp / EndpointFactory.cpp / Result.cpp
├── RpcCodec.cpp
├── tcp/
│   ├── Endpoint_Tcp.cpp
│   └── TcpCommunicator.{h,cpp}
└── ucx/
    ├── Endpoint_Ucx.cpp
    └── UcxCommunicator.{h,cpp}
```

`Protocol.h` is **header-only and dependency-free** (just `<cstdint>`) —
the UCX communicator includes it for its RmaSetup/RmaPosted side-channel
without pulling the codec in. `RpcCodec.{h,cpp}` is the only translation
unit that depends on both the wire format and the `Communicator` interface.

---

## What didn't change

- **Plugin ABI**: every `cudaXxx`/`cublasXxx`/`cudnnXxx` handler signature
  and the `Frontend::AddVariableForArguments` / `AddHostPointerForArguments` /
  `AddHostPointerForArgumentsDirect` plugin-facing API are byte-for-byte
  unchanged. `cudart`'s `CudaRt_memory.cpp` call sites were not edited.
- **Wire format**: with no `AddRef` segments, the bytes on the wire are
  identical to the pre-refactor `Add<T*>` path.
- **TCP behaviour**: TCP gets its `WriteFrame`/`TryAcquireFrame` from
  the `Communicator` base class — no TCP-specific code was added for the
  framing; the length-prefix scheme is the natural default.

---

## Tests

| Test | Drives | Why it matters |
|---|---|---|
| `test_buffer_iov` | `Buffer::AddRef` / `GetIov` / `Dump` framing | Proves the IoV model is wire-identical to the copy path. |
| `test_protocol_loopback` | All four codec entry points + base-class framing | End-to-end TCP-equivalent path with no real network. |

Both compile without CUDA, UCX, or log4cplus, so they catch regressions
in the codec layer before anything hits the dev container.

---

## Net diff (refactor commits only)

```
 ~40 files changed
 +1100 / −5200 lines (approx, including communicator deletions)
```

The codebase is now **smaller, more modular, and more readable**, with a
single source of truth for the RPC wire format and a single,
transport-agnostic call/dispatch path.
