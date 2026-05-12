# UCX Implementation Review

## Architecture Comparison: TCP vs UCX

### TCP Communication Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TCP COMMUNICATOR PATH                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FRONTEND (client node)              BACKEND (GPU node)                     │
│  ┌──────────────────────┐            ┌──────────────────────┐              │
│  │  CUDA Application    │            │  Plugin (.so)         │              │
│  │  cudaMalloc(...)     │            │  real cudaMalloc()    │              │
│  └──────────┬───────────┘            └──────────▲───────────┘              │
│             │                                   │                           │
│             ▼                                   │                           │
│  ┌──────────────────────┐            ┌──────────────────────┐              │
│  │  Frontend::Execute() │            │  Process::Start()     │              │
│  │  ┌────────────────┐  │            │  ┌────────────────┐  │              │
│  │  │ Write routine  │──┼──TCP───────┼─▶│ Read routine   │  │              │
│  │  │ Write in_buf sz│──┼──TCP───────┼─▶│ Read in_buf sz │  │              │
│  │  │ Write in_buf   │──┼──TCP───────┼─▶│ Read in_buf    │  │              │
│  │  │                │  │            │  │                │  │              │
│  │  │ Read exit_code │◀─┼──TCP───────┼──│ Write exit_code│  │              │
│  │  │ Read out_buf sz│◀─┼──TCP───────┼──│ Write out_buf  │  │              │
│  │  │ Read out_buf   │◀─┼──TCP───────┼──│   sz + data    │  │              │
│  │  └────────────────┘  │            │  └────────────────┘  │              │
│  └──────────────────────┘            └──────────────────────┘              │
│                                                                             │
│  Syscalls per RPC:  6 send() + 6 recv() = 12 kernel transitions            │
│  Copies per RPC:    app→user buf→kernel buf→NIC (×2 directions) = 4+ copies│
│  Latency drivers:   TCP/IP stack, Nagle, ACK delays, context switches      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### UCX Active Message Path (Current Implementation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UCX ACTIVE MESSAGE PATH                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FRONTEND (client node)              BACKEND (GPU node)                     │
│  ┌──────────────────────┐            ┌──────────────────────┐              │
│  │  CUDA Application    │            │  Plugin (.so)         │              │
│  │  cudaMalloc(...)     │            │  real cudaMalloc()    │              │
│  └──────────┬───────────┘            └──────────▲───────────┘              │
│             │                                   │                           │
│             ▼                                   │                           │
│  ┌──────────────────────┐            ┌──────────────────────┐              │
│  │  Frontend::Execute() │            │  Process::Start()     │              │
│  │  ┌────────────────┐  │            │  ┌────────────────┐  │              │
│  │  │ Build envelope │  │            │  │                │  │              │
│  │  │ [HDR|routine|  │  │            │  │ AM callback    │  │              │
│  │  │  payload]      │  │  single    │  │ fires with     │  │              │
│  │  │                │──┼──AM send───┼─▶│ complete frame  │  │              │
│  │  │                │  │            │  │                │  │              │
│  │  │ AM recv gets   │◀─┼──AM send───┼──│ [HDR|payload]  │  │              │
│  │  │ response frame │  │  single    │  │ single write   │  │              │
│  │  └────────────────┘  │            │  └────────────────┘  │              │
│  └──────────────────────┘            └──────────────────────┘              │
│                                                                             │
│  Syscalls per RPC:  1 send + 1 recv = 2 (or 0 with RDMA)                   │
│  Copies per RPC:    app→UCX buf→NIC (eager) or app→NIC (zcopy) = 1-2 copies│
│  Latency drivers:   UCX protocol negotiation (one-time), progress calls     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Wire-Level Comparison (Single CUDA RPC Call)

```
TCP (6 separate socket writes per direction):
─────────────────────────────────────────────────────────────────────────

  Time ──────────────────────────────────────────────────────────▶

  Frontend          Network              Backend
  ────────          ───────              ───────
  send(routine)  ──▶ [TCP segment 1] ──▶ recv()
  send(in_size)  ──▶ [TCP segment 2] ──▶ recv()
  send(in_data)  ──▶ [TCP segment 3] ──▶ recv()    ← execute()
                                                      │
  recv() ◀────────── [TCP segment 4] ◀── send(exit)   │
  recv() ◀────────── [TCP segment 5] ◀── send(size)   │
  recv() ◀────────── [TCP segment 6] ◀── send(data)   ▼

  Total: 6 network round-trip-capable segments
  Worst case with Nagle/delayed ACK: extra 40ms+ per segment


UCX AM (1 message per direction):
─────────────────────────────────────────────────────────────────────────

  Time ──────────────────────────────────────────────────────────▶

  Frontend          Network              Backend
  ────────          ───────              ───────
  am_send(frame) ──▶ [single AM msg] ──▶ am_recv_handler()
                                           │
                                           ├── execute()
                                           │
  am_recv()  ◀─────── [single AM msg] ◀── am_send(response)

  Total: 2 network messages (request + response)
  With RDMA: zero-copy, kernel-bypass, no TCP overhead
```

### Transport Stack Depth

```
┌────────────────────────────────┐    ┌────────────────────────────────┐
│         TCP STACK              │    │         UCX STACK (RDMA)       │
├────────────────────────────────┤    ├────────────────────────────────┤
│                                │    │                                │
│  GVirtuS Buffer               │    │  GVirtuS Envelope Frame       │
│         │                      │    │         │                      │
│         ▼                      │    │         ▼                      │
│  write() syscall               │    │  ucp_am_send_nbx()            │
│         │                      │    │         │                      │
│         ▼                      │    │         ▼                      │
│  TCP send buffer (kernel)      │    │  UCX protocol layer            │
│         │                      │    │    (eager/rndv decision)       │
│         ▼                      │    │         │                      │
│  IP layer (routing, frag)      │    │         ▼                      │
│         │                      │    │  ┌──────────────────────┐     │
│         ▼                      │    │  │  RDMA verb (ibv_post) │     │
│  Ethernet framing              │    │  │  OR tcp sendmsg      │     │
│         │                      │    │  └──────────┬───────────┘     │
│         ▼                      │    │             │                  │
│  NIC driver (interrupt)        │    │             ▼                  │
│         │                      │    │  NIC hardware (RDMA: no CPU)  │
│         ▼                      │    │                                │
│  NIC hardware                  │    │                                │
│                                │    │                                │
│  Layers: 6                     │    │  Layers: 3-4                   │
│  Kernel transitions: 2+        │    │  Kernel transitions: 0 (RDMA) │
│  Copies: 2-3 (user→kern→NIC)  │    │  Copies: 0-1 (zcopy possible) │
│                                │    │                                │
└────────────────────────────────┘    └────────────────────────────────┘
```

### Performance Gain Potential

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EXPECTED PERFORMANCE GAINS                               │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│  Metric              │  TCP                 │  UCX (RDMA)                   │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│  Latency (small msg) │  ~50-100 µs          │  ~1-5 µs (10-50× better)     │
│  Throughput (bulk)   │  ~10-25 Gbps         │  ~100-200 Gbps (ConnectX-6)  │
│  CPU overhead/msg    │  High (syscalls,     │  Low (userspace, polling)     │
│                      │   interrupts, copies)│                               │
│  Messages per RPC    │  6 (3 req + 3 resp)  │  2 (1 req + 1 resp)          │
│  Kernel involvement  │  Every send/recv     │  None (RDMA) or minimal      │
│  Memory copies       │  2-3 per direction   │  0-1 per direction (zcopy)   │
│  Connection setup    │  TCP 3-way handshake │  UCX endpoint (one-time)      │
├──────────────────────┴──────────────────────┴───────────────────────────────┤
│                                                                             │
│  WHERE GAINS ARE MOST VISIBLE:                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Small CUDA calls (cudaSetDevice, cudaGetLastError, etc.)           │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  TCP:  dominated by per-message overhead (6 syscalls)               │   │
│  │  UCX:  near wire-speed, single AM round-trip                        │   │
│  │  Gain: 10-50× latency reduction                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Large transfers (cudaMemcpy with multi-MB buffers)                 │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  TCP:  limited by kernel copy + NIC speed (~25 Gbps max)            │   │
│  │  UCX:  rendezvous + zero-copy RDMA (100+ Gbps on mlx5)             │   │
│  │  Gain: 4-8× throughput improvement                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  High-frequency call bursts (training loops, many small kernels)    │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  TCP:  CPU-bound on send/recv syscall overhead                      │   │
│  │  UCX:  userspace polling, no context switches                       │   │
│  │  Gain: 5-20× higher call throughput (calls/sec)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WHERE GAINS ARE LIMITED (current implementation):                   │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  • GPU memory still staged through host (no GPUDirect RDMA yet)     │   │
│  │  • All calls are synchronous (no async pipeline)                    │   │
│  │  • Busy-poll loop wastes CPU under low load                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Future: GPU-Direct RDMA Path (Not Yet Implemented)

```
CURRENT (host-staged):
═══════════════════════════════════════════════════════════════════

  Client Node                          GPU Node
  ┌─────────┐                          ┌─────────┐
  │ App buf │──copy──▶ UCX send ═══════▶ UCX recv ──copy──▶│ GPU mem │
  │ (host)  │         (host buf)       (host buf)          │ (device)│
  └─────────┘                          └─────────┘
                    2 copies + 1 RDMA transfer


FUTURE (GPU-aware UCX, GPUDirect RDMA):
═══════════════════════════════════════════════════════════════════

  Client Node                          GPU Node
  ┌─────────┐                          ┌─────────┐
  │ App buf │────────── UCX RDMA ═══════════════──────────▶│ GPU mem │
  │ (host)  │          (zero-copy,                         │ (device)│
  └─────────┘           NIC reads                          └─────────┘
                        directly from
                        registered mem)
                    0 copies + 1 RDMA transfer

  Requirements:
  • UCX built with CUDA support (cuda_copy in UCX_TLS)
  • nv_peer_mem or nvidia-peermem kernel module
  • ucp_mem_map() on GPU allocations
  • Envelope extended with memory_type field
```

---

## What Has Been Done

The project has a **complete Active Message (AM)-based UCX transport** fully integrated into GVirtuS.

### Core Transport (`src/communicators/ucx/UcxCommunicator.cpp`)
- UCX context initialized with `UCP_FEATURE_AM` and multi-threaded worker (`UCS_THREAD_MODE_MULTI`)
- Server path: listener → accept → child communicator sharing context/worker
- Client path: client-server endpoint with error handler
- AM handler registered at ID 1 with `UCP_AM_FLAG_WHOLE_MSG`
- **Rendezvous support**: detects `UCP_AM_RECV_ATTR_FLAG_RNDV` and uses `ucp_am_recv_data_nbx()` for large payloads
- Sync via `worker_flush_nbx()`

### Protocol (`include/gvirtus/communicators/UcxAmProtocol.h`)
- Structured envelope: magic (`0x4756414d` / "GVAM"), version, message type (Request/Response/Error), request ID, routine/payload sizes
- Frame format: `[EnvelopeHeader] + [routine_name] + [payload]`

### Frontend Integration (`src/frontend/Frontend.cpp`)
- Detects UCX mode via `to_string() == "ucxcommunicator"`
- Assembles request frame (header + routine + serialized args), sends as single AM write
- Reads response, validates magic/version/request_id, extracts output buffer

### Backend Integration (`src/backend/Process.cpp`)
- Separate AM request loop that reads envelope → dispatches to plugin handler → sends AM response
- Helper functions `read_ucx_am_request()` / `write_ucx_am_response()` for clean framing

### Infrastructure
- `CommunicatorFactory` and `EndpointFactory` both handle `"ucx"` suite
- Docker image installs UCX 1.20.0 (MOFED+CUDA12 build)
- Makefile exposes all UCX env vars (`UCX_TLS`, `UCX_NET_DEVICES`, `UCX_IB_GID_INDEX`, etc.)
- Benchmark script (`examples/simple_matrix/benchmark.sh`) supports `tcp`, `ucx-tcp`, `ucx-rdma`, `ucx-mixed` modes
- Frontend script auto-detects RDMA availability and falls back to TCP

---

## Review: Configuration & Design Issues

| Issue | Detail |
|-------|--------|
| **UCX mode detection via string comparison** | `to_string() == "ucxcommunicator"` in both Frontend and Process is fragile. A virtual method like `isAmMode()` or an enum would be cleaner and avoid silent breakage if the string changes. |
| **Busy-poll `Read()` loop** | `Read()` calls `ucp_worker_progress()` in a tight loop with no backoff. Under low load this burns CPU. A `ucp_worker_wait()` or event-fd poll would reduce idle CPU. |
| **Single AM ID for everything** | All messages share AM ID 1. This works for request-response but prevents multiplexing (e.g., out-of-band health checks, async notifications) without protocol changes. |
| **Worker mutex contention** | `Write()` locks the worker mutex for the full send+wait cycle. Under concurrent frontend threads, this serializes all writes through a single lock. |
| **No message size limits** | Neither frontend nor backend validates `payload_size` before allocating `std::vector<unsigned char>(payload_size)`. A malicious or corrupt header could trigger OOM. |
| **`header_size` mismatch on send vs. receive** | Frontend sets `header_size = 0` (default-initialized) but backend validates `header_size == sizeof(EnvelopeHeader)`. This works only because the backend code path seen doesn't always check it, but it's an inconsistency. |
| **Transport selection is env-var only** | `UCX_TLS`, `UCX_NET_DEVICES` etc. are only controllable via environment variables. There's no way to set them in `properties_ucx.json`, meaning the JSON config tells you *where* to connect but not *how*. |

---

## Recommended Next Steps

### 1. Robustness & Safety
- **Add payload size cap**: Validate `header.payload_size` against a configurable max (e.g., 256 MB) before allocating. Reject with `MessageType::Error`.
- **Fix `header_size` consistency**: Frontend should set `req_header.header_size = sizeof(EnvelopeHeader)` to match backend validation.
- **Add connection-level keepalive/heartbeat**: Currently a silently dead connection is only detected on the next `Read()` timeout. A periodic AM ping would detect failures faster.

### 2. Performance
- **Replace busy-poll with event-driven progress**: Use `ucp_worker_arm()` + epoll/event-fd to avoid spinning when idle. This is critical for multi-client backends.
- **Reduce worker mutex scope**: Separate progress/send paths, or use per-endpoint workers to allow parallel writes from concurrent frontend threads.
- **Zero-copy receive path**: The current AM handler copies data into `std::vector`. For large payloads, investigate UCX memory pools (`ucp_mem_map`) to avoid the copy.
- **Batch small messages**: Multiple small CUDA calls (e.g., `cudaSetDevice`, `cudaGetLastError`) could be batched into a single AM frame to reduce round-trip overhead.

### 3. Supporting Virtual CUDA Calls Better
- **Asynchronous call support**: CUDA async APIs (`cudaMemcpyAsync`, kernel launches) currently block on the round-trip. Implement a **fire-and-forget request path** with deferred response collection, matching CUDA stream semantics. The `request_id` in the envelope already supports this — responses just need to be matched asynchronously.
- **UCX transport config in JSON**: Move `UCX_TLS`, `UCX_NET_DEVICES`, `UCX_IB_GID_INDEX` into `properties_ucx.json` so a single config file controls both addressing and transport. Example:
  ```json
  "endpoint": {
      "suite": "ucx",
      "protocol": "ucx",
      "server_address": "25.25.25.1",
      "port": "32222",
      "tls": "rc_mlx5,ud_mlx5,tcp,self",
      "net_devices": "mlx5_1:1,ens1f1np1",
      "ib_gid_index": 3
  }
  ```
- **GPU-aware transport path**: Currently GVirtuS serializes everything to host buffers. For `cudaMemcpy` with large device buffers, the flow is: GPU→host copy → serialize → UCX send (host) → UCX recv (host) → deserialize → host→GPU copy. If UCX is built with CUDA support (`cuda_copy` in `UCX_TLS`), the communicator could pass GPU pointers directly to UCX, eliminating two host↔GPU copies. This requires:
  1. Extending the envelope with a `memory_type` field (host vs. device)
  2. Using `ucp_mem_map()` for GPU-registered memory on both sides
  3. Adding `cuda_copy` to `UCX_TLS` in the config
- **Multi-endpoint support**: The backend already reads an array of communicators from JSON, but the UCX path only uses one. Supporting multiple UCX endpoints (e.g., one for control, one for bulk data) would allow separating latency-sensitive calls from bandwidth-heavy transfers.

### 4. Observability
- **Per-call latency metrics**: The backend already measures `server_exec_sec`. Adding transport-level timing (serialize, send, recv, deserialize) would help identify bottlenecks.
- **UCX lane verification in logs**: Log which UCX transport lane was actually selected. Querying `ucp_ep_print_info()` at connection time and logging it would make this visible in GVirtuS logs instead of relying on external `UCX_LOG_LEVEL=info`.
