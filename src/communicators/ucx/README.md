# UCX Communicator

This directory holds the **UCX** transport implementation for GVirtuS. It
satisfies the `Communicator` interface (see
[`include/gvirtus/communicators/Communicator.h`](../../../include/gvirtus/communicators/Communicator.h))
using OpenUCX active messages for control + small payloads and `ucp_put_nbx`
RDMA for bulk data — with optional GPUDirect when the NIC + GPU + driver
allow it.

From the outside, this is just another `Communicator`: the frontend and
backend never see a UCX symbol. Everything in this directory is the
"infrastructure UCX makes you build yourself" that the kernel ships for
free with TCP — async completion, pinned RX pools, RMA rkey exchange, CUDA
memory registration.

---

## Files

| File | Lines | Purpose |
|---|---|---|
| [`UcxCommunicator.h`](UcxCommunicator.h) | 303 | Public class declaration + nested types (PinnedSlot, RemoteSlot, PooledMsg, AmState) |
| [`UcxCommunicator.cpp`](UcxCommunicator.cpp) | 1094 | The `Communicator` interface methods: init/destroy, Connect/Serve/Accept, Read/Write/WriteIov/WriteFrame, TryAcquireFrame/ReleaseFrame, Sync/Close, AM receive callback |
| [`UcxRma.cpp`](UcxRma.cpp) | 639 | The RDMA fast path: `send_rma_setup`, `handle_rma_setup_am`, `destroy_rma_state`, `WriteIovRma` |
| [`UcxRxPool.cpp`](UcxRxPool.cpp) | 274 | Pinned RX slot pool + TX scratch buffer: `init/destroy_rx_pool`, `acquire/release_rx_slot`, `map/unmap_slot_to_ucp`, `ensure/release_tx_scratch_locked` |
| [`UcxGpu.cpp`](UcxGpu.cpp) | 223 | CUDA dlopen helpers + GPUDirect probe and flag (`alloc_pinned_host`, `alloc_gpu_slot`, `is_gpu_pointer`, `probe_gpudirect`, `set/get_gpudirect_enabled`) |
| [`UcxInternal.h`](UcxInternal.h) | 70 | Shared helper API used by all four `.cpp` files (no `<ucp/api/ucp.h>` pulled in) |
| [`Endpoint_Ucx.cpp`](Endpoint_Ucx.cpp) | 82 | JSON config parser for `"suite": "ucx"` — pure data, no UCX symbols |

All `Ucx*.cpp` files compile into `libgvirtus-communicators-ucx.so`
(see [`CMakeLists.txt`](../../../CMakeLists.txt)).

---

## Layered view

```
                      ┌──────────────────────────────┐
                      │  Frontend.cpp / Process.cpp  │  ← never see UCX
                      └──────────────┬───────────────┘
                                     │  Communicator interface
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │            UcxCommunicator.cpp                     │
        │   Connect / Serve / Accept / Read / Write          │
        │   WriteFrame / WriteIov / TryAcquireFrame          │
        │   am_recv_handler  ◄── all incoming AMs land here  │
        └─────┬──────────────┬─────────────────┬─────────────┘
              │              │                 │
              ▼              ▼                 ▼
     ┌──────────────┐  ┌────────────┐   ┌─────────────────┐
     │  UcxRma.cpp  │  │UcxRxPool   │   │  UcxGpu.cpp     │
     │              │  │.cpp        │   │                 │
     │ RDMA fast    │  │ Pinned RX  │   │ CUDA dlopen     │
     │ path:        │  │ slots +    │   │ + GPUDirect     │
     │ rkey exch.   │  │ TX scratch │   │ probe + flag    │
     │ ucp_put_nbx  │  │            │   │ is_gpu_pointer  │
     └──────────────┘  └────────────┘   └─────────────────┘
              └─────────── UcxInternal.h ───────────┘
                  (debug-log + GPU helper API)
```

The arrows are one-way: `UcxCommunicator.cpp` calls into the other three
TUs; they don't call back into it. `UcxInternal.h` carries the tiny
shared helper surface so each `once_flag` / atomic has exactly one home.

---

## How a single call flows through

### Small CUDA call (e.g. `cudaSetDevice(0)`)

```
Frontend::Execute
  └─► am::WriteRequest        (RpcCodec)
       └─► UcxCommunicator::WriteFrame
            └─► WriteIov  ───►  ucp_am_send_nbx (eager AM)
                                          │
                                          │ (network)
                                          ▼
                          UcxCommunicator::am_recv_handler
                                          │
                                          │ enqueue PooledMsg from RX pool
                                          ▼
                          Process dispatch loop
                            └─► am::ReadRequest
                                 └─► TryAcquireFrame (zero-copy view)
```

20 B envelope + payload, one AM round trip, no memcpy on the receive side.

### Large `cudaMemcpy(64 MB H2D)` — RMA path

```
Frontend::Execute
  └─► am::WriteRequest
       └─► UcxCommunicator::WriteFrame
            └─► WriteIov
                 │ payload ≥ 64 KB and rma_setup_received_? → YES
                 ▼
            WriteIovRma  (in UcxRma.cpp)
                 │
                 │ pick a free remote slot (round-robin)
                 │ ucp_put_nbx user buffer  ──►  remote pinned slot
                 │   (or, with GPUDirect, peer's GPU shadow region)
                 │ ucp_am_send_nbx RmaPosted  ──► slot_idx + RmaPostedBody
                 ▼
                                  am_recv_handler sees RmaPosted
                                    │ wrap slot as PooledMsg with
                                    │ gpu_data / gpu_size set when split
                                    ▼
                                  TryAcquireFrame returns the slot view
                                  Handler does cudaMemcpyDeviceToDevice
                                    directly from slot.gpu_addr
```

64 MB never crosses host RAM on the receive side when GPUDirect is up.

---

## The "RMA vs AM" threshold

[`UcxCommunicator::WriteIov`](UcxCommunicator.cpp) picks between paths:

```cpp
if (total >= (64u * 1024u) && rma_setup_received_.load()) {
    size_t put = WriteIovRma(iov, iov_count, total);
    if (put == total) return put;
}
// else: ucp_am_send_nbx with UCP_DATATYPE_IOV
```

Hardcoded **64 KB** — small enough that bulk transfers always take RMA,
large enough that tiny control calls aren't burdened with rkey lookup.

Inside RMA, [`WriteIovRma`](UcxRma.cpp) picks between `staged` and
`zerocopy` modes via the `GVIRTUS_RMA_ZEROCOPY` env var (default on).

---

## GPUDirect activation chain

1. Backend launches with `GVIRTUS_GPUDIRECT=1` (typically set in
   [`etc/ucx.env`](../../../etc/ucx.env)).
2. `init_ucx()` in `UcxCommunicator.cpp`:
   - reads `UCX_TLS` — if no CUDA-capable transport (rc/dc/ud/ib), bails
   - sets `UCX_RCACHE_ENABLE=n UCX_MEMTYPE_CACHE=n` (probe needs this)
   - calls `probe_gpudirect(ctx)` from `UcxGpu.cpp` — try cudaMalloc(4K)
     + ucp_mem_map(CUDA) + cleanup
   - on success: `set_gpudirect_enabled(true)`; logs `[GVS] GPUDirect=enabled`
3. `init_rx_pool()` (in `UcxRxPool.cpp`) checks `gpudirect_enabled()` and
   allocates a GPU shadow region for each host slot via `alloc_gpu_slot`.
4. `send_rma_setup()` (in `UcxRma.cpp`) packs rkeys for both host slots
   AND their GPU shadows into the `RmaSetup` AM (per-slot flag bit
   `kHasGpuShadow`).
5. The client's `handle_rma_setup_am` unpacks both into `RemoteSlot`
   entries.
6. On a `cudaMemcpy` H2D, `WriteIovRma` checks `is_gpu_pointer(src)` and
   if true, splits the payload — small prefix to host slot, big GPU
   portion to the peer's GPU shadow.
7. `am_recv_handler` recognizes the GPU-split `RmaPosted`, builds a
   `PooledMsg` with `gpu_data`/`gpu_size` set, and the cudart handler
   does `cudaMemcpyDeviceToDevice` directly from the shadow — no host
   bounce.

If any step fails, the path falls back gracefully: probe fail → host-only;
no RDMA lane → AM-IOV path; no `RmaSetup` received → AM-IOV path.

---

## Why so much code

UCX is async-everything and gives you nothing for free. Roughly:

| Bucket | Lines | TCP equivalent |
|---|---|---|
| UCX bootstrap (`init_ucx`/`destroy_ucx`) | ~280 | `socket()` + `bind()` |
| AM dispatch (callback, RX queue, RNDV poll) | ~150 | a `read()` syscall |
| Pinned RX pool | 274 | kernel skb pool |
| RMA setup + data path | 639 | none — no RDMA in TCP |
| CUDA dlopen + GPUDirect | 223 | none |
| Communicator API (Read/Write/Sync/Close/...) | ~400 | ~250 lines in TCP |

About **70 %** of the UCX code is infrastructure UCX requires you to build
that the kernel provides transparently for sockets. None of this leaks
above the `Communicator` interface, which is the entire point of the
refactor: from `Frontend.cpp` and `Process.cpp` both transports look
identical.

---

## Conventions

- **No transport leaks above the `Communicator` interface.** If you find
  yourself wanting to call a UCX-specific symbol from the frontend or
  backend, add a new virtual method to `Communicator` instead.
- **CUDA is never linked statically.** Every `cuda*` symbol is `dlopen`'d
  via the resolvers in `UcxGpu.cpp`. The plugin is one library — it
  should not refuse to load on a CUDA-less host.
- **One `once_flag` per resolver.** All CUDA dlopen state lives in
  `UcxGpu.cpp`; if you need a new `cuda*` symbol, add it there and expose
  it via `UcxInternal.h`.
- **All debug logging via `ucx_debug_log`.** It's a no-op unless
  `GVIRTUS_LOGLEVEL` is DEBUG (10000) or TRACE (0).
- **The four RMA fields in the envelope are not duplicated here.** Wire
  types live in [`Protocol.h`](../../../include/gvirtus/communicators/Protocol.h);
  the codec lives in [`RpcCodec.cpp`](../RpcCodec.cpp).

---

## Adding a feature

Most additions land in **one** of these files, not across all four:

| Adding... | Goes in |
|---|---|
| A new `cuda*` symbol you need to dlopen | `UcxGpu.cpp` + `UcxInternal.h` |
| A new control message type | `Protocol.h` (the enum) + `am_recv_handler` in `UcxCommunicator.cpp` |
| A new RMA optimisation (e.g. message coalescing) | `UcxRma.cpp` |
| A new slot lifecycle hook (e.g. NUMA-aware slots) | `UcxRxPool.cpp` |
| Tweaking the RMA-vs-AM threshold | one line in `UcxCommunicator::WriteIov` |
| A new env var | the file that owns the state it gates |

If a change spans multiple files, you're probably crossing a layering
boundary — stop and reconsider.

---

## Further reading

- [`docs/UCX_GUIDE.md`](../../../docs/UCX_GUIDE.md) — user-facing
  configuration and tuning guide.
- [`docs/UCX_OPTIMIZATIONS.md`](../../../docs/UCX_OPTIMIZATIONS.md) — the
  optimisation phases and their measured impact.
- [`docs/GPUDIRECT.md`](../../../docs/GPUDIRECT.md) — the GPUDirect
  rollout in detail (Variant B steps B1 → B4).
- [`docs/IOV_REFACTOR.md`](../../../docs/IOV_REFACTOR.md) — the IoV
  transport refactor that produced this layout.
