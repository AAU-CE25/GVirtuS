/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRtHandler.h"
#include "gvirtus/communicators/Communicator.h"
#include "CudaUtil.h"

#include <gvirtus/communicators/Communicator.h>

#include <algorithm>
#include <cstdlib>

using namespace log4cplus;
using namespace std;

using gvirtus::common::mappedPointer;

namespace {
// GPUDirect gate. Two layers:
//
//   (1) Process-wide: GVIRTUS_GPUDIRECT_ACTIVE env var, set by the UCX
//       communicator's init_ucx based on the cudaMalloc + ucp_mem_map(CUDA)
//       probe. False unless the backend was started with GVIRTUS_GPUDIRECT=1
//       AND nvidia-peermem is loaded AND UCX can register CUDA memory.
//
//   (2) Per-connection (Option 2 — supersedes the prior process-only gate):
//       Process.cpp's UCX-AM dispatch loop sets
//       gvirtus::communicators::tls_connection_supports_cuda just before
//       calling Handler::Execute, derived from the active endpoint's
//       negotiated transport. False on UCX-TCP endpoints (which cannot
//       peer-DMA from CUDA memory) and on the brief window before
//       Process.cpp sets it.
//
// Both must be true for GPUDirect to activate on a given call. (1) is
// cached at process start; (2) is read fresh each call so a single
// backend with UCX_TLS=rc_mlx5,ud_mlx5,tcp,self can serve mixed
// RDMA + TCP frontends, only routing through the GPU shadow on the
// connections that negotiated an RDMA lane.
bool gvirtus_gpudirect_enabled() {
    static const bool process_active = []() {
        const char *v = std::getenv("GVIRTUS_GPUDIRECT_ACTIVE");
        return v != nullptr && v[0] == '1';
    }();
    return process_active &&
           gvirtus::communicators::tls_connection_supports_cuda;
}

// D2H GPUDirect gate — separate from the H2D one on purpose, and auto-detected.
// The D2H "GPU scratch + attach device fragment to the Result" path needs the
// backend to ucp_put that device fragment into the client's RX slot (backend GPU
// -> client host slot). That only works when the client advertised RMA slots
// whose rkey THIS backend's UCX could unpack. A native frontend whose UCX
// exposes no RMA-unpackable memory domain fails that unpack -> the client's
// remote-slot rkeys come back null -> WriteIovRma falls through and the device
// fragment would be pushed through the eager AM path -> ucp_am_send of device
// memory errors -> the connection resets. So gate the D2H scratch on whether the
// client is actually RMA-put-capable (tls_client_rma_put_capable, set per
// connection by Process.cpp from the communicator): Docker/all-RDMA frontends
// keep the zero-copy D2H GPUDirect; CPU/native frontends transparently fall back
// to the host path (correct everywhere, no crash). H2D GPUDirect (backend GPU
// shadow receive) is UNAFFECTED — this gates only the D2H response.
bool gvirtus_gpudirect_d2h_enabled() {
    return gvirtus_gpudirect_enabled() &&
           gvirtus::communicators::tls_client_rma_put_capable;
}

// Thread-local GPU scratch used by the D2H handler when GPUDirect is active.
// Grows on demand (2× growth to amortize cudaMalloc cost). Each backend
// worker thread has its own scratch so there's no cross-thread synchronization.
// Lifetime: process lifetime (cudaFree at destructor).
thread_local void *tls_gpu_scratch       = nullptr;
thread_local size_t tls_gpu_scratch_size = 0;

void *get_tls_gpu_scratch(size_t needed) {
    if (tls_gpu_scratch_size >= needed && tls_gpu_scratch != nullptr) {
        return tls_gpu_scratch;
    }
    if (tls_gpu_scratch != nullptr) {
        cudaFree(tls_gpu_scratch);
        tls_gpu_scratch = nullptr;
        tls_gpu_scratch_size = 0;
    }
    const size_t new_size = std::max(needed, tls_gpu_scratch_size * 2);
    if (cudaMalloc(&tls_gpu_scratch, new_size) != cudaSuccess) {
        tls_gpu_scratch = nullptr;
        tls_gpu_scratch_size = 0;
        return nullptr;
    }
    tls_gpu_scratch_size = new_size;
    return tls_gpu_scratch;
}

// Pool of thread-local GPU scratches for ASYNC (deferred) D2H via client-GET.
// The synchronous D2H (get_tls_gpu_scratch, single buffer) is GET'd immediately,
// so one buffer suffices. An async/deferred D2H is GET'd by the client LATER (at
// the next stream sync), so its scratch must survive until then — a single
// buffer would be overwritten by the next async D2H. With N buffers round-
// robined, up to N async D2H can be in flight concurrently; the frontend's flow
// control caps in-flight at N (draining+GET-ing the oldest before a slot is
// reused), so a scratch is never overwritten before its client GET completes.
static constexpr int kD2HGetPoolSize = 4;
thread_local void *tls_d2h_get_pool[kD2HGetPoolSize]       = {nullptr};
thread_local size_t tls_d2h_get_pool_size[kD2HGetPoolSize] = {0};
thread_local int tls_d2h_get_pool_next                     = 0;

void *get_d2h_get_scratch(size_t needed) {
    const int i = tls_d2h_get_pool_next;
    tls_d2h_get_pool_next = (tls_d2h_get_pool_next + 1) % kD2HGetPoolSize;
    if (tls_d2h_get_pool[i] != nullptr && tls_d2h_get_pool_size[i] >= needed) {
        return tls_d2h_get_pool[i];
    }
    if (tls_d2h_get_pool[i] != nullptr) {
        cudaFree(tls_d2h_get_pool[i]);
        tls_d2h_get_pool[i] = nullptr;
        tls_d2h_get_pool_size[i] = 0;
    }
    const size_t new_size = std::max(needed, tls_d2h_get_pool_size[i] * 2);
    if (cudaMalloc(&tls_d2h_get_pool[i], new_size) != cudaSuccess) {
        tls_d2h_get_pool[i] = nullptr;
        tls_d2h_get_pool_size[i] = 0;
        return nullptr;
    }
    tls_d2h_get_pool_size[i] = new_size;
    return tls_d2h_get_pool[i];
}

// Fase 1+2 TLS host slot for the D2H legacy path. Replaces the per-call
// `new char[count]` allocation: first call faults all pages via memset(0),
// subsequent calls reuse the warm slot. Combined with the non-owning
// Buffer(char*, size_t) view (mOwnBuffer=false), eliminates two of the
// three 64 MB copies the naive implementation does (alloc-with-page-fault,
// cudaMemcpy into it, Add<char> realloc+memmove into Buffer's mpBuffer,
// delete[]) -> handler latency drops from ~80 ms to ~4.7 ms at 64 MB.
//
// Layout: slot[0..8) = size_t prefix (count), slot[8..8+count) = data.
// Matches Add<char>(ptr, n) wire format so the frontend's Assign<char>
// reads the prefix correctly. Grows monotonically by 2x; never shrinks.
thread_local char *tls_d2h_slot      = nullptr;
thread_local size_t tls_d2h_slot_cap = 0;

char *get_tls_d2h_slot(size_t needed_bytes_for_data) {
    const size_t needed_total = sizeof(size_t) + needed_bytes_for_data;
    if (tls_d2h_slot != nullptr && tls_d2h_slot_cap >= needed_total) {
        return tls_d2h_slot;
    }
    delete[] tls_d2h_slot;
    const size_t new_cap = std::max(needed_total, tls_d2h_slot_cap * 2);
    tls_d2h_slot = new (std::nothrow) char[new_cap];
    if (tls_d2h_slot == nullptr) {
        tls_d2h_slot_cap = 0;
        return nullptr;
    }
    tls_d2h_slot_cap = new_cap;
    // Pre-fault: touch every page so subsequent cudaMemcpy doesn't pay
    // page-fault cost on first access. memset is the simplest portable way.
    std::memset(tls_d2h_slot, 0, new_cap);
    return tls_d2h_slot;
}
}  // namespace

// This is for HostRegister support
// Key: Frontend pointer that was malloc’d on client side.
// Value: Backend pinned pointer allocated via cudaHostRegister.
unordered_map<void *, void *> hostRegisteredMap;

CUDA_ROUTINE_HANDLER(MemGetInfo) {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    size_t *free = out->Delegate<size_t>();
    size_t *total = out->Delegate<size_t>();
    cudaError_t exit_code = cudaMemGetInfo(free, total);
    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(Free) {
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    cudaError_t exit_code = cudaFree(devPtr);

    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(FreeArray) {
    cudaArray *arrayPtr = input_buffer->GetFromMarshal<cudaArray *>();

    cudaError_t exit_code = cudaFreeArray(arrayPtr);

    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(GetSymbolAddress) {
    void *devPtr;
    const char *symbol = pThis->GetSymbol(input_buffer);

    cudaError_t exit_code = cudaGetSymbolAddress(&devPtr, symbol);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    if (exit_code == cudaSuccess) out->AddMarshal(devPtr);

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetSymbolSize) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        size_t *size = out->Delegate<size_t>();
        *size = *(input_buffer->Assign<size_t>());
        const char *symbol = pThis->GetSymbol(input_buffer);
        cudaError_t exit_code = cudaGetSymbolSize(size, symbol);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// testing vpelliccia

CUDA_ROUTINE_HANDLER(MemcpyPeerAsync) {
    void *dst = NULL;
    void *src = NULL;
    try {
        dst = input_buffer->GetFromMarshal<void *>();
        int dstDevice = input_buffer->Get<int>();
        src = input_buffer->GetFromMarshal<void *>();
        int srcDevice = input_buffer->Get<int>();
        size_t count = input_buffer->Get<size_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();

        cudaError_t exit_code = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocManaged) {
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "MallocManaged");

    try {
        void *hostPtr = input_buffer->Get<void *>();
        size_t size = input_buffer->Get<size_t>();
        unsigned flags = input_buffer->Get<unsigned>();
        void *devPtr;

        cudaError_t exit_code = cudaMallocManaged(&devPtr, size, flags);
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "cudaMallocManaged returned: " << exit_code);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        mappedPointer host;
        host.pointer = hostPtr;
        host.size = size;

        out->AddMarshal(devPtr);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Malloc3DArray) {
    cudaArray *array = NULL;

    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    cudaExtent extent = input_buffer->Get<cudaExtent>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaMalloc3DArray(&array, desc, extent, flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(&array);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(Malloc) {
    void *devPtr = NULL;
    try {
        size_t size = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaMalloc(&devPtr, size);
#ifdef DEBUG
        std::cout << "Allocated DevicePointer " << devPtr << " with a size of " << size
                  << std::endl;
#endif
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->AddMarshal(devPtr);
        // cout << "Malloc: allocated " << size << " bytes at " << devPtr <<
        // endl;
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocArray) {
    cudaArray *arrayPtr = NULL;
    try {
        cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
        size_t width = input_buffer->Get<size_t>();
        size_t height = input_buffer->Get<size_t>();

        cudaError_t exit_code = cudaMallocArray(&arrayPtr, desc, width, height);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->AddMarshal(arrayPtr);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocPitch) {
    void *devPtr = NULL;
    try {
        size_t pitch = input_buffer->Get<size_t>();
        size_t width = input_buffer->Get<size_t>();
        size_t height = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaMallocPitch(&devPtr, &pitch, width, height);
#ifdef DEBUG
        std::cout << "Allocated DevicePointer " << devPtr << " with a size of " << width * height
                  << std::endl;
#endif
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->AddMarshal(devPtr);
        out->Add(pitch);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

namespace {

// The stream the GPUDirect shadow->destination copies are issued on, and whether one is
// still in flight for the frame currently being consumed. Both are thread_local because
// the backend runs one thread per connection and the transport calls the drain hook on
// that same thread, right after it sends the response.
thread_local cudaStream_t g_shadow_stream = nullptr;
thread_local bool g_shadow_copy_pending = false;

// Set by the ASYNC H2D handler, which issues its shadow->destination copy on the
// CALLER's stream (it has to: subsequent work the client queues on that stream must
// see the data) rather than on g_shadow_stream. A stream sync on g_shadow_stream
// therefore cannot cover it, and the caller may have used more than one stream, so
// the drain falls back to a device-wide sync in that case. It only fires when such a
// copy is genuinely outstanding.
thread_local bool g_shadow_async_pending = false;

// ---------------------------------------------------------------------------
// H2D trace (GVIRTUS_H2D_TRACE=1). Diagnostic only, off by default.
// ---------------------------------------------------------------------------
// Prints, per H2D request, which branch the handler took and the FIRST BYTES IT
// ACTUALLY READ. The test payloads are byte[k] = tag*31 + (k>>12), so byte 0 is
// tag*31: the trace identifies which transfer's data (or none) is in the buffer
// the handler is about to copy from, which is what separates "the put landed
// somewhere else" from "the handler read the wrong place".
bool h2d_trace_enabled() {
    static const bool v = []() {
        const char *e = std::getenv("GVIRTUS_H2D_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

void h2d_trace(const char *who, const char *branch, const void *buf, bool buf_is_device,
               size_t count, const void *dst, const void *gpu_src, size_t gpu_src_size) {
    if (!h2d_trace_enabled()) return;
    unsigned char probe[8] = {0};
    if (buf != nullptr) {
        if (buf_is_device) {
            cudaMemcpy(probe, buf, sizeof(probe), cudaMemcpyDeviceToHost);
        } else {
            std::memcpy(probe, buf, sizeof(probe));
        }
    }
    std::fprintf(stderr,
                 "[H2DTRACE] %s branch=%s count=%zu dst=%p src=%p(%s) "
                 "gpu_src=%p gpu_sz=%zu first8=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                 who, branch, count, dst, buf, buf_is_device ? "dev" : "host",
                 gpu_src, gpu_src_size,
                 probe[0], probe[1], probe[2], probe[3],
                 probe[4], probe[5], probe[6], probe[7]);
    std::fflush(stderr);
}

void gvirtus_shadow_drain() {
    if (g_shadow_copy_pending && g_shadow_stream != nullptr) {
        cudaStreamSynchronize(g_shadow_stream);
        g_shadow_copy_pending = false;
    }
    // CORRECTNESS FIX (2026-07-25): the async H2D shadow copy was invisible here.
    //
    // The synchronous H2D handler issues its D2D on g_shadow_stream, flags
    // g_shadow_copy_pending, and this hook -- called from ReleaseFrame, i.e. BEFORE
    // the RX slot is released and SlotConsumed goes back -- waits for it. That is
    // what keeps the peer from peer-DMA'ing the next transfer over a shadow the copy
    // engine is still reading.
    //
    // The async handler used a different mechanism: tls_async_gpu_pending, drained by
    // Communicator::drain_device_if_async_pending(), which Process.cpp calls only on
    // the RESPONSE path. A fire-and-forget cudaMemcpyAsync sends no response, so for
    // a burst of them nothing drained at all: ReleaseFrame freed the slot, the client
    // reused it, and the next peer-DMA landed on the shadow mid-copy. The sync path
    // was fixed for exactly this hazard; the async path was left behind.
    if (g_shadow_async_pending) {
        cudaDeviceSynchronize();
        g_shadow_async_pending = false;
        gvirtus::communicators::tls_async_gpu_pending = false;
    }
}

struct ShadowDrainRegistrar {
    ShadowDrainRegistrar() {
        gvirtus::communicators::SetFrameDrainHook(&gvirtus_shadow_drain);
    }
};
ShadowDrainRegistrar g_shadow_drain_registrar;

}  // namespace

CUDA_ROUTINE_HANDLER(Memcpy) {
    /* cudaError_t cudaError_t cudaMemcpy(void *dst, const void *src,
        size_t count, cudaMemcpyKind kind) */
    void *dst = NULL;
    void *src = NULL;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
        size_t count = input_buffer->BackGet<size_t>();

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;
        std::shared_ptr<Buffer> out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyHostToDevice: {
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                // GPUDirect Variant B Step B4: if the peer routed the payload
                // into the slot's GPU shadow (peer-DMA via peermem), the
                // Buffer carries a GPU-resident tail pointer. Skip the host
                // AssignAll (which would read garbage from the host slot's
                // hole) and cudaMemcpy D2D from the GPU shadow directly —
                // saves one PCIe round-trip per H2D call.
                void *gpu_src = input_buffer->GetGpuPayload();
                size_t gpu_src_size = input_buffer->GetGpuPayloadSize();
                if (gpu_src != nullptr && gpu_src_size >= count) {
                    // CORRECTNESS FIX (2026-07-25): cudaMemcpy D2D is ASYNC wrt the CPU
                    // thread (it enqueues on a stream and returns). The old code let the
                    // handler return -> ReleaseFrame -> SlotConsumed -> the client reused
                    // the shadow slot and the NEXT peer-DMA overwrote it WHILE the copy
                    // engine was still reading it -> dst got mixed old/new bytes -> a
                    // fraction of clients computed on corrupt poses (multi-tenant only).
                    // Issue the D2D on a per-connection stream and WAIT for completion
                    // before returning, so the shadow is owned until the copy truly ends.
                    // Per-connection stream avoids serializing other tenants kernels.
                    // PERF (2026-07-25): the stream MUST be non-blocking. A stream from
                    // cudaStreamCreate() implicitly barriers against the legacy default
                    // stream in BOTH directions, and this backend runs every tenants
                    // kernels on the legacy stream (one shared context, one thread per
                    // connection). A blocking stream therefore made each GPUDirect
                    // consume (a) wait for all pending work of all tenants and
                    // (b) stall every subsequent launch until the copy drained --
                    // a tax the host-staged path never pays, which is why GPUDirect
                    // measured ~0.5% SLOWER than staged RDMA. cudaStreamNonBlocking
                    // removes both barriers; ordering after the callers previously
                    // issued work is preserved explicitly with an event, and the
                    // cudaStreamSynchronize below still owns the shadow slot until
                    // the copy truly completes (the correctness fix above).
                    thread_local cudaEvent_t _shadow_prior = nullptr;
                    // Create the stream and the event independently: the D2H handler
                    // below shares g_shadow_stream, so whichever path runs first would
                    // otherwise leave the other's event null -> cudaEventRecord(nullptr)
                    // fails silently and the ordering guarantee quietly disappears.
                    if (g_shadow_stream == nullptr) {
                        cudaStreamCreateWithFlags(&g_shadow_stream, cudaStreamNonBlocking);
                    }
                    if (_shadow_prior == nullptr) {
                        cudaEventCreateWithFlags(&_shadow_prior, cudaEventDisableTiming);
                    }
                    // Order the copy AFTER work already issued on the default stream
                    // (cudaMemcpy is defined to be ordered wrt previously issued work),
                    // without making later launches wait for us.
                    cudaEventRecord(_shadow_prior, 0);
                    cudaStreamWaitEvent(g_shadow_stream, _shadow_prior, 0);
                    h2d_trace("Memcpy", "gpu_shadow", gpu_src, /*device*/ true,
                              count, dst, gpu_src, gpu_src_size);
                    exit_code = cudaMemcpyAsync(dst, gpu_src, count,
                                                cudaMemcpyDeviceToDevice, g_shadow_stream);
                    // Do NOT wait here. Measured, this wait was 163-211 us of a 2.82 ms
                    // 64 MB transfer (6.4%) and the client was blocked behind all of it
                    // for no reason: the only way to observe dst is another request on
                    // this connection, which this same thread processes strictly later.
                    // Hand the wait to the transport, which runs it after the response
                    // has gone out and before it frees the slot (Communicator.h,
                    // gvirtus_shadow_drain below).
                    if (exit_code == cudaSuccess) g_shadow_copy_pending = true;
                    result = std::make_shared<Result>(exit_code);
                    break;
                }

                try {
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                h2d_trace("Memcpy", "host_slot", src, /*device*/ false,
                          count, dst, gpu_src, gpu_src_size);
                // This payload was device-destined but had to bounce through the host
                // slot: no GPU shadow on this connection. Report it so the transport can
                // decide the shadow is worth its device memory here (see
                // Communicator::NoteDeviceDestinedPayload). Process.cpp forwards it.
                gvirtus::communicators::tls_device_destined_bytes = count;
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = std::make_shared<Result>(exit_code);
                break;
            }
            case cudaMemcpyDeviceToHost: {
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                // GPUDirect fast path: cudaMemcpy D2D into a TLS GPU scratch,
                // attach the GPU pointer to the Result as a side-channel. The
                // UCX-AM response writer splits iov into [host_prefix][gpu_payload]
                // and WriteIovRma puts the GPU fragment directly via peer-DMA.
                // Saves the backend D2H copy through host pinned mem (~4.78 ms
                // warm at 64 MB on L40S/Gen4) and the redundant host buffer.
                //
                // 4 MB threshold (mirrors Variant B's threshold in WriteIovRma).
                // Below this size the per-call setup cost (TLS scratch alloc/check,
                // cudaMemcpy D2D launch overhead, dual-iov response orchestration
                // in Process.cpp) exceeds the savings from skipping the host
                // bounce. Empirically observed: at N=256 (256 KB) and N=512 (1 MB)
                // host_ms regressed from ~0.7/1.1 ms (pre-GPUDirect) to ~1.5/1.9 ms
                // with GPUDirect on. 4 MB matches Variant B and protects small RPCs.
                constexpr size_t kGpuDirectD2HThreshold = 4u * 1024u * 1024u;
                // Auto-detected: take the GPU-scratch path only when the client
                // is RMA-put-capable (its rkey unpacked), else fall through to
                // the host path below (no device fragment on the wire, so no
                // AM-path connection reset on native/CPU frontends).
                if (gvirtus_gpudirect_d2h_enabled() && count >= kGpuDirectD2HThreshold) {
                    void *gpu_scratch = get_tls_gpu_scratch(count);
                    if (gpu_scratch != nullptr) {
                        // CORRECTNESS FIX (2026-07-25) — the D2H twin of the H2D fix
                        // above, and the cause of the long-hunted "got == want - 31"
                        // corruption that five rounds attributed to the H2D/RMA slot
                        // path.
                        //
                        // cudaMemcpy D2D performs NO host-side synchronization. CUDA
                        // Runtime API, "API synchronization behavior", rule 4: "For
                        // transfers from device memory to device memory, no host-side
                        // synchronization is performed." It enqueues the copy and
                        // returns immediately.
                        //
                        // The handler then returned, Process.cpp registered gpu_scratch
                        // with ucp_mem_map and shipped the client a GET descriptor, and
                        // the client issued an RDMA READ straight out of gpu_scratch --
                        // while the copy engine was still filling it. get_tls_gpu_scratch
                        // hands back ONE reused buffer per thread, so the bytes the NIC
                        // served from the not-yet-overwritten head were precisely the
                        // PREVIOUS D2H's. Measured signature: got == want - 31 with the
                        // 31-per-transfer test pattern (i.e. transfer n-1, never n-2),
                        // a handful of samples, always inside the first ~64 KB -- the
                        // window where the 24 GB/s RDMA READ can outrun a D2D that was
                        // scheduled a few microseconds late, before the much faster
                        // copy overtakes it for the remaining 64 MB.
                        //
                        // Unlike the H2D case this CANNOT be deferred to the transport
                        // drain hook: there the shadow only has to survive until the
                        // slot is released, here the client reads the scratch the moment
                        // it sees the response. The copy must be complete BEFORE the
                        // response leaves this thread.
                        //
                        // Same non-blocking stream + prior-work event as the H2D path,
                        // so a tenant's copy is ordered after its own previously issued
                        // work without erecting a legacy-default-stream barrier across
                        // every other tenant on this backend.
                        thread_local cudaEvent_t _d2h_prior = nullptr;
                        if (g_shadow_stream == nullptr) {
                            cudaStreamCreateWithFlags(&g_shadow_stream,
                                                      cudaStreamNonBlocking);
                        }
                        if (_d2h_prior == nullptr) {
                            cudaEventCreateWithFlags(&_d2h_prior,
                                                     cudaEventDisableTiming);
                        }
                        cudaEventRecord(_d2h_prior, 0);
                        cudaStreamWaitEvent(g_shadow_stream, _d2h_prior, 0);
                        exit_code = cudaMemcpyAsync(gpu_scratch, src, count,
                                                    cudaMemcpyDeviceToDevice,
                                                    g_shadow_stream);
                        if (exit_code == cudaSuccess) {
                            exit_code = cudaStreamSynchronize(g_shadow_stream);
                        }
                        if (exit_code == cudaSuccess) {
                            try {
                                // Wire format matches Add<char>(p,n): [size_t count][bytes].
                                // We emit only the size_t prefix here; the count bytes
                                // come via Result::GetGpuPayload() in Process.cpp.
                                out = std::make_shared<Buffer>();
                                out->Add<size_t>(count);
                            } catch (const std::exception &e) {
                                cerr << e.what() << endl;
                                return std::make_shared<Result>(cudaErrorMemoryAllocation);
                            }
                            result = std::make_shared<Result>(exit_code, out);
                            result->SetGpuPayload(gpu_scratch, count);
                            break;
                        }
                        // cudaMemcpy D2D failed: fall through to host path.
                    }
                }

                // Legacy host path (also used when GPUDirect probe failed
                // or scratch alloc failed). Uses Fase 1+2 TLS pre-faulted
                // slot to avoid per-call new[]+page-faults, and a non-owning
                // Buffer view to skip Add<char>'s 64 MB memmove into mpBuffer.
                //
                // Wire layout matches Add<char>(p, n):
                //   slot[0..8)        = size_t prefix == count
                //   slot[8..8+count)  = data bytes
                // Buffer is constructed non-owning over slot for length
                // sizeof(size_t)+count; dtor will NOT free the slot.
                char *slot = get_tls_d2h_slot(count);
                if (slot == nullptr) {
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                *reinterpret_cast<size_t *>(slot) = count;
                exit_code = cudaMemcpy(slot + sizeof(size_t), src, count, kind);
                try {
                    out = std::make_shared<Buffer>(slot, sizeof(size_t) + count);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                result = std::make_shared<Result>(exit_code, out);
                break;
            }
            case cudaMemcpyDeviceToDevice:
                dst = input_buffer->GetFromMarshal<void *>();
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpy(dst, src, count, kind);
                // Third of the three directions (2026-07-25). cudaMemcpy D2D performs
                // no host-side synchronization, so this returned with the copy still
                // running and the RPC then told the client the call was complete.
                //
                // Today that survives only by accident: the two places that read
                // device memory on a GVirtuS-private stream (the H2D shadow copy and
                // the D2H GET scratch copy) each record an event on the legacy default
                // stream first, so they happen to be ordered after this copy. Any
                // future internal stream that forgets that event would silently race.
                // Make the invariant hold by construction instead: when a synchronous
                // RPC returns, its device work is done. The client is blocked on this
                // reply anyway, so the wait costs it nothing it was not already paying.
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(0);
                    if (sync_err != cudaSuccess) exit_code = sync_err;
                }
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
//

CUDA_ROUTINE_HANDLER(Memcpy2DFromArray) {
    void *dst = NULL;
    cudaArray *src = NULL;
    size_t dpitch;
    size_t height;
    size_t width;
    size_t wOffset, hOffset;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;
        std::shared_ptr<Buffer> out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
            case cudaMemcpyHostToDevice:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();  // fittizio
                    src = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                dst = new char[dpitch * height];
                exit_code =
                    cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
                try {
                    out = std::make_shared<Buffer>();
                    out->Add<char>((char *)dst, dpitch * height);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                delete[] (char *)dst;
                result = std::make_shared<Result>(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code =
                    cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memcpy2DToArray) {
    void *src = NULL;
    cudaArray *dst = NULL;
    size_t spitch = 0;
    size_t height = 0;
    size_t width = 0;
    size_t wOffset = 0;
    size_t hOffset = 0;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
            case cudaMemcpyDeviceToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    dst = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    height = input_buffer->BackGet<size_t>();
                    width = input_buffer->BackGet<size_t>();
                    spitch = input_buffer->BackGet<size_t>();
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    src = input_buffer->GetFromMarshal<void *>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
        }
        exit_code = cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memcpy3D) {
    void *src = NULL;

    try {
        cudaMemcpy3DParms *p = input_buffer->Assign<cudaMemcpy3DParms>();
        src = input_buffer->AssignAll<char>();
        unsigned int width = p->extent.width;
        p->srcPtr.ptr = src;

        cudaError_t exit_code = cudaMemcpy3D(p);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(p, 1);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memcpy2D) {
    void *dst = NULL;
    void *src = NULL;
    size_t dpitch;
    size_t spitch;
    size_t height;
    size_t width;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;
        std::shared_ptr<Buffer> out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyHostToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->AssignAll<char>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
                result = std::make_shared<Result>(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                dst = new char[dpitch * height];
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
                try {
                    out = std::make_shared<Buffer>();
                    out->Add<char>((char *)dst, dpitch * height);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                delete[] (char *)dst;
                result = std::make_shared<Result>(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
                // Same rule-4 hazard as the 1-D D2D above: cudaMemcpy2D D2D does not
                // synchronize with the host either.
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(0);
                    if (sync_err != cudaSuccess) exit_code = sync_err;
                }
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyAsync) {
    void *dst = NULL;
    void *src = NULL;

    try {
        cudaStream_t stream = input_buffer->BackGet<cudaStream_t>();
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
        size_t count = input_buffer->BackGet<size_t>();

        cudaError_t exit_code = cudaSuccess;
        std::shared_ptr<Buffer> out;
        std::shared_ptr<Result> result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                /*
                 * Original behavior preserved.
                 * In GVirtuS this path does not need a real backend CUDA copy.
                 */
                result = std::make_shared<Result>(cudaSuccess);
                break;

            case cudaMemcpyHostToDevice: {
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                // GPUDirect Step B4 (async peer-DMA): if the peer routed the
                // payload into the slot's GPU shadow (peer-DMA via peermem), copy
                // D2D straight from it — no host staging (mirrors the synchronous
                // Memcpy handler). This is the real peer-DMA path for
                // cudaMemcpyAsync H2D.
                void *gpu_src = input_buffer->GetGpuPayload();
                size_t gpu_src_size = input_buffer->GetGpuPayloadSize();
                if (gpu_src != nullptr && gpu_src_size >= count) {
                    h2d_trace("MemcpyAsync", "gpu_shadow", gpu_src, /*device*/ true,
                              count, dst, gpu_src, gpu_src_size);
                    exit_code = cudaMemcpyAsync(dst, gpu_src, count,
                                                cudaMemcpyDeviceToDevice, stream);
                    // Phase 3 (true async): do NOT synchronize here — consecutive
                    // fire-and-forget copies overlap. Flag that a GPU copy is in
                    // flight still reading its shadow slot; the backend drains the
                    // device (Communicator::drain_device_if_async_pending) before
                    // the next response-bearing reply, which is the only point at
                    // which the frontend's flow control may reuse a remote RMA
                    // slot. This keeps the GPU-shadow source alive for the copy's
                    // full lifetime without a per-copy stall.
                    if (exit_code == cudaSuccess) {
                        gvirtus::communicators::tls_async_gpu_pending = true;
                        // ALSO flag it for the frame drain hook. tls_async_gpu_pending
                        // alone is drained only before a response-bearing reply, and a
                        // fire-and-forget copy never produces one -- so without this the
                        // slot was released, acked, and reused while this D2D was still
                        // reading the shadow. See gvirtus_shadow_drain().
                        g_shadow_async_pending = true;
                    }
                    result = std::make_shared<Result>(exit_code);
                    break;
                }

                // Host fallback: GPUDirect off, small copy below the RMA
                // threshold, or a non-RDMA transport. Stage from the host slot.
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                "cudaMemcpyAsync HostToDevice: dst: "
                                    << dst << ", src: " << src << ", count: " << count
                                    << ", kind: " << kind << ", stream: " << stream);

                h2d_trace("MemcpyAsync", "host_slot", src, /*device*/ false,
                          count, dst, gpu_src, gpu_src_size);
                // Same signal as the synchronous handler: device-destined bytes staged
                // through the host slot for want of a GPU shadow.
                gvirtus::communicators::tls_device_destined_bytes = count;
                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);

                /*
                 * IMPORTANT:
                 * The source buffer belongs to the marshalled input buffer / UCX slot.
                 * If we return immediately, the slot may be released or reused while
                 * the GPU is still reading from it.
                 *
                 * This makes cudaMemcpyAsync effectively synchronous for now,
                 * but it validates whether the crash is caused by async lifetime issues.
                 */
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(stream);
                    if (sync_err != cudaSuccess) {
                        exit_code = sync_err;
                    }
                }

                result = std::make_shared<Result>(exit_code);
                break;
            }

            case cudaMemcpyDeviceToHost: {
                try {
                    input_buffer->Assign<char>();  // skip fake host ptr
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                // ASYNC D2H via client-GET (24 GB/s, symmetric with H2D) — mirrors
                // the synchronous Memcpy D2H handler. For a large copy on a
                // GPUDirect-capable client: D2D into a POOLED GPU scratch (several
                // async D2H may be in flight, each GET'd later by the client at its
                // stream sync — get_d2h_get_scratch + the frontend flow control cap
                // in-flight so a scratch is never overwritten pre-GET), synchronize
                // so the scratch holds the data, then hand it to the client via
                // SetGpuPayload: the deferred reply carries the GET descriptor and
                // the frontend's DrainPendingD2H issues the RDMA GET into dst.
                constexpr size_t kGpuDirectD2HThreshold = 4u * 1024u * 1024u;
                if (gvirtus_gpudirect_d2h_enabled() && count >= kGpuDirectD2HThreshold) {
                    void *gpu_scratch = get_d2h_get_scratch(count);
                    if (gpu_scratch != nullptr) {
                        exit_code = cudaMemcpyAsync(gpu_scratch, src, count,
                                                    cudaMemcpyDeviceToDevice, stream);
                        if (exit_code == cudaSuccess)
                            exit_code = cudaStreamSynchronize(stream);
                        if (exit_code == cudaSuccess) {
                            try {
                                out = std::make_shared<Buffer>();
                                out->Add<size_t>(count);
                            } catch (const std::exception &e) {
                                cerr << e.what() << endl;
                                return std::make_shared<Result>(cudaErrorMemoryAllocation);
                            }
                            result = std::make_shared<Result>(exit_code, out);
                            result->SetGpuPayload(gpu_scratch, count);
                            break;
                        }
                        // D2D failed → fall through to the host-staged path.
                    }
                }

                // Legacy host-staged path (GPUDirect off / small copy / D2D failed).
                dst = new char[count];
                LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                "cudaMemcpyAsync DeviceToHost: dst: "
                                    << dst << ", src: " << src << ", count: " << count
                                    << ", kind: " << kind << ", stream: " << stream);
                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(stream);
                    if (sync_err != cudaSuccess) {
                        exit_code = sync_err;
                    }
                }
                try {
                    out = std::make_shared<Buffer>();
                    if (exit_code == cudaSuccess) {
                        out->Add<char>((char *)dst, count);
                        result = std::make_shared<Result>(exit_code, out);
                    } else {
                        result = std::make_shared<Result>(exit_code);
                    }
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    delete[] (char *)dst;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                delete[] (char *)dst;
                break;
            }

            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                "cudaMemcpyAsync DeviceToDevice: dst: "
                                    << dst << ", src: " << src << ", count: " << count
                                    << ", kind: " << kind << ", stream: " << stream);

                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);

                /*
                 * Debug/stability mode: force completion before returning.
                 * Later this can be replaced with cudaEventRecord + pending operation tracking.
                 */
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(stream);
                    if (sync_err != cudaSuccess) {
                        exit_code = sync_err;
                    }
                }

                result = std::make_shared<Result>(exit_code);
                break;

            default:
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
        }

        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyFromSymbol) {
    try {
        void *dst = input_buffer->GetFromMarshal<void *>();
        char *handler = input_buffer->AssignString();
        char *symbol = input_buffer->AssignString();
        handler = (char *)CudaUtil::UnmarshalPointer(handler);
        size_t count = input_buffer->Get<size_t>();
        size_t offset = input_buffer->Get<size_t>();
        cudaMemcpyKind kind = input_buffer->Get<cudaMemcpyKind>();

        size_t size;

        if (cudaGetSymbolSize(&size, symbol) != cudaSuccess) {
            symbol = handler;
            cudaGetLastError();
        }

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;
        std::shared_ptr<Buffer> out = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyDeviceToHost:
                try {
                    out = std::make_shared<Buffer>(count);
                    dst = out->Delegate<char>(count);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
                result = std::make_shared<Result>(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyToArray) {
    void *src = NULL;

    try {
        cudaArray *dst = input_buffer->GetFromMarshal<cudaArray *>();
        size_t wOffset = input_buffer->Get<size_t>();
        size_t hOffset = input_buffer->Get<size_t>();
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
        size_t count = input_buffer->BackGet<size_t>();

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                /* Achtung: this isn't strictly correct because here we assign
                 * just a pointer to one character, any successive assign should
                 * take inaxpectated result ... but it works here!
                 */
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
                result = std::make_shared<Result>(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyDeviceToDevice:
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyFromArray) {
    /* cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) */

    void *dst = NULL;
    cudaArray *src = NULL;
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
    size_t count = input_buffer->BackGet<size_t>();
    size_t hOffset = input_buffer->BackGet<size_t>();
    size_t wOffset = input_buffer->BackGet<size_t>();

#ifdef DEBUG
    std::cout << "wOffset " << wOffset << " hOffset " << hOffset << std::endl;
#endif
    cudaError_t exit_code;
    std::shared_ptr<Result> result = NULL;
    std::shared_ptr<Buffer> out;

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
            // This should never happen
            result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
            break;

        case cudaMemcpyDeviceToHost:
            // FIXME: use buffer delegate
            dst = new char[count];
            /* skipping a char for fake host pointer */
            input_buffer->Assign<char>();  //???
            src = (cudaArray *)input_buffer->GetFromMarshal<void *>();

            exit_code = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
            out = std::make_shared<Buffer>();
            out->Add<char>((char *)dst, count);
            delete[] (char *)dst;
            result = std::make_shared<Result>(exit_code, out);
            break;

        case cudaMemcpyDeviceToDevice:
            dst = input_buffer->GetFromMarshal<void *>();
            src = (cudaArray *)input_buffer->GetFromMarshal<void *>();
            // src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
            result = std::make_shared<Result>(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyArrayToArray) {
    cudaArray *src = NULL;
    cudaArray *dst = NULL;
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
    size_t count;
    size_t hOffsetDst, hOffsetSrc;
    size_t wOffsetDst, wOffsetSrc;
    cudaError_t exit_code;
    std::shared_ptr<Result> result = NULL;

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
            // This should never happen
            result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
            break;

        case cudaMemcpyDeviceToDevice:
            dst = (cudaArray *)input_buffer->GetFromMarshal<void *>();
            wOffsetDst = input_buffer->Get<size_t>();
            hOffsetDst = input_buffer->Get<size_t>();
            src = (cudaArray *)input_buffer->GetFromMarshal<void *>();
            // src = input_buffer->GetFromMarshal<void *>();
            wOffsetSrc = input_buffer->Get<size_t>();
            hOffsetSrc = input_buffer->Get<size_t>();
            count = input_buffer->Get<size_t>();
            exit_code = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc,
                                               hOffsetSrc, count, kind);
            result = std::make_shared<Result>(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyToSymbol) {
    void *src = NULL;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind>();
        size_t offset = input_buffer->BackGet<size_t>();
        size_t count = input_buffer->BackGet<size_t>();
        char *handler = input_buffer->AssignString();
        char *symbol = input_buffer->AssignString();

        handler = (char *)CudaUtil::UnmarshalPointer(handler);
        size_t size;

        if (cudaGetSymbolSize(&size, symbol) != cudaSuccess) {
            symbol = handler;
            cudaGetLastError();
        }

        cudaError_t exit_code;
        std::shared_ptr<Result> result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
                result = std::make_shared<Result>(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // This should never happen
                result = std::make_shared<Result>(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyDeviceToDevice:
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
                result = std::make_shared<Result>(exit_code);
                break;
        }
        return result;
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memset) {
    try {
        void *devPtr = input_buffer->GetFromMarshal<void *>();
        int value = input_buffer->Get<int>();
        size_t count = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaMemset(devPtr, value, count);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memset2D) {
    try {
        void *devPtr = input_buffer->GetFromMarshal<void *>();
        size_t pitch = input_buffer->Get<size_t>();
        int value = input_buffer->Get<int>();
        size_t width = input_buffer->Get<size_t>();
        size_t height = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaMemset2D(devPtr, pitch, value, width, height);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(HostRegister) {
    try {
        void *frontend_ptr = input_buffer->GetFromMarshal<void *>();
        size_t size = input_buffer->Get<size_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();

        void *backend_ptr = malloc(size);

        cudaError_t exit_code = cudaHostRegister(backend_ptr, size, flags);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->AddMarshal(backend_ptr);  // send the memory address of the backend pointer
        hostRegisteredMap[frontend_ptr] = backend_ptr;  // Store the mapping
        return std::make_shared<Result>(exit_code, out);

    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(HostUnregister) {
    void *frontend_ptr = input_buffer->GetFromMarshal<void *>();
    try {
        void *backend_ptr = hostRegisteredMap.at(frontend_ptr);
        cudaError_t exit_code = cudaHostUnregister(backend_ptr);
        free(backend_ptr);
        hostRegisteredMap.erase(frontend_ptr);
        return std::make_shared<Result>(exit_code);
    } catch (const std::out_of_range &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorHostMemoryNotRegistered);
    }
}

CUDA_ROUTINE_HANDLER(PointerGetAttributes) {
    /* cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) */
    try {
        const void *ptr = input_buffer->Get<const void *>();
        cudaPointerAttributes attrs = {};
        cudaError_t exit_code = cudaPointerGetAttributes(&attrs, ptr);
        std::shared_ptr<Buffer> output_buffer = std::make_shared<Buffer>();
        output_buffer->Add(&attrs);
        return std::make_shared<Result>(exit_code, output_buffer);
    } catch (const std::exception &e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "cudaPointerGetAttributes failed: " << e.what());
        return std::make_shared<Result>(cudaErrorInvalidValue);
    }
}