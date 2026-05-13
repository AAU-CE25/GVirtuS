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
                    exit_code = cudaMemcpy(dst, gpu_src, count,
                                           cudaMemcpyDeviceToDevice);
                    result = std::make_shared<Result>(exit_code);
                    break;
                }

                try {
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
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
                if (gvirtus_gpudirect_enabled() && count >= kGpuDirectD2HThreshold) {
                    void *gpu_scratch = get_tls_gpu_scratch(count);
                    if (gpu_scratch != nullptr) {
                        exit_code = cudaMemcpy(gpu_scratch, src, count,
                                               cudaMemcpyDeviceToDevice);
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

            case cudaMemcpyHostToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->AssignAll<char>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                "cudaMemcpyAsync HostToDevice: dst: "
                                    << dst << ", src: " << src << ", count: " << count
                                    << ", kind: " << kind << ", stream: " << stream);

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

            case cudaMemcpyDeviceToHost:
                /*
                 * Allocate temporary host output buffer.
                 * This buffer must NOT be copied into the Result or freed until
                 * cudaMemcpyAsync has actually completed.
                 */
                dst = new char[count];

                try {
                    /*
                     * Skipping a char for fake host pointer.
                     */
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    delete[] (char *)dst;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }

                LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                "cudaMemcpyAsync DeviceToHost: dst: "
                                    << dst << ", src: " << src << ", count: " << count
                                    << ", kind: " << kind << ", stream: " << stream);

                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);

                /*
                 * We must wait before:
                 *   1. serializing dst into the output Buffer
                 *   2. deleting dst
                 *
                 * Otherwise the GPU may still be writing into freed or stale memory.
                 */
                if (exit_code == cudaSuccess) {
                    cudaError_t sync_err = cudaStreamSynchronize(stream);
                    if (sync_err != cudaSuccess) {
                        exit_code = sync_err;
                    }
                }

                try {
                    out = std::make_shared<Buffer>();

                    /*
                     * Only add output payload if the CUDA copy completed correctly.
                     * If it failed, return just the CUDA error.
                     */
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