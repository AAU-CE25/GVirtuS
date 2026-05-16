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

#include <algorithm>
#include <cstdlib>

using namespace log4cplus;
using namespace std;

using gvirtus::common::mappedPointer;

namespace {
// GPUDirect status (Step 3 of GVIRTUS_GPUDIRECT rollout). The UCX communicator
// sets the env var GVIRTUS_GPUDIRECT_ACTIVE to "1" or "0" at backend startup
// based on the probe result. We read it once per process and cache. Going via
// env var (instead of an extern "C" symbol) avoids RTLD_GLOBAL coupling
// between libgvirtus-communicators-ucx.so and this dlopen'd plugin.
bool gvirtus_gpudirect_enabled() {
    static const bool active = []() {
        const char *v = std::getenv("GVIRTUS_GPUDIRECT_ACTIVE");
        return v != nullptr && v[0] == '1';
    }();
    return active;
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
                // or scratch alloc failed).
                // FIXME: use buffer delegate
                dst = new char[count];
                exit_code = cudaMemcpy(dst, src, count, kind);
                try {
                    out = std::make_shared<Buffer>();
                    out->Add<char>((char *)dst, count);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    delete[] (char *)dst;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                delete[] (char *)dst;
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

        cudaError_t exit_code;
        std::shared_ptr<Buffer> out;
        std::shared_ptr<Result> result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
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
                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);
                result = std::make_shared<Result>(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                dst = new char[count];
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);
                try {
                    LOG4CPLUS_DEBUG(Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS")),
                                    "cudaMemcpyAsync HostToDevice: dst: "
                                        << dst << ", src: " << src << ", count: " << count
                                        << ", kind: " << kind << ", stream: " << stream);
                    out = std::make_shared<Buffer>();
                    out->Add<char>((char *)dst, count);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>(cudaErrorMemoryAllocation);
                }
                delete[] (char *)dst;
                result = std::make_shared<Result>(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                dst = input_buffer->GetFromMarshal<void *>();
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);
                result = std::make_shared<Result>(exit_code);
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