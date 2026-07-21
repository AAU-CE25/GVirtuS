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
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CudaRt.h"

#include <cstdint>
#include <map>
#include <mutex>

using namespace std;
using gvirtus::common::mappedPointer;

// Phase 3 async dispatcher: registry of the frontend's pinned host allocations
// (cudaHostAlloc / cudaMallocHost). A deferred async D2H is only safe when its
// destination is one of these tracked buffers -- pageable dst must stay
// synchronous because real CUDA fills it synchronously (deferring would let a
// no-sync reader observe stale bytes). Ranges are [base, base+size).
//
// Function-local statics (Meyers singletons); allocated with new and never
// freed so they survive BOTH early static init (the communicator's init_rx_pool
// calls cudaHostAlloc during CudaRtFrontend's static constructor) AND static
// destruction at exit (the communicator's rx_pool teardown calls cudaFreeHost
// after libcudart's statics would otherwise be destroyed). Both fiascos crash a
// plain file-scope or plain local-static container; the tiny one-time leak here
// is the standard, safe fix.
namespace {
std::mutex &pinned_mu() {
    static std::mutex *m = new std::mutex();
    return *m;
}
std::map<uintptr_t, size_t> &pinned_map() {
    static std::map<uintptr_t, size_t> *m = new std::map<uintptr_t, size_t>();  // base -> size
    return *m;
}

void gvirtus_track_pinned(void *p, size_t size) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lk(pinned_mu());
    pinned_map()[reinterpret_cast<uintptr_t>(p)] = size;
}
void gvirtus_untrack_pinned(void *p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lk(pinned_mu());
    pinned_map().erase(reinterpret_cast<uintptr_t>(p));
}
// True iff [dst, dst+count) lies entirely within one tracked pinned allocation.
bool gvirtus_is_pinned(const void *dst, size_t count) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(dst);
    std::lock_guard<std::mutex> lk(pinned_mu());
    auto &m = pinned_map();
    auto it = m.upper_bound(a);
    if (it == m.begin()) return false;
    --it;  // greatest base <= a
    return a >= it->first && (a + count) <= (it->first + it->second);
}
}  // namespace

cudaMemcpyKind inferMemcpyKind(void *dst, const void *src) {
    if (CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToDevice;
    } else if (CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyHostToDevice;
    } else if (!CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToHost;
    } else {
        // Ambos desconocidos para el registry. Tus workloads reservan device con
        // cudaMalloc (registrado -> ramas de arriba), así que no llegan aquí con un
        // host-host Default. CuPy reserva device por driver-API (no registrado) ->
        // ambos extremos son device -> D2D, no HostToHost (que haría memmove de
        // direcciones device en espacio host -> SIGSEGV).
        bool dstDev = CudaRtFrontend::isInDeviceRange(dst);
        bool srcDev = CudaRtFrontend::isInDeviceRange(src);
        if (srcDev && !dstDev) return cudaMemcpyDeviceToHost;
        if (dstDev && !srcDev) return cudaMemcpyHostToDevice;
        return cudaMemcpyDeviceToDevice;
    }
}

cudaMemcpyKind inferMemcpyKindFromDevice(void *dst) {
    if (CudaRtFrontend::isDevicePointer(dst)) {
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyDeviceToHost;
}

cudaMemcpyKind inferMemcpyKindToDevice(const void *src) {
    if (CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyHostToDevice;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
    // cout << "cudaMemGetInfo called" << endl;
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaMemGetInfo");
    if (CudaRtFrontend::Success()) {
        *free = *CudaRtFrontend::GetOutputHostPointer<size_t>();
        *total = *CudaRtFrontend::GetOutputHostPointer<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    if (CudaRtFrontend::isMappedMemory(devPtr)) {
        void *hostPointer = devPtr;

#ifdef DEBUG
        cerr << "Mapped pointer detected" << endl;
#endif

        mappedPointer remotePointer = CudaRtFrontend::getMappedPointer(devPtr);

        free(devPtr);
        devPtr = remotePointer.pointer;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::Execute("cudaFree");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray *array) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments((void *)array);
    CudaRtFrontend::Execute("cudaFreeArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    gvirtus_untrack_pinned(ptr);
    free(ptr);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    CudaRtFrontend::Prepare();
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments((char *)symbol);
    CudaRtFrontend::Execute("cudaGetSymbolAddress");
    if (CudaRtFrontend::Success())
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(size);
    CudaRtFrontend::AddSymbolForArguments((char *)symbol);
    CudaRtFrontend::Execute("cudaGetSymbolSize");
    if (CudaRtFrontend::Success()) *size = *(CudaRtFrontend::GetOutputHostPointer<size_t>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size,
                                                        unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostAlloc\n");
#endif
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here. Real cudaHostAlloc returns memory aligned to at least 256
    // bytes; plain malloc() only guarantees 16 bytes on glibc, which breaks
    // callers that assert stronger alignment (e.g. ggml's TENSOR_ALIGNMENT=32,
    // causing intermittent crashes). Use a 256-byte aligned allocation.
    if (posix_memalign(ptr, 256, size ? size : 1) != 0) {
        *ptr = NULL;
        return cudaErrorMemoryAllocation;
    }
    gvirtus_track_pinned(*ptr, size);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                                   unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostGetDevicePointer\n");
#endif
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
#ifdef DEBUG
    printf("Requesting cudaHostGetFlags\n");
#endif
    // Achtung: falling back to the simplest method because we can't map memory
    *pFlags = cudaHostAllocDefault;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(size);
    // cout << "cudaMalloc frontend size: " << size << endl;
    CudaRtFrontend::Execute("cudaMalloc");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        // cout << "cudaMalloc frontend devPtr: " << *devPtr << endl;
        CudaRtFrontend::addDevicePointer(*devPtr);
        CudaRtFrontend::addDeviceRange(*devPtr, size);
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
                                                       cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                                  const cudaChannelFormatDesc *desc,
                                                  cudaExtent extent, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(extent);
    CudaRtFrontend::AddVariableForArguments(flags);

    CudaRtFrontend::Execute("cudaMalloc3DArray");
    if (CudaRtFrontend::Success()) *array = *(CudaRtFrontend::GetOutputHostPointer<cudaArray_t>());
    return CudaRtFrontend::GetExitCode();
}
// FIXME: new mapping way

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
                                                          const cudaChannelFormatDesc *desc,
                                                          size_t width, size_t height,
                                                          unsigned int flags) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);

    CudaRtFrontend::Execute("cudaMallocArray");
    if (CudaRtFrontend::Success())
        *arrayPtr = (cudaArray *)CudaRtFrontend::GetOutputDevicePointer();

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here. Match real CUDA's >=256-byte alignment (plain malloc only
    // gives 16 on glibc, breaking ggml's TENSOR_ALIGNMENT=32 assert intermittently).
    if (posix_memalign(ptr, 256, size ? size : 1) != 0) {
        *ptr = NULL;
        return cudaErrorMemoryAllocation;
    }
    gvirtus_track_pinned(*ptr, size);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                                          size_t width, size_t height) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(*pitch);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMallocPitch");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        *pitch = CudaRtFrontend::GetOutputVariable<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                                              const void *src, int srcDevice,
                                                              size_t count, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(dst);
    CudaRtFrontend::AddVariableForArguments(dstDevice);
    CudaRtFrontend::AddDevicePointerForArguments(src);
    CudaRtFrontend::AddVariableForArguments(srcDevice);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::AddDevicePointerForArguments(stream);

    // Stream-ordered device-to-device peer copy carrying only pointers (no bulk
    // payload, no return data) -> fire-and-forget under async dispatch.
    CudaRtFrontend::ExecuteMaybeAsync("cudaMemcpyPeerAsync");

    //    if (CudaRtFrontend::Success()) {
    //        // *dst = CudaRtFrontend::GetOutputDevicePointer();
    //        // *src = CudaRtFrontend::GetOutputDevicePointer();
    //    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ CUDARTAPI cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                                            unsigned flags) {
    *devPtr = malloc(size);

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaMallocManaged");

    if (CudaRtFrontend::Success()) {
        void *remotePointer = CudaRtFrontend::GetOutputDevicePointer();

        mappedPointer host;
        host.pointer = remotePointer;
        host.size = size;

#ifdef DEBUG
        cerr << "device: " << std::hex << hp << " host: " << *devPtr << endl;
#endif
        CudaRtFrontend::addMappedPointer(*devPtr, host);
    } else {
        free(*devPtr);
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const cudaMemcpy3DParms *p) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(p);

    unsigned int width = p->extent.width;
    unsigned int num_faces = p->extent.depth;
    unsigned int num_layers = 1;
    unsigned int cubemap_size = width * width * num_faces;
    unsigned int size = cubemap_size * num_layers * (p->srcPtr.pitch / p->extent.width);

    CudaRtFrontend::AddHostPointerForArguments<char>(
        static_cast<char *>(const_cast<void *>(p->srcPtr.ptr)), size);
    CudaRtFrontend::Execute("cudaMemcpy3D");
    if (CudaRtFrontend::Success()) {
        return CudaRtFrontend::GetExitCode();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
                                                     cudaMemcpyKind kind) {
    if (count == 0) return cudaSuccess;
    if (dst == nullptr || src == nullptr) {
        cerr << "[GVirtuS WARN] cudaMemcpy: NULL pointer (dst=" << dst
             << ", src=" << src << ", count=" << count << ", kind=" << kind << ")" << endl;
        return cudaErrorInvalidValue;
    }
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }
    // RAPIDS interop GATEADO: reclasifica D2H->D2D SOLO si NINGUNO es device registrado
    // (= punteros driver-API de CuPy) y comparten arena. Tus D2H (src device registrado)
    // NUNCA entran aquí -> Fase 4 intacta.
    if (kind == cudaMemcpyDeviceToHost &&
        !CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        uintptr_t _dst_hi = reinterpret_cast<uintptr_t>(dst) >> 32;
        uintptr_t _src_hi = reinterpret_cast<uintptr_t>(src) >> 32;
        if (_dst_hi == _src_hi && _dst_hi >= 0x7f00ULL) kind = cudaMemcpyDeviceToDevice;
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            if (memmove(dst, src, count) == NULL) return cudaErrorInvalidValue;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Fase 5
            CudaRtFrontend::AddHostPointerForArgumentsDirect<char>(
                static_cast<const char *>(src), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            // Fase 4
            CudaRtFrontend::SetOutputDestination(dst, count);
            CudaRtFrontend::Execute("cudaMemcpy");
            if (CudaRtFrontend::Success() && !CudaRtFrontend::DirectOutputConsumed()) {
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            }
            CudaRtFrontend::ClearOutputDestination();
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                                       size_t spitch, size_t width, size_t height,
                                                       cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    char *dst_bytes = static_cast<char *>(dst);
    const char *src_bytes = static_cast<const char *>(src);

    switch (kind) {
        case cudaMemcpyHostToHost:
            if (dpitch < width) return cudaErrorInvalidValue;

            if (dpitch == spitch) {
                if (memcpy(dst_bytes, src_bytes, spitch * height) == NULL)
                    return cudaErrorInvalidValue;
            } else {
                for (int i = 0; i < height; i++) {
                    if (memcpy(dst_bytes + (dpitch * i), src_bytes + (spitch * i), width) == NULL)
                        return cudaErrorInvalidValue;
                }
            }
            return cudaSuccess;
        case cudaMemcpyHostToDevice:
            // Use original pointers or cast again if needed for frontend calls
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(const_cast<char *>(src_bytes),
                                                             spitch * height);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
        case cudaMemcpyDeviceToHost:
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            if (CudaRtFrontend::Success())
                memmove(dst_bytes, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                        dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
    size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                                            const void *src, size_t spitch,
                                                            size_t width, size_t height,
                                                            cudaMemcpyKind kind,
                                                            cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                                                cudaArray_const_t src,
                                                                size_t wOffset, size_t hOffset,
                                                                size_t width, size_t height,
                                                                cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
            break;

        case cudaMemcpyDeviceToHost:
            // pass contenuto source
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                        dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void *dst, size_t dpitch, const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
                                                              size_t hOffset, const void *src,
                                                              size_t spitch, size_t width,
                                                              size_t height, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;

        case cudaMemcpyHostToDevice:
            // pass contenuto source
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), spitch * height);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            break;
    }
    CudaRtFrontend::Execute("cudaMemcpy2DToArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
                                                            cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                          cudaMemcpyKind kind,
                                                          cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (dst == nullptr || src == nullptr) {
        cerr << "[GVirtuS WARN] cudaMemcpyAsync: NULL pointer (dst=" << dst
             << ", src=" << src << ", count=" << count << ", kind=" << kind << ")" << endl;
        return cudaErrorInvalidValue;
    }
    // Async dispatch: H2D and D2D carry no return data. Small copies
    // fire-and-forget on the AM path; large H2D uses the RMA path with the
    // frontend's slot flow control (Phase 2). D2H still needs deferred
    // completion (Phase 3) and stays synchronous here.
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }
    // Mismo gate que cudaMemcpy: no toca los D2H de tus workloads (src device registrado).
    if (kind == cudaMemcpyDeviceToHost &&
        !CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        uintptr_t _dst_hi = reinterpret_cast<uintptr_t>(dst) >> 32;
        uintptr_t _src_hi = reinterpret_cast<uintptr_t>(src) >> 32;
        if (_dst_hi == _src_hi && _dst_hi >= 0x7f00ULL) kind = cudaMemcpyDeviceToDevice;
    }

    CudaRtFrontend::Prepare();
    // cout << "cudaMemcpyAsync frontend: "
    //      << "dst: " << dst << ", src: " << src << ", count: " << count
    //      << ", kind: " << kind << ", stream: " << stream << endl;

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (memmove(dst, src, count) == NULL) {
                return cudaErrorInvalidValue;
            }
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            // cout << "cudaMemcpyAsync HostToDevice" << endl;
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Fase 5 (peer-DMA): splice the user src straight into the WriteIov
            // iov instead of staging a full copy into mpInputBuffer. Mirrors the
            // synchronous cudaMemcpy H2D. With GPUDirect the big fragment is
            // peer-DMA'd into the backend GPU shadow slot (no host bounce), and
            // the backend copies D2D from there. Caller must not mutate src until
            // Execute returns — guaranteed here (fire-and-forget waits for local
            // send completion, i.e. the RDMA read of src has drained).
            CudaRtFrontend::AddHostPointerForArgumentsDirect<char>(
                static_cast<const char *>(src), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            // H2D returns no data. Small copies fire-and-forget on the AM path;
            // large copies take the RMA path where the frontend's slot flow
            // control keeps them async while a free remote slot exists and
            // transparently demotes to synchronous (draining the ring) otherwise.
            CudaRtFrontend::ExecuteMaybeAsync("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            // cout << "cudaMemcpyAsync DeviceToHost" << endl;
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            // D2H returns data. When the async gate is on AND dst is one of our
            // tracked pinned buffers, defer: the frontend writes dst at the next
            // sync point (Phase 3). Otherwise (gate off, or pageable dst) copy
            // synchronously as before.
            if (CudaRtFrontend::AsyncDispatchEnabled() && gvirtus_is_pinned(dst, count)) {
                CudaRtFrontend::ExecuteDeferredD2H("cudaMemcpyAsync", dst, count);
            } else {
                CudaRtFrontend::Execute("cudaMemcpyAsync");
                if (CudaRtFrontend::Success()) {
                    memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
                }
            }
            break;
        case cudaMemcpyDeviceToDevice:
            // cout << "cudaMemcpyAsync DeviceToDevice" << endl;
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            // D2D carries only pointers (small AM payload) and returns no data,
            // so it is always safe to fire-and-forget when the gate is on.
            CudaRtFrontend::ExecuteMaybeAsync("cudaMemcpyAsync");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const cudaArray *src,
                                                              size_t wOffset, size_t hOffset,
                                                              size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
            // pass contenuto source
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src,
                       size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToDevice:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffsetDst);
            CudaRtFrontend::AddVariableForArguments(hOffsetDst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(hOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyArrayToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
                                                                   size_t wOffset, size_t hOffset,
                                                                   size_t count,
                                                                   cudaMemcpyKind kind,
                                                                   cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                                               size_t count, size_t offset,
                                                               cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKindFromDevice(dst);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
            // Achtung: adding a fake host pointer
            CudaRtFrontend::AddDevicePointerForArguments((void *)0x666);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                                                    size_t count, size_t offset,
                                                                    cudaMemcpyKind kind,
                                                                    cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
                                                            size_t hOffset, const void *src,
                                                            size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *)dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *)dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
                                                                 size_t hOffset, const void *src,
                                                                 size_t count, cudaMemcpyKind kind,
                                                                 cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src,
                                                             size_t count, size_t offset,
                                                             cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKindToDevice(src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol,
                                                                  const void *src, size_t count,
                                                                  size_t offset,
                                                                  cudaMemcpyKind kind,
                                                                  cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int c, size_t count) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
                                                            size_t width, size_t height,
                                                            cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(pitch);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMemset2DAsync");
    // TODO: backend side needs implementation
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int c, size_t count,
                                                          cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::ExecuteMaybeAsync("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value,
                                                       size_t width, size_t height) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(pitch);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMemset2D");
    // TO-DO si deve fare sul serio solo la parte di backend ovviamente
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
                                                       cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size,
                                                           unsigned int flags) {
    if (ptr == NULL || size == 0) {
        return cudaErrorInvalidValue;
    } else if (CudaRtFrontend::isMappedMemory(ptr)) {
        // Memory is already registered
        return cudaErrorHostMemoryAlreadyRegistered;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(ptr);  // send the memory address
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaHostRegister");
    if (CudaRtFrontend::Success()) {
        // backend_ptr tracking is done here and also in the backend
        void *backend_ptr = CudaRtFrontend::GetOutputDevicePointer();
        mappedPointer host;
        host.pointer = backend_ptr;
        host.size = size;
        CudaRtFrontend::addMappedPointer(ptr, host);
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) {
    if (!CudaRtFrontend::isMappedMemory(ptr)) {
        return cudaErrorHostMemoryNotRegistered;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(ptr);
    CudaRtFrontend::Execute("cudaHostUnregister");
    if (CudaRtFrontend::Success()) {
        CudaRtFrontend::removeMappedPointer(ptr);
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
    // GVirtuS short-circuit: ptr from a known device allocation (cudaMalloc / RMM pool).
    // The RPC handler below is unvalidated (// TODO) and crashes cupy's
    // UnownedMemory(device_id=-1) path (rmm_cupy_allocator) behind cuDF .values/.to_cupy.
    // isInDeviceRange is populated by cudaMalloc via addDeviceRange (b56a77f).
    if (attributes != nullptr && CudaRtFrontend::isInDeviceRange(ptr)) {
        memset(attributes, 0, sizeof(cudaPointerAttributes));
        attributes->type = cudaMemoryTypeDevice;
        attributes->device = 0;
        attributes->devicePointer = const_cast<void *>(ptr);
        attributes->hostPointer = nullptr;
        return cudaSuccess;
    }
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(ptr);
    CudaRtFrontend::Execute("cudaPointerGetAttributes");
    if (CudaRtFrontend::Success() && attributes != nullptr) {
        *attributes = *(CudaRtFrontend::GetOutputHostPointer<cudaPointerAttributes>());
    }
    return CudaRtFrontend::GetExitCode();
}

// GVirtuS PTDS forwarders (RAPIDS compiles libcugraph/raft with --default-stream per-thread).
extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
}
extern "C" __host__ cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaMemsetAsync(devPtr, value, count, stream);
}
