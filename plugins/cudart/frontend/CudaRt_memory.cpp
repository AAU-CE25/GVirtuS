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

#include <atomic>
#include "CudaRt.h"
#include "PtdsExplicit.h"
#include "CaptureMirror.h"

// Definida en CudaRt_graph.cpp. I12, punto de observacion IMPLICITO: una operacion sincrona en
// el stream legacy sincroniza los streams blocking, asi que las salidas D2H capturadas de un
// grafo lanzado en uno de ellos pasan a ser legalmente legibles aqui. Sale inmediatamente si
// no hay ningun lanzamiento con salidas pendientes, que es el caso normal.
void gvs_recoge_por_stream_legacy();
#include "CudaRt_lazyfatbin.h"
#include <dlfcn.h>

#include <cstdint>
#include <map>
#include <mutex>

#include "gvirtus/communicators/Communicator.h"

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

// Registration-cacheability hook for the transport (see Communicator.h).
//
// The UCX communicator caches the ucp_mem_h of a large H2D SOURCE buffer keyed by its
// address, because re-registering per put halves H2D bandwidth (23.4 -> 11.5 GB/s
// measured). That is only safe for a buffer whose free we will report, so answer true
// only for allocations this frontend tracks:
//
//   * device pointers  - cudaFree comes through us
//   * pinned host      - cudaFreeHost comes through us and untracks it
//
// A plain malloc'd host buffer answers false and is registered per call. glibc mmaps
// large allocations and readily returns the same address after a free, so caching one
// would recreate the stale-handle corruption fixed on the D2H side in a86b1ec.
static bool gvirtus_registration_cacheable(const void *addr, size_t len) {
    if (addr == nullptr || len == 0) return false;
    if (CudaRtFrontend::isDevicePointer(const_cast<void *>(addr))) return true;
    if (gvirtus_is_pinned(addr, len)) return true;
    // Host pageable: cacheable SOLO si el transporte instalo el hook de UCM y por tanto
    // se entera de los free que no pasan por nosotros. Sin el, seguimos respondiendo
    // false y se registra por llamada, que es lo correcto para una direccion cuya vida
    // no controlamos. Ver Communicator.h.
    return gvirtus::communicators::HostUnmapTrackingActive();
}

namespace {
struct RegCacheableRegistrar {
    RegCacheableRegistrar() {
        gvirtus::communicators::SetRegistrationCacheableHook(&gvirtus_registration_cacheable);
        // El mismo mapa de intervalos que ya se mantiene para la cacheabilidad, expuesto
        // ahora tambien a la politica de colocacion. Sin RPC: es una busqueda local.
        gvirtus::communicators::SetHostPinnedHook(
            [](const void *p, size_t n) { return gvirtus_is_pinned(p, n); });
    }
};
RegCacheableRegistrar g_reg_cacheable_registrar;
}  // namespace


/* Clasifica un puntero preguntando al Driver API. Devuelve true si el driver lo
 * reconoce, y deja en *is_host si es memoria de host. Resuelto por dlsym: sin
 * dependencia de enlace entre libcudart y libcuda. */
static bool gvs_pointer_is_host(const void *p, bool *is_host) {
    typedef int (*gvs_pga_t)(void *, int, unsigned long long);
    static gvs_pga_t pga =
        reinterpret_cast<gvs_pga_t>(dlsym(RTLD_DEFAULT, "cuPointerGetAttribute"));
    if (pga == nullptr || p == nullptr) return false;
    unsigned int memtype = 0;
    /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2, CU_MEMORYTYPE_HOST = 1 */
    if (pga(&memtype, 2, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p))) != 0)
        return false;
    *is_host = (memtype == 1u);
    return true;
}

static cudaMemcpyKind inferMemcpyKind_impl(void *dst, const void *src) {
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

        // Antes de conjeturar D2D, preguntar al driver. La memoria host de
        // cuMemHostAlloc (Driver API) no esta en NINGUN registro del runtime, y
        // asumirla device hacia que el backend recibiera un puntero host como
        // si fuera de dispositivo -> cudaErrorInvalidValue. Tumbaba 12 de las
        // 22 consultas de PDS-H (2026-07-28).
        bool dstHost = false, srcHost = false;
        const bool dstKnown = gvs_pointer_is_host(dst, &dstHost);
        const bool srcKnown = gvs_pointer_is_host(src, &srcHost);
        if (dstKnown && srcKnown) {
            if (dstHost && srcHost) return cudaMemcpyHostToHost;
            if (dstHost && !srcHost) return cudaMemcpyDeviceToHost;
            if (!dstHost && srcHost) return cudaMemcpyHostToDevice;
            return cudaMemcpyDeviceToDevice;
        }
        // Un solo extremo reconocido: ese manda, el otro es el contrario.
        // Faltaban las dos ramas de abajo (driver dice DEVICE), que son
        // precisamente las que aparecen en PDS-H.
        if (dstKnown && !srcKnown) {
            return dstHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        }
        if (srcKnown && !dstKnown) {
            return srcHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        }

        return cudaMemcpyDeviceToDevice;
    }
}


static bool gvs_memcpy_trace() {
    static const bool on = [] {
        const char *v = getenv("GVIRTUS_MEMCPY_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

static const char *gvs_kind_name(cudaMemcpyKind k) {
    switch (k) {
        case cudaMemcpyHostToHost:     return "H2H";
        case cudaMemcpyHostToDevice:   return "H2D";
        case cudaMemcpyDeviceToHost:   return "D2H";
        case cudaMemcpyDeviceToDevice: return "D2D";
        case cudaMemcpyDefault:        return "DEFAULT";
        default:                       return "?";
    }
}

cudaMemcpyKind inferMemcpyKind(void *dst, const void *src) {
    const cudaMemcpyKind k = inferMemcpyKind_impl(dst, src);
    if (gvs_memcpy_trace()) {
        bool dh = false, sh = false;
        const bool dk = gvs_pointer_is_host(dst, &dh);
        const bool sk = gvs_pointer_is_host(src, &sh);
        fprintf(stderr,
                "[GVS MEMCPY] dst=%p src=%p -> %s | registro: dst=%d src=%d | "
                "rango: dst=%d src=%d | driver: dst=%s src=%s\n",
                dst, src, gvs_kind_name(k),
                (int)CudaRtFrontend::isDevicePointer(dst),
                (int)CudaRtFrontend::isDevicePointer(src),
                (int)CudaRtFrontend::isInDeviceRange(dst),
                (int)CudaRtFrontend::isInDeviceRange(src),
                dk ? (dh ? "HOST" : "DEV") : "desconocido",
                sk ? (sh ? "HOST" : "DEV") : "desconocido");
        fflush(stderr);
    }
    return k;
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

    // Same reasoning as cudaFreeHost: a device source registration cached at >= 2 MB
    // must not outlive the allocation it describes.
    gvirtus::communicators::RunRegistrationInvalidate(devPtr);

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
    // Drop any transport registration covering this address BEFORE the memory goes
    // back to the allocator. Without this the next allocation can land on the same
    // address and inherit a handle describing the old mapping.
    gvirtus::communicators::RunRegistrationInvalidate(ptr);
    free(ptr);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    // Ruta de simbolo: no pasa por un lanzamiento, asi que no sabemos a que
    // fatbin toca. Se envian todos: conservador, pero nunca se consulta un
    // modulo que el backend no conoce.
    if (gvirtus_lazyfat::enabled()) gvirtus_lazyfat::flush_all();
    CudaRtFrontend::Prepare();
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments((char *)symbol);
    CudaRtFrontend::Execute("cudaGetSymbolAddress");
    if (CudaRtFrontend::Success())
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    // Ruta de simbolo: no pasa por un lanzamiento, asi que no sabemos a que
    // fatbin toca. Se envian todos: conservador, pero nunca se consulta un
    // modulo que el backend no conoce.
    if (gvirtus_lazyfat::enabled()) gvirtus_lazyfat::flush_all();
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
    // `isInDeviceRange` ANADIDO, y esta es la correccion: `isDevicePointer` es una busqueda por
    // direccion EXACTA, asi que un puntero con desplazamiento dentro de una asignacion propia
    // -- `base + offset`, que es lo que pasa ggml_backend_cuda_buffer_get_tensor -- daba false y
    // caia dentro de la puerta. El argumento de seguridad original ("tus D2H con src device
    // registrado nunca entran aqui") solo valia para la direccion base.
    // Reproducido deterministicamente en tests/semantic/d2hreclass.cu: con offset != 0 y dst en
    // la misma ventana de 4 GiB, un D2H se ejecutaba como D2D -> cudaErrorInvalidValue y 64 KiB
    // sin escribir. En produccion la coincidencia de ventana es azar del reparto de direcciones,
    // que es por lo que el fallo se veia intermitente.
    // El caso de RAPIDS se conserva: un puntero driver-API de CuPy no esta en ningun rango
    // registrado, luego `isInDeviceRange` es false y la puerta sigue disparando para el.
    if (kind == cudaMemcpyDeviceToHost &&
        !CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        uintptr_t _dst_hi = reinterpret_cast<uintptr_t>(dst) >> 32;
        uintptr_t _src_hi = reinterpret_cast<uintptr_t>(src) >> 32;
        // GVS_RECLASS_FORCE=1 trata la prueba de ventana como si SIEMPRE coincidiera. En
        // produccion esa coincidencia es azar del reparto de direcciones y aparece en ~1 de cada
        // 9 corridas, lo que obliga a cazar un evento raro para probar causalidad. Forzandola, el
        // par {legado, arreglado} se vuelve determinista sobre la carga real.
        static const bool fuerza = [] {
            const char *e = std::getenv("GVS_RECLASS_FORCE");
            return e != nullptr && e[0] == '1';
        }();
        if (fuerza || (_dst_hi == _src_hi && _dst_hi >= 0x7f00ULL)) {
            // Perilla de ABLACION: restaura el criterio anterior al arreglo. Existe para poder
            // exhibir el defecto a voluntad -- una vez arreglado, un reproductor que ya no
            // reproduce no sirve de prueba de nada. Por defecto APAGADA.
            static const bool legado = [] {
                const char *e = std::getenv("GVS_RECLASS_LEGACY");
                return e != nullptr && e[0] == '1';
            }();
            if (!legado && CudaRtFrontend::isInDeviceRange(src)) {
                // RAMA NUEVA = el arreglo. Y se CUENTA aparte a proposito: "cuantas veces habria
                // disparado la puerta con el criterio viejo" es la unica forma de decir con una
                // medida, y no con un argumento, si este cambio altera una carga concreta.
                // suppressed=0 en una carga significa que el arreglo no la toca.
                static std::atomic<unsigned long> suprimidos{0};
                const unsigned long k = suprimidos.fetch_add(1) + 1;
                bool imprime = false;
                for (unsigned long p = 1; p <= k; p *= 10) if (p == k) { imprime = true; break; }
                if (imprime)
                    fprintf(stderr, "[GVS RECLASS] suppressed=%lu: D2H kept as D2H because src=%p is "
                                    "inside a registered cudaMalloc range (dst=%p count=%zu). The old "
                                    "exact-address test would have reclassified this to D2D.\n",
                            k, const_cast<void *>(src), dst, count);
            } else {
                // Sigue siendo una HEURISTICA: "misma ventana de 4 GiB" es evidencia debil de que
                // dst tambien sea memoria de dispositivo. Observable para poder afirmar en que
                // cargas dispara, en vez de suponerlo.
                static std::atomic<unsigned long> avisos{0};
                const unsigned long k = avisos.fetch_add(1) + 1;
                bool imprime = false;
                for (unsigned long p = 1; p <= k; p *= 10) if (p == k) { imprime = true; break; }
                if (imprime)
                    fprintf(stderr, "[GVS RECLASS] fired=%lu: D2H reclassified as D2D (RAPIDS "
                                    "driver-API heuristic): dst=%p src=%p count=%zu\n",
                            k, dst, const_cast<void *>(src), count);
                kind = cudaMemcpyDeviceToDevice;
            }
        }
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
    cudaError_t rc = CudaRtFrontend::GetExitCode();
    // I12: cudaMemcpy es sincrona y va por el stream legacy, de modo que sincroniza los
    // streams blocking. Si un grafo lanzado en uno de ellos dejo salidas D2H capturadas, aqui
    // el host ya tiene derecho a leerlas.
    if (rc == cudaSuccess) gvs_recoge_por_stream_legacy();
    return rc;
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
    stream = gvs_ptds::traduce(stream);
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
    // `isInDeviceRange` ANADIDO, y esta es la correccion: `isDevicePointer` es una busqueda por
    // direccion EXACTA, asi que un puntero con desplazamiento dentro de una asignacion propia
    // -- `base + offset`, que es lo que pasa ggml_backend_cuda_buffer_get_tensor -- daba false y
    // caia dentro de la puerta. El argumento de seguridad original ("tus D2H con src device
    // registrado nunca entran aqui") solo valia para la direccion base.
    // Reproducido deterministicamente en tests/semantic/d2hreclass.cu: con offset != 0 y dst en
    // la misma ventana de 4 GiB, un D2H se ejecutaba como D2D -> cudaErrorInvalidValue y 64 KiB
    // sin escribir. En produccion la coincidencia de ventana es azar del reparto de direcciones,
    // que es por lo que el fallo se veia intermitente.
    // El caso de RAPIDS se conserva: un puntero driver-API de CuPy no esta en ningun rango
    // registrado, luego `isInDeviceRange` es false y la puerta sigue disparando para el.
    if (kind == cudaMemcpyDeviceToHost &&
        !CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        uintptr_t _dst_hi = reinterpret_cast<uintptr_t>(dst) >> 32;
        uintptr_t _src_hi = reinterpret_cast<uintptr_t>(src) >> 32;
        // GVS_RECLASS_FORCE=1 trata la prueba de ventana como si SIEMPRE coincidiera. En
        // produccion esa coincidencia es azar del reparto de direcciones y aparece en ~1 de cada
        // 9 corridas, lo que obliga a cazar un evento raro para probar causalidad. Forzandola, el
        // par {legado, arreglado} se vuelve determinista sobre la carga real.
        static const bool fuerza = [] {
            const char *e = std::getenv("GVS_RECLASS_FORCE");
            return e != nullptr && e[0] == '1';
        }();
        if (fuerza || (_dst_hi == _src_hi && _dst_hi >= 0x7f00ULL)) {
            // Perilla de ABLACION: restaura el criterio anterior al arreglo. Existe para poder
            // exhibir el defecto a voluntad -- una vez arreglado, un reproductor que ya no
            // reproduce no sirve de prueba de nada. Por defecto APAGADA.
            static const bool legado = [] {
                const char *e = std::getenv("GVS_RECLASS_LEGACY");
                return e != nullptr && e[0] == '1';
            }();
            if (!legado && CudaRtFrontend::isInDeviceRange(src)) {
                // RAMA NUEVA = el arreglo. Y se CUENTA aparte a proposito: "cuantas veces habria
                // disparado la puerta con el criterio viejo" es la unica forma de decir con una
                // medida, y no con un argumento, si este cambio altera una carga concreta.
                // suppressed=0 en una carga significa que el arreglo no la toca.
                static std::atomic<unsigned long> suprimidos{0};
                const unsigned long k = suprimidos.fetch_add(1) + 1;
                bool imprime = false;
                for (unsigned long p = 1; p <= k; p *= 10) if (p == k) { imprime = true; break; }
                if (imprime)
                    fprintf(stderr, "[GVS RECLASS] suppressed=%lu: D2H kept as D2H because src=%p is "
                                    "inside a registered cudaMalloc range (dst=%p count=%zu). The old "
                                    "exact-address test would have reclassified this to D2D.\n",
                            k, const_cast<void *>(src), dst, count);
            } else {
                // Sigue siendo una HEURISTICA: "misma ventana de 4 GiB" es evidencia debil de que
                // dst tambien sea memoria de dispositivo. Observable para poder afirmar en que
                // cargas dispara, en vez de suponerlo.
                static std::atomic<unsigned long> avisos{0};
                const unsigned long k = avisos.fetch_add(1) + 1;
                bool imprime = false;
                for (unsigned long p = 1; p <= k; p *= 10) if (p == k) { imprime = true; break; }
                if (imprime)
                    fprintf(stderr, "[GVS RECLASS] fired=%lu: D2H reclassified as D2D (RAPIDS "
                                    "driver-API heuristic): dst=%p src=%p count=%zu\n",
                            k, dst, const_cast<void *>(src), count);
                kind = cudaMemcpyDeviceToDevice;
            }
        }
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
            // Con una captura abierta esta copia NO se ejecuta: se graba. El nodo leera su
            // origen en cada lanzamiento, asi que hay que recordar (src, count) en orden para
            // refrescar el staging del backend antes de lanzar. Se anota DESPUES de enviar:
            // solo cuenta lo que el backend efectivamente registro.
            if (gvs_capmirror::capturando()) gvs_capmirror::anota_h2d(src, count);
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
            // Con una captura abierta esta copia se GRABA: el backend la dirige a un buffer
            // suyo y no adjunta datos a la respuesta, asi que aqui no hay nada que copiar.
            // Se anota (dst, count) para recogerlo tras sincronizar.
            if (gvs_capmirror::capturando()) {
                CudaRtFrontend::Execute("cudaMemcpyAsync");
                gvs_capmirror::anota_d2h(dst, count);
                break;
            }
            if (CudaRtFrontend::AsyncDispatchEnabled() && gvirtus_is_pinned(dst, count)) {
                // TRUE async: non-blocking. The backend delivers via the client-GET
                // (GPUDirect, 24 GB/s) — the GET is issued at the next stream sync
                // (DrainPendingD2H), not here, so the call returns immediately and
                // overlaps with subsequent stream work. Race-safety = the drain runs
                // at every synchronization point before the caller observes dst.
                CudaRtFrontend::ExecuteDeferredD2H("cudaMemcpyAsync", dst, count);
            } else {
                // Registrar el destino igual que la ruta sincrona. El backend
                // activa el camino client-GET segun el payload y la conexion, sin
                // saber si el cliente registro destino; sin esto, con GPUDirect
                // llegaba el descriptor y no habia donde depositarlo
                // ("D2H-GET response without output destination", q10 de PDS-H).
                CudaRtFrontend::SetOutputDestination(dst, count);
                CudaRtFrontend::Execute("cudaMemcpyAsync");
                if (CudaRtFrontend::Success() && !CudaRtFrontend::DirectOutputConsumed()) {
                    memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
                }
                CudaRtFrontend::ClearOutputDestination();
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
    cudaError_t rc = CudaRtFrontend::GetExitCode();
    if (rc == cudaSuccess) gvs_recoge_por_stream_legacy();
    return rc;
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
    stream = gvs_ptds::traduce(stream);
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
    return cudaMemcpyAsync(dst, src, count, kind, stream ? stream : cudaStreamPerThread);
}
extern "C" __host__ cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaMemsetAsync(devPtr, value, count, stream ? stream : cudaStreamPerThread);
}
