/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Memoria host del Driver API: cuMemHostAlloc, cuMemAllocHost, cuMemFreeHost y
 * cuMemHostGetFlags, con el registro que las respalda.
 *
 * Reconstruido el 2026-07-29; ver la cabecera de CudaDr_hostmem.h para la
 * procedencia. Los detalles que NO son elegibles --se leyeron del binario
 * superviviente-- van marcados abajo con "[del .o]".
 *
 * Estas cuatro funciones no hablan con el backend. Se comprobo en el objeto
 * original: sus unicos simbolos externos de asignacion son posix_memalign y
 * free, y no referencia a CudaDrFrontend en absoluto. Es lo correcto --el buffer
 * host vive en el cliente-- pero deja dos obligaciones que el registro cubre:
 * cuMemFreeHost debe distinguir un puntero suyo de un malloc ajeno, y
 * cuPointerGetAttribute debe poder describir el buffer sin preguntar a un
 * backend que no lo ha visto nunca.
 */

#include "CudaDr.h"
#include "CudaDr_hostmem.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

/* cuda.h renombra cuMemAllocHost -> cuMemAllocHost_v2 por macro. Hay que
 * deshacerlo para poder definir AMBOS deletreos con el mismo comportamiento.
 *
 * No es un detalle de estilo. Antes de la campana del 28 de julio esta API tenia
 * tres simbolos exportados con tres comportamientos distintos (plano =
 * NOT_SUPPORTED, _v2 = "not yet implemented", _v2_ptsz = NOT_SUPPORTED), y cual
 * veia el llamante dependia de que cadena pasara a dlsym. Importa porque KvikIO y
 * nvcomp resuelven el driver por dlopen + dlsym, no enlazando, y bajo la receta de
 * GVirtuS ese dlopen cae en esta biblioteca y no en el driver real. */
#undef cuMemAllocHost

namespace gvirtus_cudadr {
namespace {

/* Estaticos locales de funcion, no globales: los nombres mangled del objeto
 * original (registry_mu()::m, registry()::m) muestran que asi era, y ademas
 * evita depender del orden de inicializacion estatica entre unidades. */
std::mutex &registry_mu() {
    static std::mutex m;
    return m;
}

/* Clave = direccion base como entero. [del .o] el tipo de la clave se leyo del
 * nombre mangled de _Rb_tree<unsigned long, pair<const unsigned long, ...>>. */
std::map<unsigned long, DriverHostAllocation> &registry() {
    static std::map<unsigned long, DriverHostAllocation> m;
    return m;
}

bool trace_on() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_CUDADR_HOSTMEM_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

long trace_max() {
    static const long m = [] {
        const char *v = std::getenv("GVIRTUS_CUDADR_HOSTMEM_TRACE_MAX");
        return v != nullptr ? std::strtol(v, nullptr, 10) : 64L;
    }();
    return m;
}

std::atomic<long> &trace_count() {
    static std::atomic<long> c{0};
    return c;
}

/* Acotada por GVIRTUS_CUDADR_HOSTMEM_TRACE_MAX: son fprintf sincronos, y cuDF
 * hace suficientes asignaciones host como para que una traza sin limite cambie
 * el propio tiempo que se esta midiendo. */
__attribute__((format(printf, 1, 2))) void trace(const char *fmt, ...) {
    if (!trace_on()) return;
    if (trace_count().fetch_add(1) + 1 > trace_max()) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[GVS CUDADR HOSTMEM] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::fflush(stderr);
}

/* Los unicos flags que cuMemHostAlloc define: PORTABLE(1) | DEVICEMAP(2) |
 * WRITECOMBINED(4). [del .o] el original enmascaraba con 0xfffffff8 y saltaba a
 * error si quedaba algo, que es exactamente esta comprobacion. */
constexpr unsigned int kKnownHostAllocFlags = CU_MEMHOSTALLOC_PORTABLE |
                                              CU_MEMHOSTALLOC_DEVICEMAP |
                                              CU_MEMHOSTALLOC_WRITECOMBINED;

/* [del .o] alineacion 256, leida del `mov $0x100,%esi` que precede a la llamada
 * a posix_memalign. */
constexpr size_t kHostAllocAlignment = 256;

CUresult mem_host_alloc_impl(void **pp, size_t bytesize, unsigned int flags,
                             DriverHostAllocationOrigin origin) {
    if (pp == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *pp = nullptr;

    if ((flags & ~kKnownHostAllocFlags) != 0) {
        trace("alloc: flags desconocidos 0x%x -> INVALID_VALUE", flags);
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Medido contra el driver real: cuMemHostAlloc(&p, 0, 0) devuelve exito con
     * p == NULL. Es de las dos que habrian salido mal si se suponen en vez de
     * medirlas -- lo intuitivo seria INVALID_VALUE. */
    if (bytesize == 0) {
        trace("alloc: tamano 0 -> SUCCESS con p == NULL (semantica del driver real)");
        return CUDA_SUCCESS;
    }

    void *p = nullptr;
    if (posix_memalign(&p, kHostAllocAlignment, bytesize) != 0 || p == nullptr) {
        trace("alloc: posix_memalign(%zu) fallo -> OUT_OF_MEMORY", bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    DriverHostAllocation a;
    a.base  = reinterpret_cast<unsigned long>(p);
    a.size  = bytesize;
    /* Verbatim, sin anadir DEVICEMAP implicito. El driver real si lo anade (pides
     * 0x5 y lees 0x7) porque en una maquina sola la pagina host tiene de verdad un
     * alias de dispositivo. Aqui el buffer esta en otra maquina que la GPU, asi que
     * afirmarlo seria mentir al llamante. */
    a.flags  = (origin == kDriverHostAllocOriginMemAllocHost) ? 0u : flags;
    a.origin = origin;

    {
        std::lock_guard<std::mutex> lk(registry_mu());
        registry()[a.base] = a;
    }

    *pp = p;
    trace("alloc: %p tamano=%zu flags=0x%x origen=%d", p, bytesize, a.flags,
          static_cast<int>(origin));
    return CUDA_SUCCESS;
}

/* Devuelve true y saca la entrada del registro solo si addr es EXACTAMENTE una
 * base viva. Un puntero interior, un malloc ajeno o algo ya liberado dan false, que
 * es lo que hace que cuMemFreeHost responda INVALID_VALUE en esos tres casos --
 * medido contra el driver real. */
bool erase_host_allocation(const void *p, DriverHostAllocation *out) {
    const unsigned long addr = reinterpret_cast<unsigned long>(p);
    std::lock_guard<std::mutex> lk(registry_mu());
    auto it = registry().find(addr);
    if (it == registry().end()) return false;
    if (out != nullptr) *out = it->second;
    registry().erase(it);
    return true;
}

}  // namespace

bool find_host_allocation_exact(const void *p, DriverHostAllocation *out) {
    if (p == nullptr) return false;
    const unsigned long addr = reinterpret_cast<unsigned long>(p);

    std::lock_guard<std::mutex> lk(registry_mu());
    auto it = registry().find(addr);
    if (it == registry().end()) return false;
    if (out != nullptr) *out = it->second;
    return true;
}

bool find_host_allocation_containing(const void *p, DriverHostAllocation *out) {
    if (p == nullptr) return false;
    const unsigned long addr = reinterpret_cast<unsigned long>(p);

    std::lock_guard<std::mutex> lk(registry_mu());
    auto &reg = registry();
    /* La primera base ESTRICTAMENTE mayor que addr; la candidata es la anterior.
     * Es el unico predecesor posible porque las asignaciones no se solapan. */
    auto it = reg.upper_bound(addr);
    if (it == reg.begin()) return false;
    --it;
    if (addr - it->first >= it->second.size) return false;
    if (out != nullptr) *out = it->second;
    return true;
}

}  // namespace gvirtus_cudadr

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    return gvirtus_cudadr::mem_host_alloc_impl(
        pp, bytesize, Flags, gvirtus_cudadr::kDriverHostAllocOriginMemHostAlloc);
}

/* cuMemAllocHost equivale nativamente a cuMemHostAlloc con flags = 0. Se exportan
 * los dos deletreos porque cuda.h renombra por macro pero dlsym no: ver el #undef
 * de arriba. */
extern "C" CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    return gvirtus_cudadr::mem_host_alloc_impl(
        pp, bytesize, 0u, gvirtus_cudadr::kDriverHostAllocOriginMemAllocHost);
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    return gvirtus_cudadr::mem_host_alloc_impl(
        pp, bytesize, 0u, gvirtus_cudadr::kDriverHostAllocOriginMemAllocHost);
}

extern "C" CUresult cuMemFreeHost(void *p) {
    /* No-op documentado, y medido: el driver real devuelve exito. */
    if (p == nullptr) return CUDA_SUCCESS;

    gvirtus_cudadr::DriverHostAllocation a;
    if (!gvirtus_cudadr::erase_host_allocation(p, &a)) {
        gvirtus_cudadr::trace("free: %p no es una base viva -> INVALID_VALUE", p);
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::free(p);
    gvirtus_cudadr::trace("free: %p tamano=%zu", p, a.size);
    return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    if (pFlags == nullptr || p == nullptr) return CUDA_ERROR_INVALID_VALUE;

    gvirtus_cudadr::DriverHostAllocation a;
    /* Interior legal: medido, cuMemHostGetFlags(base + 64) devuelve exito en el
     * driver real. Por eso "containing" y no "exact". */
    if (!gvirtus_cudadr::find_host_allocation_containing(p, &a)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pFlags = a.flags;
    return CUDA_SUCCESS;
}
