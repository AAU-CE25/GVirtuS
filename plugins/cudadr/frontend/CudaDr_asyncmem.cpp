/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Variantes asincronas de las copias del Driver API: cuMemcpyDtoHAsync y
 * cuMemcpyHtoDAsync, en sus tres deletreos cada una.
 *
 * Reconstruido el 2026-07-29; ver la cabecera de CudaDr_hostmem.h para la
 * procedencia.
 *
 * QUE HACEN REALMENTE, Y POR QUE IMPORTA PARA LAS MEDIDAS
 * ------------------------------------------------------
 * Son envoltorios finos sobre las copias SINCRONAS. No hay asincronia de extremo
 * a extremo: la peticion cruza la red, el backend copia, y los datos vuelven en la
 * respuesta. Esto se confirmo en el objeto original, cuyos unicos simbolos CUDA
 * externos son exactamente cuMemcpyDtoH_v2, cuMemcpyHtoD_v2 y cuStreamSynchronize.
 *
 * De ahi se sigue algo que corrige el encuadre de la campana de cuDF: anadir
 * cuMemcpyDtoHAsync NO podia mejorar nada por si solo, porque desemboca en el
 * mismo handler sincrono. Lo que decide el coste es el camino de cuMemcpyDtoH.
 * Por eso el trabajo de optimizacion va ahi y no aqui.
 *
 * Lo que si aportan: existir. cuDF consulta el driver por cuGetProcAddress y
 * descarta la ruta si el simbolo falta, asi que su ausencia cambiaba que codigo
 * tomaba cuDF, no lo rapido que corria.
 */

#include "CudaDr.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

/* cuda.h renombra por macro:
 *     cuMemcpyDtoHAsync -> __CUDA_API_PTSZ(cuMemcpyDtoHAsync_v2)
 *     cuMemcpyHtoDAsync -> __CUDA_API_PTSZ(cuMemcpyHtoDAsync_v2)
 * Se deshace para exportar los tres deletreos con comportamiento identico, por la
 * misma razon que en CudaDr_hostmem.cpp: KvikIO y nvcomp resuelven por dlsym y la
 * cadena que pasen decide que simbolo ven. */
#undef cuMemcpyDtoHAsync
#undef cuMemcpyHtoDAsync

/* OJO: hay que nombrar las _v2 EXPLICITAMENTE.
 *
 * Tras los #undef de arriba, escribir `cuMemcpyDtoH(...)` ya no se expande a
 * `cuMemcpyDtoH_v2`: enlazaria con el simbolo plano `cuMemcpyDtoH`, que lo define
 * CudaDr_compat_stubs.cpp y devuelve CUDA_ERROR_NOT_SUPPORTED. Es decir, toda
 * copia asincrona fallaria -- y lo haria en silencio para quien no comprueba el
 * codigo de retorno. Las implementaciones de verdad, las que hablan con el
 * backend, estan en CudaDr_memory.cpp y exportan el deletreo _v2. */
extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);

namespace {

bool trace_on() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_CUDADR_ASYNCMEM_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

__attribute__((format(printf, 1, 2))) void trace(const char *fmt, ...) {
    if (!trace_on()) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[GVS CUDADR ASYNCMEM] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::fflush(stderr);
}

/* El sincronizado va ANTES de la copia, no despues.
 *
 * El origen de un D2H suele producirlo un kernel que todavia esta en ese stream, y
 * el destino de un H2D puede estarlo leyendo una operacion anterior del mismo
 * stream. Como la copia que hacemos es bloqueante y sale por una ruta distinta
 * (una RPC al backend, que copia en SU contexto), no queda ordenada respecto al
 * stream del llamante por si sola: hay que esperar a que el stream se vacie
 * primero. Sincronizar despues dejaria leer datos que aun no se han escrito.
 *
 * No basta con confiar en el stream nulo: cuDF trabaja con streams
 * non-blocking / per-thread, que no se sincronizan implicitamente con el. */
CUresult drain_stream(CUstream hStream) {
    return cuStreamSynchronize(hStream);
}

CUresult memcpy_dtoh_async_impl(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                CUstream hStream) {
    trace("DtoH: %zu B stream=%p", ByteCount, static_cast<void *>(hStream));
    const CUresult sync_rc = drain_stream(hStream);
    if (sync_rc != CUDA_SUCCESS) return sync_rc;
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult memcpy_htod_async_impl(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                CUstream hStream) {
    trace("HtoD: %zu B stream=%p", ByteCount, static_cast<void *>(hStream));
    const CUresult sync_rc = drain_stream(hStream);
    if (sync_rc != CUDA_SUCCESS) return sync_rc;
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

}  // namespace

extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
    return memcpy_dtoh_async_impl(dstHost, srcDevice, ByteCount, hStream);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                         CUstream hStream) {
    return memcpy_dtoh_async_impl(dstHost, srcDevice, ByteCount, hStream);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2_ptsz(void *dstHost, CUdeviceptr srcDevice,
                                              size_t ByteCount, CUstream hStream) {
    return memcpy_dtoh_async_impl(dstHost, srcDevice, ByteCount, hStream);
}

extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
    return memcpy_htod_async_impl(dstDevice, srcHost, ByteCount, hStream);
}

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                                         size_t ByteCount, CUstream hStream) {
    return memcpy_htod_async_impl(dstDevice, srcHost, ByteCount, hStream);
}

extern "C" CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void *srcHost,
                                              size_t ByteCount, CUstream hStream) {
    return memcpy_htod_async_impl(dstDevice, srcHost, ByteCount, hStream);
}
