/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Registro de asignaciones de memoria host del Driver API.
 *
 * Reconstruido el 2026-07-29. El fichero original (2026-07-28) se perdio: nunca
 * llego a git --el stash "campana 28jul" solo capturo directorios de build-- y
 * fue borrado del arbol de trabajo. Se ha reconstruido a partir de:
 *   - el conjunto de simbolos de build_cudadr_hostmem/.../CudaDr_hostmem.cpp.o,
 *   - los nombres mangled, que fijan las firmas y revelan que el registro es un
 *     std::map<unsigned long, DriverHostAllocation> y que registry_mu/registry
 *     son estaticos locales de funcion,
 *   - los usos supervivientes en CudaDr_unified.cpp, que fijan los campos,
 *   - el desensamblado, que fija alineacion 256 y la validacion de flags,
 *   - la semantica MEDIDA contra el driver real documentada en la nota
 *     cudadr-hostmem-pointer-attrs.
 *
 * Por que existe este registro. La memoria host del Driver API se asigna en el
 * CLIENTE: no hay nada que hacer en el backend, porque el buffer vive en esta
 * maquina. Pero cuPointerGetAttribute(s) tiene que poder responder sobre ella
 * sin preguntar al backend --que no la conoce-- y cuMemFreeHost tiene que
 * distinguir un puntero suyo de un malloc cualquiera. De ahi el registro.
 */

#ifndef GVIRTUS_CUDADR_HOSTMEM_H
#define GVIRTUS_CUDADR_HOSTMEM_H

#include <cstddef>

namespace gvirtus_cudadr {

/* Que llamada creo la asignacion. Importa para cuMemHostGetFlags: una
 * asignacion hecha con cuMemAllocHost no tiene flags propios (nativamente
 * equivale a cuMemHostAlloc con flags = 0), mientras que una de cuMemHostAlloc
 * conserva los que pidio el llamante. */
enum DriverHostAllocationOrigin {
    kDriverHostAllocOriginMemHostAlloc = 0,
    kDriverHostAllocOriginMemAllocHost = 1
};

struct DriverHostAllocation {
    /* Entera, no void*: CudaDr_unified.cpp hace static_cast<CUdeviceptr>(a.base)
     * para responder RANGE_START_ADDR, y eso no se puede hacer desde un puntero. */
    unsigned long base;
    size_t size;
    unsigned int flags;
    DriverHostAllocationOrigin origin;
};

/* Coincidencia EXACTA con la direccion base. Es la que usa cuMemFreeHost: liberar
 * por un puntero interior es un error, igual que en el driver real. */
bool find_host_allocation_exact(const void *p, DriverHostAllocation *out);

/* La asignacion que CONTIENE p, sea p la base o un puntero interior. Es la que
 * usan cuMemHostGetFlags y cuPointerGetAttribute(s): preguntar por el interior de
 * un buffer es legal --medido contra el driver real, base + 64 devuelve exito. */
bool find_host_allocation_containing(const void *p, DriverHostAllocation *out);

}  // namespace gvirtus_cudadr

#endif  // GVIRTUS_CUDADR_HOSTMEM_H
