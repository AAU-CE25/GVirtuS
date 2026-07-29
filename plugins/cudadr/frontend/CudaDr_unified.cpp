/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"
#include "CudaDr_hostmem.h"

using namespace std;

/** Width the driver writes for each attribute. 0 means "do not forward":
 *  either unknown to this build, or wider than the 8-byte buffer the backend
 *  handler writes into (P2P_TOKENS). */
static size_t gvs_pointer_attribute_size(CUpointer_attribute attribute) {
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT:                    return sizeof(CUcontext);
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:                return sizeof(unsigned int);
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:             return sizeof(CUdeviceptr);
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:               return sizeof(void *);
        case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:                return sizeof(int);
        case CU_POINTER_ATTRIBUTE_BUFFER_ID:                  return sizeof(unsigned long long);
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:                 return sizeof(int);
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:             return sizeof(int);
        case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE: return sizeof(int);
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:           return sizeof(CUdeviceptr);
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:                 return sizeof(size_t);
        case CU_POINTER_ATTRIBUTE_MAPPED:                     return sizeof(int);
        case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:       return sizeof(unsigned int);
        case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: return sizeof(int);
        case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:               return sizeof(unsigned int);
        case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:             return sizeof(CUmemoryPool);
        case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:               return sizeof(size_t);
        case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:          return sizeof(void *);
        case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:            return sizeof(unsigned long long);
        default:                                              return 0;
    }
}

/* Host memory allocated by cuMemHostAlloc exists ONLY in this process. The
 * backend has never seen the address and its own address space may well have
 * something unrelated at the same numeric value, so forwarding the query would
 * ask the wrong machine and get a confidently wrong answer. These pointers are
 * therefore answered from the local registry, and only unknown addresses - the
 * virtual device pointers - go to the backend as before.
 *
 * Which attributes are answered locally follows what the real driver does for a
 * cuMemHostAlloc'd buffer, measured on an L40S (native_ptr.c, 2026-07-28):
 *
 *     MEMORY_TYPE     SUCCESS, CU_MEMORYTYPE_HOST
 *     HOST_POINTER    SUCCESS, and for an interior address it returns THAT
 *                     address, not the base
 *     IS_MANAGED      SUCCESS, 0
 *     DEVICE_POINTER  SUCCESS natively, returning the host address itself
 *     DEVICE_ORDINAL  SUCCESS natively, 0
 *     CONTEXT         SUCCESS natively, the current context
 *
 * The last three are deliberately NOT replicated. Natively they are answers
 * about a unified address space where a host page really does have a device
 * alias. Here the buffer lives in a different machine from the GPU, so
 * returning the host address as a device pointer would hand the caller
 * something that is not a device address and would be used as one. Reporting
 * INVALID_VALUE makes the caller take the copy path, which works. Inventing a
 * plausible answer would make it take the mapped path, which cannot. Same
 * reasoning as cuMemHostGetDevicePointer, which keeps refusing.
 */
static CUresult gvs_host_pointer_attribute(void *data, CUpointer_attribute attribute,
                                           CUdeviceptr ptr,
                                           const gvirtus_cudadr::DriverHostAllocation &a,
                                           bool *handled) {
    *handled = true;
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
            unsigned int t = CU_MEMORYTYPE_HOST;
            memcpy(data, &t, sizeof(t));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
            void *h = reinterpret_cast<void *>(static_cast<uintptr_t>(ptr));
            memcpy(data, &h, sizeof(h));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
            int m = 0;
            memcpy(data, &m, sizeof(m));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: {
            CUdeviceptr base = static_cast<CUdeviceptr>(a.base);
            memcpy(data, &base, sizeof(base));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE: {
            size_t sz = a.size;
            memcpy(data, &sz, sizeof(sz));
            return CUDA_SUCCESS;
        }
        default:
            /* CONTEXT, DEVICE_ORDINAL, DEVICE_POINTER and the rest: see above. */
            return CUDA_ERROR_INVALID_VALUE;
    }
}

extern "C" CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
    if (data == nullptr) return CUDA_ERROR_INVALID_VALUE;

    gvirtus_cudadr::DriverHostAllocation a;
    if (gvirtus_cudadr::find_host_allocation_containing(
            reinterpret_cast<const void *>(static_cast<uintptr_t>(ptr)), &a)) {
        bool handled = false;
        const CUresult rc = gvs_host_pointer_attribute(data, attribute, ptr, a, &handled);
        if (handled) return rc;
    }

    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(attribute);
    CudaDrFrontend::AddVariableForArguments(ptr);

    CudaDrFrontend::Execute("cuPointerGetAttribute");

    if (CudaDrFrontend::Success()) {
        /* El backend siempre devuelve 8 bytes: su handler escribe en un
         * char[sizeof(void*)] y manda el buffer entero. Pero el destino del
         * caller solo mide la anchura del atributo, y para MEMORY_TYPE eso es
         * un CUmemorytype de 4 bytes -- que es exactamente lo que pasan cuDF y
         * nvcomp. Copiar 8 bytes ahi pisa 4 bytes contiguos.
         *
         * No es teorico: con este memcpy sin acotar, la misma configuracion dio
         * tres desenlaces distintos en tres ejecuciones identicas (cuelgue
         * girando, abort con core, y "corrupted double-linked list" dentro de
         * nvcomp), que es la firma de una escritura fuera de rango cuyo efecto
         * depende de donde caiga.
         *
         * Se copia solo lo que el atributo vale. Un atributo que la tabla no
         * conoce no se copia en absoluto: adivinar la anchura es precisamente
         * el fallo que se esta corrigiendo. */
        const size_t width = gvs_pointer_attribute_size(attribute);
        if (width == 0 || width > sizeof(void *)) return CUDA_ERROR_INVALID_VALUE;
        memcpy(data, CudaDrFrontend::GetOutputHostPointer<void>(), width);
    }

    return CudaDrFrontend::GetExitCode();
}
#include <cstring>

/* ---------------------------------------------------------------------------
 * cuPointerGetAttributes - the plural form.
 *
 * Why it is here. KvikIO ships inside libcudf and does not link the driver: it
 * resolves it with dlopen("libcuda.so.1") + dlsym. Under the GVirtuS run recipe
 * (LD_PRELOAD=libcuda.so.1, LD_LIBRARY_PATH=.../lib/frontend) that dlopen
 * returns THIS library rather than the real driver - verified with dladdr on
 * 2026-07-28, the symbol resolved to /opt/GVirtuS/lib/frontend/libcuda.so.1.
 * KvikIO calls the plural form to decide whether a pointer is host or device
 * memory. It was a generated stub returning CUDA_ERROR_NOT_SUPPORTED, KvikIO's
 * CUDA_DRIVER_TRY turned that into an exception, and every cuDF-Polars query
 * died before the frontend issued a single RPC (79 bytes on the wire for a
 * whole run).
 *
 * Implemented as a loop over the singular, which already forwards to the
 * backend, rather than as a new RPC opcode: callers ask for one or two
 * attributes, so one round trip each is an honest cost, and it needs no
 * wire-protocol change on a backend that is mid-campaign.
 *
 * Two hazards the singular gets away with and this must not:
 *
 *   - The singular memcpy's sizeof(void *) into the caller's buffer whatever
 *     the attribute is. KvikIO passes &memtype, a 4-byte CUmemorytype, so an
 *     8-byte write would corrupt its stack. Here the reply lands in a local
 *     8-byte scratch and only the attribute's own width reaches the caller.
 *   - P2P_TOKENS wants 16 bytes while the backend handler writes into an
 *     8-byte buffer, so it is refused instead of forwarded.
 *
 * Semantics follow the driver: the plural does NOT fail when an attribute is
 * inapplicable to the pointer; that entry is zeroed and the call still reports
 * success. That is precisely what makes host pointers work - the real driver
 * answers CUDA_ERROR_INVALID_VALUE for them, we zero the entry, and KvikIO
 * reads memtype == 0 as "host memory", which is the correct answer.
 * ------------------------------------------------------------------------- */


extern "C" CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                           CUpointer_attribute *attributes,
                                           void **data, CUdeviceptr ptr) {
    if (numAttributes == 0) return CUDA_SUCCESS;
    if (attributes == nullptr || data == nullptr) return CUDA_ERROR_INVALID_VALUE;

    for (unsigned int i = 0; i < numAttributes; ++i) {
        if (data[i] == nullptr) continue;

        const size_t width = gvs_pointer_attribute_size(attributes[i]);
        /* Unknown or oversized: leave the caller's entry exactly as it was.
         * Zeroing would be a write of a length we do not know. */
        if (width == 0 || width > sizeof(void *)) continue;

        unsigned char scratch[sizeof(void *)] = {0};
        if (cuPointerGetAttribute(scratch, attributes[i], ptr) == CUDA_SUCCESS) {
            memcpy(data[i], scratch, width);
        } else {
            /* Inapplicable attribute: the driver's contract for the plural form
             * is to keep going, not to fail the whole call. */
            memset(data[i], 0, width);
        }
    }

    return CUDA_SUCCESS;
}
