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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include "CudaDrHandler.h"

#include <gvirtus/communicators/Communicator.h>
#include <gvirtus/common/PathStats.h>


/* Contadores de ruta del driver API. Fichero propio para no pisar el del
 * runtime API. */
namespace {
constexpr const char *kDrStatsEnv = "GVIRTUS_PATHSTATS_DR";
inline void dr_count(gvirtus::common::pathstats::Path p, size_t n) {
    gvirtus::common::pathstats::count(kDrStatsEnv, p, n);
}
}  // namespace

using namespace std;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

/*Frees device memory.*/

// ---------------------------------------------------------------------------
// Slot TLS pre-tocado para el D2H del driver API.
//
// El handler original hacia `new char[ByteCount]` en CADA cuMemcpyDtoH. Con
// columnas de 62,5 MiB eso son 16.000 fallos de pagina por llamada, ocho veces
// por batch. El plugin cudart resolvio esto hace tiempo con un slot reutilizado
// y pre-tocado; aqui se replica la misma idea.
//
// El slot crece al doble para amortizar, se pre-toca pagina a pagina la primera
// vez, y vive lo que viva el hilo. Reserva sizeof(size_t) extra al principio
// porque el formato de cable es [size_t count][bytes] -- el mismo que emite
// Add<char>(p, n) -- y asi la cabecera y los datos son contiguos y se pueden
// entregar con un Buffer no propietario, sin copiar.
// ---------------------------------------------------------------------------
namespace {

thread_local char  *gvs_dtoh_slot     = nullptr;
thread_local size_t gvs_dtoh_slot_cap = 0;

thread_local bool gvs_dtoh_slot_pinned = false;

char *gvs_get_dtoh_slot(size_t bytes) {
    const size_t need = bytes + sizeof(size_t);
    if (gvs_dtoh_slot != nullptr && gvs_dtoh_slot_cap >= need) return gvs_dtoh_slot;

    if (gvs_dtoh_slot != nullptr) {
        if (gvs_dtoh_slot_pinned) cuMemFreeHost(gvs_dtoh_slot);
        else                      delete[] gvs_dtoh_slot;
        gvs_dtoh_slot = nullptr; gvs_dtoh_slot_cap = 0;
    }
    size_t cap = gvs_dtoh_slot_cap ? gvs_dtoh_slot_cap : (1u << 20);
    while (cap < need) cap *= 2;

    // Pinned primero. cuMemcpyDtoH hacia memoria paginable obliga al driver a
    // hacer staging interno por su propio bufer pinned: dos copias y ~6 GB/s.
    // Hacia pinned va por DMA directo, ~20 GB/s. Ademas la pinned es registrable
    // para RDMA, cosa que la paginable no es de forma estable.
    void *p = nullptr;
    if (cuMemHostAlloc(&p, cap, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS && p != nullptr) {
        gvs_dtoh_slot = static_cast<char *>(p);
        gvs_dtoh_slot_pinned = true;
        gvs_dtoh_slot_cap = cap;
        return gvs_dtoh_slot;   // ya viene residente y fijada: no hace falta pre-tocar
    }

    // Sin pinned: comportamiento anterior. Peor, pero correcto.
    gvs_dtoh_slot = new (std::nothrow) char[cap];
    if (gvs_dtoh_slot == nullptr) { gvs_dtoh_slot_cap = 0; return nullptr; }
    gvs_dtoh_slot_pinned = false;
    gvs_dtoh_slot_cap = cap;
    for (size_t i = 0; i < cap; i += 4096) gvs_dtoh_slot[i] = 0;
    return gvs_dtoh_slot;
}

// ---------------------------------------------------------------------------
// D2H por GET del cliente para el Driver API (2026-07-29).
//
// El slot pinned de arriba dejo el handler en ~2 ms de server_exec, pero la
// respuesta sigue cruzando la red como carga util del mensaje: 14 ms por columna
// de 62,5 MiB, que son 4.464 MiB/s sobre un enlace que da 24.000 medidos con OSU.
// Lo que sobra no es la copia, es el camino.
//
// En vez de meter los bytes en el Buffer, se copian D2D a un scratch de GPU y se
// adjuntan al Result como payload de dispositivo. Process.cpp registra ese scratch
// y manda al cliente un descriptor [count][addr][rkey]; el frontend hace un RDMA
// GET directo a su destino. Es exactamente lo que ya hace el plugin cudart en su
// handler sincrono de Memcpy D2H.
//
// POR QUE UN SOLO SCRATCH Y NO UN POOL. cuMemcpyDtoH es sincrona: el frontend se
// bloquea hasta completar el GET antes de poder emitir la siguiente. Como el
// scratch es thread_local y el hilo que atiende es el mismo, no puede haber dos
// GET vivos sobre el a la vez. El pool de cuatro que usa cudart es para la ruta
// DIFERIDA (ExecuteDeferredD2H), donde el frontend deja hasta cuatro sin drenar;
// esa ruta no se usa aqui.
// ---------------------------------------------------------------------------

// Umbral: por debajo, el viaje de ida y vuelta del registro RDMA cuesta mas que
// los bytes que ahorra. Es el mismo valor que usa cudart.
constexpr size_t kGpuDirectD2HThreshold = 4u * 1024u * 1024u;

bool gvs_dr_gpudirect_d2h_enabled() {
    static const bool process_active = []() {
        const char *v = std::getenv("GVIRTUS_GPUDIRECT_ACTIVE");
        return v != nullptr && v[0] == '1';
    }();
    static const bool d2h_disabled = []() {
        const char *e = std::getenv("GVIRTUS_GPUDIRECT_D2H");
        return e != nullptr && e[0] == '0';
    }();
    if (!process_active || d2h_disabled) return false;
    // Las dos banderas son por conexion y las pone Process.cpp. La segunda es la
    // que importa aqui: si el cliente no puede responder a un RDMA READ, adjuntar
    // un payload de dispositivo haria que el fragmento saliera por la via eager y
    // la conexion se reiniciaria. Un frontend nativo cae asi al camino de host,
    // que es correcto en todas partes.
    return gvirtus::communicators::tls_connection_supports_cuda &&
           gvirtus::communicators::tls_client_rma_put_capable;
}

// Diagnostico del camino rapido (GVIRTUS_CUDADR_D2H_TRACE=1). Existe porque los
// tres modos de fallo de aqui --scratch nulo, D2D fallido, sincronizacion
// fallida-- caen todos al camino de host sin ruido: desde fuera se ven igual que
// "la puerta esta cerrada", y las dos cosas se arreglan de forma distinta.
void gvs_dr_note_fastpath(const char *what, int rc) {
    static const bool on = []() {
        const char *v = std::getenv("GVIRTUS_CUDADR_D2H_TRACE");
        return v != nullptr && v[0] == '1';
    }();
    if (!on) return;
    static int n = 0;
    if (n++ >= 12) return;
    std::fprintf(stderr, "[GVS DR D2H] camino_rapido: %s rc=%d\n", what, rc);
    std::fflush(stderr);
}

thread_local CUdeviceptr gvs_dr_gpu_scratch = 0;
thread_local size_t      gvs_dr_gpu_scratch_cap = 0;

CUdeviceptr gvs_get_dr_gpu_scratch(size_t needed) {
    if (gvs_dr_gpu_scratch != 0 && gvs_dr_gpu_scratch_cap >= needed) {
        return gvs_dr_gpu_scratch;
    }
    if (gvs_dr_gpu_scratch != 0) {
        // Sin invalidacion cruzada de registros: el scratch es thread_local y las
        // conexiones son hilos de ESTE proceso, asi que soltar la direccion en
        // todos los contextos UCX destruiria el registro vivo de otro hilo que ya
        // hubiera recibido esta misma direccion de cuMemAlloc. PrepareGpuGet
        // re-mapea solo cuando el tamano crece, que es justo este caso.
        cuMemFree(gvs_dr_gpu_scratch);
        gvs_dr_gpu_scratch = 0;
        gvs_dr_gpu_scratch_cap = 0;
    }
    const size_t new_cap = std::max(needed, gvs_dr_gpu_scratch_cap * 2);
    if (cuMemAlloc(&gvs_dr_gpu_scratch, new_cap) != CUDA_SUCCESS) {
        gvs_dr_gpu_scratch = 0;
        gvs_dr_gpu_scratch_cap = 0;
        return 0;
    }
    gvs_dr_gpu_scratch_cap = new_cap;
    return gvs_dr_gpu_scratch;
}

}  // namespace

CUDA_DRIVER_HANDLER(MemFree) {
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr>();
    CUresult exit_code = cuMemFree(dptr);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

/*Allocates device memory.*/
CUDA_DRIVER_HANDLER(MemAlloc) {
    CUdeviceptr dptr = 0;
    size_t bytesize = input_buffer->Get<size_t>();
    CUresult exit_code = cuMemAlloc(&dptr, bytesize);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(dptr);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemRelease) {
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr>();
    CUresult exit_code = cuMemRelease(dptr);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

CUDA_DRIVER_HANDLER(MemAddressFree) {
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr>();
    size_t size = input_buffer->Get<size_t>();
    CUresult exit_code = cuMemAddressFree(dptr, size);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

CUDA_DRIVER_HANDLER(MemMap) {
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr>();
    size_t size = input_buffer->Get<size_t>();
    size_t offset = input_buffer->Get<size_t>();
    CUmemGenericAllocationHandle handle = input_buffer->Get<CUmemGenericAllocationHandle>();
    unsigned long long flags = input_buffer->Get<unsigned long long>();
    CUresult exit_code = cuMemMap(dptr, size, offset, handle, flags);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

/*Copies memory from Device to Host. */
CUDA_DRIVER_HANDLER(MemcpyDtoH) {
    CUdeviceptr srcDevice = input_buffer->Get<CUdeviceptr>();
    size_t ByteCount = input_buffer->Get<size_t>();

    // Diagnostico de la puerta, apagado por defecto (GVIRTUS_CUDADR_D2H_TRACE=1).
    // Sirve para distinguir "el GET no se toma" de "el GET se toma y no ayuda",
    // que desde fuera se parecen: en los dos casos el tiempo no baja.
    {
        static const bool trace_gate = []() {
            const char *v = std::getenv("GVIRTUS_CUDADR_D2H_TRACE");
            return v != nullptr && v[0] == '1';
        }();
        if (trace_gate) {
            static const bool env_active = []() {
                const char *v = std::getenv("GVIRTUS_GPUDIRECT_ACTIVE");
                return v != nullptr && v[0] == '1';
            }();
            static int n = 0;
            if (n++ < 8) {
                std::fprintf(stderr,
                             "[GVS DR D2H] bytes=%zu env_active=%d cuda=%d rma_put=%d "
                             "sobre_umbral=%d\n",
                             ByteCount, (int)env_active,
                             (int)gvirtus::communicators::tls_connection_supports_cuda,
                             (int)gvirtus::communicators::tls_client_rma_put_capable,
                             (int)(ByteCount >= kGpuDirectD2HThreshold));
                std::fflush(stderr);
            }
        }
    }

    // Camino rapido: D2D a un scratch de GPU y GET del cliente. Ver la nota larga
    // junto a gvs_get_dr_gpu_scratch. Si algo falla se cae al camino de host de
    // abajo, que es correcto en cualquier transporte.
    if (gvs_dr_gpudirect_d2h_enabled() && ByteCount >= kGpuDirectD2HThreshold) {
        CUdeviceptr scratch = gvs_get_dr_gpu_scratch(ByteCount);
        if (scratch == 0) gvs_dr_note_fastpath("scratch=0 (cuMemAlloc fallo)", 0);
        if (scratch != 0) {
            CUresult rc = cuMemcpyDtoD(scratch, srcDevice, ByteCount);
            if (rc != CUDA_SUCCESS) gvs_dr_note_fastpath("cuMemcpyDtoD", (int)rc);
            // HAY QUE SINCRONIZAR. cuMemcpyDtoD es asincrona respecto al host: solo
            // encola la copia. Sin esta espera, el handler devolveria, Process.cpp
            // registraria el scratch y le mandaria el descriptor al cliente, y el
            // cliente leeria por RDMA un bufer que el motor de copia todavia esta
            // llenando. No es hipotetico: es la causa raiz de la corrupcion "got ==
            // want - 31" que se atribuyo cinco veces al camino H2D/RMA antes de
            // encontrarse aqui, en el gemelo de cudart.
            if (rc == CUDA_SUCCESS) {
                rc = cuCtxSynchronize();
                if (rc != CUDA_SUCCESS) gvs_dr_note_fastpath("cuCtxSynchronize", (int)rc);
            }
            if (rc == CUDA_SUCCESS) {
                gvs_dr_note_fastpath("OK: payload de GPU adjuntado", 0);
                /* Se cuenta AQUI, no antes de la copia: solo cuando ya sabemos que el
                 * payload de GPU sale de verdad. Contarlo antes inflaria la ruta
                 * GPUDirect con intentos que acabaron cayendo al camino de host. */
                dr_count(gvirtus::common::pathstats::kD2hGpu, ByteCount);
                std::shared_ptr<Buffer> out;
                try {
                    out = std::make_shared<Buffer>();
                    out->Add<size_t>(ByteCount);
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return std::make_shared<Result>((cudaError_t)CUDA_ERROR_OUT_OF_MEMORY);
                }
                auto result = std::make_shared<Result>((cudaError_t)rc, out);
                result->SetGpuPayload(reinterpret_cast<void *>(scratch), ByteCount);
                return result;
            }
        }
    }

    // Slot reutilizado y pre-tocado en vez de `new char[ByteCount]`, y Buffer no
    // propietario sobre el en vez de Add<char>. Elimina, por columna de 62,5 MiB:
    // una asignacion con sus fallos de pagina y una copia host-a-host completa.
    // Layout de cable identico al de Add<char>(p, n):
    //   slot[0 .. 8)               = size_t con ByteCount
    //   slot[8 .. 8+ByteCount)     = datos
    /* Escala en host: estos bytes SI tocan memoria de host en el backend. Bajo
     * GPUDirect las columnas >=4 MiB de esta fila tienen que quedar a cero. */
    dr_count(gvirtus::common::pathstats::kD2hHost, ByteCount);
    char *slot = gvs_get_dtoh_slot(ByteCount);
    if (slot == nullptr) {
        return std::make_shared<Result>((cudaError_t)CUDA_ERROR_OUT_OF_MEMORY);
    }
    *reinterpret_cast<size_t *>(slot) = ByteCount;
    CUresult exit_code = cuMemcpyDtoH(slot + sizeof(size_t), srcDevice, ByteCount);

    std::shared_ptr<Buffer> out;
    try {
        out = std::make_shared<Buffer>(slot, sizeof(size_t) + ByteCount);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>((cudaError_t)CUDA_ERROR_OUT_OF_MEMORY);
    }
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Copies memory from Device to Host, enqueued on a stream.*/
CUDA_DRIVER_HANDLER(MemcpyDtoHAsync) {
    CUdeviceptr srcDevice = input_buffer->Get<CUdeviceptr>();
    size_t ByteCount = input_buffer->Get<size_t>();
    CUstream hStream = input_buffer->Get<CUstream>();

    dr_count(gvirtus::common::pathstats::kD2hHost, ByteCount);
    char *slot = gvs_get_dtoh_slot(ByteCount);
    if (slot == nullptr) {
        return std::make_shared<Result>((cudaError_t)CUDA_ERROR_OUT_OF_MEMORY);
    }
    *reinterpret_cast<size_t *>(slot) = ByteCount;

    CUresult exit_code = cuMemcpyDtoHAsync(slot + sizeof(size_t), srcDevice,
                                           ByteCount, hStream);
    // Hay que sincronizar antes de responder: los datos viajan en la respuesta,
    // asi que tienen que estar completos. La ganancia frente a la version
    // sincrona no es asincronia de extremo a extremo -- es que la copia entra en
    // el stream del llamante y puede solaparse con lo que ese stream ya tenga.
    if (exit_code == CUDA_SUCCESS) exit_code = cuStreamSynchronize(hStream);

    std::shared_ptr<Buffer> out;
    try {
        out = std::make_shared<Buffer>(slot, sizeof(size_t) + ByteCount);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>((cudaError_t)CUDA_ERROR_OUT_OF_MEMORY);
    }
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Copies memory from Host to Device.*/
CUDA_DRIVER_HANDLER(MemcpyHtoD) {
    void *srcHost = NULL;
    size_t ByteCount = input_buffer->Get<size_t>();
    CUdeviceptr dstDevice = input_buffer->Get<CUdeviceptr>();
    srcHost = input_buffer->Assign<char>(ByteCount);
    dr_count(gvirtus::common::pathstats::kH2dHost, ByteCount);
    CUresult exit_code = cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

/*Creates a 1D or 2D CUDA array. */
CUDA_DRIVER_HANDLER(ArrayCreate) {
    CUarray pHandle;
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray =
        input_buffer->Assign<const CUDA_ARRAY_DESCRIPTOR>();
    CUresult exit_code = cuArrayCreate(&pHandle, pAllocateArray);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pHandle);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Creates a 3D CUDA array.*/
CUDA_DRIVER_HANDLER(Array3DCreate) {
    CUarray pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray =
        input_buffer->Assign<const CUDA_ARRAY3D_DESCRIPTOR>();
    CUresult exit_code = cuArray3DCreate(&pHandle, pAllocateArray);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pHandle);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Copies memory for 2D arrays. */
CUDA_DRIVER_HANDLER(Memcpy2D) {
    CUDA_MEMCPY2D *pCopy = input_buffer->AssignAll<CUDA_MEMCPY2D>();
    int flag = input_buffer->Get<int>();
    if (flag == 1) pCopy->srcHost = input_buffer->AssignAll<char>();
    CUresult exit_code = cuMemcpy2D(pCopy);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

/*Destroys a CUDA array.*/
CUDA_DRIVER_HANDLER(ArrayDestroy) {
    CUarray hArray = input_buffer->Get<CUarray>();
    CUresult exit_code = cuArrayDestroy(hArray);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

/*Allocates pitched device memory.*/
CUDA_DRIVER_HANDLER(MemAllocPitch) {
    CUdeviceptr dptr;
    size_t pitch;
    size_t WidthInBytes = input_buffer->Get<size_t>();
    size_t Height = input_buffer->Get<size_t>();
    unsigned int ElementSizeBytes = input_buffer->Get<unsigned int>();
    CUresult exit_code = cuMemAllocPitch(&dptr, &pitch, WidthInBytes, Height, ElementSizeBytes);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(dptr);
    out->Add(pitch);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Get information on memory allocations.*/
CUDA_DRIVER_HANDLER(MemGetAddressRange) {
    CUdeviceptr pbase;
    size_t psize;
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr>();
    CUresult exit_code = cuMemGetAddressRange(&pbase, &psize, dptr);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pbase);
    out->Add(psize);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

/*Gets free and total memory.*/
CUDA_DRIVER_HANDLER(MemGetInfo) {
    size_t free;
    size_t total;
    CUresult exit_code = cuMemGetInfo(&free, &total);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(free);
    out->Add(total);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemsetD32Async) {
    CUdeviceptr dstDevice = input_buffer->Get<CUdeviceptr>();
    unsigned int ui = input_buffer->Get<unsigned int>();
    size_t N = input_buffer->Get<size_t>();
    CUstream hStream = input_buffer->Get<CUstream>();

    CUresult exit_code = cuMemsetD32Async(dstDevice, ui, N, hStream);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemsetD32Async executed for dstDevice: "
                                            << dstDevice << ", ui: " << ui << ", N: " << N);
    return std::make_shared<Result>((cudaError_t)exit_code);
}