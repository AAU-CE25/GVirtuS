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
 */

#include "CudaRt.h"
#include "PtdsExplicit.h"

// Definidas en CudaRt_graph.cpp.
void gvs_recoge_salidas(cudaStream_t solo_este, bool todos);
void gvs_anota_evento(cudaEvent_t ev, cudaStream_t s);
void gvs_olvida_evento(cudaEvent_t ev);
bool gvs_stream_de_evento(cudaEvent_t ev, cudaStream_t &out);

// I12, puntos de observacion. Una copia D2H capturada deja sus bytes en un buffer del backend
// hasta que el cliente los pide. "En la siguiente sincronizacion" no es UN punto: CUDA declara
// el resultado visible al host en varios, y hasta 2026-08-03 este frontend solo recogia en
// cudaStreamSynchronize y cudaDeviceSynchronize. Medido con tests/semantic/graphvis.cu: los
// otros tres devolvian los bytes VIEJOS con cudaSuccess -- el mismo modo de fallo silencioso
// que la semantica de relanzamiento.
//
// En los dos QUERY se recoge SOLO si la llamada devolvio cudaSuccess. El handler de recogida
// sincroniza el stream en el backend, de modo que recoger tras un cudaErrorNotReady
// convertiria una consulta no bloqueante en una espera.
using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaEventCreate");
    if (CudaRtFrontend::Success()) *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event,
                                                                   unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaEventCreateWithFlags");
    if (CudaRtFrontend::Success()) *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    gvs_olvida_evento(event);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventDestroy");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                                               cudaEvent_t end) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(ms);
    CudaRtFrontend::AddDevicePointerForArguments(start);
    CudaRtFrontend::AddDevicePointerForArguments(end);
    CudaRtFrontend::Execute("cudaEventElapsedTime");
    if (CudaRtFrontend::Success()) *ms = *(CudaRtFrontend::GetOutputHostPointer<float>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventQuery");
    cudaError_t rc = CudaRtFrontend::GetExitCode();
    // Solo en exito: cudaSuccess aqui significa que el trabajo previo al record termino, que
    // es exactamente cuando el host tiene derecho a leer el destino de la copia.
    cudaStream_t s;
    if (rc == cudaSuccess && gvs_stream_de_evento(event, s)) gvs_recoge_salidas(s, false);
    return rc;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    stream = gvs_ptds::traduce(stream);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::ExecuteMaybeAsync("cudaEventRecord");
    gvs_anota_evento(event, stream);
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::ExecuteMaybeAsync("cudaEventRecordWithFlags");
    gvs_anota_evento(event, stream);
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventSynchronize");
    cudaError_t rc = CudaRtFrontend::GetExitCode();
    // Se recoge SOLO el stream al que se grabo el evento: cudaEventSynchronize no espera a
    // los demas y esto tampoco debe hacerlo.
    cudaStream_t s;
    if (rc == cudaSuccess && gvs_stream_de_evento(event, s)) gvs_recoge_salidas(s, false);
    return rc;
}
