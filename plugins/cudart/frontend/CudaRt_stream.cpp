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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRt.h"
#include "PtdsExplicit.h"
#include "CaptureMirror.h"

// Definida en CudaRt_graph.cpp.
void gvs_recoge_salidas(cudaStream_t solo_este, bool todos);
void gvs_anota_stream(cudaStream_t s, unsigned int flags);
void gvs_olvida_stream(cudaStream_t s);

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* pStream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaStreamCreate");
    if (CudaRtFrontend::Success()) {
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
        gvs_anota_stream(*pStream, cudaStreamDefault);   // cudaStreamCreate crea BLOCKING
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                                                    unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaStreamCreateWithFlags");
    if (CudaRtFrontend::Success()) {
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
        gvs_anota_stream(*pStream, flags);
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    gvs_olvida_stream(stream);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamDestroy");
    return CudaRtFrontend::GetExitCode();
}

/*cudaError_t cudaStreamWaitEvent	(	cudaStream_t 	stream,
cudaEvent_t 	event,
unsigned int 	flags
)	*/
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                              cudaEvent_t event,
                                                              unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::ExecuteMaybeAsync("cudaStreamWaitEvent");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    stream = gvs_ptds::traduce(stream);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamQuery");
    cudaError_t rc = CudaRtFrontend::GetExitCode();
    // Solo en exito. cudaSuccess = el stream esta vacio, asi que la sincronizacion que el
    // handler de recogida hace en el backend no bloquea y la consulta sigue sin bloquear.
    if (rc == cudaSuccess) gvs_recoge_salidas(stream, false);
    return rc;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* pStream,
                                                                       unsigned int flags,
                                                                       int priority) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::AddVariableForArguments(priority);
    CudaRtFrontend::Execute("cudaStreamCreateWithPriority");
    if (CudaRtFrontend::Success()) {
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
        gvs_anota_stream(*pStream, flags);
    }
    return CudaRtFrontend::GetExitCode();
}

namespace gvs_cli { void vuelca(const char *quien, int rc); }

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const void *crudo = (const void *)stream;
    stream = gvs_ptds::traduce(stream);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamSynchronize");
    // El stream ya se sincronizo: si algun grafo lanzado en el dejo salidas D2H capturadas en
    // el backend, este es el momento en que el cliente espera verlas en su buffer.
    gvs_recoge_salidas(stream, false);
    const cudaError_t rc = CudaRtFrontend::GetExitCode();
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[GVS CLI] cudaStreamSynchronize raw=%p translated=%p -> rc=%d\n",
                     crudo, (const void *)stream, (int)rc);
        gvs_cli::vuelca("cudaStreamSynchronize", (int)rc);
    }
    return rc;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments<cudaStreamCaptureMode>(*mode);
    CudaRtFrontend::Execute("cudaThreadExchangeStreamCaptureMode");
    if (CudaRtFrontend::Success())
        *mode = CudaRtFrontend::GetOutputVariable<cudaStreamCaptureMode>();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,
                                                                 cudaStreamCaptureMode mode) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // cout << "Stream starts capturing." << endl;
    CudaRtFrontend::AddVariableForArguments(mode);
    CudaRtFrontend::Execute("cudaStreamBeginCapture");
    // Espejo de captura: a partir de aqui se anota cada H2D para poder refrescar el staging
    // del backend antes de cada lanzamiento (ver CaptureMirror.h).
    if (CudaRtFrontend::Success()) gvs_capmirror::abre();
    
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI   cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamIsCapturing");
    if (CudaRtFrontend::Success()){
        *pCaptureStatus = CudaRtFrontend::GetOutputVariable<cudaStreamCaptureStatus>();
        // cout << "Stream is capturing. " << *pCaptureStatus << endl;
    }
        
    return CudaRtFrontend::GetExitCode();
}
// cudaStreamGetId
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, 
                                                                cudaStreamCaptureStatus* captureStatus_out, 
                                                                unsigned long long* id_out, 
                                                                cudaGraph_t* graph_out, 
                                                                const cudaGraphNode_t **dependencies_out, 
                                                                // const cudaGraphEdgeData **edgeData_out,
                                                                size_t *numDependencies_out) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // cout << "Frontend" << *captureStatus_out << endl;
    // cout << "Frontend" << *id_out << endl;
    // 
    // cout << "Frontend" << *numDependencies_out << endl;
    CudaRtFrontend::AddHostPointerForArguments(graph_out);
    CudaRtFrontend::AddHostPointerForArguments<const cudaGraphNode_t*>(dependencies_out);
    CudaRtFrontend::AddHostPointerForArguments(numDependencies_out);

    cudaGraph_t local_graph = nullptr;
    size_t local_numDeps = 0;
    const cudaGraphNode_t* local_deps = nullptr;

    CudaRtFrontend::Execute("cudaStreamGetCaptureInfo");
    if (CudaRtFrontend::Success()){


        // cout << "Stream capture info." << endl;
        *captureStatus_out = CudaRtFrontend::GetOutputVariable<cudaStreamCaptureStatus>();
        // cout << "Frontend Status_out " << *captureStatus_out << endl;

        *id_out = CudaRtFrontend::GetOutputVariable<unsigned long long>();
        // cout << "Frontend id_out " << *id_out << endl;
        
        local_graph   = (cudaGraph_t)CudaRtFrontend::GetOutputDevicePointer();
        // cout << "Frontend graph_out " << local_graph  << endl;
        
        local_deps = (const cudaGraphNode_t *)CudaRtFrontend::GetOutputDevicePointer();
        // cout << "Frontend dependencies_out " << (void*)local_deps << endl;
        local_numDeps = CudaRtFrontend::GetOutputVariable<size_t>();
        // cout << "Frontend num dependencies_out " << local_numDeps << endl;

        
        if (graph_out) *graph_out = local_graph;
        // cout << "Frontend graph_out " << local_graph << " "<<*graph_out << endl;
        if (dependencies_out) *dependencies_out = local_deps;
        // cout << "Frontend dependencies_out " << *dependencies_out << endl;
        if (numDependencies_out) *numDependencies_out = local_numDeps;

    }
        
        
    return CudaRtFrontend::GetExitCode();
}


// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream,
                                                               cudaGraph_t* pGraph) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // cout << "End capture." << *pGraph << endl;
    CudaRtFrontend::Execute("cudaStreamEndCapture");
    if (CudaRtFrontend::Success()) *pGraph = CudaRtFrontend::GetOutputVariable<cudaGraph_t>();
    // Se cierra el espejo pase lo que pase: si EndCapture fallo (captura invalidada) hay que
    // soltar la lista igual, o la siguiente captura de este hilo heredaria sus entradas.
    gvs_capmirror::cierra(CudaRtFrontend::Success() ? *pGraph : nullptr);
    // cout << "End capture." << *pGraph << endl;
    
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs fixing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
                                                                cudaStreamCallback_t callback,
                                                                void* userData,
                                                                unsigned int flags) {
    // IMPORTANT: cudaStreamCallback_t is a function pointer type
    // not sure how to handle it in the frontend
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // CudaRtFrontend::AddDevicePointerForArguments(callback);
    CudaRtFrontend::AddDevicePointerForArguments(userData);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaStreamAddCallback");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                                int* priority) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(hStream);
    CudaRtFrontend::Execute("cudaStreamGetPriority");
    if (CudaRtFrontend::Success()) *priority = CudaRtFrontend::GetOutputVariable<int>();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    return cudaStreamSynchronize(stream ? stream : cudaStreamPerThread);
}
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    return cudaStreamQuery(stream ? stream : cudaStreamPerThread);
}
