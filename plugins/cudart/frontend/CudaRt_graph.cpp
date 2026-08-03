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
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 *
 *             Ting-Hui Cheng <tinghc@es.aau.dk>
 *             Department of Electronic Systems, Aalborg University
 */

#include "CudaRt.h"
#include "CaptureMirror.h"
#include <cstring>

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph,
                                                            cudaGraphNode_t* nodes,
                                                            size_t* numNodes) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddHostPointerForArguments(nodes);
    CudaRtFrontend::Execute("cudaGraphGetNodes");

    if (CudaRtFrontend::Success()) {
        *numNodes = CudaRtFrontend::GetOutputVariable<size_t>();
        // cout << "Get a node for graph." << endl;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graphExec);
    CudaRtFrontend::Execute("cudaGraphExecDestroy");
    gvs_capmirror::destruye_exec(graphExec);
    // cout << "Destroy graph execution." << endl;
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaGraphInstantiate");

    if (CudaRtFrontend::Success()) {
        *pGraphExec = CudaRtFrontend::GetOutputVariable<cudaGraphExec_t>();
        gvs_capmirror::instancia(graph, *pGraphExec);
        // cout << "Creates an executable graph from a graph." << endl;
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                                                        cudaGraph_t graph,
                                                                        unsigned long long flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaGraphInstantiateWithFlags");
    // cout << "Graph:" << graph << endl;                                                                           
    if (CudaRtFrontend::Success()) {
        *pGraphExec = CudaRtFrontend::GetOutputVariable<cudaGraphExec_t>();
        gvs_capmirror::instancia(graph, *pGraphExec);
        // cout << "Creates an executable graph from a graph." << endl;
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph,
                                                                 const char* path,
                                                                 unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddStringForArguments(path);
    CudaRtFrontend::AddVariableForArguments<unsigned int>(flags);
    CudaRtFrontend::Execute("cudaGraphDebugDotPrint");

    return CudaRtFrontend::GetExitCode();
}


// Refresca, en el backend, el staging de cada nodo H2D que se capturo en este ejecutable.
//
// Un nodo H2D capturado lee su origen EN EL LANZAMIENTO. Como el origen vive en este proceso,
// sin esto el backend relanzaria siempre la foto que tomo al capturar -- medido, y en
// silencio. El payload viaja por el camino de datos normal (splice directo en el iov), o sea
// RMA cuando el tamano lo justifica: esto no rodea el pool de slots, lo usa.
//
// Coste cero para quien no captura copias H2D (llama.cpp captura kernels): la lista esta
// vacia y no se emite ningun mensaje.
static void gvs_refresca_staging(cudaGraphExec_t graphExec) {
    const auto entradas = gvs_capmirror::entradas_de(graphExec);
    for (size_t k = 0; k < entradas.size(); ++k) {
        const auto &e = entradas[k];
        if (e.src == nullptr || e.bytes == 0) continue;
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddDevicePointerForArguments(graphExec);
        CudaRtFrontend::AddVariableForArguments(k);
        CudaRtFrontend::AddVariableForArguments(e.bytes);
        CudaRtFrontend::AddHostPointerForArgumentsDirect<char>(
            static_cast<const char *>(e.src), e.bytes);
        // El destino es el buffer de staging del backend, que es memoria de HOST: sin esto el
        // payload se peer-DMA a la sombra de GPU y el slot de host llega con un hueco.
        gvirtus::frontend::Frontend::GetFrontend()->MarkDirectInputHostDestined();
        // Mismo modo de despacho que el propio lanzamiento: el orden en el cable se conserva,
        // asi que el refresco esta puesto en el staging antes de que el backend encole el
        // grafo. Un error se reconcilia en el siguiente punto de sincronizacion, como
        // cualquier otro fallo asincrono.
        CudaRtFrontend::ExecuteMaybeAsync("cudaGraphStagingRefresh");
    }
}

// Recoge, en el punto de sincronizacion, las salidas de los nodos D2H capturados que se
// lanzaron desde este hilo. El backend sincroniza el stream dentro del handler, que es donde
// ya es legal hacerlo: dentro de la ventana de captura no lo era.
//
// Se llama desde cudaStreamSynchronize / cudaDeviceSynchronize. Un grafo sin nodos D2H
// capturados no registra nada, asi que esto no cuesta nada en el caso normal (llama.cpp).
void gvs_recoge_salidas(cudaStream_t solo_este, bool todos) {
    auto &pend = gvs_capmirror::lanzados();
    if (pend.empty()) return;
    std::vector<gvs_capmirror::Lanzado> quedan;
    for (const auto &L : pend) {
        if (!todos && L.stream != solo_este) { quedan.push_back(L); continue; }
        const auto ss = gvs_capmirror::salidas_de(L.exec);
        for (size_t j = 0; j < ss.size(); ++j) {
            if (ss[j].dst == nullptr || ss[j].bytes == 0) continue;
            CudaRtFrontend::Prepare();
            CudaRtFrontend::AddDevicePointerForArguments(L.exec);
            CudaRtFrontend::AddVariableForArguments(j);
            CudaRtFrontend::AddVariableForArguments(ss[j].bytes);
            CudaRtFrontend::AddDevicePointerForArguments(L.stream);
            CudaRtFrontend::Execute("cudaGraphStagingFetch");
            if (CudaRtFrontend::Success())
                memmove(ss[j].dst,
                        CudaRtFrontend::GetOutputHostPointer<char>(ss[j].bytes),
                        ss[j].bytes);
        }
    }
    pend.swap(quedan);
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec,
                                                          cudaStream_t stream) {
    gvs_refresca_staging(graphExec);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graphExec);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // Stream-ordered graph launch, no return data -> fire-and-forget under
    // GVIRTUS_ASYNC_DISPATCH (like cudaLaunchKernel). Errors surface at the next
    // synchronous call. Backend handles the no-response flag generically.
    CudaRtFrontend::ExecuteMaybeAsync("cudaGraphLaunch");
    // Si el grafo tiene nodos D2H capturados, sus bytes estan en un buffer del backend hasta
    // que el cliente sincronice. Se anota el lanzamiento; la recogida ocurre en el sync.
    if (!gvs_capmirror::salidas_de(graphExec).empty())
        gvs_capmirror::lanzados().push_back(gvs_capmirror::Lanzado{graphExec, stream});
    // cout << "Graph Launch" << endl;                                                        
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    // cout << "Graph is Create" << *pGraph << endl;
    CudaRtFrontend::Execute("cudaGraphCreate");
    
    if (CudaRtFrontend::Success()) *pGraph = CudaRtFrontend::GetOutputVariable<cudaGraph_t>();
    
    return CudaRtFrontend::GetExitCode();
}


extern "C" __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::Execute("cudaGraphDestroy");
    gvs_capmirror::destruye_grafo(graph);
    // cout << "Graph is Destroy" << endl;
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graphExec);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // Fix: was mistakenly invoking the "cudaGraphLaunch" backend handler; the
    // correct routine is "cudaGraphUpload" (a real handler exists). Also
    // stream-ordered + no return data -> fire-and-forget under async dispatch.
    CudaRtFrontend::ExecuteMaybeAsync("cudaGraphUpload");
    
    return CudaRtFrontend::GetExitCode();
}


// Real cudaGraphExecUpdate (CUDA 12 form). Forwards the two backend handles,
// runs the real in-place update on the backend, and marshals the
// cudaGraphExecUpdateResultInfo back. Returning cudaSuccess lets llama.cpp reuse
// an instantiated graph across decode tokens WITHOUT re-instantiating; returning
// cudaErrorGraphExecUpdateFailure makes it re-instantiate (its documented path).
// This is the key to CUDA-graph token-generation running at ~baremetal over GVirtuS.
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                                             cudaGraph_t hGraph,
                                                             cudaGraphExecUpdateResultInfo *resultInfo) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(hGraphExec);
    CudaRtFrontend::AddDevicePointerForArguments(hGraph);
    CudaRtFrontend::Execute("cudaGraphExecUpdate");
    if (resultInfo != NULL) {
        *resultInfo = CudaRtFrontend::GetOutputVariable<cudaGraphExecUpdateResultInfo>();
    }
    return CudaRtFrontend::GetExitCode();
}
