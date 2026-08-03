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
 * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,
 *             Department of Electronic Systems, Aalborg University, Denmark
 */

#include "CudaRtHandler.h"
#include "gvirtus/communicators/CaptureStaging.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

CUDA_ROUTINE_HANDLER(GraphCreate) {
    try {
        cudaGraph_t pGraph;
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaGraphCreate(&pGraph, flags);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaGraph_t>(pGraph);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GraphDestroy) {
    try {
        cudaGraph_t graph = input_buffer->Get<cudaGraph_t>();
        cudaError_t exit_code = cudaGraphDestroy(graph);
        // Suelta la referencia del grafo al lote de staging. Sin esto el lote no se libera
        // NUNCA: la cadena captura->grafo->ejecutable estaba escrita pero no conectada, asi
        // que cada grafo capturado dejaba sus buffers vivos hasta el fin del proceso.
        gvs_capture::grafo_destruido(graph);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// Refresco del staging de un nodo H2D capturado, previo a cada cudaGraphLaunch.
//
// POR QUE EXISTE. Un nodo H2D capturado lee su origen EN EL LANZAMIENTO (medido, nativo y
// aqui). El origen del cliente vive en OTRO PROCESO, asi que el buffer que el backend copio
// al capturar entrega los MISMOS bytes en cada relanzamiento. Medido sobre GVirtuS antes de
// este handler: capturar 1 KiB, cambiar el origen y relanzar devolvia los bytes viejos, con
// cudaSuccess en todas las llamadas. Un resultado mal en silencio.
//
// El frontend envia el payload por el MISMO camino de datos que cualquier H2D (splice directo
// en el iov -> RMA o mensaje activo segun tamano), asi que el refresco no rodea el diseno de
// slots: lo usa.
CUDA_ROUTINE_HANDLER(GraphStagingRefresh) {
    try {
        cudaGraphExec_t graphExec = input_buffer->Get<cudaGraphExec_t>();
        size_t k     = input_buffer->Get<size_t>();
        size_t bytes = input_buffer->Get<size_t>();
        // El payload puede llegar por el slot de host o, si algun frontend lo marca como
        // device-destined, por la sombra de GPU. Leer siempre el slot de host encontraria el
        // HUECO que deja el peer-DMA y refrescaria con ceros -- medido, y en silencio.
        void  *gpu_src = input_buffer->GetGpuPayload();
        size_t gpu_sz  = input_buffer->GetGpuPayloadSize();
        bool ok;
        if (gpu_src != nullptr && gpu_sz >= bytes) {
            ok = gvs_capture::refresca_desde_device(graphExec, k, gpu_src, bytes);
        } else {
            char *src = input_buffer->AssignAll<char>();
            if (src == nullptr) return std::make_shared<Result>(cudaErrorInvalidValue);
            // El indice y el tamano se validan contra lo que se registro al capturar: si los
            // dos lados se desalinean, se rechaza en vez de escribir en el buffer que no es.
            ok = gvs_capture::refresca(graphExec, k, src, bytes);
        }
        if (!ok) return std::make_shared<Result>(cudaErrorInvalidValue);
        return std::make_shared<Result>(cudaSuccess);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// Recogida del staging de SALIDA de un nodo D2H capturado, en el punto de sincronizacion.
//
// Sincroniza el stream AQUI, no en el handler de la copia: dentro de la ventana de captura
// sincronizar la invalida, y fuera de ella es exactamente lo que el cliente pidio con su
// cudaStreamSynchronize. Devuelve los bytes que el grafo dejo en el buffer del backend.
CUDA_ROUTINE_HANDLER(GraphStagingFetch) {
    try {
        cudaGraphExec_t graphExec = input_buffer->Get<cudaGraphExec_t>();
        size_t j       = input_buffer->Get<size_t>();
        size_t bytes   = input_buffer->Get<size_t>();
        cudaStream_t st = input_buffer->Get<cudaStream_t>();
        void *buf = gvs_capture::salida_de(graphExec, j, bytes);
        if (buf == nullptr) return std::make_shared<Result>(cudaErrorInvalidValue);
        cudaError_t exit_code = cudaStreamSynchronize(st);
        if (exit_code != cudaSuccess) return std::make_shared<Result>(exit_code);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<char>((char *)buf, bytes);
        return std::make_shared<Result>(cudaSuccess, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GraphGetNodes) {
    try {
        cudaGraph_t pGraph = input_buffer->Get<cudaGraph_t>();
        cudaGraphNode_t* nodes = input_buffer->Assign<cudaGraphNode_t>();
        size_t numNodes;
        cudaError_t exit_code = cudaGraphGetNodes(pGraph, nodes, &numNodes);
        // Debugging output
        // std::cout << "GraphGetNodes " << nodes << " with a size of "
        //     << numNodes << std::endl;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<size_t>(numNodes);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// cudaGraphInstantiate signature changed in CUDA 12.
CUDA_ROUTINE_HANDLER(GraphInstantiate) {
    try {
        cudaGraphExec_t pGraphExec;
        cudaGraph_t graph = input_buffer->Get<cudaGraph_t>();
        unsigned long long flags = input_buffer->Get<unsigned long long>();
        cudaError_t exit_code = cudaGraphInstantiate(&pGraphExec, graph, flags);
        // El ejecutable pasa a compartir el lote de staging del grafo: sus nodos apuntan a
        // esos buffers, asi que tienen que sobrevivirle.
        if (exit_code == cudaSuccess) gvs_capture::grafo_instanciado(graph, pGraphExec);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaGraphExec_t>(pGraphExec);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GraphInstantiateWithFlags) {
    try {
        cudaGraphExec_t pGraphExec;
        cudaGraph_t graph = input_buffer->Get<cudaGraph_t>();
        unsigned long long flags = input_buffer->Get<unsigned long long>();

        cudaError_t exit_code = cudaGraphInstantiateWithFlags(&pGraphExec, graph, flags);
        if (exit_code == cudaSuccess) gvs_capture::grafo_instanciado(graph, pGraphExec);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaGraphExec_t>(pGraphExec);
        // std::cout << "execution: " << pGraphExec << " Graph: "<< graph << std::endl;
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// No Testing
CUDA_ROUTINE_HANDLER(GraphLaunch) {
    try {
        cudaGraphExec_t graphExec = input_buffer->Get<cudaGraphExec_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaGraphLaunch(graphExec, stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GraphExecDestroy) {
    try {
        cudaGraphExec_t graphExec = input_buffer->Get<cudaGraphExec_t>();
        cudaError_t exit_code = cudaGraphExecDestroy(graphExec);
        gvs_capture::exec_destruido(graphExec);   // ultimo eslabon: aqui muere el lote
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GraphUpload) {
    try {
        cudaGraphExec_t graphExec = input_buffer->Get<cudaGraphExec_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaGraphUpload(graphExec, stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}


CUDA_ROUTINE_HANDLER(GraphExecUpdate) {
    cudaGraphExecUpdateResultInfo resultInfo = {};
    cudaError_t exit_code;
    try {
        cudaGraphExec_t hGraphExec = input_buffer->Get<cudaGraphExec_t>();
        cudaGraph_t hGraph = input_buffer->Get<cudaGraph_t>();
        exit_code = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        exit_code = cudaErrorMemoryAllocation;
    }
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add<cudaGraphExecUpdateResultInfo>(resultInfo);
    return std::make_shared<Result>(exit_code, out);
}
