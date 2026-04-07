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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(StreamCreate) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaStream_t pStream;
        cudaError_t exit_code = cudaStreamCreate(&pStream);
        out->Add<cudaStream_t>(pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamCreateWithPriority) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        cudaStream_t pStream;
        unsigned int flags = input_buffer->Get<unsigned int>();
        int priority = input_buffer->Get<int>();
        cudaError_t exit_code = cudaStreamCreateWithPriority(&pStream, flags, priority);
        out->Add((pointer_t)pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamCreateWithFlags) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaStream_t pStream;
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaStreamCreateWithFlags(&pStream, flags);
        out->Add<cudaStream_t>(pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamDestroy) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamDestroy(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamWaitEvent) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        return std::make_shared<Result>(cudaStreamWaitEvent(stream, event, flags));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamQuery) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamQuery(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamSynchronize) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamSynchronize(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(ThreadExchangeStreamCaptureMode) {
    try {
        cudaStreamCaptureMode mode = input_buffer->Get<cudaStreamCaptureMode>();
        cudaError_t exit_code = cudaThreadExchangeStreamCaptureMode(&mode);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaStreamCaptureMode>(mode);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamIsCapturing) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCaptureStatus captureStatus;
        cudaError_t exit_code = cudaStreamIsCapturing(stream, &captureStatus);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaStreamCaptureStatus>(captureStatus);
        // std::cout << "Stream Status: " << captureStatus <<  std::endl;
        
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamBeginCapture) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCaptureMode mode = input_buffer->Get<cudaStreamCaptureMode>();
        // std::cout << "Begin capturing\n";
        return std::make_shared<Result>(cudaStreamBeginCapture(stream, mode));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamEndCapture) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaGraph_t pGraph;
        cudaError_t exit_code = cudaStreamEndCapture(stream, &pGraph);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaGraph_t>(pGraph);
        // std::cout <<" Graph: "<< pGraph << std::endl;
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

// void PrintBufferContents(const std::shared_ptr<gvirtus::communicators::Buffer>& buf) {
//     try {
//         // Use the Buffer public API to read values in the SAME order you wrote them.
//         // NOTE: these call sites assume Buffer implements Get<T>() and GetFromMarshal<T>().
//         // If your Buffer API uses different names, adapt accordingly.

//         // rewind / create a reader view if Buffer supports it; many Buffer implementations
//         // maintain an internal read cursor, so this assumes a fresh buffer or that
//         // the cursor is at the beginning. If your Buffer has a Rewind() or ResetRead()
//         // call, invoke it here:
//         // buf->Rewind();

//         cudaStreamCaptureStatus captureStatus = buf->Get<cudaStreamCaptureStatus>();
//         unsigned long long id = buf->Get<unsigned long long>();
//         cudaGraph_t graph = buf->Get<cudaGraph_t>();

//         // Read marshalled dependency pointer array (this returns the pointer stored in the buffer)
//         // Adjust the template if your API expects a different signature.
//         const cudaGraphNode_t* deps = buf->GetFromMarshal<const cudaGraphNode_t*>();
//         size_t numDeps = buf->Get<size_t>();

//         // Print nicely:
//         std::cout << "=== Buffer contents ===\n";
//         std::cout << "Capture status: " << static_cast<int>(captureStatus) << " (enum value)\n";
//         std::cout << "ID: " << id << '\n';
//         std::cout << "Graph handle: " << graph << '\n';
//         std::cout << "  Captured Dependency: " << (void*)deps << std::endl;
//         std::cout << "Num dependencies: " << numDeps << '\n';

//         // Print dependency pointers (pointer values only; not dereferenced!)
//         if (deps != nullptr && numDeps > 0) {
//             std::cout << "Dependency pointers:\n";
//             for (size_t i = 0; i < numDeps; ++i) {
//                 // note: deps is an array of cudaGraphNode_t* (or const cudaGraphNode_t*)
//                 // printing pointer values is the only safe operation here.
//                 std::cout << "  [" << i << "] = " << deps[i] << '\n';
//             }
//         } else {
//             std::cout << "No dependencies (deps == nullptr or numDeps == 0)\n";
//         }
//         std::cout << "=======================\n";
//     } catch (const std::exception& e) {
//         std::cerr << "PrintBufferContents: exception: " << e.what() << '\n';
//     }
// }

CUDA_ROUTINE_HANDLER(StreamGetCaptureInfo) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCaptureStatus captureStatus_out;
        unsigned long long id_out;
        cudaGraph_t* graph_out = input_buffer->Assign<cudaGraph_t>();
        const cudaGraphNode_t** dependencies_out = input_buffer->Assign<const cudaGraphNode_t*>();
        // const cudaGraphEdgeData* edgeData_out;
        size_t* numDependencies_out = input_buffer->Assign<size_t>();
        cout << " Graph: "<< graph_out << std::endl;
        cudaError_t exit_code = cudaStreamGetCaptureInfo(stream, &captureStatus_out,
                                                        &id_out, graph_out, dependencies_out,
                                                        nullptr, numDependencies_out);

        
        
        out->Add<cudaStreamCaptureStatus>(captureStatus_out);
        out->Add<unsigned long long>(id_out);
        out->Add<cudaGraph_t>(graph_out);
        out->Add((pointer_t)dependencies_out);
        out->Add(numDependencies_out);
        // std::cout << "Status: " << captureStatus_out << " id: " << id_out << " Graph: "<< graph_out << " Dependencies: "<< (void*)dependencies_out << std::endl;
        // PrintBufferContents(out);
        std::cerr << "DEBUG: cudaStreamEndCapture -> " << exit_code << " (" 
          << cudaGetErrorString(exit_code) << ")" << std::endl;
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
/*
CUDA_ROUTINE_HANDLER(StreamAddCallback) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCallback_t callback = input_buffer->Get<cudaStreamCallback_t>();
        void* userData = input_buffer->GetFromMarshal<void*>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        return std::make_shared<Result>(cudaStreamAddCallback(stream, callback, userData, flags));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
*/
