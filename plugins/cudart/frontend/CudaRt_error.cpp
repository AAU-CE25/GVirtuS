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

using namespace std;

// --- Frontend-local CUDA runtime state (GUSTO optimization) -------------------
// Definitions for the accessors declared in CudaRt_internal.h. Thread-local so
// each host thread has its own current-device and sticky last-error, matching
// CUDA's per-thread runtime state.
namespace cudart_state {
static thread_local cudaError_t g_last_error = cudaSuccess;
static thread_local int g_current_device = 0;  // CUDA default current device

void note_exit_code(int exit_code) {
    // CUDA only updates the sticky error on failure; success never clears it.
    // IMPORTANT: cudaErrorNotReady is NOT a sticky error — cudaStreamQuery /
    // cudaEventQuery return it while work is still in flight, and native CUDA
    // does NOT record it as the last error. Treating it as sticky (as a naive
    // "any non-success" check would) corrupts cudaGetLastError and makes callers
    // like ggml's CUDA_CHECK abort — observed specifically under GPUDirect, where
    // transfers are genuinely async so queries return cudaErrorNotReady often.
    const cudaError_t ec = static_cast<cudaError_t>(exit_code);
    if (ec != cudaSuccess && ec != cudaErrorNotReady) g_last_error = ec;
}
int take_last_error() {
    cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;  // cudaGetLastError() resets to success
    return static_cast<int>(e);
}
int peek_last_error() { return static_cast<int>(g_last_error); }
int current_device() { return g_current_device; }
void set_current_device(int device) { g_current_device = device; }
}  // namespace cudart_state

extern "C" __host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(error);
    CudaRtFrontend::Execute("cudaGetErrorString");
#ifdef _WIN32
    char* error_string = _strdup(CudaRtFrontend::GetOutputString());
#else
    char* error_string = strdup(CudaRtFrontend::GetOutputString());
#endif
    return error_string;
}
extern "C" __host__ cudaError_t CUDARTAPI cudaPeekAtLastError(void) {
    // Frontend-local: return the sticky last error without clearing it. No RPC.
    return static_cast<cudaError_t>(cudart_state::peek_last_error());
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
    // Frontend-local: return the sticky last error and reset it. No RPC. The
    // last error is tracked from the exit code of every remoted cudart call
    // (see CudaRtFrontend::Execute + cudart_state::note_exit_code), so this is
    // behaviourally identical to the old blocking round-trip.
    return static_cast<cudaError_t>(cudart_state::take_last_error());
}

extern "C" __host__ __device__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(error);
    CudaRtFrontend::Execute("cudaGetErrorName");
    return CudaRtFrontend::GetOutputString();
}