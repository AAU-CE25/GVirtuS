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
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>

using namespace std;

// cuInit reentrancy fix.
//
// UCX's libuct_cuda.so calls cuInit(0) during its module init, which happens
// inside ucp_init() while Frontend::Init() is still constructing the
// frontend. The original implementation below called Frontend::Prepare() ->
// Buffer::Reset() on a not-yet-initialised buffer -> SIGSEGV. Forwarding the
// call to the real local libcuda fixes both the reentrancy crash and lets
// libuct_cuda do its memory-type probing for future GPUDirect work.

namespace {
typedef CUresult (*cuInit_fn_t)(unsigned int);

std::once_flag g_real_cuinit_once;
std::atomic<cuInit_fn_t> g_real_cuinit{nullptr};

void gvs_load_real_cuinit() {
    const char *candidates[] = {
        "/usr/local/cuda/compat/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        nullptr,
    };
    for (int i = 0; candidates[i] != nullptr; ++i) {
        void *handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) continue;
        cuInit_fn_t fn = reinterpret_cast<cuInit_fn_t>(dlsym(handle, "cuInit"));
        if (fn != nullptr) {
            g_real_cuinit.store(fn);
            fprintf(stderr,
                    "[GVIRTUS] cuInit passthrough wired to %s\n",
                    candidates[i]);
            fflush(stderr);
            return;
        }
        dlclose(handle);
    }
    fprintf(stderr,
            "[GVIRTUS] cuInit passthrough: no real libcuda.so found, "
            "returning CUDA_SUCCESS to satisfy callers\n");
    fflush(stderr);
}
}  // namespace

/* Initialize the CUDA driver API.
 *
 * Always passthrough to the real local libcuda. The RPC path is unsafe
 * during early library initialisation (libuct_cuda calls cuInit while
 * ucp_init runs inside Frontend::Init() -> recursion -> SIGSEGV).
 */
extern "C" CUresult cuInit(unsigned int flags) {
    std::call_once(g_real_cuinit_once, gvs_load_real_cuinit);

    cuInit_fn_t fn = g_real_cuinit.load();
    if (fn != nullptr) {
        // Call the real local driver ONLY for its side effect (UCX
        // libuct_cuda memory-type probing + reentrancy safety), but DISCARD
        // its return value. In the GVirtuS model the physical GPU lives on the
        // backend; the frontend container has no local device, so the real
        // compat driver returns CUDA_ERROR_NO_DEVICE. Forwarding that made
        // cuda-python's runtime reimplementation (which calls cuInit first
        // during lazy init) abort every device query with cudaErrorNoDevice,
        // even though the GVirtuS driver RPCs (cuDeviceGetCount/cuDeviceGet)
        // correctly report the remote device. Always report SUCCESS: the real
        // devices are enumerated over the RPC path, not the local driver.
        (void)fn(flags);
    }
    // Best-effort SUCCESS so optional consumers (libuct_cuda capability
    // probing, cuda-python runtime lazy init) proceed to the RPC device path.
    return CUDA_SUCCESS;
}
