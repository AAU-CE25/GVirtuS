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
 */

/**
 * @file DeviceMemory.cpp
 *
 * Implementation of the shared, transport-agnostic CUDA device-memory
 * primitives declared in DeviceMemory.h. This is a straight extraction of
 * the "Device resolver" block that used to live only in
 * src/communicators/ucx/UcxGpu.cpp — same dlsym candidates, same
 * std::call_once caching, same short-circuit-on-disabled-gate behaviour.
 * ucx_internal::is_gpu_pointer / set_gpudirect_enabled / gpudirect_enabled /
 * alloc_gpu_slot / free_gpu_slot (UcxGpu.cpp) now forward here so there is a
 * single resolver and a single "is GPUDirect active" flag shared by UCX and
 * by Buffer/Communicator.
 */
#include "gvirtus/communicators/DeviceMemory.h"

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>

namespace gvirtus::communicators {

namespace {

// ---- Device resolver (cudaMalloc / cudaFree / cudaMemcpy /
//      cudaPointerGetAttributes), dlopen'd on first use. No static link to
//      libcudart, so builds without a driver/toolkit still compile and run
//      (every entry point simply resolves to nullptr and every public
//      function here degrades to "unavailable"). ----

using cudaMalloc_t = int (*)(void **, std::size_t);
using cudaFree_t = int (*)(void *);
using cudaPointerGetAttributes_t = int (*)(void *, const void *);
using cudaMemcpy_t = int (*)(void *, const void *, std::size_t, int);

// Mirror of cudaPointerAttributes (CUDA 11+ layout). `type` 0=Unregistered,
// 1=Host, 2=Device, 3=Managed. Only `type` is read; remaining fields kept
// for ABI alignment. Identical to the layout UcxGpu.cpp used.
struct cudaPointerAttributes_layout {
    int type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

// cudaMemcpyKind::cudaMemcpyDeviceToHost == 2 in the CUDA runtime ABI.
constexpr int kCudaMemcpyDeviceToHost = 2;

std::once_flag g_resolver_once;
std::atomic<cudaMalloc_t> g_cuda_malloc{nullptr};
std::atomic<cudaFree_t> g_cuda_free{nullptr};
std::atomic<cudaPointerGetAttributes_t> g_cuda_pointer_attrs{nullptr};
std::atomic<cudaMemcpy_t> g_cuda_memcpy{nullptr};

void load_cuda_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto m = reinterpret_cast<cudaMalloc_t>(dlsym(h, "cudaMalloc"));
        auto f = reinterpret_cast<cudaFree_t>(dlsym(h, "cudaFree"));
        auto a = reinterpret_cast<cudaPointerGetAttributes_t>(
                     dlsym(h, "cudaPointerGetAttributes"));
        auto c = reinterpret_cast<cudaMemcpy_t>(dlsym(h, "cudaMemcpy"));
        if (m && f) {
            g_cuda_malloc.store(m);
            g_cuda_free.store(f);
            g_cuda_pointer_attrs.store(a);  // may be nullptr; callers handle it
            g_cuda_memcpy.store(c);         // may be nullptr; callers handle it
            return;
        }
        dlclose(h);
    }
    // libcudart unavailable — every public function below degrades to
    // "unavailable"/false, which is the correct behaviour for a host-only
    // build or a dev container without the driver installed.
}

// Process-wide flag: true iff GVIRTUS_GPUDIRECT=1 was set AND the UCX
// startup probe (cudaMalloc + ucp_mem_map(CUDA)) succeeded.
std::atomic<bool> g_device_probe_enabled{false};

}  // namespace

void SetDeviceProbeEnabled(bool on) { g_device_probe_enabled.store(on); }
bool DeviceProbeEnabled() { return g_device_probe_enabled.load(); }

// CRITICAL: this function runs on both frontend AND backend. On the
// frontend, libcudart.so is GVirtuS's own shim, which REMOTES
// cudaPointerGetAttributes as an RPC — slow and meaningless (the frontend
// has no local GPU to ask about). The DeviceProbeEnabled() short-circuit
// avoids that RPC storm: the frontend never has the env var/probe set, so
// this returns false before ever touching the resolved function pointer.
bool IsDevicePointer(const void *p) {
    if (p == nullptr) return false;
    if (!DeviceProbeEnabled()) return false;
    std::call_once(g_resolver_once, load_cuda_funcs);
    auto fn = g_cuda_pointer_attrs.load();
    if (fn == nullptr) return false;
    cudaPointerAttributes_layout attrs{};
    if (fn(&attrs, p) != 0) return false;
    return attrs.type == 2 /*Device*/ || attrs.type == 3 /*Managed*/;
}

bool DeviceMemcpyD2H(void *dst_host, const void *src_gpu, std::size_t n) {
    if (dst_host == nullptr || src_gpu == nullptr) return false;
    if (n == 0) return true;
    std::call_once(g_resolver_once, load_cuda_funcs);
    auto fn = g_cuda_memcpy.load();
    if (fn == nullptr) return false;
    return fn(dst_host, src_gpu, n, kCudaMemcpyDeviceToHost) == 0;
}

bool AllocDeviceMemory(void **p, std::size_t n) {
    if (p == nullptr) return false;
    std::call_once(g_resolver_once, load_cuda_funcs);
    auto fn = g_cuda_malloc.load();
    if (fn == nullptr) return false;
    return fn(p, n) == 0 && *p != nullptr;
}

void FreeDeviceMemory(void *p) {
    if (p == nullptr) return;
    std::call_once(g_resolver_once, load_cuda_funcs);
    auto fn = g_cuda_free.load();
    if (fn != nullptr) fn(p);
}

}  // namespace gvirtus::communicators
