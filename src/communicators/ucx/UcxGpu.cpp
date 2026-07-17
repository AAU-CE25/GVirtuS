/*
 * UcxCommunicator CUDA dlopen helpers + GPUDirect probe + flag.
 *
 * Extracted from UcxCommunicator.cpp so the CUDA runtime resolver state
 * (once_flags + atomic function pointers) and the GPUDirect-active flag
 * live in one focused translation unit. The rest of the UCX communicator
 * touches CUDA only through the small public surface declared in
 * UcxInternal.h.
 *
 *   alloc_pinned_host / free_pinned_host  — for the RX pool
 *   alloc_gpu_slot / free_gpu_slot        — for the GPU shadow regions
 *   probe_gpudirect                        — startup check (used by init_ucx)
 *   set/get gpudirect_enabled             — process-wide flag
 *   is_gpu_pointer                         — used by the RMA fast path
 *
 * No static link to CUDA — libcudart is loaded via dlopen at first use.
 *
 * The cudaMalloc/cudaFree/cudaPointerGetAttributes dlsym resolver, the
 * process-wide gpudirect-enabled flag, and is_gpu_pointer's logic all live
 * in transport-agnostic gvirtus::communicators::{IsDevicePointer,
 * DeviceProbeEnabled,SetDeviceProbeEnabled,AllocDeviceMemory,
 * FreeDeviceMemory} (DeviceMemory.h), so that Buffer::Add() and the base
 * Communicator::WriteIov() can classify and bounce device pointers without
 * depending on any UCX header. Everything in this file that touches that
 * state is a thin forwarder to the shared primitive — same dlsym
 * candidates, same call_once caching, same short-circuit-on-disabled-gate
 * behaviour (UcxRma.cpp's is_gpu_pointer call site, init_ucx's
 * set_gpudirect_enabled call, etc. all see identical behaviour either way).
 */
#include "UcxInternal.h"

#include <gvirtus/communicators/DeviceMemory.h>

#include <cstdlib>
#include <dlfcn.h>
#include <malloc.h>
#include <mutex>
#include <atomic>

#include <ucp/api/ucp.h>

namespace gvirtus::communicators::ucx_internal {

namespace {

// ---- Pinned-host resolver (cudaHostAlloc / cudaFreeHost) ----------------
// Unrelated to the device-memory primitives (this allocates HOST memory
// that happens to be page-locked for fast DMA) — stays local to UCX, no
// other transport needs it.

using cudaHostAlloc_t = int (*)(void **, size_t, unsigned);
using cudaFreeHost_t  = int (*)(void *);

std::once_flag g_cuda_host_once;
std::atomic<cudaHostAlloc_t> g_cuda_host_alloc{nullptr};
std::atomic<cudaFreeHost_t>  g_cuda_free_host{nullptr};

void load_cuda_pinned_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto a = reinterpret_cast<cudaHostAlloc_t>(dlsym(h, "cudaHostAlloc"));
        auto f = reinterpret_cast<cudaFreeHost_t>(dlsym(h, "cudaFreeHost"));
        if (a && f) {
            g_cuda_host_alloc.store(a);
            g_cuda_free_host.store(f);
            ucx_debug_log("rx_pool: loaded cudaHostAlloc from %s", candidates[i]);
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("rx_pool: cudaHostAlloc unavailable, falling back to posix_memalign");
}

}  // namespace

// ---- Public API (see UcxInternal.h) -------------------------------------

unsigned char *alloc_pinned_host(std::size_t n, bool &is_cuda) {
    std::call_once(g_cuda_host_once, load_cuda_pinned_funcs);
    auto fn = g_cuda_host_alloc.load();
    if (fn != nullptr) {
        void *p = nullptr;
        if (fn(&p, n, /*cudaHostAllocDefault*/ 0u) == 0 && p != nullptr) {
            is_cuda = true;
            return static_cast<unsigned char *>(p);
        }
    }
    void *p = nullptr;
    if (posix_memalign(&p, 4096, n) == 0 && p != nullptr) {
        is_cuda = false;
        return static_cast<unsigned char *>(p);
    }
    return nullptr;
}

void free_pinned_host(unsigned char *p, bool is_cuda) {
    if (p == nullptr) return;
    if (is_cuda) {
        auto fn = g_cuda_free_host.load();
        if (fn != nullptr) {
            fn(p);
            return;
        }
    }
    std::free(p);
}

// alloc_gpu_slot / free_gpu_slot: thin forwarders to the shared
// AllocDeviceMemory/FreeDeviceMemory resolver (DeviceMemory.cpp) instead of
// keeping a second copy of the cudaMalloc/cudaFree dlsym logic.
unsigned char *alloc_gpu_slot(std::size_t n) {
    void *p = nullptr;
    if (!gvirtus::communicators::AllocDeviceMemory(&p, n)) return nullptr;
    return static_cast<unsigned char *>(p);
}

void free_gpu_slot(unsigned char *p) {
    gvirtus::communicators::FreeDeviceMemory(p);
}

// Thin forwarder to gvirtus::communicators::IsDevicePointer — see
// DeviceMemory.h for the (unchanged) short-circuit/dlsym behaviour.
bool is_gpu_pointer(const void *p) {
    return gvirtus::communicators::IsDevicePointer(p);
}

bool probe_gpudirect(ucp_context_h ctx, std::string &reason) {
    if (ctx == nullptr) {
        reason = "ucp_context is null";
        return false;
    }
    void *gpu = nullptr;
    if (!gvirtus::communicators::AllocDeviceMemory(&gpu, 4096)) {
        reason = "cudaMalloc(4K) failed (no GPU? OOM? libcudart missing?)";
        return false;
    }

    ucp_mem_map_params_t p{};
    p.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                    UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                    UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    p.address     = gpu;
    p.length      = 4096;
    p.memory_type = UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h memh = nullptr;
    ucs_status_t st = ucp_mem_map(ctx, &p, &memh);
    if (st != UCS_OK) {
        reason  = "ucp_mem_map(CUDA) failed: ";
        reason += ucs_status_string(st);
        gvirtus::communicators::FreeDeviceMemory(gpu);
        return false;
    }
    ucp_mem_unmap(ctx, memh);
    gvirtus::communicators::FreeDeviceMemory(gpu);
    reason.clear();
    return true;
}

// Thin forwarders to the shared atomic (DeviceMemory.cpp). Single source of
// truth for "is GPUDirect active" across UCX and Buffer::Add()'s gate.
void set_gpudirect_enabled(bool ok) { gvirtus::communicators::SetDeviceProbeEnabled(ok); }
bool gpudirect_enabled()             { return gvirtus::communicators::DeviceProbeEnabled(); }

}  // namespace gvirtus::communicators::ucx_internal
