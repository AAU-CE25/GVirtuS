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
 */
#include "UcxInternal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <malloc.h>
#include <mutex>

#include <ucp/api/ucp.h>

namespace gvirtus::communicators::ucx_internal {

namespace {

// ---- Pinned-host resolver (cudaHostAlloc / cudaFreeHost) ----------------

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

// ---- Device resolver (cudaMalloc / cudaFree / cudaPointerGetAttributes) -

using cudaMalloc_t = int (*)(void **, size_t);
using cudaFree_t   = int (*)(void *);
using cudaPointerGetAttributes_t = int (*)(void *, const void *);

// Mirror of cudaPointerAttributes (CUDA 11+ layout). `type` 0=Unregistered,
// 1=Host, 2=Device, 3=Managed. Only `type` is read; remaining fields kept for
// ABI alignment.
struct cudaPointerAttributes_layout {
    int   type;
    int   device;
    void *devicePointer;
    void *hostPointer;
};

std::once_flag                                g_cuda_dev_once;
std::atomic<cudaMalloc_t>                     g_cuda_malloc{nullptr};
std::atomic<cudaFree_t>                       g_cuda_free{nullptr};
std::atomic<cudaPointerGetAttributes_t>       g_cuda_pointer_attrs{nullptr};

void load_cuda_device_funcs() {
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
        if (m && f) {
            g_cuda_malloc.store(m);
            g_cuda_free.store(f);
            g_cuda_pointer_attrs.store(a);  // may be nullptr; is_gpu_pointer handles that
            ucx_debug_log("gpudirect: loaded cuda runtime symbols from %s "
                          "(pointer_attrs=%s)",
                          candidates[i], a ? "yes" : "no");
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("gpudirect: cudaMalloc/cudaFree unavailable (libcudart not found)");
}

// Global flag: true iff GVIRTUS_GPUDIRECT=1 and probe succeeded.
std::atomic<bool> g_gpudirect_enabled{false};

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

unsigned char *alloc_gpu_slot(std::size_t n) {
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto fn = g_cuda_malloc.load();
    if (fn == nullptr) return nullptr;
    void *p = nullptr;
    if (fn(&p, n) != 0 || p == nullptr) return nullptr;
    return static_cast<unsigned char *>(p);
}

void free_gpu_slot(unsigned char *p) {
    if (p == nullptr) return;
    auto fn = g_cuda_free.load();
    if (fn != nullptr) fn(p);
}

// CRITICAL: this function is called from both frontend AND backend
// (WriteIovRma runs on both sides). On the frontend, libcudart.so is the
// GVirtuS shim that REMOTES cudaPointerGetAttributes as an RPC — both slow
// and broken for our purposes (the frontend has no local GPU to ask
// about). The g_gpudirect_enabled short-circuit avoids that RPC storm:
// the frontend never has the env var set → returns false → no RPC. Only
// the backend (where GPUDirect probed OK) actually calls into the cuda
// runtime.
bool is_gpu_pointer(const void *p) {
    if (p == nullptr) return false;
    if (!g_gpudirect_enabled.load()) return false;
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto fn = g_cuda_pointer_attrs.load();
    if (fn == nullptr) return false;
    cudaPointerAttributes_layout attrs{};
    if (fn(&attrs, p) != 0) return false;
    return attrs.type == 2 /*Device*/ || attrs.type == 3 /*Managed*/;
}

bool probe_gpudirect(ucp_context_h ctx, std::string &reason) {
    if (ctx == nullptr) {
        reason = "ucp_context is null";
        return false;
    }
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto cmalloc = g_cuda_malloc.load();
    auto cfree   = g_cuda_free.load();
    if (cmalloc == nullptr || cfree == nullptr) {
        reason = "cudaMalloc/cudaFree symbols unavailable";
        return false;
    }

    void *gpu = nullptr;
    if (cmalloc(&gpu, 4096) != 0 || gpu == nullptr) {
        reason = "cudaMalloc(4K) failed (no GPU? OOM?)";
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
        cfree(gpu);
        return false;
    }
    ucp_mem_unmap(ctx, memh);
    cfree(gpu);
    reason.clear();
    return true;
}

void set_gpudirect_enabled(bool ok) { g_gpudirect_enabled.store(ok); }
bool gpudirect_enabled()             { return g_gpudirect_enabled.load(); }

}  // namespace gvirtus::communicators::ucx_internal
