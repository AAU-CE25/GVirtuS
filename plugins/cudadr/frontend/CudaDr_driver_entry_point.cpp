/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,
 *             Department of Electronic Systems, Aalborg University, Denmark
 */

#include "CudaDr.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <string>

// cuda.h defines:  #define cuGetProcAddress cuGetProcAddress_v2
// Undefine it so our function is exported under the base name "cuGetProcAddress".
// We then provide cuGetProcAddress_v2 as an alias below, covering both names.
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

using namespace std;

/*
 * cuGetProcAddress — resolve CUDA driver symbols to LOCAL function pointers.
 *
 * This function MUST NOT forward to the backend.
 *
 * The CUDA runtime calls cuGetProcAddress to discover driver API functions and
 * then calls the returned function pointer DIRECTLY in this process.  If we
 * forwarded to the backend, the backend would return a pointer into its own
 * address space, which is meaningless here — calling it causes garbage data,
 * which the runtime detects as an inconsistency and raises:
 *
 *   cudaErrorCallRequiresNewerDriver (error 36)
 *
 * Correct implementation: use dlopen on our own libcuda.so and dlsym to find
 * the GVirtuS frontend function, then return that local pointer.
 *
 * Versioned-name fallback:
 *   The runtime may request "cuDeviceGetUuid_v2" while we export "cuDeviceGetUuid",
 *   or request "cuCtxCreate" while we export "cuCtxCreate_v3".  We try:
 *     1. Exact name
 *     2. Strip _v2 / _v3 / _v4 suffix
 *     3. Append _v2 suffix
 */
extern "C" CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                                     cuuint64_t flags,
                                     CUdriverProcAddressQueryResult* symbolStatus) {
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
    *pfn = nullptr;

    // Find the path of this shared library (libcuda.so) using a known symbol.
    Dl_info di;
    void* self = nullptr;
    if (dladdr((void*)cuGetProcAddress, &di) && di.dli_fname) {
        // RTLD_NOLOAD: already mapped — don't re-load, just get a handle.
        self = dlopen(di.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }

    // Lookup helper: try self first, then fall back to whole-process symbol table.
    auto lookup = [&](const char* name) -> void* {
        void* fn = self ? dlsym(self, name) : nullptr;
        if (!fn) fn = dlsym(RTLD_DEFAULT, name);
        return fn;
    };

    string sym(symbol);
    void* fn = nullptr;

    // 1. Exact match (covers most functions and already-versioned exports like cuCtxCreate_v2).
    fn = lookup(sym.c_str());

    // 2. Strip _v2 / _v3 / _v4 — runtime requests versioned name but we export base name.
    if (!fn) {
        for (const char* suf : {"_v2", "_v3", "_v4"}) {
            size_t n = strlen(suf);
            if (sym.size() > n && sym.compare(sym.size() - n, n, suf) == 0) {
                fn = lookup(sym.substr(0, sym.size() - n).c_str());
                if (fn) break;
            }
        }
    }

    // 3. Append _v2 — runtime requests base name but we export the versioned form.
    if (!fn) {
        fn = lookup((sym + "_v2").c_str());
    }

    if (self) dlclose(self);

    if (fn) {
        *pfn = fn;
        if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
        return CUDA_SUCCESS;
    }

    // Symbol not implemented in GVirtuS frontend.
    // Return CUDA_SUCCESS + NOT_FOUND status — this is correct: returning an
    // error code would make the runtime think the driver itself is broken.
    fprintf(stderr, "[GVirtuS] cuGetProcAddress: unresolved '%s' (cudaVersion=%d)\n",
            symbol, cudaVersion);
    if (symbolStatus) *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}

// CUDA 11.3+ introduced cuGetProcAddress_v2 with the same signature.
extern "C" CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                                        cuuint64_t flags,
                                        CUdriverProcAddressQueryResult* symbolStatus)
    __attribute__((alias("cuGetProcAddress")));


