/*
 * GVirtuS interop: pass-through NVRTC frontend.
 *
 * NVRTC compiles CUDA C++ source to PTX/CUBIN — this is a CPU-bound
 * operation that does NOT touch the GPU. There is no benefit to
 * routing it across the network to the remote backend; in fact doing
 * so adds latency and requires implementing every NVRTC API on the
 * backend side.
 *
 * Instead, this frontend file replaces the previous Execute()-based
 * stubs with direct passthrough to the NVIDIA libnvrtc shipped inside
 * the runtime container (/usr/local/cuda/lib64/libnvrtc.so.12). The
 * compiled CUBIN that NVRTC produces is then transparently shipped to
 * the remote GPU by the next call to cuModuleLoadData, which DOES go
 * through GVirtuS.
 *
 * This file deliberately replaces (not augments) the legacy
 *   Nvrtc_compilation.cpp, Nvrtc_error.cpp, Nvrtc_giq.cpp,
 *   Nvrtc_host.cpp, Nvrtc_pch.cpp
 * to avoid multiple-definition link errors. The CMakeLists.txt has
 * been updated to compile only this file plus NvrtcFrontend.cpp.
 */

#include <dlfcn.h>
#include <cstring>
#include <stdio.h>
#include <stddef.h>

// Minimal NVRTC type/enum declarations — we deliberately avoid including
// the real <nvrtc.h> because some signatures use C++ types that pull in
// extra headers; opaque pointers are enough for forwarding.
typedef int   nvrtcResult;
typedef void *nvrtcProgram;

// Each passthrough function looks up the real symbol once and caches it
// in a static. dlopen is also cached. Thread-safe enough for the
// single-init pattern: even with a benign race, all racers compute the
// same address.

static void *gvs_nvrtc_handle = nullptr;

static void *gvs_nvrtc_dlsym(const char *sym) {
    if (!gvs_nvrtc_handle) {
        // Try common paths for the real NVIDIA NVRTC inside the container.
        const char *candidates[] = {
            "/usr/local/cuda/lib64/libnvrtc.so.12",
            "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvrtc.so.12",
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so.12",
            nullptr,
        };
        for (int i = 0; candidates[i] && !gvs_nvrtc_handle; ++i) {
            gvs_nvrtc_handle = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        }
        if (!gvs_nvrtc_handle) {
            fprintf(stderr,
                "[GVirtuS NVRTC passthrough] FATAL: cannot dlopen real "
                "libnvrtc.so.12 from any known path. Reason: %s\n",
                dlerror());
        }
    }
    void *p = gvs_nvrtc_handle ? dlsym(gvs_nvrtc_handle, sym) : nullptr;
    if (!p) {
        fprintf(stderr,
            "[GVirtuS NVRTC passthrough] symbol '%s' not found in real "
            "libnvrtc: %s\n", sym, dlerror());
    }
    return p;
}

#define NVRTC_FWD(name, sig_decl, sig_call) \
    extern "C" __attribute__((visibility("default"))) \
    nvrtcResult name sig_decl { \
        typedef nvrtcResult (*fn_t) sig_decl; \
        static fn_t real = nullptr; \
        if (!real) real = (fn_t)gvs_nvrtc_dlsym(#name); \
        if (!real) return -1; \
        return real sig_call; \
    }

// ---- Version / capability queries ----
NVRTC_FWD(nvrtcVersion,
          (int *major, int *minor),
          (major, minor))
NVRTC_FWD(nvrtcGetVersion,                       // CuPy uses this name
          (int *major, int *minor),
          (major, minor))
NVRTC_FWD(nvrtcGetNumSupportedArchs,
          (int *numArchs),
          (numArchs))
NVRTC_FWD(nvrtcGetSupportedArchs,
          (int *supportedArchs),
          (supportedArchs))

// ---- Program lifecycle ----
NVRTC_FWD(nvrtcCreateProgram,
          (nvrtcProgram *prog, const char *src, const char *name,
           int numHeaders, const char *const *headers,
           const char *const *includeNames),
          (prog, src, name, numHeaders, headers, includeNames))
NVRTC_FWD(nvrtcDestroyProgram,
          (nvrtcProgram *prog),
          (prog))
NVRTC_FWD(nvrtcCompileProgram,
          (nvrtcProgram prog, int numOptions, const char *const *options),
          (prog, numOptions, options))

// ---- Output queries (PTX / CUBIN / NVVM / LTOIR / OptiX IR) ----
NVRTC_FWD(nvrtcGetPTXSize,    (nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetPTX,        (nvrtcProgram p, char *o),   (p, o))
NVRTC_FWD(nvrtcGetCUBINSize,  (nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetCUBIN,      (nvrtcProgram p, char *o),   (p, o))
NVRTC_FWD(nvrtcGetNVVMSize,   (nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetNVVM,       (nvrtcProgram p, char *o),   (p, o))
NVRTC_FWD(nvrtcGetLTOIRSize,  (nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetLTOIR,      (nvrtcProgram p, char *o),   (p, o))
NVRTC_FWD(nvrtcGetOptiXIRSize,(nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetOptiXIR,    (nvrtcProgram p, char *o),   (p, o))

// ---- Diagnostic ----
NVRTC_FWD(nvrtcGetProgramLogSize, (nvrtcProgram p, size_t *s), (p, s))
NVRTC_FWD(nvrtcGetProgramLog,     (nvrtcProgram p, char *l),   (p, l))

// ---- Name expressions ----
NVRTC_FWD(nvrtcAddNameExpression,
          (nvrtcProgram prog, const char *nameExpression),
          (prog, nameExpression))
NVRTC_FWD(nvrtcGetLoweredName,
          (nvrtcProgram prog, const char *nameExpression,
           const char **loweredName),
          (prog, nameExpression, loweredName))

// ---- Error string (special: returns const char*, not nvrtcResult) ----
extern "C" __attribute__((visibility("default")))
const char *nvrtcGetErrorString(nvrtcResult result) {
    typedef const char *(*fn_t)(nvrtcResult);
    static fn_t real = nullptr;
    if (!real) real = (fn_t)gvs_nvrtc_dlsym("nvrtcGetErrorString");
    if (!real) return "GVirtuS: real libnvrtc unavailable";
    return real(result);
}
