/*
 * GVirtuS interop: compatibility shims for cuRAND symbols that CuPy's
 * cupy_backends/cuda/libs/curand.cpython-310.so requires at dlopen()
 * time but the GVirtuS curand frontend does not yet implement.
 *
 * Only 3 symbols are missing today:
 *   curandGetVersion              — must return a sensible version,
 *                                   otherwise CuPy treats curand as
 *                                   broken and aborts at import time
 *   curandSetGeneratorOrdering    — pseudo/quasi RNG ordering hint,
 *                                   safely ignored
 *   curandSetStream               — associates a cudaStream_t with the
 *                                   generator; we pretend success so
 *                                   CuPy doesn't fail. The Generate*
 *                                   calls already route through the
 *                                   real GVirtuS curand handlers, and
 *                                   the GPU side knows which stream is
 *                                   bound implicitly via cudaSetDevice.
 *
 * Variadic signatures avoid needing the real <curand.h> (which would
 * conflict with the rest of the GVirtuS curand frontend). The dynamic
 * loader only requires the symbol to exist.
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef int gvs_curand_status_t;
#define GVS_CURAND_SUCCESS         0
#define GVS_CURAND_NOT_SUPPORTED   102

// libcurand 10.3.x ships with cuda-12.6 — report 10301 so CuPy's version
// check passes (anything >= 10000 is accepted).
__attribute__((visibility("default")))
gvs_curand_status_t curandGetVersion(int *version) {
    if (version) *version = 10301;
    return GVS_CURAND_SUCCESS;
}

// CuPy may call this with CURAND_ORDERING_PSEUDO_DEFAULT for new
// generators. Accept and ignore — the underlying generator behavior is
// unchanged.
__attribute__((visibility("default")))
gvs_curand_status_t curandSetGeneratorOrdering(...) {
    return GVS_CURAND_SUCCESS;
}

// CuPy calls this once per generator to bind the user-facing CuPy
// stream. We accept and ignore — the GVirtuS curand backend uses the
// device's default stream for all generator ops, which is sufficient
// for correctness albeit not for stream-overlapped throughput.
__attribute__((visibility("default")))
gvs_curand_status_t curandSetStream(...) {
    return GVS_CURAND_SUCCESS;
}

#ifdef __cplusplus
}
#endif
