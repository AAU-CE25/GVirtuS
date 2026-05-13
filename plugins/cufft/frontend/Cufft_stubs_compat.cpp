/*
 * GVirtuS interop: compatibility stubs for CUFFT symbols that CuPy /
 * cuDF Python extension modules require at dlopen() time but the
 * GVirtuS cufft frontend does not implement.
 *
 * Each stub returns CUFFT_NOT_SUPPORTED (= 15) so the dynamic loader is
 * satisfied; callers either ignore the failure (rare) or fall back
 * to alternative code paths (most cuDF/CuPy code does this for Ex /
 * batched / strided / _64 variants).
 *
 * IMPORTANT: this file deliberately does NOT include the real
 * <cufft*.h> header. Including it would re-declare every cufftXxx
 * with its full signature, conflicting with our variadic stubs. The
 * dynamic loader only cares that the symbol exists; the actual
 * argument list does not have to match (we always ignore arguments
 * and return immediately).
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef int cufft_status_t_local;
#define GVS_CUFFT_NOT_SUPPORTED  15

#define GVS_CUFFT_STUB(name) \
    __attribute__((visibility("default"))) \
    cufft_status_t_local name(...) { return GVS_CUFFT_NOT_SUPPORTED; }

GVS_CUFFT_STUB(cufftXtExecDescriptorZ2Z)
GVS_CUFFT_STUB(cufftXtSetWorkArea)

GVS_CUFFT_STUB(cufftXtGetSizeMany)
#ifdef __cplusplus
}
#endif
