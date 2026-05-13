/*
 * GVirtuS interop: compatibility stubs for CUBLAS symbols that CuPy /
 * cuDF Python extension modules require at dlopen() time but the
 * GVirtuS cublas frontend does not implement.
 *
 * Each stub returns CUBLAS_STATUS_NOT_SUPPORTED (= 15) so the dynamic loader is
 * satisfied; callers either ignore the failure (rare) or fall back
 * to alternative code paths (most cuDF/CuPy code does this for Ex /
 * batched / strided / _64 variants).
 *
 * IMPORTANT: this file deliberately does NOT include the real
 * <cublas*.h> header. Including it would re-declare every cublasXxx
 * with its full signature, conflicting with our variadic stubs. The
 * dynamic loader only cares that the symbol exists; the actual
 * argument list does not have to match (we always ignore arguments
 * and return immediately).
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef int cublas_status_t_local;
#define GVS_CUBLAS_NOT_SUPPORTED  15

#define GVS_CUBLAS_STUB(name) \
    __attribute__((visibility("default"))) \
    cublas_status_t_local name(...) { return GVS_CUBLAS_NOT_SUPPORTED; }

// ---- original 16 stubs (from earlier sessions) ----
GVS_CUBLAS_STUB(cublasCdgmm)
GVS_CUBLAS_STUB(cublasCgeam)
GVS_CUBLAS_STUB(cublasCgetriBatched)
GVS_CUBLAS_STUB(cublasDdgmm)
GVS_CUBLAS_STUB(cublasDgeam)
GVS_CUBLAS_STUB(cublasDgetriBatched)
GVS_CUBLAS_STUB(cublasDtpttr)
GVS_CUBLAS_STUB(cublasDtrttp)
GVS_CUBLAS_STUB(cublasSdgmm)
GVS_CUBLAS_STUB(cublasSgeam)
GVS_CUBLAS_STUB(cublasSgetriBatched)
GVS_CUBLAS_STUB(cublasStpttr)
GVS_CUBLAS_STUB(cublasStrttp)
GVS_CUBLAS_STUB(cublasZdgmm)
GVS_CUBLAS_STUB(cublasZgeam)
GVS_CUBLAS_STUB(cublasZgetriBatched)

// ---- _64 variants and Ex/cublasLt symbols (this session) ----
GVS_CUBLAS_STUB(cublasCcopy_v2_64)
GVS_CUBLAS_STUB(cublasCgeam_64)
GVS_CUBLAS_STUB(cublasCgemmStridedBatched_64)
GVS_CUBLAS_STUB(cublasCgemm_v2_64)
GVS_CUBLAS_STUB(cublasCgemvStridedBatched_64)
GVS_CUBLAS_STUB(cublasCgemv_v2_64)
GVS_CUBLAS_STUB(cublasCgerc_v2_64)
GVS_CUBLAS_STUB(cublasCgeru_v2_64)
GVS_CUBLAS_STUB(cublasChemm_v2_64)
GVS_CUBLAS_STUB(cublasChemv_v2_64)
GVS_CUBLAS_STUB(cublasCher2k_v2_64)
GVS_CUBLAS_STUB(cublasCscal_v2_64)
GVS_CUBLAS_STUB(cublasCsrot_v2_64)
GVS_CUBLAS_STUB(cublasCsscal_v2_64)
GVS_CUBLAS_STUB(cublasCswap_v2_64)
GVS_CUBLAS_STUB(cublasCsyrkx)
GVS_CUBLAS_STUB(cublasCtrmm_v2_64)
GVS_CUBLAS_STUB(cublasCtrmv_v2_64)
GVS_CUBLAS_STUB(cublasCtrsm_v2_64)
GVS_CUBLAS_STUB(cublasDcopy_v2_64)
GVS_CUBLAS_STUB(cublasDgeam_64)
GVS_CUBLAS_STUB(cublasDgemmStridedBatched_64)
GVS_CUBLAS_STUB(cublasDgemm_v2_64)
GVS_CUBLAS_STUB(cublasDgemvStridedBatched_64)
GVS_CUBLAS_STUB(cublasDgemv_v2_64)
GVS_CUBLAS_STUB(cublasDger_v2_64)
GVS_CUBLAS_STUB(cublasDrot_v2_64)
GVS_CUBLAS_STUB(cublasDscal_v2_64)
GVS_CUBLAS_STUB(cublasDswap_v2_64)
GVS_CUBLAS_STUB(cublasDsymm_v2_64)
GVS_CUBLAS_STUB(cublasDsymv_v2_64)
GVS_CUBLAS_STUB(cublasDsyr2k_v2_64)
GVS_CUBLAS_STUB(cublasDsyrkx)
GVS_CUBLAS_STUB(cublasDtrmm_v2_64)
GVS_CUBLAS_STUB(cublasDtrmv_v2_64)
GVS_CUBLAS_STUB(cublasDtrsm_v2_64)
GVS_CUBLAS_STUB(cublasLt)
GVS_CUBLAS_STUB(cublasLtCreate)
GVS_CUBLAS_STUB(cublasLtDestroy)
GVS_CUBLAS_STUB(cublasScopy_v2_64)
GVS_CUBLAS_STUB(cublasSetAtomicsMode)
GVS_CUBLAS_STUB(cublasSetMatrixAsync)
GVS_CUBLAS_STUB(cublasSgeam_64)
GVS_CUBLAS_STUB(cublasSgemmStridedBatched_64)
GVS_CUBLAS_STUB(cublasSgemm_v2_64)
GVS_CUBLAS_STUB(cublasSgemvStridedBatched_64)
GVS_CUBLAS_STUB(cublasSgemv_v2_64)
GVS_CUBLAS_STUB(cublasSger_v2_64)
GVS_CUBLAS_STUB(cublasSrot_v2_64)
GVS_CUBLAS_STUB(cublasSscal_v2_64)
GVS_CUBLAS_STUB(cublasSswap_v2_64)
GVS_CUBLAS_STUB(cublasSsymm_v2_64)
GVS_CUBLAS_STUB(cublasSsymv_v2_64)
GVS_CUBLAS_STUB(cublasSsyr2k_v2_64)
GVS_CUBLAS_STUB(cublasSsyrkx)
GVS_CUBLAS_STUB(cublasStrmm_v2_64)
GVS_CUBLAS_STUB(cublasStrmv_v2_64)
GVS_CUBLAS_STUB(cublasStrsm_v2_64)
GVS_CUBLAS_STUB(cublasZcopy_v2_64)
GVS_CUBLAS_STUB(cublasZdrot_v2_64)
GVS_CUBLAS_STUB(cublasZdscal_v2_64)
GVS_CUBLAS_STUB(cublasZgeam_64)
GVS_CUBLAS_STUB(cublasZgemmStridedBatched_64)
GVS_CUBLAS_STUB(cublasZgemm_v2_64)
GVS_CUBLAS_STUB(cublasZgemvStridedBatched_64)
GVS_CUBLAS_STUB(cublasZgemv_v2_64)
GVS_CUBLAS_STUB(cublasZgerc_v2_64)
GVS_CUBLAS_STUB(cublasZgeru_v2_64)
GVS_CUBLAS_STUB(cublasZhemm_v2_64)
GVS_CUBLAS_STUB(cublasZhemv_v2_64)
GVS_CUBLAS_STUB(cublasZher2k_v2_64)
GVS_CUBLAS_STUB(cublasZscal_v2_64)
GVS_CUBLAS_STUB(cublasZswap_v2_64)
GVS_CUBLAS_STUB(cublasZsyrkx)
GVS_CUBLAS_STUB(cublasZtrmm_v2_64)
GVS_CUBLAS_STUB(cublasZtrmv_v2_64)
GVS_CUBLAS_STUB(cublasZtrsm_v2_64)

#ifdef __cplusplus
}
#endif
