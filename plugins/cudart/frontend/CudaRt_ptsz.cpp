/*
 * Per-thread default stream (PTDS) entry points.
 *
 * Cuando una unidad de traduccion se compila con `nvcc --default-stream per-thread`, las
 * cabeceras de CUDA sustituyen cada llamada por su variante `_ptsz`:
 *
 *     cudaEventRecord(e, s)   ->   cudaEventRecord_ptsz(e, s)
 *
 * La UNICA diferencia semantica entre `X` y `X_ptsz` es como se interpreta el stream por
 * defecto: en `X`, un `0` significa el stream legacy (compartido por todo el contexto); en
 * `X_ptsz`, un `0` significa el stream por defecto DE ESTE HILO.
 *
 * Antes de este fichero, 49 variantes `_ptsz` vivian en la tabla auto-generada de
 * CudaRt_stubs_compat.cpp y devolvian cudaErrorNotSupported (71) SIN registrar nada (STUB_LOG
 * se compila a nada salvo que se defina GVIRTUS_LOG_STUB_CALLS). Una aplicacion compilada con
 * PTDS que llamase a cudaEventRecord, cudaStreamWaitEvent o cudaStreamBeginCapture recibia un
 * 71 silencioso.
 *
 * Por que basta con reenviar:
 *
 *   1. El centinela cudaStreamPerThread (0x2) SI viaja correctamente. El frontend lo envia
 *      con AddDevicePointerForArguments y el backend lo pasa tal cual a la CUDA runtime.
 *   2. El backend usa un modelo de UN HILO POR CONEXION -- el hilo que atiende a un cliente
 *      es el mismo durante toda su vida (src/backend/Process.cpp: "one thread owns one
 *      client"). Como GVirtuS abre una conexion por hilo del cliente
 *      (Frontend::GetFrontend indexa por tid), el hilo T del cliente tiene un hilo T' del
 *      backend dedicado y estable.
 *   3. Por tanto "el stream por defecto del hilo T del cliente" se corresponde exactamente
 *      con "el stream por defecto del hilo T' del backend", y 0x2 es la traduccion correcta,
 *      no una coincidencia.
 *
 * Lo unico que hay que hacer en la frontera es sustituir el defecto: un `0` que llega por una
 * entrada `_ptsz` significa per-thread, y hay que convertirlo en 0x2 ANTES de mandarlo, o el
 * backend lo interpretara como el stream legacy. Los cinco `_ptsz` que ya existian
 * (cudaStreamSynchronize, cudaStreamQuery, cudaMemcpyAsync, cudaMemsetAsync,
 * cudaLaunchKernel) reenviaban sin esta sustitucion, de modo que el trabajo de un llamante
 * PTDS que usara el stream por defecto acababa en el stream legacy del backend --
 * compartido por TODOS los hilos cliente. Eso queda corregido aqui tambien.
 */

#include "CudaRt.h"

namespace {
// Sustitucion del defecto en la frontera `_ptsz`. Solo el 0 cambia; cualquier handle
// explicito (incluido 0x1 legacy y 0x2 per-thread) se respeta tal cual.
inline cudaStream_t ptsz_default(cudaStream_t s) {
    return (s == nullptr) ? cudaStreamPerThread : s;
}
}  // namespace

extern "C" {

// --- streams -------------------------------------------------------------------------
__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event,
                                                        unsigned int flags) {
    return cudaStreamWaitEvent(ptsz_default(stream), event, flags);
}

__host__ cudaError_t CUDARTAPI cudaStreamAddCallback_ptsz(cudaStream_t stream,
                                                          cudaStreamCallback_t callback,
                                                          void *userData, unsigned int flags) {
    return cudaStreamAddCallback(ptsz_default(stream), callback, userData, flags);
}

__host__ cudaError_t CUDARTAPI cudaStreamGetPriority_ptsz(cudaStream_t hStream, int *priority) {
    return cudaStreamGetPriority(ptsz_default(hStream), priority);
}

__host__ cudaError_t CUDARTAPI cudaStreamBeginCapture_ptsz(cudaStream_t stream,
                                                           cudaStreamCaptureMode mode) {
    return cudaStreamBeginCapture(ptsz_default(stream), mode);
}

__host__ cudaError_t CUDARTAPI cudaStreamEndCapture_ptsz(cudaStream_t stream,
                                                         cudaGraph_t *pGraph) {
    return cudaStreamEndCapture(ptsz_default(stream), pGraph);
}

__host__ cudaError_t CUDARTAPI cudaStreamIsCapturing_ptsz(cudaStream_t stream,
                                                          cudaStreamCaptureStatus *pStatus) {
    return cudaStreamIsCapturing(ptsz_default(stream), pStatus);
}

// --- eventos -------------------------------------------------------------------------
__host__ cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event,
                                                             cudaStream_t stream,
                                                             unsigned int flags) {
    return cudaEventRecordWithFlags(event, ptsz_default(stream), flags);
}

// --- graphs --------------------------------------------------------------------------
__host__ cudaError_t CUDARTAPI cudaGraphLaunch_ptsz(cudaGraphExec_t graphExec,
                                                    cudaStream_t stream) {
    return cudaGraphLaunch(graphExec, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaGraphUpload_ptsz(cudaGraphExec_t graphExec,
                                                    cudaStream_t stream) {
    return cudaGraphUpload(graphExec, ptsz_default(stream));
}

// --- memoria asincrona ---------------------------------------------------------------
__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src,
                                                      size_t spitch, size_t width, size_t height,
                                                      cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                             ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms *p,
                                                      cudaStream_t stream) {
    return cudaMemcpy3DAsync(p, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync_ptsz(const void *symbol, const void *src,
                                                            size_t count, size_t offset,
                                                            cudaMemcpyKind kind,
                                                            cudaStream_t stream) {
    return cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync_ptsz(void *dst, const void *symbol,
                                                              size_t count, size_t offset,
                                                              cudaMemcpyKind kind,
                                                              cudaStream_t stream) {
    return cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value,
                                                      size_t width, size_t height,
                                                      cudaStream_t stream) {
    return cudaMemset2DAsync(devPtr, pitch, value, width, height, ptsz_default(stream));
}

__host__ cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void **devPtr, size_t size,
                                                    cudaStream_t hStream) {
    return cudaMallocAsync(devPtr, size, ptsz_default(hStream));
}

__host__ cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void *devPtr, cudaStream_t hStream) {
    return cudaFreeAsync(devPtr, ptsz_default(hStream));
}

}  // extern "C"
