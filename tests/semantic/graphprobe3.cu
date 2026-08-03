// graphprobe3.cu -- �depende del TAMANO (y por tanto del camino RMA) o pasa siempre?
// Debajo del suelo RMA (8 KiB) la copia viaja por mensajes activos y no hay reserva de slot,
// ni registro de memoria, ni sombra. Si la captura sobrevive ahi y muere arriba, el culpable
// esta en el camino RMA de recepcion, no en la copia.
#include <cstdio>
#include <cuda_runtime.h>
static void celda(size_t nb) {
    unsigned char *h=nullptr; void *d=nullptr; cudaStream_t s=nullptr; cudaGraph_t g=nullptr;
    cudaHostAlloc((void**)&h, nb, cudaHostAllocDefault); cudaMalloc(&d, nb); cudaStreamCreate(&s);
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    cudaError_t dentro = cudaGetLastError();
    cudaError_t e = cudaStreamEndCapture(s, &g);
    std::printf("PROBE3,%8zu,end=%s,dentro=%s\n", nb, cudaGetErrorName(e), cudaGetErrorName(dentro));
    std::fflush(stdout);
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s);
}
int main(){ cudaFree(0); for (size_t n : {(size_t)1024,(size_t)4096,(size_t)8192,(size_t)65536,(size_t)1048576}) celda(n); return 0; }
