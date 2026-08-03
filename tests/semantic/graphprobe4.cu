// graphprobe4.cu -- barrido fino del umbral. graphprobe3 solo probo 1024 y 4096;
// el limite exacto es una huella: identifica QUE mecanismo cambia de camino.
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
    std::printf("PROBE4,%8zu,end=%s,dentro=%s\n", nb, cudaGetErrorName(e), cudaGetErrorName(dentro));
    std::fflush(stdout);
    if (e == cudaSuccess && g) cudaGraphDestroy(g);
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s); cudaGetLastError();
}
int main(){
    cudaFree(0);
    for (size_t n : {(size_t)256,(size_t)512,(size_t)1024,(size_t)1536,(size_t)2048,
                     (size_t)2560,(size_t)3072,(size_t)3584,(size_t)4096,(size_t)6144,
                     (size_t)8192,(size_t)16384,(size_t)65536}) celda(n);
    return 0;
}
