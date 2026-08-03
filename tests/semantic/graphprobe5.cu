// graphprobe5.cu -- �es un efecto de PRIMER TOQUE por tamano, o pasa en cada mensaje?
// Antes de capturar se hace una copia IDENTICA fuera de la ventana. Si eso lo arregla,
// el culpable es una accion que solo ocurre la primera vez que se ve ese tamano
// (crecer el slot, registrar memoria); si no, es algo que ocurre en cada mensaje.
#include <cstdio>
#include <cuda_runtime.h>
static void celda(size_t nb, int warm) {
    unsigned char *h=nullptr; void *d=nullptr; cudaStream_t s=nullptr; cudaGraph_t g=nullptr;
    cudaHostAlloc((void**)&h, nb, cudaHostAllocDefault); cudaMalloc(&d, nb); cudaStreamCreate(&s);
    if (warm) { cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s); cudaStreamSynchronize(s); }
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    cudaError_t e = cudaStreamEndCapture(s, &g);
    std::printf("PROBE5,warm=%d,%8zu,end=%s\n", warm, nb, cudaGetErrorName(e));
    std::fflush(stdout);
    if (e == cudaSuccess && g) cudaGraphDestroy(g);
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s); cudaGetLastError();
}
int main(int argc, char**argv){
    int warm = (argc > 1) ? atoi(argv[1]) : 1;
    cudaFree(0);
    for (size_t n : {(size_t)1024,(size_t)1536,(size_t)4096,(size_t)65536}) celda(n, warm);
    // segunda pasada: los tamanos ya se han visto TODOS
    std::printf("-- segunda pasada --\n");
    for (size_t n : {(size_t)1024,(size_t)1536,(size_t)4096,(size_t)65536}) celda(n, 0);
    return 0;
}
