// graphprobe2.cu -- �invalida la captura una ASIGNACION perezosa, y no la copia?
// Hipotesis: la sombra de GPU (y el pool) se materializan en la primera transferencia grande.
// Si esa materializacion cae DENTRO de la ventana de captura, el cudaMalloc/cudaHostAlloc del
// backend la invalida -- en modo ThreadLocal, una asignacion en el mismo hilo lo hace.
// Prueba: calentar con una copia grande ANTES de capturar. Si entonces pasa, la copia nunca fue
// el problema y el arreglo es "no asignar dentro de la captura".
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(unsigned char*p,size_t n){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i<n)p[i]++;}

static void celda(const char *nombre, int calentar) {
    const size_t nb = 1u<<20;
    unsigned char *h=nullptr; void *d=nullptr;
    cudaStream_t s=nullptr; cudaGraph_t g=nullptr;
    cudaHostAlloc((void**)&h, nb, cudaHostAllocDefault);
    cudaMalloc(&d, nb);
    cudaStreamCreate(&s);
    if (calentar) {                    // fuerza la materializacion FUERA de la captura
        cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
    }
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    cudaError_t dentro = cudaGetLastError();
    cudaError_t e = cudaStreamEndCapture(s, &g);
    std::printf("PROBE2,%s,warmed=%d,end=%s,inside=%s\n",
                nombre, calentar, cudaGetErrorName(e), cudaGetErrorName(dentro));
    std::fflush(stdout);
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s);
}
int main(){
    cudaFree(0);
    celda("sin_calentar", 0);
    celda("con_calentar", 1);
    celda("segunda_sin_calentar", 0);   // ya calentado por la anterior
    return 0;
}
