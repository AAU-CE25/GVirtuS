// graphprobe.cu -- que operacion invalida la captura de streams a traves del remoting.
// Tres capturas identicas salvo por lo que contienen. Si solo falla la del memcpy, el defecto
// esta acotado a esa operacion y es arreglable; si fallan todas, la captura entera no sobrevive.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(unsigned char*p,size_t n){size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i<n)p[i]++;}

static int prueba(const char *nombre, int con_kernel, int con_memcpy) {
    const size_t nb = 1u<<20;
    unsigned char *h=nullptr; void *d=nullptr;
    cudaStream_t s=nullptr; cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
    cudaHostAlloc((void**)&h, nb, cudaHostAllocDefault);
    cudaMalloc(&d, nb);
    cudaStreamCreate(&s);
    cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) { std::printf("PROBE,%s,begin,%s\n", nombre, cudaGetErrorName(e)); return 1; }
    if (con_memcpy) cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    if (con_kernel) k<<<(unsigned)((nb+255)/256),256,0,s>>>((unsigned char*)d, nb);
    cudaError_t dentro = cudaGetLastError();
    e = cudaStreamEndCapture(s, &g);
    std::printf("PROBE,%s,end,%s,inside=%s\n", nombre, cudaGetErrorName(e), cudaGetErrorName(dentro));
    if (e == cudaSuccess) {
        cudaError_t i2 = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        cudaError_t l  = ge ? cudaGraphLaunch(ge, s) : cudaErrorUnknown;
        cudaError_t sy = cudaStreamSynchronize(s);
        std::printf("PROBE,%s,run,inst=%s,launch=%s,sync=%s\n", nombre,
                    cudaGetErrorName(i2), cudaGetErrorName(l), cudaGetErrorName(sy));
    }
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s);
    return e != cudaSuccess;
}
int main(){
    cudaFree(0);
    prueba("solo_kernel", 1, 0);
    prueba("solo_memcpy", 0, 1);
    prueba("kernel+memcpy", 1, 1);
    prueba("vacia", 0, 0);
    return 0;
}
