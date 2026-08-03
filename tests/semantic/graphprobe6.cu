// graphprobe6.cu -- graph_ptds falla con UN hilo, asi que no es concurrencia: es algun
// elemento de su secuencia. Se anaden de uno en uno hasta que la ventana muera.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void kA(unsigned *p, size_t n, unsigned t){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]+=t; }
__global__ void kB(unsigned *p, size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=p[i]*2u+1u; }

static void celda(const char *nombre, int con_h2d, int con_kernel, int con_d2h, int con_upload) {
    const size_t nb = 1u<<20; const size_t n = nb/sizeof(unsigned);
    unsigned *hi=nullptr,*ho=nullptr; void *d=nullptr;
    cudaHostAlloc((void**)&hi,nb,cudaHostAllocDefault); cudaHostAlloc((void**)&ho,nb,cudaHostAllocDefault);
    cudaMalloc(&d,nb);
    cudaStream_t s=nullptr; cudaStreamCreate(&s); cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
    cudaGetLastError();
    cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);
    if (con_h2d)    cudaMemcpyAsync(d,hi,nb,cudaMemcpyHostToDevice,s);
    if (con_kernel) { kA<<<(int)((n+255)/256),256,0,s>>>((unsigned*)d,n,7u);
                      kB<<<(int)((n+255)/256),256,0,s>>>((unsigned*)d,n); }
    if (con_d2h)    cudaMemcpyAsync(ho,d,nb,cudaMemcpyDeviceToHost,s);
    cudaError_t e=cudaStreamEndCapture(s,&g);
    const char *up = "-";
    if (e==cudaSuccess && con_upload) {
        if (cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)==cudaSuccess)
            up = cudaGetErrorName(cudaGraphUpload(ge,s));
    }
    std::printf("PROBE6,%-26s end=%-34s upload=%s\n", nombre, cudaGetErrorName(e), up);
    std::fflush(stdout);
    if (ge) cudaGraphExecDestroy(ge);
    if (g)  cudaGraphDestroy(g);
    cudaFree(d); cudaFreeHost(hi); cudaFreeHost(ho); cudaStreamDestroy(s); cudaGetLastError();
}
int main(){
    cudaFree(0);
    celda("kernels only",        0,1,0,0);
    celda("H2D",                 1,0,0,0);
    celda("H2D+kernels",         1,1,0,0);
    celda("D2H only",            0,0,1,0);
    celda("H2D+kernels+D2H",     1,1,1,0);
    celda("all + GraphUpload",  1,1,1,1);
    return 0;
}
