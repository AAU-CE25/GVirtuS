// graphsem.cu -- �el grafo capturado LEE EN EL LANZAMIENTO, tambien sobre GVirtuS?
// Nativo ya dice que si. Si GVirtuS hace foto en la captura, EndCapture=cudaSuccess es
// una senal verde sobre algo roto: el relanzamiento copia bytes viejos.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
int main(){
    const size_t N = 1024;
    cudaFree(0);
    unsigned char *h=nullptr; cudaHostAlloc((void**)&h,N,cudaHostAllocDefault);
    std::memset(h,0xAA,N);
    void *d=nullptr; cudaMalloc(&d,N);
    cudaStream_t s=nullptr; cudaStreamCreate(&s);
    cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
    cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d,h,N,cudaMemcpyHostToDevice,s);
    cudaError_t e=cudaStreamEndCapture(s,&g);
    std::printf("SEM,EndCapture=%s\n",cudaGetErrorName(e));
    if(e!=cudaSuccess) return 1;
    cudaError_t ei=cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
    std::printf("SEM,Instantiate=%s\n",cudaGetErrorName(ei));
    if(ei!=cudaSuccess) return 1;
    // lanzamiento 1: origen sigue en 0xAA
    cudaError_t el=cudaGraphLaunch(ge,s);
    cudaError_t es=cudaStreamSynchronize(s);
    unsigned char out=0; cudaMemcpy(&out,d,1,cudaMemcpyDeviceToHost);
    std::printf("SEM,launch1=%s sync=%s device=0x%02X (esperado 0xAA)\n",
                cudaGetErrorName(el),cudaGetErrorName(es),out);
    // lanzamiento 2: el origen CAMBIA
    std::memset(h,0xBB,N);
    cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
    out=0; cudaMemcpy(&out,d,1,cudaMemcpyDeviceToHost);
    std::printf("SEM,launch2 device=0x%02X -> %s\n", out,
                out==0xBB?"CORRECTO (lee en el lanzamiento)":"INCORRECTO (bytes viejos)");
    return 0;
}
