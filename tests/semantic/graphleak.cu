// graphleak.cu -- la cadena captura->grafo->ejecutable no estaba conectada: los buffers de
// staging no se liberaban NUNCA. Ahora GraphDestroy/GraphExecDestroy sueltan su referencia.
// Esto lo comprueba: 100 ciclos de capturar/instanciar/lanzar/destruir con 4 MiB de entrada y
// 4 MiB de salida. Si el lote se filtrara, el backend crece ~800 MiB.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
int main(){
    const size_t nb = 4u<<20;
    cudaFree(0);
    unsigned char *hi=nullptr,*ho=nullptr; void *d=nullptr;
    cudaHostAlloc((void**)&hi,nb,cudaHostAllocDefault);
    cudaHostAlloc((void**)&ho,nb,cudaHostAllocDefault);
    cudaMalloc(&d,nb);
    int malos=0;
    for (int it=0; it<100; ++it) {
        std::memset(hi,(unsigned char)(it&0xFF),nb);
        cudaStream_t s=nullptr; cudaStreamCreate(&s);
        cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
        cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);
        cudaMemcpyAsync(d,hi,nb,cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(ho,d,nb,cudaMemcpyDeviceToHost,s);
        cudaError_t e=cudaStreamEndCapture(s,&g);
        if (e!=cudaSuccess) { ++malos; }
        else if (cudaGraphInstantiate(&ge,g,nullptr,nullptr,0)==cudaSuccess) {
            cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            if (ho[0]!=(unsigned char)(it&0xFF) || ho[nb-1]!=(unsigned char)(it&0xFF)) ++malos;
        } else ++malos;
        if (ge) cudaGraphExecDestroy(ge);
        if (g)  cudaGraphDestroy(g);
        cudaStreamDestroy(s); cudaGetLastError();
        if ((it+1)%25==0) { std::printf("LEAK,iter=%d bad=%d\n",it+1,malos); std::fflush(stdout); }
    }
    std::printf("LEAK,END bad=%d\n",malos);
    cudaFree(d); cudaFreeHost(hi); cudaFreeHost(ho);
    return malos==0?0:1;
}
