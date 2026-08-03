// capturetruth3.cu -- casos que decide el diseno del arreglo.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
static cudaStream_t g_otro=nullptr; static void *g_dev=nullptr,*g_pin=nullptr; static void *g_pag=nullptr;
#define CASO(nombre, accion) do { \
    cudaStream_t s=nullptr; cudaGraph_t g=nullptr; cudaStreamCreate(&s); cudaGetLastError(); \
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); \
    cudaMemcpyAsync(g_dev,g_pin,256,cudaMemcpyHostToDevice,s); { accion; } \
    cudaError_t e=cudaStreamEndCapture(s,&g); \
    std::printf("%-52s %s\n", nombre, e==cudaSuccess?"OK  (no invalida)":cudaGetErrorName(e)); \
    if(e==cudaSuccess&&g) cudaGraphDestroy(g); cudaStreamDestroy(s); cudaGetLastError(); } while(0)
int main(){
    cudaFree(0); cudaStreamCreateWithFlags(&g_otro,cudaStreamNonBlocking);
    cudaMalloc(&g_dev,16<<20); cudaHostAlloc(&g_pin,16<<20,cudaHostAllocDefault);
    g_pag = std::malloc(16<<20);
    CASO("cudaStreamQuery(OTRO stream)", (void)cudaStreamQuery(g_otro));
    CASO("cudaMemcpyAsync D2H a dst PAGINABLE en OTRO stream",
         cudaMemcpyAsync(g_pag,g_dev,8<<20,cudaMemcpyDeviceToHost,g_otro));
    CASO("cudaMemcpyAsync D2H a dst FIJADO en OTRO stream",
         cudaMemcpyAsync(g_pin,g_dev,8<<20,cudaMemcpyDeviceToHost,g_otro));
    CASO("cudaLaunchHostFunc(OTRO stream)", {
        cudaLaunchHostFunc(g_otro, [](void*){}, nullptr); });
    // �Bloquea el D2H a paginable? (si bloquea, da la espera exacta gratis)
    cudaDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now();
    cudaMemcpyAsync(g_pag,g_dev,8<<20,cudaMemcpyDeviceToHost,g_otro);
    auto t1=std::chrono::steady_clock::now();
    cudaStreamSynchronize(g_otro);
    auto t2=std::chrono::steady_clock::now();
    auto us=[](auto a,auto b){return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count();};
    std::printf("\nD2H 8MiB a paginable: retorno de la llamada=%ldus, sync posterior=%ldus -> %s\n",
                us(t0,t1), us(t1,t2), us(t0,t1) > 10*us(t1,t2) ? "BLOQUEA (espera gratis)" : "NO bloquea");
    return 0;
}
