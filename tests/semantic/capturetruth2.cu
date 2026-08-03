// capturetruth2.cu -- �Record o Query? y �que semantica tiene un nodo H2D capturado?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
static cudaStream_t g_otro=nullptr; static void *g_dev=nullptr,*g_dev2=nullptr,*g_pin=nullptr;
static cudaEvent_t g_ev=nullptr;
#define CASO(nombre, accion) do { \
    cudaStream_t s=nullptr; cudaGraph_t g=nullptr; cudaStreamCreate(&s); cudaGetLastError(); \
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); \
    cudaMemcpyAsync(g_dev,g_pin,256,cudaMemcpyHostToDevice,s); { accion; } \
    cudaError_t e=cudaStreamEndCapture(s,&g); \
    std::printf("%-46s %s\n", nombre, e==cudaSuccess?"OK  (no invalida)":cudaGetErrorName(e)); \
    if(e==cudaSuccess&&g) cudaGraphDestroy(g); cudaStreamDestroy(s); cudaGetLastError(); } while(0)

int main(){
    cudaFree(0); cudaStreamCreateWithFlags(&g_otro,cudaStreamNonBlocking);
    cudaMalloc(&g_dev,1<<20); cudaMalloc(&g_dev2,1<<20); cudaHostAlloc(&g_pin,1<<20,cudaHostAllocDefault);
    cudaEventCreateWithFlags(&g_ev,cudaEventDisableTiming);
    std::printf("=== eventos, desglosado ===\n");
    CASO("cudaEventRecord(OTRO) solo", cudaEventRecord(g_ev,g_otro));
    CASO("cudaEventQuery solo (ev ya grabado antes)", (void)cudaEventQuery(g_ev));
    CASO("cudaEventSynchronize solo", cudaEventSynchronize(g_ev));
    CASO("cudaEventRecord(stream por defecto 0)", cudaEventRecord(g_ev,0));

    std::printf("\n=== SEMANTICA de un nodo H2D capturado ===\n");
    // Se captura una copia H2D desde un buffer de host; luego se CAMBIA el buffer y se lanza.
    // Si el grafo lee en el LANZAMIENTO, se vera el valor nuevo. Si hace foto en la captura,
    // se vera el viejo. Esto decide si un staging en el backend puede ser correcto.
    unsigned char *h=nullptr; cudaHostAlloc((void**)&h,4096,cudaHostAllocDefault);
    std::memset(h,0xAA,4096);
    void *d=nullptr; cudaMalloc(&d,4096);
    cudaStream_t s=nullptr; cudaStreamCreate(&s); cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
    cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d,h,4096,cudaMemcpyHostToDevice,s);
    cudaError_t e=cudaStreamEndCapture(s,&g);
    std::printf("EndCapture=%s\n",cudaGetErrorName(e));
    if(e==cudaSuccess){
        cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
        std::memset(h,0xBB,4096);           // el origen CAMBIA despues de capturar
        cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
        unsigned char out=0; cudaMemcpy(&out,d,1,cudaMemcpyDeviceToHost);
        std::printf("tras lanzar con origen cambiado a 0xBB: device=0x%02X -> %s\n", out,
                    out==0xBB?"LEE EN EL LANZAMIENTO":"foto en la captura");
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    return 0;
}
