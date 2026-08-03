// capturetruth.cu -- NATIVO. Tabla de verdad: que llamadas invalidan de verdad una captura
// ThreadLocal en ESTE driver. Todo el diseno del arreglo se apoya en esta tabla, asi que se
// mide en vez de deducirla del manual.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static cudaStream_t g_otro = nullptr;   // stream interno, NO capturado
static void *g_dev = nullptr, *g_dev2 = nullptr;
static void *g_pin = nullptr;

#define CASO(nombre, accion)                                                  \
    do {                                                                      \
        cudaStream_t s = nullptr; cudaGraph_t g = nullptr;                    \
        cudaStreamCreate(&s); cudaGetLastError();                             \
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);          \
        cudaMemcpyAsync(g_dev, g_pin, 256, cudaMemcpyHostToDevice, s);        \
        { accion; }                                                           \
        cudaError_t e = cudaStreamEndCapture(s, &g);                          \
        std::printf("%-46s %s\n", nombre,                                     \
            e == cudaSuccess ? "OK  (no invalida)" : cudaGetErrorName(e));    \
        if (e == cudaSuccess && g) cudaGraphDestroy(g);                       \
        cudaStreamDestroy(s); cudaGetLastError();                             \
    } while (0)

int main() {
    cudaFree(0);
    cudaStreamCreateWithFlags(&g_otro, cudaStreamNonBlocking);
    cudaMalloc(&g_dev, 1 << 20); cudaMalloc(&g_dev2, 1 << 20);
    cudaHostAlloc(&g_pin, 1 << 20, cudaHostAllocDefault);
    std::printf("=== TABLA DE VERDAD DE CAPTURA (ThreadLocal, hilo que captura) ===\n");

    CASO("nada (control)", (void)0);
    CASO("cudaMalloc / cudaFree", { void*p=nullptr; cudaMalloc(&p,4096); cudaFree(p); });
    CASO("cudaMalloc solo", { void*p=nullptr; cudaMalloc(&p,4096); (void)p; });
    CASO("cudaFree solo", { void*p=nullptr; cudaMalloc(&p,4096); cudaStreamEndCapture; cudaFree(p); });
    CASO("cudaHostAlloc solo", { void*p=nullptr; cudaHostAlloc(&p,4096,0); (void)p; });
    CASO("cudaFreeHost solo", { void*p=nullptr; cudaHostAlloc(&p,4096,0); cudaFreeHost(p); });
    CASO("malloc/free (host puro)", { void*p=std::malloc(4096); std::free(p); });
    CASO("cudaMemcpyAsync en OTRO stream", cudaMemcpyAsync(g_dev2,g_pin,4096,cudaMemcpyHostToDevice,g_otro));
    CASO("cudaMemcpyAsync D2D en OTRO stream", cudaMemcpyAsync(g_dev2,g_dev,4096,cudaMemcpyDeviceToDevice,g_otro));
    CASO("cudaStreamSynchronize(OTRO stream)", cudaStreamSynchronize(g_otro));
    CASO("cudaDeviceSynchronize", cudaDeviceSynchronize());
    CASO("cudaStreamCreateWithFlags/Destroy", { cudaStream_t x; cudaStreamCreateWithFlags(&x,cudaStreamNonBlocking); cudaStreamDestroy(x); });
    CASO("cudaEventCreateWithFlags/Destroy", { cudaEvent_t x; cudaEventCreateWithFlags(&x,cudaEventDisableTiming); cudaEventDestroy(x); });
    CASO("cudaEventRecord(OTRO)+Query hasta listo", {
        cudaEvent_t x; cudaEventCreateWithFlags(&x,cudaEventDisableTiming);
        cudaMemcpyAsync(g_dev2,g_dev,4096,cudaMemcpyDeviceToDevice,g_otro);
        cudaEventRecord(x,g_otro);
        while (cudaEventQuery(x)==cudaErrorNotReady) {}
        cudaEventDestroy(x); });
    CASO("cudaPointerGetAttributes", { cudaPointerAttributes a; cudaPointerGetAttributes(&a,g_dev); });
    CASO("cudaGetLastError", cudaGetLastError());
    CASO("cudaMemcpy SINCRONO (H2D)", cudaMemcpy(g_dev2,g_pin,4096,cudaMemcpyHostToDevice));
    return 0;
}
