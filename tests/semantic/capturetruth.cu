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
            e == cudaSuccess ? "OK  (does not invalidate)" : cudaGetErrorName(e));    \
        if (e == cudaSuccess && g) cudaGraphDestroy(g);                       \
        cudaStreamDestroy(s); cudaGetLastError();                             \
    } while (0)

int main() {
    cudaFree(0);
    cudaStreamCreateWithFlags(&g_otro, cudaStreamNonBlocking);
    cudaMalloc(&g_dev, 1 << 20); cudaMalloc(&g_dev2, 1 << 20);
    cudaHostAlloc(&g_pin, 1 << 20, cudaHostAllocDefault);
    std::printf("=== CAPTURE TRUTH TABLE (ThreadLocal, capturing thread) ===\n");

    CASO("nothing (control)", (void)0);
    CASO("cudaMalloc / cudaFree", { void*p=nullptr; cudaMalloc(&p,4096); cudaFree(p); });
    CASO("cudaMalloc alone", { void*p=nullptr; cudaMalloc(&p,4096); (void)p; });
    CASO("cudaFree alone", { void*p=nullptr; cudaMalloc(&p,4096); cudaStreamEndCapture; cudaFree(p); });
    CASO("cudaHostAlloc alone", { void*p=nullptr; cudaHostAlloc(&p,4096,0); (void)p; });
    CASO("cudaFreeHost alone", { void*p=nullptr; cudaHostAlloc(&p,4096,0); cudaFreeHost(p); });
    CASO("malloc/free (pure host)", { void*p=std::malloc(4096); std::free(p); });
    CASO("cudaMemcpyAsync on ANOTHER stream", cudaMemcpyAsync(g_dev2,g_pin,4096,cudaMemcpyHostToDevice,g_otro));
    CASO("cudaMemcpyAsync D2D on ANOTHER stream", cudaMemcpyAsync(g_dev2,g_dev,4096,cudaMemcpyDeviceToDevice,g_otro));
    CASO("cudaStreamSynchronize(OTHER stream)", cudaStreamSynchronize(g_otro));
    CASO("cudaDeviceSynchronize", cudaDeviceSynchronize());
    CASO("cudaStreamCreateWithFlags/Destroy", { cudaStream_t x; cudaStreamCreateWithFlags(&x,cudaStreamNonBlocking); cudaStreamDestroy(x); });
    CASO("cudaEventCreateWithFlags/Destroy", { cudaEvent_t x; cudaEventCreateWithFlags(&x,cudaEventDisableTiming); cudaEventDestroy(x); });
    CASO("cudaEventRecord(OTHER)+Query until ready", {
        cudaEvent_t x; cudaEventCreateWithFlags(&x,cudaEventDisableTiming);
        cudaMemcpyAsync(g_dev2,g_dev,4096,cudaMemcpyDeviceToDevice,g_otro);
        cudaEventRecord(x,g_otro);
        while (cudaEventQuery(x)==cudaErrorNotReady) {}
        cudaEventDestroy(x); });
    CASO("cudaPointerGetAttributes", { cudaPointerAttributes a; cudaPointerGetAttributes(&a,g_dev); });
    CASO("cudaGetLastError", cudaGetLastError());
    CASO("cudaMemcpy SYNCHRONOUS (H2D)", cudaMemcpy(g_dev2,g_pin,4096,cudaMemcpyHostToDevice));
    return 0;
}
