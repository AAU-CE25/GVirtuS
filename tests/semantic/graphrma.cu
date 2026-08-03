// graphrma.cu -- captura de grafos SOBRE EL CAMINO RMA, con verificacion de datos.
// Tamanos por encima del suelo RMA por defecto (4 MiB) para que la copia viaje por
// ucp_put contra un slot del pool y no por mensaje activo. Comprueba tres cosas:
//   1. la ventana de captura sobrevive,
//   2. el primer lanzamiento entrega los bytes correctos,
//   3. el RELANZAMIENTO tras cambiar el origen entrega los bytes NUEVOS (semantica),
// y ademas una captura con VARIOS nodos H2D, que es lo que ejercita el contrato de indices
// entre el espejo del frontend y el lote del backend.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

static int fallos = 0;
static void comprueba(const char *que, unsigned char got, unsigned char exp) {
    const bool ok = (got == exp);
    if (!ok) ++fallos;
    std::printf("  %-34s got=0x%02X exp=0x%02X %s\n", que, got, exp, ok ? "OK" : "**WRONG**");
}
static unsigned char primer_byte(void *d) {
    unsigned char b = 0; cudaMemcpy(&b, d, 1, cudaMemcpyDeviceToHost); return b;
}
static unsigned char byte_en(void *d, size_t off) {
    unsigned char b = 0; cudaMemcpy(&b, (char*)d + off, 1, cudaMemcpyDeviceToHost); return b;
}

static void celda(size_t nb) {
    std::printf("RMA,%zu bytes\n", nb);
    unsigned char *h = nullptr; void *d = nullptr;
    cudaHostAlloc((void**)&h, nb, cudaHostAllocDefault); cudaMalloc(&d, nb);
    std::memset(h, 0x11, nb);
    cudaStream_t s = nullptr; cudaStreamCreate(&s);
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    cudaMemcpyAsync(d, h, nb, cudaMemcpyHostToDevice, s);
    cudaError_t e = cudaStreamEndCapture(s, &g);
    std::printf("  EndCapture=%s\n", cudaGetErrorName(e));
    if (e != cudaSuccess) { ++fallos; goto limpia; }
    if (cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess) { ++fallos; goto limpia; }
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    comprueba("launch 1 (source 0x11)", primer_byte(d), 0x11);
    std::memset(h, 0x22, nb);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    comprueba("launch 2 (source 0x22)", primer_byte(d), 0x22);
    comprueba("launch 2, last byte",  byte_en(d, nb - 1), 0x22);
    std::memset(h, 0x33, nb);
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    comprueba("launch 3 (source 0x33)", primer_byte(d), 0x33);
limpia:
    if (ge) cudaGraphExecDestroy(ge);
    if (g)  cudaGraphDestroy(g);
    cudaFree(d); cudaFreeHost(h); cudaStreamDestroy(s); cudaGetLastError();
}

// Varios nodos H2D en UNA captura: si los indices de los dos lados se desalinean, cada
// destino recibe el contenido del otro y esto lo ve.
static void multinodo(size_t nb) {
    std::printf("RMA,multinode 3 x %zu bytes\n", nb);
    unsigned char *h[3]; void *d[3];
    for (int i = 0; i < 3; ++i) {
        cudaHostAlloc((void**)&h[i], nb, cudaHostAllocDefault); cudaMalloc(&d[i], nb);
        std::memset(h[i], 0xA0 + i, nb);
    }
    cudaStream_t s = nullptr; cudaStreamCreate(&s);
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    cudaGetLastError();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < 3; ++i) cudaMemcpyAsync(d[i], h[i], nb, cudaMemcpyHostToDevice, s);
    cudaError_t e = cudaStreamEndCapture(s, &g);
    std::printf("  EndCapture=%s\n", cudaGetErrorName(e));
    if (e == cudaSuccess && cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) == cudaSuccess) {
        for (int i = 0; i < 3; ++i) std::memset(h[i], 0xB0 + i, nb);
        cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        for (int i = 0; i < 3; ++i) {
            char nom[48]; std::snprintf(nom, sizeof nom, "node %d after refresh", i);
            comprueba(nom, primer_byte(d[i]), (unsigned char)(0xB0 + i));
        }
    } else ++fallos;
    if (ge) cudaGraphExecDestroy(ge);
    if (g)  cudaGraphDestroy(g);
    for (int i = 0; i < 3; ++i) { cudaFree(d[i]); cudaFreeHost(h[i]); }
    cudaStreamDestroy(s); cudaGetLastError();
}

int main() {
    cudaFree(0);
    for (size_t n : {(size_t)4u*1024*1024, (size_t)8u*1024*1024, (size_t)16u*1024*1024}) celda(n);
    multinodo(8u*1024*1024);
    std::printf("RMA,RESULT failures=%d\n", fallos);
    return fallos == 0 ? 0 : 1;
}
