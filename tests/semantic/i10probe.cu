// i10probe.cu -- �existe realmente la consulta de capacidad que I10 necesitaria?
// La propuesta pide negociar NIC_TO_GPU_VISIBILITY. La pregunta previa, y barata, es si el
// driver ya expone ordering y opciones de flush, o si habria que inferirlo (que seria otra
// suposicion disfrazada). Se usa la API de driver, no la de runtime.
#include <cstdio>
#include <chrono>
#include <cuda.h>

static const char *orden(int v) {
    switch (v) {
        case 0:   return "NONE       (ninguna garantia: hace falta flush)";
        case 100: return "OWNER      (ordenado hacia la GPU duena del BAR)";
        case 200: return "ALL_DEVICES(ordenado hacia todos los dispositivos)";
        default:  return "??";
    }
}

int main() {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { std::printf("cuInit fallo %d\n", r); return 1; }
    CUdevice dev; cuDeviceGet(&dev, 0);
    char nombre[128]; cuDeviceGetName(nombre, sizeof nombre, dev);

    int soportado = -1, opciones = -1, ordering = -1;
    CUresult r1 = cuDeviceGetAttribute(&soportado,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
    CUresult r2 = cuDeviceGetAttribute(&opciones,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, dev);
    CUresult r3 = cuDeviceGetAttribute(&ordering,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, dev);

    std::printf("GPU: %s\n", nombre);
    std::printf("  GPU_DIRECT_RDMA_SUPPORTED        = %d      (rc=%d)\n", soportado, r1);
    std::printf("  GPU_DIRECT_RDMA_WRITES_ORDERING  = %-4d -> %s   (rc=%d)\n",
                ordering, orden(ordering), r3);
    std::printf("  GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 0x%x  (rc=%d)\n", opciones, r2);
    std::printf("      bit HOST   (cuFlushGPUDirectRDMAWrites desde el host) : %s\n",
                (opciones & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) ? "SI" : "no");
    std::printf("      bit MEMOPS (flush como memop en un stream)            : %s\n",
                (opciones & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS) ? "SI" : "no");

    // �Cuanto cuesta el flush del host? Es el numero que decide si el modo explicito es
    // pagable por transferencia o solo por lote.
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    if (opciones & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) {
        CUresult f = cuFlushGPUDirectRDMAWrites(
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
        std::printf("\n  cuFlushGPUDirectRDMAWrites(CURRENT_CTX, TO_OWNER) rc=%d\n", f);
        if (f == CUDA_SUCCESS) {
            const int N = 10000;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i)
                cuFlushGPUDirectRDMAWrites(
                    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()
                        / 1000.0 / N;
            std::printf("  coste medio del flush = %.3f us  (%d repeticiones)\n", us, N);
        }
    } else {
        std::printf("\n  el bit HOST no esta: cuFlushGPUDirectRDMAWrites no es aplicable aqui\n");
    }
    cuCtxDestroy(ctx);
    return 0;
}
