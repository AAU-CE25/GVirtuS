// Reproductor determinista de la hipotesis del abort de llama.
//
// Hipotesis, de lectura de codigo:
//   - llama.cpp usa el stream por hilo (PTDS). Su traza aborta en
//       ggml_backend_cuda_buffer_get_tensor -> cudaStreamSynchronize((cudaStream_t)0x2)
//     y 0x2 es cudaStreamPerThread.
//   - El shim del frontend de GVirtuS implementa SOLO tres variantes _ptsz
//     (cudaLaunchKernel_ptsz, cudaMemcpyAsync_ptsz, cudaMemsetAsync_ptsz).
//     cudaStreamSynchronize_ptsz NO existe, y el backend no traduce el pseudo-handle.
//   - PTDS es THREAD-LOCAL: el handle 0x2 significa "el stream por defecto de ESTE hilo".
//     Al cruzar la frontera de remoting, el backend lo resuelve en SU hilo, que no es el del
//     cliente. La semantica no sobrevive al viaje.
//
// El test recorre cuatro casos y dice cual falla. Si el caso PTDS falla con
// "invalid argument" y el legacy no, la hipotesis queda confirmada y el fallo pasa de
// intermitente (1 de 9) a determinista.
#include <cstdio>
#include <cuda_runtime.h>

static int fallos = 0;

static void prueba(const char *nombre, cudaStream_t s, bool async) {
    const size_t N = 1 << 20;          // 1 MiB: por encima del suelo RMA de 8 KiB
    void *d = nullptr; void *h = nullptr;
    cudaError_t e;

    if ((e = cudaMalloc(&d, N)) != cudaSuccess) {
        std::printf("  %-28s cudaMalloc FALLO: %s\n", nombre, cudaGetErrorString(e));
        ++fallos; return;
    }
    if ((e = cudaHostAlloc(&h, N, cudaHostAllocDefault)) != cudaSuccess) {
        std::printf("  %-28s cudaHostAlloc FALLO: %s\n", nombre, cudaGetErrorString(e));
        cudaFree(d); ++fallos; return;
    }

    if (async) e = cudaMemcpyAsync(d, h, N, cudaMemcpyHostToDevice, s);
    else       e = cudaMemcpy(d, h, N, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::printf("  %-28s H2D FALLO: %s\n", nombre, cudaGetErrorString(e));
        ++fallos; cudaFree(d); cudaFreeHost(h); return;
    }

    // ESTE es el punto bajo prueba: la misma llamada que aborta en llama.
    e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        std::printf("  %-28s cudaStreamSynchronize(%p) FALLO: %s   <== REPRODUCIDO\n",
                    nombre, (void *)s, cudaGetErrorString(e));
        ++fallos;
    } else {
        // Y una D2H detras, que es lo que hace get_tensor.
        e = cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);
        std::printf("  %-28s ok (sync y D2H correctos)%s\n", nombre,
                    e == cudaSuccess ? "" : " pero la D2H fallo");
        if (e != cudaSuccess) ++fallos;
    }
    cudaFree(d); cudaFreeHost(h);
}

int main() {
    std::printf("cudaStreamPerThread = %p | cudaStreamLegacy = %p\n",
                (void *)cudaStreamPerThread, (void *)cudaStreamLegacy);

    std::printf("\n-- caso 1: stream nulo (default), sincrono --\n");
    prueba("stream 0 (default)", 0, false);

    std::printf("\n-- caso 2: stream explicito creado --\n");
    cudaStream_t propio;
    if (cudaStreamCreate(&propio) == cudaSuccess) {
        prueba("stream creado", propio, true);
        cudaStreamDestroy(propio);
    } else {
        std::printf("  no se pudo crear el stream\n"); ++fallos;
    }

    std::printf("\n-- caso 3: cudaStreamLegacy (0x1) --\n");
    prueba("cudaStreamLegacy", cudaStreamLegacy, true);

    std::printf("\n-- caso 4: cudaStreamPerThread (0x2)  <-- el de llama --\n");
    prueba("cudaStreamPerThread", cudaStreamPerThread, true);

    std::printf("\n== %s ==\n", fallos ? "HAY FALLOS" : "todo correcto");
    return fallos ? 1 : 0;
}
