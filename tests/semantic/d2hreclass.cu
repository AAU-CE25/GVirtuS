// d2hreclass.cu -- reproductor DETERMINISTA de la reclasificacion D2H->D2D.
//
// LA PUERTA, en CudaRt_memory.cpp (cudaMemcpy y cudaMemcpyAsync):
//     if (kind == D2H && !isDevicePointer(dst) && !isDevicePointer(src)) {
//         if ((dst>>32) == (src>>32) && (dst>>32) >= 0x7f00) kind = D2D;
//     }
// Se anadio para punteros driver-API de CuPy. Su argumento de seguridad escrito es
// "tus D2H (src device registrado) NUNCA entran aqui". Eso es falso para un puntero con
// DESPLAZAMIENTO: `isDevicePointer` es una busqueda por direccion EXACTA en un set, asi que
// base+offset no se reconoce. Y `(dst>>32)==(src>>32)` no es "comparten arena": es "caen en la
// misma ventana de 4 GiB", que un puntero de host y uno de dispositivo pueden cumplir por
// casualidad -- de ahi que el fallo sea intermitente.
//
// ggml_backend_cuda_buffer_get_tensor hace exactamente eso:
//     cudaMemcpyAsync(data, (char*)tensor->data + offset, size, D2H, cudaStreamPerThread);
//     cudaStreamSynchronize(cudaStreamPerThread);
//
// Aqui la coincidencia de ventana se FUERZA con mmap(MAP_FIXED_NOREPLACE), de modo que lo que
// en produccion es azar se vuelve reproducible. Rejilla 2x2 con sus controles:
//
//   offset  ventana   prediccion si la hipotesis es cierta
//   0       distinta  correcto  (src es base registrada -> la puerta no dispara)
//   0       igual     correcto  (idem; controla que mmap por si solo no rompe nada)
//   4096    distinta  correcto  (la puerta exige (dst>>32)==(src>>32))
//   4096    igual     FALLA     <- unica celda que dispara la puerta
//
// Un fallo solo en la cuarta celda identifica la puerta como causa y excluye "el desplazamiento
// rompe D2H" y "mmap rompe D2H" por separado.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>

static const size_t N = 1u << 20;
static const unsigned char PATRON = 0xAB;

// Reserva en la ventana de 4 GiB pedida, probando varios desplazamientos bajos porque
// MAP_FIXED_NOREPLACE falla con EEXIST si algo ya ocupa esa direccion.
static void *reserva_en_ventana(uint64_t hi, size_t n) {
    for (uint64_t bajo = 0x10000000ULL; bajo < 0xF0000000ULL; bajo += 0x10000000ULL) {
        void *p = mmap((void *)((hi << 32) | bajo), n, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != MAP_FAILED && ((uint64_t)(uintptr_t)p >> 32) == hi) return p;
        if (p != MAP_FAILED) munmap(p, n);
    }
    return nullptr;
}

struct Celda { const char *nombre; size_t off; bool misma_ventana; };

int main() {
    void *d = nullptr;
    if (cudaMalloc(&d, N) != cudaSuccess) { printf("FAIL: cudaMalloc\n"); return 1; }
    const uint64_t hi_dev = (uint64_t)(uintptr_t)d >> 32;

    unsigned char *sem = (unsigned char *)malloc(N);
    memset(sem, PATRON, N);
    if (cudaMemcpy(d, sem, N, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FAIL: H2D de siembra\n"); return 1;
    }
    printf("device=%p  ventana_dev=0x%llx\n", d, (unsigned long long)hi_dev);

    void *dst_igual = reserva_en_ventana(hi_dev, N);
    void *dst_otra  = reserva_en_ventana(hi_dev - 1, N);
    if (!dst_igual || !dst_otra) {
        printf("FAIL: no se pudo reservar (igual=%p otra=%p) -- sin esto el test no concluye\n",
               dst_igual, dst_otra);
        return 1;
    }
    printf("dst_misma_ventana=%p  dst_otra_ventana=%p\n\n", dst_igual, dst_otra);

    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    const Celda celdas[4] = {
        {"off=0     ventana distinta", 0,    false},
        {"off=0     misma ventana   ", 0,    true},
        {"off=4096  ventana distinta", 4096, false},
        {"off=4096  misma ventana   ", 4096, true},
    };

    int fallos = 0, fallo_solo_en_celda4 = 1;
    const size_t n = 65536;
    for (int i = 0; i < 4; i++) {
        unsigned char *dst = (unsigned char *)(celdas[i].misma_ventana ? dst_igual : dst_otra);
        memset(dst, 0, n);
        cudaError_t e1 = cudaMemcpyAsync(dst, (char *)d + celdas[i].off, n,
                                         cudaMemcpyDeviceToHost, s);
        cudaError_t e2 = cudaStreamSynchronize(s);
        size_t malos = 0;
        for (size_t k = 0; k < n; k++) if (dst[k] != PATRON) malos++;
        const bool ok = (e1 == cudaSuccess && e2 == cudaSuccess && malos == 0);
        printf("  %s  memcpyAsync=%-2d sync=%-2d bytes_malos=%-6zu  %s\n",
               celdas[i].nombre, (int)e1, (int)e2, malos, ok ? "ok" : "*** FALLA");
        if (!ok) { fallos++; if (i != 3) fallo_solo_en_celda4 = 0; }
        if (e2 != cudaSuccess) printf("       sync dice: %s\n", cudaGetErrorString(e2));
        // Un error pegajoso a nivel de contexto contaminaria las celdas siguientes.
        cudaGetLastError();
    }

    printf("\n");
    if (fallos == 1 && fallo_solo_en_celda4)
        printf("VERDICT: RECLASIFICACION CONFIRMADA -- falla solo la celda que cumple las dos "
               "condiciones de la puerta (offset != 0 y misma ventana de 4 GiB)\n");
    else if (fallos == 0)
        printf("VERDICT: no reproducido -- la puerta no disparo en ninguna celda\n");
    else
        printf("VERDICT: patron NO limpio (%d celdas fallan) -- no se puede atribuir a la puerta\n",
               fallos);
    return fallos ? 2 : 0;
}
