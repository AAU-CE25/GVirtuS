// epochharm.cu -- demostracion DANINA de la colision de epoch.
//
// QUE FALTA HOY. La guarda de epoch esta demostrada ALCANZABLE: cuDF reanuncia los mismos
// server_idx [0-7] en el epoch 1 y en el epoch 2, asi que un ack del epoch viejo puede
// referirse a un slot que ya pertenece al epoch nuevo. Pero el DANO no lo produjo ninguna
// carga: 0 entregas en 186 oportunidades. La guarda es un invariante necesario, no un salvado
// observado, y eso es mas debil de lo que el paper querria.
//
// POR QUE NO SE MATERIALIZABA. Para que haga dano no basta con que el estado exista. Hacen
// falta TRES cosas a la vez:
//   1. que el reanuncio reutilice los indices (no que renumere),
//   2. que el ack viejo llegue DESPUES de instalarse el epoch nuevo,
//   3. y que el slot al que apunta este OCUPADO en ese instante.
// Los bancos anteriores cumplian 1 y 2 pero no 3: sincronizaban entre fases, asi que cuando
// el ack viejo aterrizaba no habia nada vivo que liberar.
//
// QUE HACE ESTE. Mantiene trafico en vuelo DURANTE el cambio de epoch, con el despachador
// asincrono, y valida cada byte. Se corre en dos brazos:
//   guarda ON  (por defecto)          -> se espera 0 corrupciones
//   guarda OFF (GVS_ABLATE=no_epoch)  -> se espera corrupcion o liberacion prematura
// Si el brazo OFF tampoco corrompe, eso TAMBIEN es un resultado y hay que decirlo: querria
// decir que la ventana es mas estrecha de lo que el codigo sugiere.
//
// uso: epochharm [iteraciones] [bytes_fase1] [bytes_fase2]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA FALLO %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  return 2; } } while(0)

static void llena(unsigned char *p, size_t n, int tag) {
    for (size_t i = 0; i < n; i += 512) p[i] = (unsigned char)(tag * 31 + (int)(i >> 9));
}
static long comprueba(const unsigned char *p, size_t n, int tag) {
    long mal = 0;
    for (size_t i = 0; i < n; i += 512)
        if (p[i] != (unsigned char)(tag * 31 + (int)(i >> 9))) ++mal;
    return mal;
}

int main(int argc, char **argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 40;
    const size_t b1 = (argc > 2) ? (size_t)atoll(argv[2]) : (2ull << 20);   // fase 1: 2 MiB
    const size_t b2 = (argc > 3) ? (size_t)atoll(argv[3]) : (6ull << 20);   // fase 2: 6 MiB
    const int NB = 8;   // tantos buffers como slots, para que el pool este siempre ocupado

    std::printf("epochharm: iters=%d fase1=%zu fase2=%zu buffers=%d\n", iters, b1, b2, NB);

    unsigned char *h[NB], *back[NB];
    void *d[NB];
    const size_t bmax = (b1 > b2) ? b1 : b2;
    for (int k = 0; k < NB; ++k) {
        CK(cudaHostAlloc((void **)&h[k], bmax, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&back[k], bmax, cudaHostAllocDefault));
        CK(cudaMalloc(&d[k], bmax));
    }

    long total_mal = 0;
    int fases = 0;

    // Alternar tamano cada iteracion fuerza al backend a reconstruir el pool -> cambio de
    // epoch. Y NO se sincroniza entre buffers: se lanzan los ocho y solo despues se valida,
    // de modo que cuando el epoch cambia hay transferencias vivas en varios slots.
    for (int it = 0; it < iters; ++it) {
        const size_t nb = (it % 2 == 0) ? b1 : b2;
        if (it % 2 == 0) ++fases;

        for (int k = 0; k < NB; ++k) {
            llena(h[k], nb, it * 16 + k);
            CK(cudaMemcpyAsync(d[k], h[k], nb, cudaMemcpyHostToDevice, 0));
        }
        // Aqui hay hasta 8 transferencias en vuelo justo cuando el pool puede reconstruirse.
        for (int k = 0; k < NB; ++k) {
            std::memset(back[k], 0, nb);
            CK(cudaMemcpyAsync(back[k], d[k], nb, cudaMemcpyDeviceToHost, 0));
        }
        CK(cudaStreamSynchronize(0));

        for (int k = 0; k < NB; ++k) {
            long mal = comprueba(back[k], nb, it * 16 + k);
            if (mal) {
                std::printf("CORRUPCION it=%d buf=%d bytes_malos=%ld de %zu muestras\n",
                            it, k, mal, nb / 512);
                total_mal += mal;
            }
        }
    }

    for (int k = 0; k < NB; ++k) {
        cudaFree(d[k]); cudaFreeHost(h[k]); cudaFreeHost(back[k]);
    }
    std::printf("RESULTADO iters=%d cambios_de_tamano=%d bytes_malos=%ld -> %s\n",
                iters, fases, total_mal, total_mal ? "CORRUPCION OBSERVADA" : "sin corrupcion");
    return total_mal ? 1 : 0;
}
