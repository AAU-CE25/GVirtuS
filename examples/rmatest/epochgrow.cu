// epochgrow.cu — N fases de tamano ASCENDENTE sobre una sola conexion.
//
// Cada salto de tamano obliga al backend a reconstruir su pool de slots y a re-anunciarlo,
// o sea a un epoch nuevo. Con >=3 saltos se puede comprobar si los server_idx anunciados se
// RECICLAN entre layouts, que es la unica via por la que un SlotConsumed de un epoch viejo
// podria liberar un slot vivo del epoch actual. growtest solo hace un salto y por eso no
// distingue "la guarda de epoch no hace falta" de "el experimento no la ejercita".
//
//   uso: epochgrow <reps_por_fase> <tam1> [tam2 ...]
//
// Imprime una fila CSV por transferencia, con verificacion de contenido en las dos
// direcciones (igual que growtest): fase,i,bytes,ms,GBps,bad,check
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

using namespace std::chrono;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static int g_tag = 0;

static void fase(int f, size_t sz, int reps, char *h, char *back, char *d) {
    for (int i = 0; i < reps; ++i) {
        const int t = ++g_tag;
        for (size_t k = 0; k < sz; k += 4096) h[k] = (char)(t * 31 + (int)(k >> 12));
        h[sz - 1] = (char)(0xA0 + (t & 0x3f));

        CK(cudaDeviceSynchronize());
        auto t0 = steady_clock::now();
        CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());
        auto t1 = steady_clock::now();

        std::memset(back, 0, sz);
        CK(cudaMemcpy(back, d, sz, cudaMemcpyDeviceToHost));

        long nbad = 0;
        for (size_t k = 0; k < sz; k += 4096)
            if (back[k] != (char)(t * 31 + (int)(k >> 12))) ++nbad;
        if (back[sz - 1] != (char)(0xA0 + (t & 0x3f))) ++nbad;

        double ms = duration<double, std::milli>(t1 - t0).count();
        printf("P%d,%d,%zu,%.3f,%.3f,%ld,%s\n", f, i, sz, ms,
               (double)sz / (ms / 1000.0) / 1e9, nbad, nbad ? "FAIL" : "pass");
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s <reps_por_fase> <tam1> [tam2 ...]\n", argv[0]);
        return 2;
    }
    const int reps = atoi(argv[1]);
    const int nf = argc - 2;
    size_t maxsz = 0;
    for (int i = 0; i < nf; ++i) {
        size_t v = (size_t)atoll(argv[2 + i]);
        if (v > maxsz) maxsz = v;
    }

    char *h = nullptr, *back = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&h,    maxsz, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&back, maxsz, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, maxsz));

    printf("fase,i,bytes,ms,GBps,bad,check\n");
    for (int i = 0; i < nf; ++i)
        fase(i + 1, (size_t)atoll(argv[2 + i]), reps, h, back, d);

    CK(cudaFree(d)); CK(cudaFreeHost(h)); CK(cudaFreeHost(back));
    return 0;
}
