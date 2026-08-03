// d2hpool.cu -- ¿puede un D2H SINCRONO pisar el scratch de un D2H diferido pendiente?
//
// LA SOSPECHA, de leer el codigo. El backend sirve todo D2H >= 4 MiB desde un pool de 4
// scratch de GPU en round-robin (`get_d2h_get_scratch`). El frontend limita los D2H
// DIFERIDOS en vuelo a 4 -- el mismo numero -- y esa tapa es lo que impide reutilizar un
// scratch que el cliente aun no ha leido por RDMA.
//
// Pero la tapa sólo cuenta los DIFERIDOS (`if (deferred_d2h && mPendingD2H.size() >= 4)`),
// mientras el pool lo consume CUALQUIER D2H >= 4 MiB, sincrono incluido. Un D2H sincrono
// (destino paginable) no pasa por la tapa y su drenaje ocurre en la RECEPCION, o sea despues
// de que el backend ya haya tomado su slot del round-robin -- que con 4 diferidos pendientes
// es el del mas viejo.
//
// EL EXPERIMENTO. 4 D2H diferidos (destino FIJADO, con async dispatch) con patrones distintos,
// luego un D2H SINCRONO grande (destino PAGINABLE) con otro patron, y despues se sincroniza y
// se comprueban los cuatro. Si el sincrono piso un scratch, alguno de los cuatro trae SUS
// bytes. Un fallo aqui es dato incorrecto en silencio, no un error devuelto.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); \
  return 2; } }while(0)

static const size_t N = 8u << 20;           // 8 MiB: por encima del umbral D2H de 4 MiB
static const int    K = 4;                  // = tamano del pool y de la tapa

__global__ void fill(unsigned char *d, size_t n, unsigned char v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}

int main() {
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    unsigned char *dev[K + 1];
    unsigned char *pin[K];                  // destinos FIJADOS -> camino diferido
    for (int i = 0; i < K; ++i) {
        CK(cudaMalloc((void **)&dev[i], N));
        CK(cudaHostAlloc((void **)&pin[i], N, cudaHostAllocDefault));
        fill<<<(N + 255) / 256, 256, 0, s>>>(dev[i], N, (unsigned char)(0xA0 + i));
        std::memset(pin[i], 0xEE, N);
    }
    CK(cudaMalloc((void **)&dev[K], N));
    fill<<<(N + 255) / 256, 256, 0, s>>>(dev[K], N, 0x5C);   // el patron del SINCRONO
    CK(cudaStreamSynchronize(s));

    // 1) K D2H diferidos: destino fijado. Llenan el pool y quedan pendientes.
    for (int i = 0; i < K; ++i)
        CK(cudaMemcpyAsync(pin[i], dev[i], N, cudaMemcpyDeviceToHost, s));

    // 2) un cudaMemcpyAsync D2H con destino PAGINABLE. Tiene que ser ASYNC: la copia
    //    sincrona `cudaMemcpy` va a OTRO handler del backend y usa OTRO scratch, asi que no
    //    toca este pool -- la primera version de este test cometia ese error y pasaba sin
    //    probar nada. Con destino paginable, `gvirtus_is_pinned` es falso, el frontend NO lo
    //    difiere (luego no pasa por la tapa) y el backend lo sirve desde el MISMO pool.
    unsigned char *pag = (unsigned char *)std::malloc(N);
    if (!pag) { printf("out of memory\n"); return 2; }
    CK(cudaMemcpyAsync(pag, dev[K], N, cudaMemcpyDeviceToHost, s));

    // 3) punto de sincronizacion: aqui el cliente ya tuvo que recoger los cuatro diferidos.
    CK(cudaStreamSynchronize(s));
    CK(cudaDeviceSynchronize());

    int malos = 0;
    printf("buffer,expected,got_first,got_last,verdict\n");
    for (int i = 0; i < K; ++i) {
        const unsigned char e = (unsigned char)(0xA0 + i);
        const bool ok = (pin[i][0] == e && pin[i][N - 1] == e);
        const bool pisado = (pin[i][0] == 0x5C || pin[i][N - 1] == 0x5C);
        printf("%d,0x%02X,0x%02X,0x%02X,%s\n", i, e, pin[i][0], pin[i][N - 1],
               ok ? "OK" : (pisado ? "**OVERWRITTEN BY THE SYNCHRONOUS D2H**" : "**WRONG**"));
        if (!ok) ++malos;
    }
    const bool spag = (pag[0] == 0x5C && pag[N - 1] == 0x5C);
    printf("sync,0x5C,0x%02X,0x%02X,%s\n", pag[0], pag[N - 1], spag ? "OK" : "**WRONG**");
    if (!spag) ++malos;
    printf("SUMMARY deferred=%d sync=1 failures=%d %s\n", K, malos, malos ? "FAIL" : "PASS");

    std::free(pag);
    for (int i = 0; i < K; ++i) { CK(cudaFree(dev[i])); CK(cudaFreeHost(pin[i])); }
    CK(cudaFree(dev[K])); CK(cudaStreamDestroy(s));
    return malos ? 1 : 0;
}
