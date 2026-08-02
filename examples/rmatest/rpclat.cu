// Coste por RPC del plano de control. Es el perfil de cuDF, no el de miniBUDE.
//
// Motivo: los parches de instrumentacion anaden coste POR OPERACION (un reloj monotono por
// transicion de estado, incrementos atomicos por WriteIov, y la comprobacion de intervalo de
// gusto_metric_maybe_emit en cada Read). Las cargas de control que tenia -- miniBUDE
// (compute-bound) y rma_checksum (transferencias grandes) -- no tienen ese perfil y podrian
// dar +0,0 % sin decir nada del riesgo.
//
// cuDF esta dominado por latencia de RPC: su arranque es el 98 % del total y sus operaciones
// son planas con el tamano, que es la firma de estar midiendo el plano de control. Este
// microbenchmark mide justo eso: muchas operaciones diminutas, microsegundos por operacion.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
using namespace std::chrono;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("ERROR %s en %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } }while(0)

int main(int argc, char **argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 20000;
    const size_t TAM = 64;                 // diminuto a proposito: domina la RPC, no los bytes

    unsigned char *h = nullptr, *d = nullptr;
    CK(cudaHostAlloc((void **)&h, TAM, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d, TAM));
    for (size_t i = 0; i < TAM; ++i) h[i] = (unsigned char)i;

    // calentamiento: contexto, fatbins y construccion del pool fuera de la medida
    for (int i = 0; i < 200; ++i) CK(cudaMemcpy(d, h, TAM, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    // H2D diminuta
    auto t0 = steady_clock::now();
    for (int i = 0; i < N; ++i) CK(cudaMemcpy(d, h, TAM, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    auto t1 = steady_clock::now();
    double us_h2d = duration<double, std::micro>(t1 - t0).count() / N;

    // D2H diminuta: es el otro sentido del plano de control
    auto t2 = steady_clock::now();
    for (int i = 0; i < N; ++i) CK(cudaMemcpy(h, d, TAM, cudaMemcpyDeviceToHost));
    CK(cudaDeviceSynchronize());
    auto t3 = steady_clock::now();
    double us_d2h = duration<double, std::micro>(t3 - t2).count() / N;

    // una operacion de control pura, sin datos
    auto t4 = steady_clock::now();
    for (int i = 0; i < N; ++i) { int dev; CK(cudaGetDevice(&dev)); }
    auto t5 = steady_clock::now();
    double us_ctl = duration<double, std::micro>(t5 - t4).count() / N;

    std::printf("RPCLAT n=%d h2d_us=%.3f d2h_us=%.3f ctl_us=%.3f\n", N, us_h2d, us_d2h, us_ctl);
    CK(cudaFree(d)); CK(cudaFreeHost(h));
    return 0;
}
