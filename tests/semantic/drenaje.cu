// drenaje.cu -- ¿los drenajes esperan a la GPU ENTERA o solo a su propia copia?
//
// Tres sitios del backend sincronizaban el dispositivo completo en pleno regimen, y como el
// backend comparte UN SOLO contexto CUDA entre clientes, eso esperaba a los kernels de los
// demas. Este test los ejerce con otro hilo ocupando la GPU y mide cuanto tardan.
//
//   fase 1  drenaje de la copia "dispara y olvida" (cudaMemcpyAsync H2D por la sombra de GPU).
//           Se emite la copia async y a continuacion una RPC pequena CON respuesta, que es
//           donde salta el drenaje. Lo que se cronometra es ESA RPC, no la copia: la copia
//           tiene que esperarse siempre, el drenaje no tiene por que esperar a nadie mas.
//
//   fase 2  camino rapido de D2H de la API de driver (cuMemcpyDtoH >= umbral GPUDirect), que
//           hacia cuCtxSynchronize por transferencia.
//
// Y las dos comprueban LOS BYTES. Un drenaje que se salta la espera es rapido y corrupto, asi
// que una medida de tiempo sin verificacion de datos aqui no vale para nada: exactamente la
// carrera que este drenaje existe para evitar (el slot se libera y se reutiliza mientras el
// motor de copia lo sigue leyendo) se manifiesta como bytes de otra transferencia.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std::chrono;
static double ms_desde(steady_clock::time_point t) {
    return duration<double, std::milli>(steady_clock::now() - t).count();
}

__global__ void quema(volatile unsigned long long *sink, unsigned long long vueltas) {
    unsigned long long a = 0;
    for (unsigned long long i = 0; i < vueltas; ++i) a += i ^ (a >> 3);
    if (threadIdx.x == 1024) *sink = a;
}

int main(int argc, char **argv) {
    // ~8 s en la L40S. El kernel solo tiene que durar MUCHO mas que una copia de 8 MiB.
    const unsigned long long vueltas = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1000000000ull;
    const int IT = (argc > 2) ? atoi(argv[2]) : 8;
    const size_t N = 8u << 20;             // 8 MiB: por encima del suelo RMA de 4 MiB

    unsigned long long *sink = nullptr;
    if (cudaMalloc((void **)&sink, sizeof(*sink)) != cudaSuccess) { printf("DRAIN,error=malloc\n"); return 2; }

    std::atomic<bool> lanzado{false};
    double ms_largo = -1;

    std::thread A([&] {
        cudaStream_t sa; cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking);
        auto t0 = steady_clock::now();
        quema<<<1, 32, 0, sa>>>(sink, vueltas);
        lanzado = true;
        cudaStreamSynchronize(sa);
        ms_largo = ms_desde(t0);
        cudaStreamDestroy(sa);
    });

    // ---- preparacion del hilo B, ANTES de que A ocupe la GPU -------------------------------
    cudaStream_t sb; cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);
    unsigned char *d = nullptr, *h = nullptr, *v = nullptr;
    if (cudaMalloc((void **)&d, N) != cudaSuccess ||
        cudaHostAlloc((void **)&h, N, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void **)&v, N, cudaHostAllocDefault) != cudaSuccess) {
        printf("DRAIN,error=prep\n"); return 2;
    }
    cudaStreamSynchronize(sb);   // calienta la conexion

    while (!lanzado.load()) std::this_thread::yield();
    std::this_thread::sleep_for(milliseconds(300));   // A ya esta de verdad dentro del kernel

    // ---- fase 1: drenaje de la copia async -------------------------------------------------
    double peor_rpc = -1.0, peor_copia = -1.0;
    int malos1 = 0;
    for (int it = 0; it < IT; ++it) {
        const unsigned char tag = (unsigned char)(0xA0 + it);
        std::memset(h, tag, N);
        auto t0 = steady_clock::now();
        cudaMemcpyAsync(d, h, N, cudaMemcpyHostToDevice, sb);   // dispara y olvida
        const double ms_copia = ms_desde(t0);
        // RPC pequena CON respuesta: aqui es donde el backend drena antes de contestar.
        auto t1 = steady_clock::now();
        cudaError_t q = cudaStreamQuery(sb);
        const double ms_rpc = ms_desde(t1);
        (void)q;
        if (ms_copia > peor_copia) peor_copia = ms_copia;
        if (ms_rpc  > peor_rpc)    peor_rpc   = ms_rpc;
        // ¿llegaron MIS bytes?
        cudaStreamSynchronize(sb);
        std::memset(v, 0, N);
        if (cudaMemcpy(v, d, N, cudaMemcpyDeviceToHost) != cudaSuccess ||
            v[0] != tag || v[N / 2] != tag || v[N - 1] != tag) {
            printf("  fase1 it %d: esperaba 0x%02X, leyo 0x%02X/0x%02X/0x%02X\n",
                   it, tag, v[0], v[N / 2], v[N - 1]);
            ++malos1;
        }
    }

    // ---- fase 2: camino rapido de D2H de la API de driver -----------------------------------
    // cuMemcpyDtoH sobre un puntero de cudaMalloc: el runtime y el driver comparten el espacio
    // de direcciones, asi que esto entra por CudaDrHandler_memory (el gemelo que hacia
    // cuCtxSynchronize por transferencia).
    double peor_d2h = -1.0;
    int malos2 = 0, err_dr = 0;
    for (int it = 0; it < IT; ++it) {
        const unsigned char tag = (unsigned char)(0x50 + it);
        std::memset(h, tag, N);
        if (cudaMemcpy(d, h, N, cudaMemcpyHostToDevice) != cudaSuccess) { ++err_dr; continue; }
        std::memset(v, 0, N);
        auto t0 = steady_clock::now();
        CUresult rc = cuMemcpyDtoH(v, (CUdeviceptr)d, N);
        const double ms = ms_desde(t0);
        if (ms > peor_d2h) peor_d2h = ms;
        if (rc != CUDA_SUCCESS) { ++err_dr; continue; }
        if (v[0] != tag || v[N / 2] != tag || v[N - 1] != tag) {
            printf("  fase2 it %d: esperaba 0x%02X, leyo 0x%02X/0x%02X/0x%02X\n",
                   it, tag, v[0], v[N / 2], v[N - 1]);
            ++malos2;
        }
    }

    A.join();
    cudaFree(sink); cudaFree(d); cudaFreeHost(h); cudaFreeHost(v); cudaStreamDestroy(sb);

    // Sin kernel largo de verdad no hay nada que discriminar y el verde no significa nada.
    const bool vacuo = (ms_largo < 1000.0);
    // Umbral: una copia de 8 MiB son milisegundos; el kernel, segundos. 500 ms separa las dos
    // hipotesis con dos ordenes de magnitud de margen.
    const bool espera_a_todos = (!vacuo && (peor_rpc > 500.0 || peor_d2h > 500.0));
    printf("DRAIN,long_ms=%.1f,it=%d,worst_async_rpc_ms=%.1f,worst_launch_copy_ms=%.1f,"
           "worst_drv_d2h_ms=%.1f,bad_h2d=%d,bad_d2h=%d,drv_err=%d,verdict=%s\n",
           ms_largo, IT, peor_rpc, peor_copia, peor_d2h, malos1, malos2, err_dr,
           vacuo ? "**VACUOUS: the long kernel was too short**"
                 : (espera_a_todos ? "**DEVICE-WIDE: a drain waited for another client's kernel**"
                                   : "scoped (drains wait only for their own copy)"));
    const int fallos = malos1 + malos2 + err_dr + (espera_a_todos ? 1 : 0) + (vacuo ? 1 : 0);
    printf("DRAIN_SUMMARY %s\n", fallos ? "FAIL" : "PASS");
    return fallos ? 1 : 0;
}
