// ptds_mt.cu -- cudaStreamPerThread con VARIOS hilos: el paso que quedo pendiente el 2026-08-02.
//
// POR QUE ESTE TEST. `cudaStreamPerThread` (el literal 0x2) es THREAD-LOCAL: cada hilo tiene su
// propio stream. Pero por el cable viaja el literal, y en el backend se resuelve al PTDS del
// hilo que atiende la conexion, no al del hilo del cliente. Si el mapeo hilo-de-cliente ->
// conexion no es uno a uno, los streams de N hilos colapsan en uno.
//
// `ptds_repro.cu` ya cubria PTDS, pero con UN SOLO HILO, que es exactamente la limitacion que
// el handoff del 2026-08-02 senalaba al dejar esto pendiente. El abort de llama aparece con
// --parallel 8 y guardado de estado concurrente.
//
// DOS FASES, y la segunda es la que discrimina:
//   A. correccion bajo concurrencia: N hilos x R iteraciones, cada uno con su patron, D2H
//      grande (por encima del suelo RMA) y sync de SU PTDS. Cruce de datos = colapso.
//   B. discriminador de serializacion: el hilo 0 encola un kernel LARGO en su PTDS; el hilo 1
//      encola uno corto en el suyo y cronometra su propio sync. Si los PTDS son de verdad
//      por hilo, el hilo 1 vuelve enseguida.
//   C. lo mismo con streams EXPLICITOS. Sin esta fase, un resultado de B se leeria como
//      "PTDS colapsado" -- que es lo que yo escribi antes de correrla, y es falso: C tambien
//      serializa, luego la causa no es la resolucion de cudaStreamPerThread.
//
// RESULTADO (2026-08-04, medido): nativo solapa (sync corto en 0,0 ms sobre un kernel de 50 s);
// Gusto NO (50,1 s en B y 50,2 s en C). Excluido por medicion: resolucion de PTDS (fase C),
// conexion compartida (11 establecidas), hilo de backend compartido (13 conn distintos, 20 RPC
// cada uno, reparto exacto), objeto Frontend compartido (habria concentrado los RPC en uno),
// sincronizacion de dispositivo en camino comun (los dos cudaDeviceSynchronize del backend son
// thread_local o pedidos por el cliente) y cerrojo global de despacho (no hay).
// El mecanismo NO esta identificado. Lo que si es nuevo: esto es DETERMINISTA, al contrario
// que el abort de 1 entre 9.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

using namespace std::chrono;

static int NH  = 8;                       // hilos, como --parallel 8 de llama
static int IT  = 12;                      // iteraciones por hilo
static size_t N = 6u << 20;               // 6 MiB: por encima del suelo RMA de 4 MiB

__global__ void fill(unsigned char *d, size_t n, unsigned char v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}
// Kernel deliberadamente lento, para la fase B.
__global__ void quema(volatile unsigned long long *sink, unsigned long long vueltas) {
    unsigned long long a = 0;
    for (unsigned long long i = 0; i < vueltas; ++i) a += i ^ (a >> 3);
    if (threadIdx.x == 1024) *sink = a;    // nunca se cumple: solo evita que se optimice
}

static std::atomic<int> listos{0}, malos{0}, errores{0}, cruces{0};

static void trabajador(int id) {
    unsigned char *d = nullptr, *h = nullptr;
    if (cudaMalloc((void **)&d, N) != cudaSuccess ||
        cudaHostAlloc((void **)&h, N, cudaHostAllocDefault) != cudaSuccess) {
        ++errores; ++listos; return;
    }
    const unsigned char mio = (unsigned char)(0x40 + id);
    ++listos;
    while (listos.load() < NH) std::this_thread::yield();   // barrera: maximo solapamiento

    for (int it = 0; it < IT; ++it) {
        std::memset(h, 0xEE, N);
        fill<<<(N + 255) / 256, 256, 0, cudaStreamPerThread>>>(d, N, mio);
        cudaError_t e1 = cudaMemcpyAsync(h, d, N, cudaMemcpyDeviceToHost, cudaStreamPerThread);
        cudaError_t e2 = cudaStreamSynchronize(cudaStreamPerThread);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            std::printf("  hilo %d it %d: memcpy=%s sync=%s\n", id, it,
                        cudaGetErrorName(e1), cudaGetErrorName(e2));
            ++errores; continue;
        }
        // ¿Trae MIS bytes, o los de otro hilo?
        if (h[0] != mio || h[N / 2] != mio || h[N - 1] != mio) {
            const unsigned char g = h[0];
            const bool de_otro = (g >= 0x40 && g < (unsigned char)(0x40 + NH) && g != mio);
            std::printf("  hilo %d it %d: esperaba 0x%02X, leyo 0x%02X %s\n",
                        id, it, mio, g, de_otro ? "<== DE OTRO HILO" : "");
            if (de_otro) ++cruces;
            ++malos;
        }
    }
    cudaFree(d); cudaFreeHost(h);
}

int main(int argc, char **argv) {
    if (argc > 1) NH = std::atoi(argv[1]);
    if (argc > 2) IT = std::atoi(argv[2]);
    std::printf("== fase A: %d hilos x %d iteraciones, %zu MiB por D2H ==\n", NH, IT, N >> 20);
    std::vector<std::thread> hs;
    for (int i = 0; i < NH; ++i) hs.emplace_back(trabajador, i);
    for (auto &t : hs) t.join();
    std::printf("A,threads=%d,iters=%d,wrong=%d,cross_thread=%d,errors=%d\n",
                NH, IT, malos.load(), cruces.load(), errores.load());

    // ---- fase B: ¿el PTDS de un hilo espera al trabajo de otro? -------------------------
    std::printf("\n== fase B: discriminador de serializacion ==\n");
    unsigned long long *sink = nullptr;
    if (cudaMalloc((void **)&sink, sizeof(*sink)) != cudaSuccess) return 2;
    std::atomic<bool> lanzado{false};
    double ms_corto = -1.0, ms_largo = -1.0;

    std::thread largo([&] {
        auto t0 = steady_clock::now();
        quema<<<1, 32, 0, cudaStreamPerThread>>>(sink, 6000000000ull);
        lanzado = true;
        cudaStreamSynchronize(cudaStreamPerThread);
        ms_largo = duration<double, std::milli>(steady_clock::now() - t0).count();
    });
    std::thread corto([&] {
        while (!lanzado.load()) std::this_thread::yield();
        std::this_thread::sleep_for(milliseconds(50));   // que el largo este de verdad corriendo
        unsigned char *d2 = nullptr;
        if (cudaMalloc((void **)&d2, 4096) != cudaSuccess) return;
        auto t0 = steady_clock::now();
        fill<<<1, 256, 0, cudaStreamPerThread>>>(d2, 4096, 0x11);
        cudaStreamSynchronize(cudaStreamPerThread);
        ms_corto = duration<double, std::milli>(steady_clock::now() - t0).count();
        cudaFree(d2);
    });
    corto.join(); largo.join(); cudaFree(sink);

    // Umbral: el kernel largo dura del orden de segundos. Si el corto tarda >500 ms es que
    // espero al largo, o sea que los dos PTDS son el MISMO stream.
    // Si el kernel largo termino ANTES de que el corto midiera, no hubo nada a lo que
    // esperar y el veredicto "independiente" no significa nada. Un discriminador que no
    // discrimina tiene que decirlo, no salir en verde.
    const bool vacuo = (ms_largo < 200.0);
    const bool serializado = (!vacuo && ms_corto > 500.0);
    std::printf("B,long_kernel_ms=%.1f,short_sync_ms=%.1f,verdict=%s\n", ms_largo, ms_corto,
                vacuo ? "**VACUOUS: the long kernel finished too fast to overlap**"
                      : (serializado ? "**SERIALIZED: one thread waited for another's kernel**"
                                     : "independent"));

    // ---- fase C: el mismo experimento con streams EXPLICITOS ---------------------------
    // Discrimina la CAUSA. Si con streams creados a mano tampoco hay solapamiento, lo que
    // serializa no es la resolucion de cudaStreamPerThread sino el transporte (una conexion
    // compartida, un cerrojo de worker), y el arreglo es otro.
    std::printf("\n== fase C: mismo experimento con streams EXPLICITOS ==\n");
    unsigned long long *sink2 = nullptr;
    if (cudaMalloc((void **)&sink2, sizeof(*sink2)) != cudaSuccess) return 2;
    std::atomic<bool> lanzado2{false};
    double ms_corto2 = -1.0, ms_largo2 = -1.0;
    std::thread largo2([&] {
        cudaStream_t sa; cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking);
        auto t0 = steady_clock::now();
        quema<<<1, 32, 0, sa>>>(sink2, 6000000000ull);
        lanzado2 = true;
        cudaStreamSynchronize(sa);
        ms_largo2 = duration<double, std::milli>(steady_clock::now() - t0).count();
        cudaStreamDestroy(sa);
    });
    std::thread corto2([&] {
        while (!lanzado2.load()) std::this_thread::yield();
        std::this_thread::sleep_for(milliseconds(50));
        cudaStream_t sb; cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);
        unsigned char *d3 = nullptr;
        if (cudaMalloc((void **)&d3, 4096) != cudaSuccess) return;
        auto t0 = steady_clock::now();
        fill<<<1, 256, 0, sb>>>(d3, 4096, 0x22);
        cudaStreamSynchronize(sb);
        ms_corto2 = duration<double, std::milli>(steady_clock::now() - t0).count();
        cudaFree(d3); cudaStreamDestroy(sb);
    });
    corto2.join(); largo2.join(); cudaFree(sink2);
    const bool vacuo2 = (ms_largo2 < 200.0);
    const bool serializado2 = (!vacuo2 && ms_corto2 > 500.0);
    std::printf("C,long_kernel_ms=%.1f,short_sync_ms=%.1f,verdict=%s\n", ms_largo2, ms_corto2,
                vacuo2 ? "**VACUOUS**"
                       : (serializado2 ? "**ALSO serialized: the cause is NOT PTDS resolution**"
                                       : "independent -> the serialization is SPECIFIC to cudaStreamPerThread"));

    const int fallos = malos.load() + errores.load() + (serializado ? 1 : 0) + (vacuo ? 1 : 0);
    std::printf("\nSUMMARY threads=%d wrong=%d cross_thread=%d errors=%d serialized=%s %s\n",
                NH, malos.load(), cruces.load(), errores.load(), serializado ? "yes" : "no",
                fallos ? "FAIL" : "PASS");
    return fallos ? 1 : 0;
}
