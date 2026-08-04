// serialrepro.cu -- reproductor MINIMO y RAPIDO de la serializacion entre hilos.
//
// ptds_mt.cu la reproduce, pero cuesta ~150 s por corrida (fases A-D) y el kernel dura 50 s.
// Para iterar sobre el mecanismo hace falta algo que quepa en 15 s y que ademas diga MAS:
// no solo "el hilo B tardo", sino CUANTAS RPC hizo B antes de atascarse y cual fue la primera
// que se atasco. Esa serie temporal distingue "B nunca progresa" de "B progresa hasta que pasa
// algo concreto y a partir de ahi se queda".
//
// Hilo A: kernel largo (~10 s) en su propio stream + sync.
// Hilo B: calienta su conexion ANTES (para excluir establecimiento tardio), espera a que A este
//         de verdad corriendo, y luego cronometra RPC individuales.
//
// Salida (una linea por magnitud, facil de gsub en un arnes):
//   REPRO,long_ms=..,warm_ms=..,n=..,ok_before_stall=..,max_ms=..,total_ms=..,verdict=..
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
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
__global__ void toca(unsigned char *d) { d[threadIdx.x] = 1; }

int main(int argc, char **argv) {
    // ~10 s en la L40S; se puede acortar por argumento para iterar aun mas rapido.
    unsigned long long vueltas = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1200000000ull;
    const int N = (argc > 2) ? atoi(argv[2]) : 60;

    unsigned long long *sink = nullptr;
    if (cudaMalloc((void **)&sink, sizeof(*sink)) != cudaSuccess) { printf("REPRO,error=malloc\n"); return 2; }

    std::atomic<bool> lanzado{false}, fin_b{false};
    std::atomic<cudaStream_t> stream_a{nullptr};
    double ms_largo = -1, ms_warm = -1, ms_max = -1, ms_max_lanz = -1, ms_total = -1;
    int ok_antes = -1;
    std::vector<double> serie;

    std::thread A([&] {
        cudaStream_t sa; cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking);
        stream_a.store(sa);
        auto t0 = steady_clock::now();
        quema<<<1, 32, 0, sa>>>(sink, vueltas);
        lanzado = true;
        cudaStreamSynchronize(sa);
        ms_largo = ms_desde(t0);
        cudaStreamDestroy(sa);
    });

    std::thread B([&] {
        // Calentamiento: crea la conexion y el stream ANTES de que A ocupe la GPU.
        cudaStream_t sb; cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking);
        unsigned char *d = nullptr;
        if (cudaMalloc((void **)&d, 256) != cudaSuccess) { fin_b = true; return; }
        auto tw = steady_clock::now();
        cudaStreamSynchronize(sb);
        ms_warm = ms_desde(tw);

        while (!lanzado.load()) std::this_thread::yield();
        std::this_thread::sleep_for(milliseconds(300));   // A ya esta de verdad dentro del kernel

        // --- discriminadores, ANTES de medir la serie ---------------------------------------
        // (1) ¿Son el MISMO stream en el backend? Los handles son punteros del backend, asi que
        //     compararlos aqui contesta la pregunta de identidad sin tocar el servidor.
        // (2) Sync de un stream VACIO (sin lanzar nada): si esto ya tarda, la espera no es de
        //     trabajo de este hilo, y sobra toda discusion sobre donde fue a parar su kernel.
        // (3) Consulta en vez de espera: cudaStreamQuery responde por ESTADO, no por tiempo.
        auto tv = steady_clock::now();
        cudaError_t rc_vacio = cudaStreamSynchronize(sb);
        const double ms_vacio = ms_desde(tv);
        auto tq = steady_clock::now();
        cudaError_t rc_q = cudaStreamQuery(sb);
        const double ms_q = ms_desde(tq);
        std::printf("REPRO_ID,stream_A=%p,stream_B=%p,same=%s,"
                    "empty_sync_ms=%.1f(rc=%d),query_ms=%.1f(rc=%d,%s)\n",
                    (void *)stream_a.load(), (void *)sb,
                    ((void *)stream_a.load() == (void *)sb) ? "SI" : "no",
                    ms_vacio, (int)rc_vacio, ms_q, (int)rc_q, cudaGetErrorName(rc_q));
        std::fflush(stdout);

        auto tt = steady_clock::now();
        serie.reserve(N);
        for (int i = 0; i < N; ++i) {
            // SEPARAR lanzamiento de sync es lo unico que discrimina, y por no hacerlo estuve
            // a punto de dar por bueno un arreglo mirando un numero que no cambia:
            //   - el SYNC espera a que el kernel de A acabe TAMBIEN EN NATIVO (medido: 9 818,9 ms
            //     nativo frente a 9 818,6 ms sobre Gusto). No es un defecto de Gusto.
            //   - el LANZAMIENTO es asincrono: en nativo vuelve en microsegundos pase lo que pase.
            //     Si sobre Gusto tarda segundos, eso SI es de Gusto, y es la serializacion.
            // Los dos acaban cuando acaba el kernel de A, o sea que sumados dan el mismo total.
            auto t0 = steady_clock::now();
            toca<<<1, 32, 0, sb>>>(d);            // RPC: cudaLaunchKernel (deberia ser asincrona)
            const double dt_lanz = ms_desde(t0);
            auto t1 = steady_clock::now();
            cudaStreamSynchronize(sb);            // RPC: cudaStreamSynchronize
            const double dt_sync = ms_desde(t1);
            const double dt = dt_lanz + dt_sync;
            serie.push_back(dt_lanz);
            if (dt_lanz > ms_max_lanz) ms_max_lanz = dt_lanz;
            if (dt > ms_max) ms_max = dt;
            if (ok_antes < 0 && dt > 500.0) ok_antes = i;   // primera atascada
        }
        ms_total = ms_desde(tt);
        cudaFree(d); cudaStreamDestroy(sb);
        fin_b = true;
    });

    B.join(); A.join(); cudaFree(sink);

    const bool vacuo = (ms_largo < 1000.0);
    // EL VEREDICTO VA SOBRE EL LANZAMIENTO, no sobre el total: el total tambien tarda en nativo.
    const bool serializado = (!vacuo && ms_max_lanz > 500.0);
    printf("REPRO,long_ms=%.1f,warm_ms=%.1f,n=%d,ok_before_stall=%d,"
           "max_launch_ms=%.1f,max_pair_ms=%.1f,total_ms=%.1f,verdict=%s\n",
           ms_largo, ms_warm, N, ok_antes, ms_max_lanz, ms_max, ms_total,
           vacuo ? "VACUOUS"
                 : (serializado ? "**SERIALIZED: an async launch waited for another client**"
                                : "launch independent (any residual wait is the sync, as native)"));
    printf("REPRO_SERIE_LANZ");
    for (size_t i = 0; i < serie.size() && i < 12; ++i) printf(",%.2f", serie[i]);
    printf("\n");
    return serializado ? 1 : 0;
}
