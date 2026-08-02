// stress4 -- los dos escenarios de estres que ningun driver existente alcanza.
//
//   h2d_d2h   H2D y D2H CONCURRENTES sobre la misma conexion. Los drivers actuales
//             (pool_churn, concgrow, rma_checksum) hacen H2D y luego D2H en serie, asi que
//             las dos direcciones nunca tienen slots vivos a la vez. Con el despachador
//             asincrono se encadenan sin sincronizar en medio, de modo que un GET iniciado
//             por el cliente coexiste con un PUT hacia el backend.
//
//   disorder  Consumo DESORDENADO entre conexiones. GVirtuS mapea una conexion por HILO
//             (Frontend::GetFrontend indexa por tid), asi que N hilos son N conexiones. Se
//             les dan tamanos MUY distintos a proposito -- de 4 a 64 MiB -- para que sus
//             transferencias terminen en un orden que no es el de emision y los acks de
//             consumo lleguen entrelazados entre conexiones.
//
// Correctitud por viaje de ida y vuelta, no por codigo de retorno: cada iteracion rellena su
// origen con un patron dependiente de (hilo, iteracion), copia H2D, copia de vuelta D2H a un
// buffer DISTINTO y compara. Un fallo de slot que entregue los bytes de otra transferencia se
// ve como discrepancia; mirar solo el cudaError_t no veria nada.
//
// Uso: stress4 <h2d_d2h|disorder> [hilos] [segundos] [iteraciones_por_lote]

#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                            \
    do {                                                                                 \
        cudaError_t e_ = (x);                                                            \
        if (e_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,             \
                         cudaGetErrorString(e_));                                        \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

static std::atomic<unsigned long long> g_xfers{0};
static std::atomic<unsigned long long> g_bad{0};
static std::atomic<bool> g_stop{false};

// Patron dependiente de (hilo, iteracion): si un slot entrega bytes de OTRA transferencia,
// el desajuste identifica de quien eran.
static void fill(unsigned char *p, size_t n, unsigned tid, unsigned it) {
    const unsigned char a = (unsigned char)(0x40 + (tid & 0x3f));
    const unsigned char b = (unsigned char)(it & 0xff);
    for (size_t i = 0; i < n; i += 4096) p[i] = a;
    for (size_t i = 2048; i < n; i += 4096) p[i] = b;
    p[0] = a;
    p[n - 1] = b;
}

static long verify(const unsigned char *p, size_t n, unsigned tid, unsigned it) {
    const unsigned char a = (unsigned char)(0x40 + (tid & 0x3f));
    const unsigned char b = (unsigned char)(it & 0xff);
    long bad = 0;
    for (size_t i = 0; i < n; i += 4096) if (p[i] != a) ++bad;
    for (size_t i = 2048; i < n; i += 4096) if (p[i] != b) ++bad;
    if (p[0] != a) ++bad;
    if (p[n - 1] != b) ++bad;
    return bad;
}

// --- H2D y D2H concurrentes -------------------------------------------------------------
// Dos streams no bloqueantes en el MISMO hilo, es decir la misma conexion. Se emiten las dos
// direcciones sin sincronizar en medio: con GVIRTUS_ASYNC_DISPATCH=1 el H2D sale
// fire-and-forget y el D2H se solapa con el. La sincronizacion va DESPUES de las dos.
static void worker_h2d_d2h(unsigned tid, size_t bytes, int lote) {
    cudaStream_t s_up, s_down;
    CK(cudaStreamCreateWithFlags(&s_up, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_down, cudaStreamNonBlocking));

    unsigned char *h_src = nullptr, *h_dst = nullptr, *d_a = nullptr, *d_b = nullptr;
    CK(cudaHostAlloc((void **)&h_src, bytes, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&h_dst, bytes, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&d_a, bytes));
    CK(cudaMalloc((void **)&d_b, bytes));

    // Sembrar d_b para que el D2H concurrente tenga contenido conocido desde la primera vuelta.
    fill(h_src, bytes, tid, 0);
    CK(cudaMemcpy(d_b, h_src, bytes, cudaMemcpyHostToDevice));

    for (unsigned it = 1; !g_stop.load(); ++it) {
        for (int k = 0; k < lote && !g_stop.load(); ++k) {
            fill(h_src, bytes, tid, it);
            // Las dos direcciones EN VUELO a la vez sobre la misma conexion.
            CK(cudaMemcpyAsync(d_a, h_src, bytes, cudaMemcpyHostToDevice, s_up));
            CK(cudaMemcpyAsync(h_dst, d_b, bytes, cudaMemcpyDeviceToHost, s_down));
            CK(cudaStreamSynchronize(s_up));
            CK(cudaStreamSynchronize(s_down));

            // El D2H debe traer lo que se sembro en d_b (iteracion 0 de este hilo).
            g_bad.fetch_add((unsigned long long)verify(h_dst, bytes, tid, 0));
            // Y el H2D debe haber dejado en d_a lo de ESTA iteracion: se comprueba con un
            // D2H aparte y sincrono, que viaja por otro camino y no se cree la palabra del
            // anterior.
            CK(cudaMemcpy(h_dst, d_a, bytes, cudaMemcpyDeviceToHost));
            g_bad.fetch_add((unsigned long long)verify(h_dst, bytes, tid, it));
            g_xfers.fetch_add(3);
        }
    }

    CK(cudaFreeHost(h_src));
    CK(cudaFreeHost(h_dst));
    CK(cudaFree(d_a));
    CK(cudaFree(d_b));
    CK(cudaStreamDestroy(s_up));
    CK(cudaStreamDestroy(s_down));
}

// --- consumo desordenado entre conexiones ------------------------------------------------
// Cada hilo (= conexion) lleva un tamano distinto, de 4 a 64 MiB. Una transferencia de 64 MiB
// tarda ~16x lo que una de 4, asi que los acks de consumo llegan en un orden que no guarda
// relacion con el de emision, y cada conexion ve entrelazados los de las demas.
static void worker_disorder(unsigned tid, size_t bytes, int lote) {
    unsigned char *h_src = nullptr, *h_dst = nullptr, *dev = nullptr;
    CK(cudaHostAlloc((void **)&h_src, bytes, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&h_dst, bytes, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&dev, bytes));

    for (unsigned it = 1; !g_stop.load(); ++it) {
        for (int k = 0; k < lote && !g_stop.load(); ++k) {
            fill(h_src, bytes, tid, it);
            std::memset(h_dst, 0, bytes);
            CK(cudaMemcpy(dev, h_src, bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(h_dst, dev, bytes, cudaMemcpyDeviceToHost));
            g_bad.fetch_add((unsigned long long)verify(h_dst, bytes, tid, it));
            g_xfers.fetch_add(2);
        }
    }

    CK(cudaFreeHost(h_src));
    CK(cudaFreeHost(h_dst));
    CK(cudaFree(dev));
}

int main(int argc, char **argv) {
    const std::string modo = (argc > 1) ? argv[1] : "h2d_d2h";
    const unsigned hilos = (argc > 2) ? (unsigned)std::atoi(argv[2]) : 4;
    const double segundos = (argc > 3) ? std::atof(argv[3]) : 20.0;
    const int lote = (argc > 4) ? std::atoi(argv[4]) : 4;

    if (modo != "h2d_d2h" && modo != "disorder") {
        std::fprintf(stderr, "stress4: modo desconocido '%s'\n", modo.c_str());
        return 2;
    }
    std::fprintf(stderr, "stress4: modo=%s hilos=%u segundos=%.0f lote=%d\n",
                 modo.c_str(), hilos, segundos, lote);

    // Tamanos muy dispares en `disorder` (4..64 MiB, todos POR ENCIMA del suelo de 4 MiB,
    // que es condicion para que tomen slots RMA); fijo de 8 MiB en `h2d_d2h`.
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < hilos; ++t) {
        size_t mb = (modo == "disorder") ? (size_t)(4u << (t % 5)) : 8u;   // 4,8,16,32,64
        size_t bytes = mb << 20;
        ts.emplace_back(modo == "h2d_d2h" ? worker_h2d_d2h : worker_disorder, t, bytes, lote);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds((long)(segundos * 1000)));
    g_stop.store(true);
    for (auto &th : ts) th.join();

    const unsigned long long x = g_xfers.load(), b = g_bad.load();
    std::printf("STRESS4_RESULT mode=%s threads=%u transfers=%llu bad=%llu %s\n",
                modo.c_str(), hilos, x, b,
                (b == 0 && x > 0) ? "PASS" : (x == 0 ? "NO_TRANSFERS" : "FAIL"));
    return (b == 0 && x > 0) ? 0 : 1;
}
