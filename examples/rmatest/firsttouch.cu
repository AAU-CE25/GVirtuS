// firsttouch.cu — N hilos que hacen su PRIMERA llamada CUDA en el mismo instante.
//
// GVirtuS mapea una conexion por hilo (Frontend::GetFrontend indexa por tid) y la construye
// perezosamente en la primera llamada. Una barrera antes de esa llamada concentra TODAS las
// inicializaciones de Frontend en la misma ventana, que es exactamente donde ThreadSanitizer
// senala la carrera sobre el mapa tid->Frontend (find() sin cerrojo concurrente con insert()).
//
// El sintoma que se busca NO es lentitud: es un hilo que se queda con un Frontend equivocado
// o a medio construir. Por eso cada hilo verifica un viaje de ida y vuelta con un patron que
// depende de SU identificador: si dos hilos comparten conexion o uno lee el estado de otro,
// los bytes no cuadran y se ve, en vez de fallar en silencio o mucho despues.
//
//   uso: firsttouch <hilos> [bytes]
//   salida: FIRSTTOUCH hilos=N ok=K err_cuda=E malos=B  -> rc 0 solo si E=0 y B=0
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static std::atomic<int> g_listos{0};
static std::atomic<bool> g_salida{false};
static std::atomic<int> g_err{0};
static std::atomic<long> g_malos{0};
static std::atomic<int> g_ok{0};

static void hilo(int id, size_t bytes) {
    std::vector<unsigned char> src(bytes), dst(bytes);
    const unsigned char pat = (unsigned char)(0x40 + (id & 0x3f));
    std::memset(src.data(), pat, bytes);
    src[bytes - 1] = (unsigned char)(id & 0xff);

    // Barrera: nadie toca CUDA hasta que todos los hilos estan listos.
    g_listos.fetch_add(1);
    while (!g_salida.load(std::memory_order_acquire)) std::this_thread::yield();

    cudaStream_t s = nullptr;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "thread %d: cudaStreamCreateWithFlags -> %s\n", id,
                     cudaGetErrorString(e));
        g_err.fetch_add(1);
        return;
    }
    void *d = nullptr;
    e = cudaMalloc(&d, bytes);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "thread %d: cudaMalloc -> %s\n", id, cudaGetErrorString(e));
        g_err.fetch_add(1);
        return;
    }
    e = cudaMemcpy(d, src.data(), bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "thread %d: H2D -> %s\n", id, cudaGetErrorString(e));
        g_err.fetch_add(1);
        return;
    }
    // Sincronizacion sobre el stream por hilo: es la llamada que aborta en llama
    // (cudaStreamSynchronize sobre cudaStreamPerThread) cuando el estado por hilo no es
    // el que el hilo cree.
    e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "thread %d: cudaStreamSynchronize -> %s\n", id,
                     cudaGetErrorString(e));
        g_err.fetch_add(1);
        return;
    }
    e = cudaMemcpy(dst.data(), d, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "thread %d: D2H -> %s\n", id, cudaGetErrorString(e));
        g_err.fetch_add(1);
        return;
    }
    long malos = 0;
    for (size_t k = 0; k < bytes; k += 4096)
        if (dst[k] != pat) ++malos;
    if (dst[bytes - 1] != (unsigned char)(id & 0xff)) ++malos;
    if (malos) {
        std::fprintf(stderr, "thread %d: %ld positions hold another thread's data\n", id, malos);
        g_malos.fetch_add(malos);
    }
    cudaFree(d);
    cudaStreamDestroy(s);
    g_ok.fetch_add(1);
}

int main(int argc, char **argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 16;
    const size_t bytes = (argc > 2) ? (size_t)atoll(argv[2]) : (1u << 20);

    std::vector<std::thread> hs;
    hs.reserve(n);
    for (int i = 0; i < n; ++i) hs.emplace_back(hilo, i, bytes);

    while (g_listos.load() < n) std::this_thread::yield();
    g_salida.store(true, std::memory_order_release);

    for (auto &t : hs) t.join();

    std::printf("FIRSTTOUCH hilos=%d ok=%d err_cuda=%d malos=%ld\n", n, g_ok.load(),
                g_err.load(), g_malos.load());
    std::fflush(stdout);
    return (g_err.load() == 0 && g_malos.load() == 0) ? 0 : 1;
}
