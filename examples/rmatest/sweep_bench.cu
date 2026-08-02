// sweep_bench -- barrido de tamano para justificar el umbral de 4 MiB.
//
// La pregunta que contesta: ¿donde se cruzan de verdad las curvas de los tres caminos de
// datos, es 4 MiB conservador, y dependen los resultados principales de ese valor exacto?
//
// El binario NO conoce los caminos. AM, host RMA y GPU RMA se seleccionan desde fuera con
// las variables de GVirtuS (GVIRTUS_RMA_MIN_BYTES, GVIRTUS_GPUDIRECT, GVIRTUS_RMA_ZEROCOPY),
// de modo que el MISMO binario mide los tres y ninguna diferencia puede venir del programa.
//
// Ejes que barre, todos por entorno:
//   SIZES     lista de tamanos en bytes (por defecto 4 KiB .. 64 MiB, doblando)
//   DIRS      h2d, d2h o ambos
//   MEM       pinned | pageable        (cudaHostAlloc frente a malloc)
//   REG       cached | cold            (reutilizar el buffer frente a reasignarlo por iteracion)
//   ITERS     repeticiones medidas por punto           WARMUP  descartadas antes
//   OUT       fichero CSV de salida
//
// Sobre REG=cold: reasigna Y LIBERA el buffer de host en cada iteracion, para que el
// asignador recicle direcciones y ninguna entrada de la cache de registro sea reutilizable.
// Es el mismo eje que necesita el coste del protocolo de vida ("throughput con reuso frente
// a registrar cada vez"), asi que una sola medida sirve a las dos secciones.
//
// Se publican mediana, minimo y maximo. La mediana porque estas distribuciones tienen cola
// por la primera transferencia (registro en frio); una media sobre eso describe mal el
// regimen estacionario, y en esta campana ya hubo un caso de media publicada sobre una
// poblacion multimodal.

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,           \
                         cudaGetErrorString(e_));                                      \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

static const char *env_or(const char *k, const char *d) {
    const char *v = std::getenv(k);
    return (v && v[0]) ? v : d;
}

static std::vector<size_t> parse_sizes(const char *s) {
    std::vector<size_t> v;
    std::string cur;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) v.push_back(std::strtoull(cur.c_str(), nullptr, 10));
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return v;
}

int main() {
    const bool pinned = std::strcmp(env_or("MEM", "pinned"), "pinned") == 0;
    const bool cold = std::strcmp(env_or("REG", "cached"), "cold") == 0;
    const int iters = std::atoi(env_or("ITERS", "20"));
    const int warmup = std::atoi(env_or("WARMUP", "5"));
    const char *dirs = env_or("DIRS", "both");
    const char *out = env_or("OUT", "sweep.csv");
    const char *tag = env_or("TAG", "run");

    std::string defsizes;
    for (size_t n = 4ull << 10; n <= (64ull << 20); n <<= 1) {
        if (!defsizes.empty()) defsizes.push_back(',');
        defsizes += std::to_string(n);
    }
    std::vector<size_t> sizes = parse_sizes(env_or("SIZES", defsizes.c_str()));
    if (sizes.empty()) {
        std::fprintf(stderr, "sin tamanos\n");
        return 2;
    }
    const size_t maxn = *std::max_element(sizes.begin(), sizes.end());

    CK(cudaSetDevice(0));
    void *dev = nullptr;
    CK(cudaMalloc(&dev, maxn));

    // En modo `cached` el buffer de host se asigna UNA vez y se reutiliza en todas las
    // iteraciones y tamanos: es el caso para el que existe la cache de registro. En `cold`
    // se asigna y libera dentro del bucle, asi que cada transferencia registra de cero.
    void *host_fijo = nullptr;
    if (!cold) {
        if (pinned) CK(cudaHostAlloc(&host_fijo, maxn, cudaHostAllocDefault));
        else        host_fijo = std::malloc(maxn);
        if (host_fijo == nullptr) { std::fprintf(stderr, "sin memoria de host\n"); return 2; }
        std::memset(host_fijo, 0xA5, maxn);
    }

    FILE *f = std::fopen(out, "w");
    if (f == nullptr) { std::fprintf(stderr, "no puedo escribir %s\n", out); return 2; }
    std::fprintf(f, "tag,bytes,direction,mem,reg,iters,median_s,gbytes_per_s,min_s,max_s\n");

    std::printf("  sweep_bench  mem=%s reg=%s iters=%d (+%d)  %zu tamanos\n",
                pinned ? "pinned" : "pageable", cold ? "cold" : "cached", iters, warmup,
                sizes.size());
    std::printf("  %12s %5s %13s %10s\n", "size", "dir", "median (ms)", "GB/s");

    for (size_t n : sizes) {
        for (int d = 0; d < 2; ++d) {
            const bool h2d = (d == 0);
            if (h2d && std::strcmp(dirs, "d2h") == 0) continue;
            if (!h2d && std::strcmp(dirs, "h2d") == 0) continue;

            std::vector<double> ts;
            ts.reserve(iters);
            for (int i = 0; i < warmup + iters; ++i) {
                void *h = host_fijo;
                if (cold) {
                    if (pinned) CK(cudaHostAlloc(&h, n, cudaHostAllocDefault));
                    else        h = std::malloc(n);
                    if (h == nullptr) { std::fprintf(stderr, "sin memoria\n"); return 2; }
                    std::memset(h, 0xA5, n);
                }
                void *src = h2d ? h : dev;
                void *dst = h2d ? dev : h;
                const cudaMemcpyKind kind =
                    h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;

                CK(cudaDeviceSynchronize());
                const auto t0 = std::chrono::steady_clock::now();
                CK(cudaMemcpy(dst, src, n, kind));
                CK(cudaDeviceSynchronize());
                const auto t1 = std::chrono::steady_clock::now();

                if (i >= warmup)
                    ts.push_back(std::chrono::duration<double>(t1 - t0).count());

                if (cold) {
                    // Se libera DENTRO del bucle a proposito: asi el asignador recicla
                    // direcciones y ninguna entrada cacheada sigue siendo valida.
                    if (pinned) CK(cudaFreeHost(h));
                    else        std::free(h);
                }
            }

            std::sort(ts.begin(), ts.end());
            const double med = ts[ts.size() / 2];
            const double gbs = (double)n / med / 1e9;
            const char *dname = h2d ? "h2d" : "d2h";
            std::fprintf(f, "%s,%zu,%s,%s,%s,%d,%.9f,%.4f,%.9f,%.9f\n", tag, n, dname,
                         pinned ? "pinned" : "pageable", cold ? "cold" : "cached", iters,
                         med, gbs, ts.front(), ts.back());
            std::fflush(f);
            if (n >= (1u << 20))
                std::printf("  %10zu M %5s %13.4f %10.2f\n", n >> 20, dname, med * 1e3, gbs);
            else
                std::printf("  %10zu K %5s %13.4f %10.2f\n", n >> 10, dname, med * 1e3, gbs);
        }
    }

    std::fclose(f);
    if (host_fijo) { if (pinned) CK(cudaFreeHost(host_fijo)); else std::free(host_fijo); }
    CK(cudaFree(dev));
    std::printf("\n  escrito: %s\n", out);
    return 0;
}
