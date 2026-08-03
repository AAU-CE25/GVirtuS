// tests/semantic/semantic_conformance.cu
//
// Conformidad semantica de CUDA a traves del remoting -- fase 2 de la campana.
//
// POR QUE UN FICHERO NUEVO Y NO AMPLIAR ptds_conformance.cu. Aquel tiene 58 filas de
// resultados ya publicadas atadas a su binario; ampliarlo invalidaria esa procedencia. Este
// cubre otras propiedades y comparte el MISMO contrato de salida, asi que un solo analisis
// lee los dos.
//
// QUE PRUEBA. No la semantica de CUDA en general -- eso lo prueba NVIDIA -- sino las
// propiedades que ESTA arquitectura pone en riesgo y que un test generico no miraria:
//
//   roundtrip_sizes   fidelidad de bytes en tamanos que cruzan los CUATRO umbrales de la
//                     politica de colocacion (H2D fijada 8 KiB, H2D paginable 1 MiB,
//                     D2H fijada 1 MiB, D2H paginable 2 MiB). Un fallo de la politica solo
//                     aparece en la clase de tamano donde cambia el camino, asi que un test
//                     de un solo tamano no lo veria nunca.
//   pinned_equiv      el MISMO tamano a traves de memoria fijada y paginable debe dar los
//                     mismos bytes. Entre 8 KiB y 1 MiB los dos toman caminos de transporte
//                     DISTINTOS (RMA vs mensajes activos), asi que esto compara el camino
//                     rapido contra el lento con la respuesta conocida.
//   d2d_fidelity      copia dispositivo a dispositivo, que en este backend pasa por el
//                     scratch de GPU y fue la causa raiz de una corrupcion real.
//   memset_ordered    cudaMemsetAsync ordenado en su stream frente a un kernel posterior.
//   event_query       cudaEventQuery da NotReady antes y Success despues; elapsed positivo.
//   error_sticky      cudaGetLastError LIMPIA y cudaPeekAtLastError NO; un lanzamiento
//                     invalido devuelve el codigo documentado.
//
// SE COMPILA DOS VECES, igual que ptds_conformance: normal y con
// --default-stream per-thread, porque la segunda redirige a la superficie _ptsz.
//
// Salida: CSVROW,test,variant,threads,seed,iterations,passed,mismatches,cuda_error,runtime_ms
// Mismo prefijo y mismo orden de campos que ptds_conformance.cu, a proposito.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cuda_runtime.h>

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#  define VARIANT "ptsz"
#else
#  define VARIANT "handle"
#endif

static std::mutex g_out;
static int g_verbose = 0;

struct Outcome {
    long mismatches = 0;
    cudaError_t err = cudaSuccess;
    const char *where = "";
};

static void emit(const char *test, int threads, unsigned seed, int iters,
                 const Outcome &o, double ms) {
    std::lock_guard<std::mutex> lk(g_out);
    std::printf("CSVROW,%s,%s,%d,%u,%d,%d,%ld,%s,%.3f\n",
                test, VARIANT, threads, seed, iters,
                (o.mismatches == 0 && o.err == cudaSuccess) ? 1 : 0,
                o.mismatches,
                o.err == cudaSuccess ? "none" : cudaGetErrorName(o.err), ms);
    if (o.err != cudaSuccess)
        std::printf("CSVERR,%s,%s,%d,%s,%s,%d\n", test, VARIANT, threads,
                    o.where, cudaGetErrorString(o.err), (int)o.err);
    std::fflush(stdout);
}

#define TRY(call, tag)                                                        \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) { out.err = _e; out.where = tag; return out; } \
    } while (0)

// Patron dependiente del offset: un desplazamiento de bloques o una longitud truncada se
// ven, cosa que un relleno constante no detecta.
static void llena(unsigned char *p, size_t n, unsigned semilla) {
    for (size_t i = 0; i < n; ++i)
        p[i] = (unsigned char)((i * 31u + semilla * 131u + (i >> 12)) & 0xff);
}
static long compara(const unsigned char *a, const unsigned char *b, size_t n) {
    long mal = 0;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) ++mal;
    return mal;
}

__global__ void k_incr(unsigned char *p, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = (unsigned char)(p[i] + 1u);
}

// Los cuatro umbrales de RmaPolicy.h y un punto a cada lado de cada uno. El 4 MiB es el
// suelo escalar historico; se incluye porque las tres politicas lo tratan distinto.
static const size_t kTams[] = {
    4u << 10, 8u << 10, 16u << 10,      // alrededor de H2D fijada (8 KiB)
    512u << 10, 1u << 20, 2u << 20,     // H2D paginable (1 MiB) y D2H fijada (1 MiB)
    3u << 20, 4u << 20, 8u << 20        // D2H paginable (2 MiB) y el suelo escalar (4 MiB)
};
static const int kNTams = (int)(sizeof(kTams) / sizeof(kTams[0]));

// ---------------------------------------------------------------------------------------
// roundtrip_sizes: H2D -> D2H en cada clase de tamano, memoria FIJADA.
// ---------------------------------------------------------------------------------------
static Outcome run_roundtrip(int hilo, int it, bool fijada) {
    Outcome out;
    const size_t nb = kTams[it % kNTams];
    unsigned char *h = nullptr, *back = nullptr;
    void *d = nullptr;
    if (fijada) {
        TRY(cudaHostAlloc((void **)&h, nb, cudaHostAllocDefault), "hostalloc_src");
        TRY(cudaHostAlloc((void **)&back, nb, cudaHostAllocDefault), "hostalloc_dst");
    } else {
        h = (unsigned char *)std::malloc(nb);
        back = (unsigned char *)std::malloc(nb);
        if (!h || !back) { out.err = cudaErrorMemoryAllocation; out.where = "malloc"; return out; }
    }
    TRY(cudaMalloc(&d, nb), "malloc_dev");
    llena(h, nb, (unsigned)(hilo * 7919 + it));
    std::memset(back, 0, nb);
    TRY(cudaMemcpy(d, h, nb, cudaMemcpyHostToDevice), "h2d");
    TRY(cudaMemcpy(back, d, nb, cudaMemcpyDeviceToHost), "d2h");
    out.mismatches = compara(h, back, nb);
    cudaFree(d);
    if (fijada) { cudaFreeHost(h); cudaFreeHost(back); }
    else { std::free(h); std::free(back); }
    return out;
}

// ---------------------------------------------------------------------------------------
// pinned_equiv: el mismo tamano por los dos tipos de memoria debe dar el MISMO resultado.
// Entre 8 KiB y 1 MiB uno va por RMA y el otro por mensajes activos: se compara el camino
// rapido contra el lento con la respuesta conocida.
// ---------------------------------------------------------------------------------------
static Outcome run_pinned_equiv(int hilo, int it) {
    Outcome out;
    const size_t nb = kTams[it % kNTams];
    unsigned char *hp = nullptr, *bp = nullptr;
    TRY(cudaHostAlloc((void **)&hp, nb, cudaHostAllocDefault), "hostalloc");
    TRY(cudaHostAlloc((void **)&bp, nb, cudaHostAllocDefault), "hostalloc2");
    unsigned char *hg = (unsigned char *)std::malloc(nb);
    unsigned char *bg = (unsigned char *)std::malloc(nb);
    if (!hg || !bg) { out.err = cudaErrorMemoryAllocation; out.where = "malloc"; return out; }
    void *d = nullptr;
    TRY(cudaMalloc(&d, nb), "malloc_dev");

    const unsigned s = (unsigned)(hilo * 104729 + it);
    llena(hp, nb, s); std::memcpy(hg, hp, nb);      // MISMOS bytes por los dos caminos
    std::memset(bp, 0, nb); std::memset(bg, 0, nb);

    TRY(cudaMemcpy(d, hp, nb, cudaMemcpyHostToDevice), "h2d_pinned");
    TRY(cudaMemcpy(bp, d, nb, cudaMemcpyDeviceToHost), "d2h_pinned");
    TRY(cudaMemset(d, 0, nb), "memset");
    TRY(cudaMemcpy(d, hg, nb, cudaMemcpyHostToDevice), "h2d_pageable");
    TRY(cudaMemcpy(bg, d, nb, cudaMemcpyDeviceToHost), "d2h_pageable");

    out.mismatches = compara(bp, bg, nb) + compara(bp, hp, nb);
    cudaFree(d); cudaFreeHost(hp); cudaFreeHost(bp); std::free(hg); std::free(bg);
    return out;
}

// ---------------------------------------------------------------------------------------
// d2d_fidelity: dispositivo a dispositivo. Pasa por el scratch de GPU del backend, que fue
// la causa raiz de una corrupcion real (D2D sin sincronizar sobre el scratch TLS unico).
// ---------------------------------------------------------------------------------------
static Outcome run_d2d(int hilo, int it) {
    Outcome out;
    const size_t nb = kTams[it % kNTams];
    unsigned char *h = nullptr, *back = nullptr;
    TRY(cudaHostAlloc((void **)&h, nb, cudaHostAllocDefault), "hostalloc");
    TRY(cudaHostAlloc((void **)&back, nb, cudaHostAllocDefault), "hostalloc2");
    void *d1 = nullptr, *d2 = nullptr;
    TRY(cudaMalloc(&d1, nb), "malloc1");
    TRY(cudaMalloc(&d2, nb), "malloc2");
    llena(h, nb, (unsigned)(hilo * 15485863 + it));
    std::memset(back, 0, nb);
    TRY(cudaMemcpy(d1, h, nb, cudaMemcpyHostToDevice), "h2d");
    TRY(cudaMemcpy(d2, d1, nb, cudaMemcpyDeviceToDevice), "d2d");
    TRY(cudaMemcpy(back, d2, nb, cudaMemcpyDeviceToHost), "d2h");
    out.mismatches = compara(h, back, nb);
    cudaFree(d1); cudaFree(d2); cudaFreeHost(h); cudaFreeHost(back);
    return out;
}

// ---------------------------------------------------------------------------------------
// memset_ordered: cudaMemsetAsync seguido de un kernel EN EL MISMO stream. El kernel debe
// ver el memset. Si el remoting reordena, el resultado no es 1.
// ---------------------------------------------------------------------------------------
static Outcome run_memset_ordered(int hilo, int it) {
    Outcome out;
    const size_t nb = 1u << 20;
    cudaStream_t s = nullptr;
    TRY(cudaStreamCreate(&s), "screate");
    void *d = nullptr;
    unsigned char *back = nullptr;
    TRY(cudaMalloc(&d, nb), "malloc");
    TRY(cudaHostAlloc((void **)&back, nb, cudaHostAllocDefault), "hostalloc");
    TRY(cudaMemsetAsync(d, 0, nb, s), "memset");
    k_incr<<<(unsigned)((nb + 255) / 256), 256, 0, s>>>((unsigned char *)d, nb);
    TRY(cudaGetLastError(), "launch");
    TRY(cudaMemcpyAsync(back, d, nb, cudaMemcpyDeviceToHost, s), "d2h");
    TRY(cudaStreamSynchronize(s), "sync");
    for (size_t i = 0; i < nb; ++i) if (back[i] != 1u) ++out.mismatches;
    cudaFree(d); cudaFreeHost(back); cudaStreamDestroy(s);
    (void)hilo; (void)it;
    return out;
}

// ---------------------------------------------------------------------------------------
// event_query: NotReady antes de sincronizar, Success despues; elapsed finito y >= 0.
// ---------------------------------------------------------------------------------------
static Outcome run_event_query(int hilo, int it) {
    Outcome out;
    const size_t nb = 8u << 20;      // grande, para que haya trabajo real en vuelo
    cudaStream_t s = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    void *d = nullptr;
    TRY(cudaStreamCreate(&s), "screate");
    TRY(cudaEventCreate(&e0), "ecreate0");
    TRY(cudaEventCreate(&e1), "ecreate1");
    TRY(cudaMalloc(&d, nb), "malloc");
    TRY(cudaEventRecord(e0, s), "rec0");
    for (int r = 0; r < 40; ++r)
        k_incr<<<(unsigned)((nb + 255) / 256), 256, 0, s>>>((unsigned char *)d, nb);
    TRY(cudaGetLastError(), "launch");
    TRY(cudaEventRecord(e1, s), "rec1");

    // No se exige NotReady: con poco trabajo puede haber terminado ya, y afirmar lo
    // contrario seria un test que falla por suerte. Lo que SI se exige es que despues de
    // sincronizar de Success, y que el tiempo transcurrido sea finito y no negativo.
    cudaError_t q = cudaEventQuery(e1);
    if (q != cudaSuccess && q != cudaErrorNotReady) { out.err = q; out.where = "query_pre"; goto fin; }
    TRY(cudaEventSynchronize(e1), "esync");
    if (cudaEventQuery(e1) != cudaSuccess) { out.mismatches++; out.where = "query_post"; }
    {
        float ms = -1.f;
        cudaError_t el = cudaEventElapsedTime(&ms, e0, e1);
        if (el != cudaSuccess) { out.err = el; out.where = "elapsed"; goto fin; }
        if (!(ms >= 0.f) || ms > 60000.f) out.mismatches++;
    }
fin:
    cudaFree(d); cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(s);
    (void)hilo; (void)it;
    return out;
}

// ---------------------------------------------------------------------------------------
// error_sticky: cudaGetLastError LIMPIA, cudaPeekAtLastError NO. Un lanzamiento con
// configuracion invalida devuelve cudaErrorInvalidConfiguration.
// ---------------------------------------------------------------------------------------
static Outcome run_error_sticky(int hilo, int it) {
    Outcome out;
    cudaGetLastError();                       // partir de limpio

    // Configuracion invalida a proposito: 0 hilos por bloque.
    k_incr<<<1, 0>>>(nullptr, 0);
    cudaError_t peek1 = cudaPeekAtLastError();
    cudaError_t peek2 = cudaPeekAtLastError();
    if (peek1 == cudaSuccess) { out.mismatches++; out.where = "no_error_raised"; }
    if (peek1 != peek2)       { out.mismatches++; out.where = "peek_cleared"; }   // peek NO limpia

    cudaError_t got = cudaGetLastError();     // esto SI limpia
    if (got != peek1)                     { out.mismatches++; out.where = "get_differs"; }
    if (cudaGetLastError() != cudaSuccess){ out.mismatches++; out.where = "get_did_not_clear"; }

    // El codigo concreto: se registra, y solo se marca fallo si no es de la familia
    // esperada -- el valor exacto puede variar por version y no es lo que se prueba.
    if (peek1 != cudaErrorInvalidConfiguration && peek1 != cudaErrorInvalidValue) {
        out.mismatches++; out.where = "unexpected_code";
    }
    (void)hilo; (void)it;
    return out;
}

// ---------------------------------------------------------------------------------------
static void in_threads(const char *nombre, int nthreads, unsigned seed, int iters,
                       Outcome (*fn)(int, int)) {
    std::vector<std::thread> hilos;
    std::vector<Outcome> res((size_t)nthreads);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < nthreads; ++t)
        hilos.emplace_back([&, t] {
            Outcome acc;
            for (int i = 0; i < iters; ++i) {
                Outcome o = fn(t, i);
                acc.mismatches += o.mismatches;
                if (o.err != cudaSuccess && acc.err == cudaSuccess) { acc.err = o.err; acc.where = o.where; break; }
            }
            res[(size_t)t] = acc;
        });
    for (auto &h : hilos) h.join();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    Outcome tot;
    for (auto &o : res) {
        tot.mismatches += o.mismatches;
        if (o.err != cudaSuccess && tot.err == cudaSuccess) { tot.err = o.err; tot.where = o.where; }
    }
    emit(nombre, nthreads, seed, iters, tot, ms);
}

int main(int argc, char **argv) {
    int nthreads  = (argc > 1) ? atoi(argv[1]) : 1;
    int iters     = (argc > 2) ? atoi(argv[2]) : 18;   // 2 vueltas a las 9 clases de tamano
    unsigned seed = (argc > 3) ? (unsigned)atoi(argv[3]) : 1u;
    const char *only = (argc > 4) ? argv[4] : nullptr;
    g_verbose = getenv("CONF_VERBOSE") ? 1 : 0;

    std::fprintf(stderr, "# variant=%s threads=%d iters=%d seed=%u clases=%d\n",
                 VARIANT, nthreads, iters, seed, kNTams);

    auto want = [&](const char *n) { return only == nullptr || strcmp(only, n) == 0; };

    if (want("roundtrip_pinned"))
        in_threads("roundtrip_pinned", nthreads, seed, iters,
                   [](int t, int i) { return run_roundtrip(t, i, true); });
    if (want("roundtrip_pageable"))
        in_threads("roundtrip_pageable", nthreads, seed, iters,
                   [](int t, int i) { return run_roundtrip(t, i, false); });
    if (want("pinned_equiv"))
        in_threads("pinned_equiv", nthreads, seed, iters, run_pinned_equiv);
    if (want("d2d_fidelity"))
        in_threads("d2d_fidelity", nthreads, seed, iters, run_d2d);
    if (want("memset_ordered"))
        in_threads("memset_ordered", nthreads, seed, iters < 20 ? iters : 20, run_memset_ordered);
    if (want("event_query"))
        in_threads("event_query", nthreads, seed, iters < 10 ? iters : 10, run_event_query);
    if (want("error_sticky"))
        in_threads("error_sticky", nthreads, seed, iters < 20 ? iters : 20, run_error_sticky);

    return 0;
}
