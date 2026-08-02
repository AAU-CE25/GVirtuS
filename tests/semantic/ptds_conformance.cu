// tests/semantic/ptds_conformance.cu
//
// Conformidad del stream por defecto por hilo (PTDS) a traves del remoting.
//
// Se compila DOS veces y las dos variantes prueban cosas distintas. Esto es el nucleo del
// test y la razon de que el reproductor de un solo hilo de la sesion anterior no encontrara
// nada:
//
//   variante "handle"  (compilacion normal)
//       Las llamadas van a los puntos de entrada legacy y se les pasa el centinela 0x2
//       explicitamente:  cudaMemcpyAsync(..., cudaStreamPerThread).
//       Ejercita: la traduccion del valor centinela a traves del cable.
//
//   variante "ptsz"    (compilada con --default-stream per-thread)
//       Las cabeceras de CUDA redirigen cudaMemcpyAsync -> cudaMemcpyAsync_ptsz,
//       cudaEventRecord -> cudaEventRecord_ptsz, cudaGraphLaunch -> cudaGraphLaunch_ptsz...
//       Ejercita: la SUPERFICIE _ptsz del frontend, que hoy tiene 5 puntos de entrada
//       reales y 49 stubs silenciosos que devuelven cudaErrorNotSupported (71).
//
// La variante "handle" pasa hoy. La variante "ptsz" es la que puede fallar, y falla en
// silencio: los stubs solo registran si se compila con GVIRTUS_LOG_STUB_CALLS.
//
// Salida: lineas CSV con prefijo fijo. Nada de parsear texto para humanos.
//   CSVROW,test,variant,threads,seed,iterations,passed,mismatches,cuda_error,runtime_ms

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
#include <cuda.h>

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
#  define VARIANT "ptsz"
#else
#  define VARIANT "handle"
#endif

// ---------------------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------------------
static std::mutex g_out;
static int g_drv_code;      // definido junto a run_driver_ptds; declarado aqui para emit()
static int g_iters = 1000;
static size_t g_bytes = 1u << 20;   // 1 MiB por defecto: por encima del suelo RMA de 8 KiB
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
                o.err == cudaSuccess ? "none" : cudaGetErrorName(o.err),
                ms);
    if (o.err != cudaSuccess) {
        std::printf("CSVERR,%s,%s,%d,%s,%s,%d,drv=%d\n", test, VARIANT, threads,
                    o.where, cudaGetErrorString(o.err), (int)o.err, g_drv_code);
    }
    std::fflush(stdout);
}

#define TRY(call, tag)                                                            \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) { out.err = _e; out.where = tag; return out; }     \
    } while (0)

// ---------------------------------------------------------------------------------------
// Kernels. B depende de A: si el orden no se preserva, el resultado cambia.
// ---------------------------------------------------------------------------------------
__global__ void kA(unsigned *d, size_t n, unsigned tag) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] + tag;
}
__global__ void kB(unsigned *d, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] * 2u + 1u;
}

// ---------------------------------------------------------------------------------------
// Secuencia base: H2D async -> kA -> kB -> D2H async -> sync -> validacion byte a byte.
// `s` es el stream bajo prueba.
// ---------------------------------------------------------------------------------------
static Outcome run_sequence(cudaStream_t s, int tid, int iters, bool sync_via_stream) {
    Outcome out;
    const size_t n = g_bytes / sizeof(unsigned);
    unsigned *h_in = nullptr, *h_out = nullptr, *d = nullptr;

    TRY(cudaHostAlloc((void **)&h_in, g_bytes, cudaHostAllocDefault), "hostalloc_in");
    TRY(cudaHostAlloc((void **)&h_out, g_bytes, cudaHostAllocDefault), "hostalloc_out");
    TRY(cudaMalloc((void **)&d, g_bytes), "malloc");

    const int block = 256;
    const int grid = (int)((n + block - 1) / block);

    for (int it = 0; it < iters; ++it) {
        const unsigned tag = (unsigned)(tid * 1000003u + it * 7919u + 1u);
        for (size_t i = 0; i < n; ++i) h_in[i] = (unsigned)(i * 2654435761u + tag);

        TRY(cudaMemcpyAsync(d, h_in, g_bytes, cudaMemcpyHostToDevice, s), "h2d");
        kA<<<grid, block, 0, s>>>(d, n, tag);
        kB<<<grid, block, 0, s>>>(d, n);
        TRY(cudaMemcpyAsync(h_out, d, g_bytes, cudaMemcpyDeviceToHost, s), "d2h");

        if (sync_via_stream) TRY(cudaStreamSynchronize(s), "streamsync");
        else                 TRY(cudaDeviceSynchronize(), "devsync");

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) { out.err = le; out.where = "lasterror"; goto done; }

        for (size_t i = 0; i < n; ++i) {
            const unsigned expect = ((unsigned)(i * 2654435761u + tag) + tag) * 2u + 1u;
            if (h_out[i] != expect) {
                ++out.mismatches;
                if (out.mismatches == 1 && g_verbose)
                    std::fprintf(stderr, "[MISMATCH] tid=%d it=%d i=%zu got=%u want=%u\n",
                                 tid, it, i, h_out[i], expect);
                break;   // una por iteracion basta para contar la iteracion como mala
            }
        }
    }
done:
    cudaFree(d); cudaFreeHost(h_in); cudaFreeHost(h_out);
    return out;
}

// ---------------------------------------------------------------------------------------
// Dependencia entre streams: productor en s1, cudaEventRecord, cudaStreamWaitEvent en s2,
// consumidor en s2. En la variante ptsz, cudaEventRecord y cudaStreamWaitEvent son STUBS.
// ---------------------------------------------------------------------------------------
static Outcome run_event_dep(int tid, int iters) {
    Outcome out;
    const size_t n = g_bytes / sizeof(unsigned);
    unsigned *h_in = nullptr, *h_out = nullptr, *d = nullptr;
    cudaStream_t s1 = nullptr, s2 = nullptr;
    cudaEvent_t ev = nullptr;

    TRY(cudaHostAlloc((void **)&h_in, g_bytes, cudaHostAllocDefault), "hostalloc_in");
    TRY(cudaHostAlloc((void **)&h_out, g_bytes, cudaHostAllocDefault), "hostalloc_out");
    TRY(cudaMalloc((void **)&d, g_bytes), "malloc");
    TRY(cudaStreamCreate(&s1), "screate1");
    TRY(cudaStreamCreate(&s2), "screate2");
    TRY(cudaEventCreate(&ev), "evcreate");

    const int block = 256;
    const int grid = (int)((n + block - 1) / block);

    for (int it = 0; it < iters; ++it) {
        const unsigned tag = (unsigned)(tid * 1000003u + it * 7919u + 1u);
        for (size_t i = 0; i < n; ++i) h_in[i] = (unsigned)(i * 2654435761u + tag);

        TRY(cudaMemcpyAsync(d, h_in, g_bytes, cudaMemcpyHostToDevice, s1), "h2d");
        kA<<<grid, block, 0, s1>>>(d, n, tag);
        TRY(cudaEventRecord(ev, s1), "eventrecord");
        TRY(cudaStreamWaitEvent(s2, ev, 0), "streamwaitevent");
        kB<<<grid, block, 0, s2>>>(d, n);
        TRY(cudaMemcpyAsync(h_out, d, g_bytes, cudaMemcpyDeviceToHost, s2), "d2h");
        TRY(cudaStreamSynchronize(s2), "sync2");

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) { out.err = le; out.where = "lasterror"; goto done; }

        for (size_t i = 0; i < n; ++i) {
            const unsigned expect = ((unsigned)(i * 2654435761u + tag) + tag) * 2u + 1u;
            if (h_out[i] != expect) { ++out.mismatches; break; }
        }
    }
done:
    if (ev) cudaEventDestroy(ev);
    if (s1) cudaStreamDestroy(s1);
    if (s2) cudaStreamDestroy(s2);
    cudaFree(d); cudaFreeHost(h_in); cudaFreeHost(h_out);
    return out;
}

// ---------------------------------------------------------------------------------------
// Graph capturado en un stream explicito y LANZADO sobre PTDS. En la variante ptsz esto
// llega a cudaGraphLaunch_ptsz, que hoy es un stub.
// ---------------------------------------------------------------------------------------
static Outcome run_graph_ptds(int tid, int iters) {
    Outcome out;
    const size_t n = g_bytes / sizeof(unsigned);
    unsigned *h_in = nullptr, *h_out = nullptr, *d = nullptr;
    cudaStream_t cap = nullptr;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;

    TRY(cudaHostAlloc((void **)&h_in, g_bytes, cudaHostAllocDefault), "hostalloc_in");
    TRY(cudaHostAlloc((void **)&h_out, g_bytes, cudaHostAllocDefault), "hostalloc_out");
    TRY(cudaMalloc((void **)&d, g_bytes), "malloc");
    TRY(cudaStreamCreate(&cap), "screate");

    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    const unsigned tag0 = (unsigned)(tid * 1000003u + 1u);

    TRY(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal), "begincapture");
    cudaMemcpyAsync(d, h_in, g_bytes, cudaMemcpyHostToDevice, cap);
    kA<<<grid, block, 0, cap>>>(d, n, tag0);
    kB<<<grid, block, 0, cap>>>(d, n);
    cudaMemcpyAsync(h_out, d, g_bytes, cudaMemcpyDeviceToHost, cap);
    TRY(cudaStreamEndCapture(cap, &g), "endcapture");
    TRY(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
    TRY(cudaGraphUpload(ge, cap), "upload");

    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < n; ++i) h_in[i] = (unsigned)(i * 2654435761u + tag0);
        // El lanzamiento va sobre el stream por defecto por hilo.
        TRY(cudaGraphLaunch(ge, cudaStreamPerThread), "graphlaunch");
        TRY(cudaStreamSynchronize(cudaStreamPerThread), "sync");

        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) { out.err = le; out.where = "lasterror"; goto done; }

        for (size_t i = 0; i < n; ++i) {
            const unsigned expect = ((unsigned)(i * 2654435761u + tag0) + tag0) * 2u + 1u;
            if (h_out[i] != expect) { ++out.mismatches; break; }
        }
    }
done:
    if (ge) cudaGraphExecDestroy(ge);
    if (g)  cudaGraphDestroy(g);
    if (cap) cudaStreamDestroy(cap);
    cudaFree(d); cudaFreeHost(h_in); cudaFreeHost(h_out);
    return out;
}

// ---------------------------------------------------------------------------------------
// Driver API sobre CU_STREAM_PER_THREAD.
// ---------------------------------------------------------------------------------------
// El contexto primario hay que RETENERLO y hacerlo corriente en ESTE hilo. Sin eso la
// llamada falla con CUDA_ERROR_NOT_SUPPORTED (801) tambien en nativo, que es un fallo del
// test y no del sistema bajo prueba. Se detecto validando en nativo antes de acusar a nadie.
static Outcome run_driver_ptds(int tid, int iters) {
    Outcome out;
    (void)tid;
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuInit"; return out; }

    CUdevice dev;
    r = cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuDeviceGet"; return out; }

    CUcontext ctx = nullptr;
    r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuDevicePrimaryCtxRetain"; return out; }
    r = cuCtxSetCurrent(ctx);
    if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuCtxSetCurrent"; goto release; }

    // Trabajo real sobre el stream por defecto por hilo de la Driver API, no solo un sync
    // vacio: reservar, H2D, D2H y comparar.
    {
        CUdeviceptr d = 0;
        const size_t nb = 65536;
        std::vector<unsigned char> hin(nb), hout(nb, 0);
        for (size_t i = 0; i < nb; ++i) hin[i] = (unsigned char)(i * 31u + 7u);

        r = cuMemAlloc(&d, nb);
        if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuMemAlloc"; goto release; }

        for (int it = 0; it < iters; ++it) {
            r = cuMemcpyHtoDAsync(d, hin.data(), nb, CU_STREAM_PER_THREAD);
            if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuMemcpyHtoDAsync(PER_THREAD)"; break; }
            r = cuMemcpyDtoHAsync(hout.data(), d, nb, CU_STREAM_PER_THREAD);
            if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuMemcpyDtoHAsync(PER_THREAD)"; break; }
            r = cuStreamSynchronize(CU_STREAM_PER_THREAD);
            if (r != CUDA_SUCCESS) { g_drv_code = r; out.err = cudaErrorNotSupported; out.where = "cuStreamSynchronize(PER_THREAD)"; break; }
            if (std::memcmp(hin.data(), hout.data(), nb) != 0) ++out.mismatches;
        }
        cuMemFree(d);
    }
release:
    cuDevicePrimaryCtxRelease(dev);
    return out;
}

// ---------------------------------------------------------------------------------------
// Lanzador multihilo. Cada hilo tiene sus propios buffers.
// ---------------------------------------------------------------------------------------
template <typename F>
static void in_threads(const char *test, int nthreads, unsigned seed, int iters, F fn) {
    std::vector<Outcome> res(nthreads);
    auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> th;
        th.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t)
            th.emplace_back([&, t]() { res[t] = fn(t, iters); });
        for (auto &x : th) x.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Outcome agg;
    for (auto &r : res) {
        agg.mismatches += r.mismatches;
        if (agg.err == cudaSuccess && r.err != cudaSuccess) { agg.err = r.err; agg.where = r.where; }
    }
    emit(test, nthreads, seed, iters, agg, ms);
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 1;
    g_iters      = (argc > 2) ? atoi(argv[2]) : 100;
    g_bytes      = (argc > 3) ? (size_t)atoll(argv[3]) : (1u << 20);
    unsigned seed= (argc > 4) ? (unsigned)atoi(argv[4]) : 1u;
    const char *only = (argc > 5) ? argv[5] : nullptr;
    g_verbose    = getenv("PTDS_VERBOSE") ? 1 : 0;

    std::fprintf(stderr, "# variant=%s threads=%d iters=%d bytes=%zu seed=%u\n",
                 VARIANT, nthreads, g_iters, g_bytes, seed);
    std::fprintf(stderr, "# cudaStreamPerThread=%p cudaStreamLegacy=%p\n",
                 (void *)cudaStreamPerThread, (void *)cudaStreamLegacy);

    auto want = [&](const char *n) { return only == nullptr || strcmp(only, n) == 0; };

    if (want("order_ptds"))
        in_threads("order_ptds", nthreads, seed, g_iters,
                   [](int t, int it) { return run_sequence(cudaStreamPerThread, t, it, true); });
    if (want("order_null"))
        in_threads("order_null", nthreads, seed, g_iters,
                   [](int t, int it) { return run_sequence(0, t, it, true); });
    if (want("order_legacy"))
        in_threads("order_legacy", nthreads, seed, g_iters,
                   [](int t, int it) { return run_sequence(cudaStreamLegacy, t, it, true); });
    if (want("order_explicit"))
        in_threads("order_explicit", nthreads, seed, g_iters, [](int t, int it) {
            Outcome o; cudaStream_t s = nullptr;
            cudaError_t e = cudaStreamCreate(&s);
            if (e != cudaSuccess) { o.err = e; o.where = "screate"; return o; }
            o = run_sequence(s, t, it, true);
            cudaStreamDestroy(s);
            return o;
        });
    if (want("event_crossstream"))
        in_threads("event_crossstream", nthreads, seed, g_iters, run_event_dep);
    if (want("graph_ptds"))
        in_threads("graph_ptds", nthreads, seed, g_iters < 20 ? g_iters : 20, run_graph_ptds);
    if (want("driver_ptds"))
        in_threads("driver_ptds", nthreads, seed, g_iters < 50 ? g_iters : 50, run_driver_ptds);

    return 0;
}
