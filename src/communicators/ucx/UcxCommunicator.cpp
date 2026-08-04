#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include "UcxCommunicator.h"
#include "gvirtus/communicators/Visibility.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <execinfo.h>
#include <functional>
#include <signal.h>
#include <unordered_map>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <ucm/api/ucm.h>
#include <limits>
#include <malloc.h>
#include <mutex>
#include <netdb.h>
#include <sys/socket.h>
#include <stdexcept>
#include <string>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"
#include "gvirtus/communicators/UcxAmProtocol.h"
#include "gvirtus/communicators/AblationGate.h"
#include "gvirtus/communicators/RmaPolicy.h"
using gvirtus::communicators::Ablation;
using gvirtus::communicators::ablated;

using gvirtus::communicators::UcxCommunicator;

namespace {
constexpr unsigned kUcxAmId = 1;

bool ucx_debug_enabled() {
    const char *lvl = std::getenv("GVIRTUS_LOGLEVEL");
    if (lvl == nullptr) return false;

    char *end = nullptr;
    long val = std::strtol(lvl, &end, 10);
    if (end == lvl) return false;
    return val <= 10000;  // DEBUG or TRACE
}

void ucx_debug_log(const char *fmt, ...) {
    if (!ucx_debug_enabled()) return;

    std::fprintf(stderr, "[UCX DEBUG] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

// Pure AM path: tag transport is intentionally removed.

// libcudart resolver — dlopen at runtime so the communicator does not need
// to link against CUDA. Used only to allocate pinned host memory for the RX
// pool; if CUDA isn't available we fall back to plain posix_memalign and
// just lose the auto-DMA-fast-path benefit (cudaMemcpy from non-registered
// memory) — the zero-init avoidance still applies.
using cudaHostAlloc_t = int (*)(void **, size_t, unsigned);
using cudaFreeHost_t  = int (*)(void *);

std::once_flag g_cuda_once;
std::atomic<cudaHostAlloc_t> g_cuda_host_alloc{nullptr};
std::atomic<cudaFreeHost_t>  g_cuda_free_host{nullptr};

void load_cuda_pinned_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto a = reinterpret_cast<cudaHostAlloc_t>(dlsym(h, "cudaHostAlloc"));
        auto f = reinterpret_cast<cudaFreeHost_t>(dlsym(h, "cudaFreeHost"));
        if (a && f) {
            g_cuda_host_alloc.store(a);
            g_cuda_free_host.store(f);
            ucx_debug_log("rx_pool: loaded cudaHostAlloc from %s", candidates[i]);
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("rx_pool: cudaHostAlloc unavailable, falling back to posix_memalign");
}

// --- I10 / A1: ordenacion de transporte -------------------------------------------------
//
// A1 dice: el RmaPosted se observa DESPUES de que el payload haya aterrizado. NO es lo mismo
// que A2 y NO lo arregla cuFlushGPUDirectRDMAWrites. `ucp_put_nbx` completa LOCALMENTE -- que
// el buffer de origen se puede reutilizar, no que los bytes hayan llegado -- y `ucp_am_send_nbx`
// no esta ordenado contra escrituras RDMA en vuelo.
//
// A1 se sostiene en ESTE despliegue porque ambas operaciones viajan por la MISMA queue pair RC
// y InfiniBand entrega en orden dentro de una QP. Se rompe con striping multi-rail, o si los
// mensajes activos salen por un transporte distinto del PUT -- y nuestro propio
// UCX_TLS=rc_mlx5,ud_mlx5,tcp,self lo permite.
//
// Aqui no se inventa una garantia: se ofrece una POLITICA, y por defecto la de siempre para no
// cambiar en silencio lo que ya estaba medido.
//
//   assume  se asume A1 sin hacer nada, y se cuenta cuantas veces se confio en ella. NO
//           descarga la obligacion: es el brazo de ablacion con el que se mide el coste de
//           las otras tres.
//   fence   ucp_worker_fence entre el PUT y el RmaPosted. **NO DESCARGA I13**, y la version
//           anterior de este comentario decia que si: se retracta. Dos razones, una de
//           documentacion y otra medida.
//             (1) La garantia de UCP se realiza en el fence de IB, y UCX documenta ese fence
//                 como "weak - fence makes sure remote READS are ordered with respect to
//                 remote WRITES" (UCX_RC_MLX5_FENCE), con `none` = no-op y `auto` eligiendo
//                 por hardware. A1 no es lectura-tras-escritura: es un SEND (el RmaPosted)
//                 que no debe observarse antes de que el WRITE haya aterrizado. La semantica
//                 documentada no cubre la obligacion.
//             (2) Medido 2026-08-03: forzando UCX_RC_MLX5_FENCE=none -- el no-op explicito --
//                 el brazo `fence` da lo MISMO a tres decimales en 4/16/64/256 KiB. O sea que
//                 su coste (de 2 a 11 %) es el de la llamada, no el de una ordenacion que se
//                 este estableciendo. No se puede distinguir de no hacer nada.
//           Se conserva como perilla de diagnostico y para poder decir todo esto con un
//           numero detras. Quien la active se lleva un aviso por stderr.
//   flush   ucp_ep_flush_nbx antes del RmaPosted: cuando vuelve, los bytes estan en el par.
//           Es la unica INCONDICIONAL -- no supone nada sobre lanes -- y es la de por defecto.
//   strict  si no se puede establecer A1, se declina el camino directo (slot de host)
enum class A1Politica { Asumir, Fence, Flush, Estricta };

inline A1Politica gvs_a1_politica() {
    static const A1Politica v = [] {
        const char *e = std::getenv("GVIRTUS_A1_POLICY");
        // Por defecto FLUSH: la configuracion desplegada descarga A1 sin condiciones. Las
        // otras tres son para medir (assume), para acotar (fence) y para el repliegue
        // conservador (strict). Cambiado 2026-08-03; antes el defecto era `assume`, que
        // contaba la obligacion sin cumplirla.
        if (e == nullptr || e[0] == '\0') return A1Politica::Flush;
        if (std::strcmp(e, "assume") == 0) return A1Politica::Asumir;
        if (std::strcmp(e, "fence")  == 0) return A1Politica::Fence;
        if (std::strcmp(e, "flush")  == 0) return A1Politica::Flush;
        if (std::strcmp(e, "strict") == 0) return A1Politica::Estricta;
        return A1Politica::Flush;
    }();
    return v;
}
inline const char *gvs_a1_nombre(A1Politica p) {
    switch (p) {
        case A1Politica::Fence:    return "fence";
        case A1Politica::Flush:    return "flush";
        case A1Politica::Estricta: return "strict";
        default:                   return "assume";
    }
}
std::atomic<unsigned long long> g_a1_asumidas{0};
std::atomic<unsigned long long> g_a1_fences{0};
std::atomic<unsigned long long> g_a1_flushes{0};
std::atomic<unsigned long long> g_a1_declinadas{0};
// Se cuenta aparte el fence que devuelve algo distinto de UCS_OK: si la implementacion no
// soporta la operacion sobre alguno de los ifaces del worker, la politica NO esta descargando
// nada y hay que saberlo por contador, no por suposicion.
std::atomic<unsigned long long> g_a1_fence_fallos{0};

// --- Ciclo de vida de los slots bajo una captura de grafo abierta ----------------------
//
// El hilo de una conexion del backend es EL MISMO que abrio la captura, asi que cualquier
// llamada CUDA insegura que haga el pool de slots la invalida. Medido en este driver
// (tests/semantic/capturetruth.cu): cudaMalloc, cudaFree, cudaHostAlloc, cudaFreeHost,
// cudaStreamSynchronize, cudaDeviceSynchronize y cudaMemcpy sincrono INVALIDAN; malloc/free,
// cudaMemcpyAsync sobre otro stream y cudaPointerGetAttributes NO.
//
// AQUI NO SE DESACTIVA NADA DEL DISENO. El pool de slots, las epocas, el retiro diferido y la
// sombra de GPU siguen exactamente igual: lo unico que cambia es CUANDO se ejecutan sus
// llamadas CUDA. Dentro de la ventana se APLAZAN; en el primer punto seguro fuera de ella se
// ejecutan y el pool converge al mismo estado que habria tenido. Una ventana de captura es una
// fase de grabacion, no el camino caliente: el camino caliente es el lanzamiento, y ahi el RMA
// y el peer-DMA estan intactos.
//
// La bandera es GLOBAL, no por hilo, y a proposito: en cudaStreamCaptureModeGlobal la llamada
// insegura de CUALQUIER hilo invalida la captura, asi que aplazar de mas es lo correcto. El
// coste de equivocarse por exceso es memoria retenida unos milisegundos.
bool captura_abierta() {
    return gvirtus::communicators::capture_open();
}

struct LiberacionAplazada {
    unsigned char *host = nullptr;   // solo si venia de cudaHostAlloc
    unsigned char *gpu  = nullptr;
};
std::mutex                       g_aplazadas_mu;
std::vector<LiberacionAplazada>  g_aplazadas;
std::atomic<unsigned long long>  g_aplazadas_total{0};

void encola_liberacion(unsigned char *host_cuda, unsigned char *gpu) {
    if (host_cuda == nullptr && gpu == nullptr) return;
    std::lock_guard<std::mutex> lk(g_aplazadas_mu);
    g_aplazadas.push_back(LiberacionAplazada{host_cuda, gpu});
    ++g_aplazadas_total;
}

// Allocate `n` bytes of pinned host memory. is_cuda set to true if the buffer
// was allocated via cudaHostAlloc (so cudaFreeHost is needed for release).
unsigned char *alloc_pinned_host(size_t n, bool &is_cuda) {
    std::call_once(g_cuda_once, load_cuda_pinned_funcs);
    // Con una captura de grafo abierta NO se puede llamar a cudaHostAlloc: la asignacion
    // invalida la captura. Se cae a posix_memalign, que es lo que este mismo helper ya hace
    // cuando CUDA no esta disponible -- memoria paginable, mismo contrato para el camino AM.
    auto fn = captura_abierta() ? nullptr : g_cuda_host_alloc.load();
    if (fn != nullptr) {
        void *p = nullptr;
        if (fn(&p, n, /*cudaHostAllocDefault*/ 0u) == 0 && p != nullptr) {
            is_cuda = true;
            return static_cast<unsigned char *>(p);
        }
    }
    void *p = nullptr;
    if (posix_memalign(&p, 4096, n) == 0 && p != nullptr) {
        is_cuda = false;
        return static_cast<unsigned char *>(p);
    }
    return nullptr;
}

void free_pinned_host(unsigned char *p, bool is_cuda) {
    if (p == nullptr) return;
    if (is_cuda) {
        // cudaFreeHost dentro de una ventana de captura la invalida (medido). Se aplaza: el
        // buffer se libera en el primer punto seguro, no se pierde.
        if (captura_abierta()) {
            encola_liberacion(p, nullptr);
            return;
        }
        auto fn = g_cuda_free_host.load();
        if (fn != nullptr) {
            fn(p);
            return;
        }
    }
    std::free(p);   // paginable: seguro incluso capturando
}

// Allocate `n` bytes of GPU memory (cudaMalloc). Forward-declared usage —
// requires load_cuda_device_funcs() to have been called (probe_gpudirect
// triggers it via std::call_once). Returns nullptr on failure.
//
// Defined LATER in the file so it can see g_cuda_malloc; here we forward-
// declare both to avoid reordering the whole anonymous-namespace section.
unsigned char *alloc_gpu_slot(size_t n);

// The payload size at or above which the RMA path is worth its RmaSetup/RmaPosted
// handshake (measured crossover, see WriteIov). Shared by the sender, which decides
// whether to use RMA, and the receiver, which uses it to decide when a connection has
// proved it needs a slot pool at all.
// Configured full slot capacity. Slots smaller than this exist too (the AM receive
// path appends one per message, sized to that message) but they are not usable for the
// RMA path and must never be advertised as if they were.
size_t ucx_slot_cap_bytes() {
    static const size_t v = []() -> size_t {
        const char *e = std::getenv("GVIRTUS_RMA_SLOT_CAP_MB");
        size_t mb = 1025;
        if (e != nullptr && e[0] != 0) {
            char *end = nullptr;
            unsigned long long parsed = std::strtoull(e, &end, 10);
            if (end != e && parsed > 0) mb = static_cast<size_t>(parsed);
        }
        return mb * 1024u * 1024u + 64u * 1024u;  // + framing slack, see P1b
    }();
    return v;
}

size_t ucx_rma_min_bytes() {
    static const size_t v = []() -> size_t {
        // El DEFECTO se deriva de la politica activa (su umbral mas pequeno) en vez de ser un
        // 4 MiB independiente. Un suelo por encima de lo que la politica admite hace que se
        // admita trafico que no se puede servir de forma nativa, y eso costaba -1,1 % en llama
        // 7B con quadrant. Ver policy_min_threshold() en RmaPolicy.h para la medida del escalon.
        const size_t derivado = gvirtus::communicators::policy_min_threshold();
        const char *e = std::getenv("GVIRTUS_RMA_MIN_BYTES");
        if (e == nullptr || e[0] == 0) {
            fprintf(stderr, "[GVS POOL] rma_min_bytes = %zu (derivado de la politica)\n", derivado);
            return derivado;
        }
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(e, &end, 10);
        const size_t v = (end != e) ? static_cast<size_t>(parsed) : derivado;
        if (v > derivado)
            fprintf(stderr, "[GVS POOL] AVISO: rma_min_bytes=%zu supera el umbral minimo de la "
                            "politica (%zu): se admitira trafico que el pool no puede servir\n",
                    v, derivado);
        else
            fprintf(stderr, "[GVS POOL] rma_min_bytes = %zu (env)\n", v);
        return v;
    }();
    return v;
}
void           free_gpu_slot(unsigned char *p);

// ---------------------------------------------------------------------------
// GPUDirect probe + flag (Step 1 of GVIRTUS_GPUDIRECT rollout)
// ---------------------------------------------------------------------------
// Runtime resolvers for cudaMalloc/cudaFree/cudaMemcpy/cudaPointerGetAttributes
// (dlopen, no static link to CUDA). cudaPointerGetAttributes lets WriteIovRma
// detect whether an iov fragment lives on the GPU so it can pass
// UCS_MEMORY_TYPE_CUDA to ucp_mem_map (required when rcache is disabled).
// cudaMemcpy is used by am_recv_handler in Step B3 to consolidate a GPU-split
// payload back into the host slot (temporary — B4 removes this consolidation).
using cudaMalloc_t = int (*)(void **, size_t);
using cudaFree_t   = int (*)(void *);
using cudaMemcpy_t = int (*)(void *, const void *, size_t, int /*cudaMemcpyKind*/);
using cudaPointerGetAttributes_t = int (*)(void *, const void *);
using cudaDeviceSynchronize_t = int (*)();
using cudaGetLastError_t = int (*)();

// cudaMemcpyKind values (from cuda_runtime_api.h, stable across CUDA versions).
constexpr int kCudaMemcpyHostToHost     = 0;
constexpr int kCudaMemcpyHostToDevice   = 1;
constexpr int kCudaMemcpyDeviceToHost   = 2;
constexpr int kCudaMemcpyDeviceToDevice = 3;

// Mirror of cudaPointerAttributes (CUDA 11+ layout). `type` 0=Unregistered,
// 1=Host, 2=Device, 3=Managed. Only `type` is read; remaining fields kept for
// ABI alignment.
struct cudaPointerAttributes_layout {
    int   type;
    int   device;
    void *devicePointer;
    void *hostPointer;
};

std::atomic<cudaMalloc_t> g_cuda_malloc{nullptr};
std::atomic<cudaFree_t>   g_cuda_free{nullptr};
std::atomic<cudaMemcpy_t> g_cuda_memcpy{nullptr};
std::atomic<cudaPointerGetAttributes_t> g_cuda_pointer_attrs{nullptr};
std::atomic<cudaDeviceSynchronize_t> g_cuda_device_sync{nullptr};
std::atomic<cudaGetLastError_t> g_cuda_get_last_error{nullptr};
std::once_flag            g_cuda_dev_once;

void load_cuda_device_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto m  = reinterpret_cast<cudaMalloc_t>(dlsym(h, "cudaMalloc"));
        auto f  = reinterpret_cast<cudaFree_t>(dlsym(h, "cudaFree"));
        auto mc = reinterpret_cast<cudaMemcpy_t>(dlsym(h, "cudaMemcpy"));
        auto a  = reinterpret_cast<cudaPointerGetAttributes_t>(
                     dlsym(h, "cudaPointerGetAttributes"));
        auto ds = reinterpret_cast<cudaDeviceSynchronize_t>(
                     dlsym(h, "cudaDeviceSynchronize"));
        auto gle = reinterpret_cast<cudaGetLastError_t>(
                     dlsym(h, "cudaGetLastError"));
        if (m && f) {
            g_cuda_malloc.store(m);
            g_cuda_free.store(f);
            g_cuda_memcpy.store(mc);          // may be nullptr
            g_cuda_pointer_attrs.store(a);    // may be nullptr; is_gpu_pointer handles that
            g_cuda_device_sync.store(ds);     // may be nullptr; drain_device_if_async_pending checks
            g_cuda_get_last_error.store(gle); // may be nullptr; clears the probe's sticky last error
            ucx_debug_log("gpudirect: loaded cuda runtime symbols from %s "
                          "(memcpy=%s pointer_attrs=%s)",
                          candidates[i], mc ? "yes" : "no", a ? "yes" : "no");
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("gpudirect: cudaMalloc/cudaFree unavailable (libcudart not found)");
}

// ---------------------------------------------------------------------------
// Aciertos y fallos de la cache de registro H2D. Globales como la propia cache: la comparten
// todas las conexiones del proceso.
static std::atomic<unsigned long long> g_reg_cache_hits{0};
static std::atomic<unsigned long long> g_reg_cache_misses{0};

// H2D source registration cache (invalidatable).
// ---------------------------------------------------------------------------
// Registering a 64 MB source per put costs half the H2D bandwidth (measured:
// 23.4 GB/s cached vs 11.5 GB/s uncached), so the cache has to stay. What it did NOT
// have was any way to be invalidated when the application freed the buffer, which is
// the same defect that silently corrupted the D2H destination path (a86b1ec): the
// allocator hands the same address back, the cached handle still describes the old
// mapping, and the NIC reads the wrong pages.
//
// This is now process-global rather than a function-static thread_local, so a free on
// any thread invalidates it, and it stores the owning context so entries from
// different communicators can be unmapped correctly. Entries are only created for
// addresses the frontend has told us it will report the free of -- see
// RegistrationCacheable().
// Keyed by (CONTEXT, address), not by address alone.
//
// The backend forks once per configured endpoint and then serves every connection as a
// detached std::thread in that one process (Backend.cpp fork over _children,
// Process.cpp `std::thread(execute, client).detach()`), so all connections share this
// map while each holds its OWN ucp_context_h. Keying by address alone let one
// connection be handed a ucp_mem_h created in another connection's context -- reachable
// because the D2H GPU scratch is thread_local, so a finished thread's scratch is freed
// and a new thread's cudaMalloc readily returns the same address. Using a memh with the
// wrong context is what surfaced as "D2H-GET failed" followed by a connection reset,
// killing ~30% of tenants at concurrency 8.
struct SrcRegKey {
    ucp_context_h ctx;
    const void *addr;
    bool operator==(const SrcRegKey &o) const { return ctx == o.ctx && addr == o.addr; }
};
struct SrcRegKeyHash {
    size_t operator()(const SrcRegKey &k) const {
        return std::hash<const void *>()(k.addr) ^
               (std::hash<const void *>()(static_cast<const void *>(k.ctx)) << 1);
    }
};
struct SrcReg {
    ucp_mem_h memh{nullptr};
    size_t len{0};
};

// Contadores de registros RDMA (2026-07-28, diagnostico de fuga de MRs).
// GVIRTUS_MEMMAP_TRACE=1. Ver [[backend_mr_leak]].
namespace {
struct GvsMmSite { std::string name; std::atomic<long> maps{0}; std::atomic<long> unmaps{0}; };
std::mutex gvs_mm_mu;
std::vector<GvsMmSite *> gvs_mm_sites;

inline bool gvs_mm_trace() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_MEMMAP_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline void gvs_mm_count(const char *name, bool is_map) {
    if (!gvs_mm_trace()) return;
    static std::atomic<long> total{0};
    GvsMmSite *site = nullptr;
    {
        std::lock_guard<std::mutex> lk(gvs_mm_mu);
        for (auto *x : gvs_mm_sites) if (x->name == name) { site = x; break; }
        if (site == nullptr) { site = new GvsMmSite{name}; gvs_mm_sites.push_back(site); }
    }
    if (is_map) site->maps.fetch_add(1); else site->unmaps.fetch_add(1);
    const long n = total.fetch_add(1) + 1;
    if (n % 2000 != 0) return;
    std::lock_guard<std::mutex> lk(gvs_mm_mu);
    std::fprintf(stderr, "[GVS MEMMAP] --- after %ld operations ---\n", n);
    for (auto *x : gvs_mm_sites) {
        const long m = x->maps.load(), u = x->unmaps.load();
        std::fprintf(stderr, "[GVS MEMMAP]   %-24s map=%-8ld unmap=%-8ld live=%ld%s\n",
                     x->name.c_str(), m, u, m - u, (m - u > 500) ? "   <-- LEAK" : "");
    }
    std::fflush(stderr);
}
}  // namespace

std::mutex g_src_reg_mu;
std::unordered_map<SrcRegKey, SrcReg, SrcRegKeyHash> g_src_regs;


// Instrumentacion (2026-07-28): confirmar si g_src_regs acumula registros de
// conexiones ya cerradas. Se activa con GVIRTUS_SRCREG_TRACE=1.
static bool gvs_srcreg_trace() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_SRCREG_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

// Debe llamarse SIN g_src_reg_mu tomado.
static void gvs_srcreg_report(const char *what) {
    if (!gvs_srcreg_trace()) return;
    size_t n = 0, bytes = 0, ctxs = 0;
    {
        std::lock_guard<std::mutex> lk(g_src_reg_mu);
        n = g_src_regs.size();
        std::vector<ucp_context_h> seen;
        for (const auto &kv : g_src_regs) {
            bytes += kv.second.len;
            bool found = false;
            for (auto c : seen) if (c == kv.first.ctx) { found = true; break; }
            if (!found) seen.push_back(kv.first.ctx);
        }
        ctxs = seen.size();
    }
    std::fprintf(stderr,
                 "[GVS SRCREG] %-22s entries=%zu  bytes=%.1f MiB  distinct_contexts=%zu\n",
                 what, n, bytes / 1048576.0, ctxs);
    std::fflush(stderr);
}


// The application freed this address, so EVERY context's registration of it is stale --
// drop them all, not just one.

// --- temporary: locate a stall by tracing entry/exit per thread ---------------------
// Enabled only when GVS_TRACE_FN=1 (a VALUE check, not a set check).
namespace {
inline bool gvs_trace_on() {
    static const bool on = []() {
        const char *v = std::getenv("GVS_TRACE_FN");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
struct GvsTraceFn {
    const char *name;
    explicit GvsTraceFn(const char *n) : name(n) {
        if (gvs_trace_on()) {
            std::fprintf(stderr, "[TRACE %lu] >> %s\n",
                         (unsigned long)pthread_self(), name);
            std::fflush(stderr);
        }
    }
    ~GvsTraceFn() {
        if (gvs_trace_on()) {
            std::fprintf(stderr, "[TRACE %lu] << %s\n",
                         (unsigned long)pthread_self(), name);
            std::fflush(stderr);
        }
    }
};
}  // namespace
#define GVS_TRACE(n) GvsTraceFn gvs_trace_guard_(n)

void src_reg_invalidate(const void *addr) {
    const char *tv = std::getenv("GVS_TRACE_FN");
    const bool tr = (tv != nullptr && tv[0] == '1');
    if (tr) { std::fprintf(stderr, "[TRACE %lu] >> src_reg_invalidate\n",
                           (unsigned long)pthread_self()); std::fflush(stderr); }
    struct Bye { bool on; ~Bye() { if (on) { std::fprintf(stderr,
        "[TRACE %lu] << src_reg_invalidate\n", (unsigned long)pthread_self());
        std::fflush(stderr); } } } bye{tr};
    std::vector<std::pair<ucp_context_h, ucp_mem_h>> doomed;
    {
        std::lock_guard<std::mutex> lk(g_src_reg_mu);
        for (auto it = g_src_regs.begin(); it != g_src_regs.end();) {
            if (it->first.addr == addr) {
                doomed.emplace_back(it->first.ctx, it->second.memh);
                it = g_src_regs.erase(it);
            } else {
                ++it;
            }
        }
    }
    gvs_mm_count("unmap:L360", false);
    // ONCE. This loop was written twice -- a diagnostic line got inserted between a loop
    // and its duplicate, and every registration was unmapped twice. ucp_mem_unmap releases
    // the handle, so the second pass is a double free of memory glibc has already recycled:
    // "free(): double free detected in tcache 2", every time an application frees a buffer
    // it had registered. cuDF hits it on its first batch.
    if (tr && !doomed.empty()) {
        std::fprintf(stderr, "[TRACE %lu] .. unmapping %zu registration(s)\n",
                     (unsigned long)pthread_self(), doomed.size());
        std::fflush(stderr);
    }
    for (auto &d : doomed) ucp_mem_unmap(d.first, d.second);
    if (tr && !doomed.empty()) {
        std::fprintf(stderr, "[TRACE %lu] .. unmapped ok\n",
                     (unsigned long)pthread_self());
        std::fflush(stderr);
    }
    if (!doomed.empty()) gvs_srcreg_report("tras invalidar");
    if (!doomed.empty()) {
        ucx_debug_log("src_reg: invalidated %zu registration(s) for %p (freed by the app)",
                      doomed.size(), addr);
    }
}

struct SrcRegHookRegistrar {
    SrcRegHookRegistrar() {
        gvirtus::communicators::SetRegistrationInvalidateHook(&src_reg_invalidate);
    }
};
SrcRegHookRegistrar g_src_reg_hook_registrar;

// ---------------------------------------------------------------------------------
// Invalidacion por UCM: poder cachear destinos que NO asignamos nosotros.
//
// La cache de registros solo admite buferes cuyo free nos llega (cudaFree/cudaFreeHost).
// Un array de numpy no cumple eso, asi que se registra en cada llamada. Con UCM nos
// enteramos de los unmap que ocurren DENTRO de glibc, cosa que interponer munmap por
// LD_PRELOAD no consigue (medido: dispara con un mmap/munmap explicito y nunca con los
// free de numpy, porque glibc desmapea por una ruta que no pasa por la PLT).
//
// El unmap real va DIFERIDO: el callback corre dentro de free(), potencialmente con los
// cerrojos del asignador tomados, asi que solo saca la entrada del mapa y encola el
// memh. Igual que hace el rcache de UCX con su lista de basura.
//
// RESULTADO MEDIDO: cierra el ancho de banda (pageable 11,4 -> 23,0 GB/s) pero EMPEORA
// el ETL de cuDF, porque to_pandas libera su destino en cada batch y la correccion
// obliga a invalidar al liberar => la cache no puede acertar nunca. Se deja apagado.
// ---------------------------------------------------------------------------------

// Protegida por g_src_reg_mu.
std::vector<std::pair<ucp_context_h, ucp_mem_h>> g_reg_garbage;

void drain_reg_garbage() {
    std::vector<std::pair<ucp_context_h, ucp_mem_h>> local;
    {
        std::lock_guard<std::mutex> lk(g_src_reg_mu);
        if (g_reg_garbage.empty()) return;
        local.swap(g_reg_garbage);
    }
    for (auto &d : local) ucp_mem_unmap(d.first, d.second);
}

// Se llama DESDE free(): nada de logs, nada de UCX, solo mover punteros.
void src_reg_invalidate_range(const void *addr, size_t len) {
    if (addr == nullptr || len == 0) return;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t hi = lo + len;
    std::lock_guard<std::mutex> lk(g_src_reg_mu);
    for (auto it = g_src_regs.begin(); it != g_src_regs.end();) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(it->first.addr);
        const uintptr_t b = a + it->second.len;
        if (a < hi && lo < b) {
            g_reg_garbage.emplace_back(it->first.ctx, it->second.memh);
            it = g_src_regs.erase(it);
        } else {
            ++it;
        }
    }
}

// Modo 2, solo diagnostico: handler instalado pero sin hacer nada y sin activar el
// cacheo, para separar el coste de la maquinaria de UCM del de nuestro callback.
bool g_ucm_noop = false;

void gvs_ucm_unmapped_cb(ucm_event_type_t type, ucm_event_t *event, void *arg) {
    (void)arg;
    if (g_ucm_noop) return;
    if ((type & UCM_EVENT_VM_UNMAPPED) == 0) return;
    src_reg_invalidate_range(event->vm_unmapped.address, event->vm_unmapped.size);
}

struct UcmInvalidateRegistrar {
    UcmInvalidateRegistrar() {
        const char *v = std::getenv("GVIRTUS_UCM_INVALIDATE");
        if (v == nullptr) return;
        if (v[0] == '2') g_ucm_noop = true;
        else if (v[0] != '1') return;
        ucs_status_t st = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, 0,
                                                gvs_ucm_unmapped_cb, nullptr);
        if (st == UCS_OK) {
            if (!g_ucm_noop) gvirtus::communicators::SetHostUnmapTrackingActive(true);
            ucx_debug_log("UCM: handler de VM_UNMAPPED instalado (noop=%d)", (int)g_ucm_noop);
        } else {
            ucx_debug_log("UCM: ucm_set_event_handler fallo (%d)", (int)st);
        }
    }
};
UcmInvalidateRegistrar g_ucm_invalidate_registrar;

// Obtain a registration for an application buffer, shared by the H2D source and the
// D2H destination -- ucp_mem_map is direction-agnostic and a buffer may well be both.
//
// Cached only when the frontend confirms it will report the free (device pointers and
// tracked pinned host). Everything else is registered per call and unmapped by the
// caller, because an address-keyed cache we cannot invalidate is exactly the defect
// that silently corrupted both directions.
ucp_mem_h acquire_app_registration(ucp_context_h ctx, const void *addr, size_t len,
                                   bool cacheable, bool is_cuda, bool &out_owned) {
    out_owned = false;
    if (ctx == nullptr || addr == nullptr || len == 0) return nullptr;

    // Contexto seguro para soltar lo que invalido el callback de UCM (ver arriba).
    drain_reg_garbage();

    auto fill = [&](ucp_mem_map_params_t &mp) {
        mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mp.address = const_cast<void *>(addr);
        mp.length = len;
        if (is_cuda) {
            mp.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            mp.memory_type = UCS_MEMORY_TYPE_CUDA;
        }
    };

    if (cacheable) {
        const SrcRegKey key{ctx, addr};
        std::lock_guard<std::mutex> lk(g_src_reg_mu);
        auto it = g_src_regs.find(key);
        if (it == g_src_regs.end()) ++g_reg_cache_misses; else ++g_reg_cache_hits;
        if (it != g_src_regs.end() && it->second.len < len) {
            gvs_mm_count("unmap:L402", false);
            ucp_mem_unmap(ctx, it->second.memh);
            g_src_regs.erase(it);
            it = g_src_regs.end();
        }
        if (it != g_src_regs.end()) return it->second.memh;

        ucp_mem_map_params_t mp{};
        fill(mp);
        ucp_mem_h m = nullptr;
        gvs_mm_count("map:L411", true);
        if (ucp_mem_map(ctx, &mp, &m) != UCS_OK) return nullptr;
        g_src_regs.emplace(key, SrcReg{m, len});
        return m;
    }

    ucp_mem_map_params_t mp{};
    fill(mp);
    ucp_mem_h m = nullptr;
    gvs_mm_count("map:L419", true);
    if (ucp_mem_map(ctx, &mp, &m) != UCS_OK) return nullptr;
    out_owned = true;
    return m;
}

// Release every registration this context owns. Called at teardown, BEFORE the UCX
// context is destroyed.
//
// Nothing used to do this. client_dst_regs_, gpu_get_regs_ and the old
// user_memh_cache were all populated with ucp_mem_map and never unmapped, so the
// process reached ucp_cleanup still holding MRs over buffers the application had long
// since freed -- which is where the "corrupted size vs. prev_size in fastbins" and
// "double free in tcache" at exit came from. user_memh_cache was worse still: being
// static thread_local, its handles outlived the context itself.
void release_registrations_for_context(ucp_context_h ctx) {
    if (ctx == nullptr) return;
    std::vector<ucp_mem_h> doomed;
    {
        std::lock_guard<std::mutex> lk(g_src_reg_mu);
        for (auto it = g_src_regs.begin(); it != g_src_regs.end();) {
            if (it->first.ctx == ctx) {
                doomed.push_back(it->second.memh);
                it = g_src_regs.erase(it);
            } else {
                ++it;
            }
        }
    }
    gvs_mm_count("unmap:L447", false);
    for (ucp_mem_h m : doomed) ucp_mem_unmap(ctx, m);
    if (!doomed.empty()) {
        ucx_debug_log("teardown: unmapped %zu cached registrations", doomed.size());
    }
}

// Global flag: true iff GVIRTUS_GPUDIRECT=1 and probe succeeded.
// Set once at backend startup by init_ucx(); read by handlers (Step 3) and
// by is_gpu_pointer below as a short-circuit guard.
std::atomic<bool> g_gpudirect_enabled{false};

// Detect whether `p` is a CUDA device or managed pointer. Returns false on
// host memory, unregistered memory, NULL, or if cudaPointerGetAttributes is
// unavailable. Used by WriteIovRma to decide whether to pass
// UCS_MEMORY_TYPE_CUDA on the ucp_mem_map call.
//
// CRITICAL: this function is called from both frontend AND backend (WriteIovRma
// runs on both sides). On the frontend, libcudart.so is the GVirtuS shim that
// REMOTES cudaPointerGetAttributes as an RPC — which is both slow and broken
// for our purposes (the frontend has no local GPU to ask about). So we
// short-circuit to false whenever GPUDirect isn't active: frontend never has
// the env var set → returns false → no RPC storm. Only the backend (where
// GPUDirect probed OK) actually calls into the cuda runtime.
bool is_gpu_pointer(const void *p) {
    if (p == nullptr) return false;
    if (!g_gpudirect_enabled.load()) return false;
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto fn = g_cuda_pointer_attrs.load();
    if (fn == nullptr) return false;
    cudaPointerAttributes_layout attrs{};
    if (fn(&attrs, p) != 0) return false;
    return attrs.type == 2 /*Device*/ || attrs.type == 3 /*Managed*/;
}

// Probe: try cudaMalloc(4K) + ucp_mem_map(CUDA) + cleanup.
// Returns true iff peermem + UCX-CUDA cooperate in this process.
// `reason` is populated with the failure description on false.
bool probe_gpudirect(ucp_context_h ctx, std::string &reason) {
    if (ctx == nullptr) {
        reason = "ucp_context is null";
        return false;
    }
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto cmalloc = g_cuda_malloc.load();
    auto cfree   = g_cuda_free.load();
    if (cmalloc == nullptr || cfree == nullptr) {
        reason = "cudaMalloc/cudaFree symbols unavailable";
        return false;
    }

    void *gpu = nullptr;
    if (cmalloc(&gpu, 4096) != 0 || gpu == nullptr) {
        reason = "cudaMalloc(4K) failed (no GPU? OOM?)";
        return false;
    }

    ucp_mem_map_params_t p{};
    p.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                   UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                   UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    p.address     = gpu;
    p.length      = 4096;
    p.memory_type = UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h memh = nullptr;
    gvs_mm_count("map:L512", true);
    ucs_status_t st = ucp_mem_map(ctx, &p, &memh);
    if (st != UCS_OK) {
        reason  = "ucp_mem_map(CUDA) failed: ";
        reason += ucs_status_string(st);
        cfree(gpu);
        return false;
    }
    gvs_mm_count("unmap:L519", false);
    ucp_mem_unmap(ctx, memh);
    cfree(gpu);
    reason.clear();
    return true;
}

// Definitions for the forward-declared helpers above. They live AFTER
// probe_gpudirect so the cuda symbol resolver runs at least once (probe
// triggers std::call_once(load_cuda_device_funcs)). If GPUDirect was never
// requested, g_cuda_malloc may still be nullptr — callers must check.
unsigned char *alloc_gpu_slot(size_t n) {
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    // cudaMalloc invalida la captura (medido). Un slot sin sombra sigue sirviendo por host:
    // es exactamente la degradacion que este mismo helper ya aplica cuando la reserva falla,
    // y el pool recupera la sombra en la primera reconstruccion fuera de la ventana.
    if (captura_abierta()) return nullptr;
    auto fn = g_cuda_malloc.load();
    if (fn == nullptr) return nullptr;
    void *p = nullptr;
    if (fn(&p, n) != 0 || p == nullptr) return nullptr;
    return static_cast<unsigned char *>(p);
}

void free_gpu_slot(unsigned char *p) {
    if (p == nullptr) return;
    if (captura_abierta()) {           // cudaFree invalida: se aplaza
        encola_liberacion(nullptr, p);
        return;
    }
    auto fn = g_cuda_free.load();
    if (fn != nullptr) fn(p);
}

// Punto seguro: ejecuta las liberaciones que se aplazaron durante una ventana de captura.
// Se sincroniza el device ANTES de liberar porque una copia de staging desde la sombra pudo
// quedar en vuelo dentro de la ventana (ver CaptureStaging.h, stage_dev): liberar la sombra
// con la copia viva seria un use-after-free en el device. Fuera de la ventana sincronizar es
// seguro, y esto solo corre cuando hay algo encolado, es decir casi nunca.
void drena_liberaciones_aplazadas() {
    if (captura_abierta()) return;
    std::vector<LiberacionAplazada> lote;
    {
        std::lock_guard<std::mutex> lk(g_aplazadas_mu);
        if (g_aplazadas.empty()) return;
        lote.swap(g_aplazadas);
    }
    if (auto sync = g_cuda_device_sync.load()) sync();
    auto libera_host = g_cuda_free_host.load();
    auto libera_gpu  = g_cuda_free.load();
    for (const auto &l : lote) {
        if (l.host != nullptr) {
            if (libera_host != nullptr) libera_host(l.host); else std::free(l.host);
        }
        if (l.gpu != nullptr && libera_gpu != nullptr) libera_gpu(l.gpu);
    }
    ucx_debug_log("capture: drained %zu deferred releases", lote.size());
}
}

UcxCommunicator::UcxCommunicator(const std::string &hostname, std::uint16_t port)
    : hostname_(hostname), port_(port) {}

UcxCommunicator::~UcxCommunicator() { Close(); }


// --- localizador de la serializacion, lado CLIENTE ------------------------------------------
// La traza del backend dice que la RPC de un hilo ENTRA 0,8 ms despues de que salga la del otro.
// Eso admite dos lecturas incompatibles y hay que separarlas: (a) el backend no ENTREGA hasta
// entonces, o (b) el backend entrega antes y el CLIENTE no procesa su recepcion mientras otro de
// sus hilos espera. Esto sella el instante en que el mensaje ATERRIZA en la cola del cliente y el
// instante en que el hilo que espera lo RECOGE. Si aterriza pronto y se recoge tarde, es (b).
// Apagado por defecto: GVS_CLIENT_TRACE=1.
static bool gvs_spin_trace() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_SPIN_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}
static bool gvs_client_trace() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_CLIENT_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}
static double gvs_cli_ms() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}
static std::atomic<double> g_ultimo_aterrizaje{0.0};

// --- ¿DONDE se queda el hilo atascado? Pila del propio hilo, sin depurador -------------------
// El muestreo de /proc dice que el hilo que se traga los 50 s esta en futex_wait con
// op=0x189 (pthread_cond_timedwait) sobre un objeto que vive en la PILA de algun hilo, y que
// eso ocurre DENTRO de ucp_worker_progress. Saber en que llamada hace falta una pila, y en el
// contenedor no hay gdb ni eu-stack (y no se van a instalar en un backend compartido).
//
// Asi que el hilo se la saca a si mismo: un vigilante mira los relojes de entrada a progress y,
// al que lleve mas de GVS_STUCK_BT_MS dentro, le manda SIGPROF; el manejador imprime su propia
// pila con backtrace_symbols_fd. Una sola vez por episodio, para no inundar ni cambiar el
// tiempo que se esta midiendo.
//
// Apagado por defecto (GVS_STUCK_BT_MS ausente o 0). El coste cuando esta apagado es un
// puñado de comparaciones por vuelta del bucle de espera.
namespace {

struct RanuraProgreso {
    std::atomic<int>    tid{0};        // 0 = libre
    std::atomic<double> t_entrada{0};  // 0 = no esta dentro de progress
    std::atomic<bool>   ya_avisado{false};
};
constexpr int kMaxRanurasProgreso = 128;
RanuraProgreso g_ranuras_progreso[kMaxRanurasProgreso];

long gvs_stuck_bt_ms() {
    static const long v = [] {
        const char *e = std::getenv("GVS_STUCK_BT_MS");
        return e != nullptr ? std::strtol(e, nullptr, 10) : 0L;
    }();
    return v;
}

void gvs_manejador_bt(int) {
    void *marcos[64];
    const int n = ::backtrace(marcos, 64);
    char cab[128];
    const int cn = std::snprintf(cab, sizeof(cab),
                                 "[GVS BT] tid=%d atascado en el camino de recepcion, profundidad=%d\n",
                                 (int)syscall(SYS_gettid), n);
    if (cn > 0) { ssize_t r = ::write(2, cab, (size_t)cn); (void)r; }
    ::backtrace_symbols_fd(marcos, n, 2);
}

// Un solo vigilante por proceso. Arranca perezosamente en el primer registro.
void gvs_arranca_vigilante_una_vez() {
    static std::once_flag una;
    std::call_once(una, [] {
        // Precalentar backtrace(): la primera llamada carga libgcc y hace malloc, cosas que no
        // se quieren dentro de un manejador de señal.
        void *m[4]; (void)::backtrace(m, 4);
        struct sigaction sa {};
        sa.sa_handler = gvs_manejador_bt;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;   // que la espera interrumpida se reanude sola
        ::sigaction(SIGPROF, &sa, nullptr);

        std::thread([] {
            const double umbral = (double)gvs_stuck_bt_ms();
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                const double ahora = gvs_cli_ms();
                for (int i = 0; i < kMaxRanurasProgreso; ++i) {
                    auto &r = g_ranuras_progreso[i];
                    const int tid = r.tid.load(std::memory_order_acquire);
                    if (tid == 0) continue;
                    const double t0 = r.t_entrada.load(std::memory_order_acquire);
                    if (t0 == 0.0) { r.ya_avisado.store(false, std::memory_order_release); continue; }
                    if (ahora - t0 < umbral) continue;
                    if (r.ya_avisado.exchange(true, std::memory_order_acq_rel)) continue;
                    std::fprintf(stderr,
                                 "[GVS BT] vigilante: tid=%d lleva %.0f ms dentro de una seccion "
                                 "vigilada del camino de recepcion; pidiendo su pila\n",
                                 tid, ahora - t0);
                    std::fflush(stderr);
                    ::syscall(SYS_tgkill, ::getpid(), tid, SIGPROF);
                }
            }
        }).detach();
    });
}

// RAII: marca la ventana en la que este hilo esta dentro de ucp_worker_progress.
struct GuardiaProgreso {
    RanuraProgreso *r = nullptr;
    GuardiaProgreso() {
        if (gvs_stuck_bt_ms() <= 0) return;
        static thread_local RanuraProgreso *mia = nullptr;
        if (mia == nullptr) {
            gvs_arranca_vigilante_una_vez();
            const int mi_tid = (int)syscall(SYS_gettid);
            for (int i = 0; i < kMaxRanurasProgreso; ++i) {
                int libre = 0;
                if (g_ranuras_progreso[i].tid.compare_exchange_strong(libre, mi_tid)) {
                    mia = &g_ranuras_progreso[i];
                    break;
                }
            }
            if (mia == nullptr) return;   // mas hilos que ranuras: no instrumentamos este
        }
        r = mia;
        r->ya_avisado.store(false, std::memory_order_release);
        r->t_entrada.store(gvs_cli_ms(), std::memory_order_release);
    }
    ~GuardiaProgreso() {
        if (r != nullptr) r->t_entrada.store(0.0, std::memory_order_release);
    }
};

}  // namespace

// --- EL LIBERADOR: ninguna llamada CUDA que sincronice el dispositivo en el camino de una RPC
// --------------------------------------------------------------------------------------------
// DEFECTO MEDIDO (2026-08-04). Pila del hilo atascado, sacada por el propio hilo (GVS_STUCK_BT_MS):
//
//   TryAcquireFrame -> ucp_worker_progress -> ucp_am_handler -> am_recv_handler
//     -> acquire_rx_slot -> cudaFreeHost -> cuMemFreeHost          [9 819,9 ms]
//
// acquire_rx_slot corre DENTRO del callback de recepcion de UCX, es decir dentro de
// ucp_worker_progress, es decir en el hilo que atiende esa conexion. Su camino de crecimiento
// llamaba a cudaFreeHost, cudaFree, cudaHostAlloc y cudaMalloc, y las cuatro sincronizan el
// DISPOSITIVO ENTERO: no vuelven hasta que la GPU drena todo el trabajo del contexto primario
// del proceso, incluidos los kernels de OTROS clientes. Un kernel de 10 s de un cliente
// congelaba 9,8 s la primera RPC de otro. Eso es "los hilos de un proceso no solapan": no era
// PTDS, ni el worker, ni el despacho -- era el asignador del pool de slots.
//
// EL ARREGLO. El camino de la RPC deja de llamar a CUDA:
//   - el buffer nuevo se reserva PAGINABLE (posix_memalign, microsegundos) y se mapea a UCX.
//     ibv_reg_mr fija las paginas igual, asi que el camino rendezvous conserva su registro;
//   - lo viejo NO se libera ahi: se encola;
//   - este hilo de fondo hace todo lo lento -- desmapear, liberar, y MEJORAR el slot a memoria
//     fijada de CUDA (+ sombra de GPU) cuando el dispositivo lo deje. Que el liberador se pase
//     50 s dentro de cudaHostAlloc da igual: no hay ninguna RPC esperandolo.
//
// El pool converge al MISMO estado que antes; lo unico que cambia es CUANDO se hacen las
// llamadas CUDA. Es el patron que este fichero ya usaba para las ventanas de captura de grafos,
// aplicado al sitio donde faltaba.
namespace {

// FUGA DELIBERADA. El liberador es un hilo `detach()`, asi que sigue vivo cuando el proceso
// empieza a salir. Si estos objetos fueran estaticos normales, sus destructores correrian
// mientras el hilo espera en el condition_variable: comportamiento indefinido en el camino de
// salida. Medido en el frontend (que SI termina ordenadamente, al contrario que el backend, al
// que mata `docker rm -f`): 3 de 3 corridas de d2hpool acababan en "dumped core", 0 de 3 con la
// lib anterior. Se reservan en el monton y no se destruyen nunca; el sistema operativo se los
// lleva al terminar el proceso.
std::mutex                         &g_tareas_mu = *new std::mutex;
std::condition_variable            &g_tareas_cv = *new std::condition_variable;
// true = terminada; false = reintentar
std::deque<std::function<bool()>>  &g_tareas    = *new std::deque<std::function<bool()>>;
std::atomic<unsigned long long>     g_liberador_hechas{0};
std::atomic<unsigned long long>     g_slot_diferidos{0};      // liberaciones sacadas del callback
std::atomic<unsigned long long>     g_slot_mejoras_ok{0};     // slots elevados a memoria fijada
std::atomic<unsigned long long>     g_slot_mejoras_fallo{0};  // no se pudo fijar: se queda paginable
std::atomic<unsigned long long>     g_slot_mejoras_saltadas{0};  // el slot cambio antes de la mejora
// Vigilancia permanente del defecto arreglado: cuantas veces acquire_rx_slot ha tardado mas de
// 100 ms, y la peor. Con el arreglo puesto tiene que quedarse en 0 -- si sube, alguna llamada
// que sincroniza el dispositivo ha vuelto al camino de la RPC.
std::atomic<unsigned long long>     g_slot_atascos{0};
std::atomic<double>                 g_slot_peor_ms{0.0};

// Ablacion en las DOS direcciones con el mismo binario: con esto a 1 vuelve el comportamiento
// viejo (CUDA dentro del callback) y la serializacion tiene que REAPARECER. Una medida que solo
// ensena la mejora no distingue el arreglo de un cambio de condiciones.
bool gvs_slot_inline_cuda() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_SLOT_INLINE_CUDA");
        const bool on = e != nullptr && e[0] == '1';
        if (on)
            std::fprintf(stderr,
                "[GVS SLOT] *** reserva/liberacion de slots CUDA EN LINEA dentro del callback de "
                "UCX -- variante con la serializacion entre clientes DE VUELTA, solo para medirla\n");
        return on;
    }();
    return v;
}

// Una tarea devuelve true cuando ha terminado. Devolver false significa "todavia no se puede,
// vuelve a intentarlo": lo necesita la mejora de un slot, que solo puede cambiarle el buffer
// cuando el slot esta libre, y justo despues de crecerlo esta EN USO por el mensaje que provoco
// el crecimiento. Sin reintento la mejora no ocurria nunca (medido: slot_pinned=0,
// slot_pin_skipped=18) y los slots se quedaban paginables para siempre -- un arreglo que
// arreglaba la latencia y degradaba en silencio el camino H2D.
std::deque<std::function<bool()>> &g_tareas_espera =   // reintentos, se reinyectan cada tick
    *new std::deque<std::function<bool()>>;            // (misma fuga deliberada de arriba)

void gvs_arranca_liberador_una_vez() {
    static std::once_flag una;
    std::call_once(una, [] {
        std::thread([] {
            for (;;) {
                std::function<bool()> t;
                {
                    std::unique_lock<std::mutex> lk(g_tareas_mu);
                    if (g_tareas.empty() && !g_tareas_espera.empty()) {
                        // Tick de reintento: se espera SIEMPRE los 50 ms aunque la lista de
                        // espera no este vacia, o el hilo giraria a tope reintentando.
                        g_tareas_cv.wait_for(lk, std::chrono::milliseconds(50));
                        for (auto &p : g_tareas_espera) g_tareas.push_back(std::move(p));
                        g_tareas_espera.clear();
                    } else {
                        g_tareas_cv.wait_for(lk, std::chrono::milliseconds(50),
                                             [] { return !g_tareas.empty(); });
                    }
                    if (g_tareas.empty()) continue;
                    // Dentro de una ventana de captura de grafo, cualquiera de estas llamadas la
                    // invalidaria. Se espera a que cierre; la cola no se pierde.
                    //
                    // El `continue` a secas giraba a tope: con la cola llena el wait_for de
                    // arriba vuelve al instante porque su predicado ya se cumple, asi que este
                    // hilo se comia un nucleo entero durante toda la ventana de captura --
                    // justo cuando la carga esta grabando un grafo y necesita la CPU.
                    if (captura_abierta()) {
                        lk.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        continue;
                    }
                    t = std::move(g_tareas.front());
                    g_tareas.pop_front();
                }
                const bool hecha = t();
                if (!hecha) {
                    std::lock_guard<std::mutex> lk(g_tareas_mu);
                    g_tareas_espera.push_back(std::move(t));
                    continue;
                }
                ++g_liberador_hechas;
            }
        }).detach();
    });
}

// Ejecuta AQUI Y AHORA todo lo que quede pendiente, en el hilo que llama. Se usa al cerrar una
// conexion: lo que el liberador no haya hecho todavia se hace antes de que desaparezcan el pool,
// el contexto y -- en el frontend -- la propia conexion por la que `cudaFreeHost` viaja como RPC.
// Los reintentos se descartan: si un slot sigue en uso mientras se cierra la conexion, cambiarle
// el buffer ya no le sirve a nadie.
void gvs_drena_tareas_ahora() {
    for (;;) {
        std::function<bool()> t;
        {
            std::lock_guard<std::mutex> lk(g_tareas_mu);
            if (!g_tareas_espera.empty()) g_tareas_espera.clear();
            if (g_tareas.empty()) return;
            t = std::move(g_tareas.front());
            g_tareas.pop_front();
        }
        t();
    }
}

void encola_tarea(std::function<bool()> t) {
    {
        std::lock_guard<std::mutex> lk(g_tareas_mu);
        g_tareas.push_back(std::move(t));
    }
    gvs_arranca_liberador_una_vez();
    g_tareas_cv.notify_one();
}

// Reserva paginable pura: NUNCA llama a CUDA, asi que es segura dentro del callback.
// alloc_pinned_host ya cae aqui cuando no hay libcudart o hay una captura abierta, de modo que
// un slot con is_cuda_host=false es un estado que el resto del fichero ya maneja.
unsigned char *alloc_host_paginable(size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 4096, n) == 0 && p != nullptr)
        return static_cast<unsigned char *>(p);
    return nullptr;
}

// Contextos UCX vivos. Diferir trabajo a otro hilo introduce una ventana que antes no existia:
// la tarea guarda un ucp_context_h y el proceso puede haber hecho ucp_cleanup entre medias, y
// entonces el ucp_mem_unmap escribe sobre memoria liberada. Solo el listener posee el contexto
// y solo lo destruye al apagarse, asi que la ventana es de cierre -- pero un cierre que casca
// se lee como un defecto igual. Se registra y se consulta; liberar memoria sin desmapear
// cuando el contexto ya no esta es inofensivo (el contexto se llevo sus registros con el).
std::mutex                 &g_ctx_vivos_mu = *new std::mutex;    // misma fuga deliberada
std::vector<ucp_context_h> &g_ctx_vivos    = *new std::vector<ucp_context_h>;

void registra_contexto(ucp_context_h c) {
    if (c == nullptr) return;
    std::lock_guard<std::mutex> lk(g_ctx_vivos_mu);
    g_ctx_vivos.push_back(c);
}
void retira_contexto(ucp_context_h c) {
    std::lock_guard<std::mutex> lk(g_ctx_vivos_mu);
    g_ctx_vivos.erase(std::remove(g_ctx_vivos.begin(), g_ctx_vivos.end(), c),
                      g_ctx_vivos.end());
}
bool contexto_vivo(ucp_context_h c) {
    if (c == nullptr) return false;
    std::lock_guard<std::mutex> lk(g_ctx_vivos_mu);
    return std::find(g_ctx_vivos.begin(), g_ctx_vivos.end(), c) != g_ctx_vivos.end();
}

// Suelta en el hilo de fondo lo que un slot dejaba de usar. El orden importa y se conserva:
// primero desmapear de UCX (mientras la memoria sigue viva), luego sincronizar el dispositivo
// (una copia D2D desde la sombra pudo quedar en vuelo) y solo entonces liberar.
void encola_suelta_de_slot(ucp_context_h ctx, ucp_mem_h memh, ucp_mem_h gpu_memh,
                           unsigned char *host, bool host_es_cuda, unsigned char *gpu) {
    if (memh == nullptr && gpu_memh == nullptr && host == nullptr && gpu == nullptr) return;
    ++g_slot_diferidos;
    encola_tarea([ctx, memh, gpu_memh, host, host_es_cuda, gpu]() -> bool {
        const bool ctx_ok = contexto_vivo(ctx);
        if (ctx_ok && memh != nullptr)     ucp_mem_unmap(ctx, memh);
        if (ctx_ok && gpu_memh != nullptr) ucp_mem_unmap(ctx, gpu_memh);
        if (gpu != nullptr) {
            if (auto sync = g_cuda_device_sync.load()) sync();
        }
        if (host != nullptr) {
            auto libera = g_cuda_free_host.load();
            if (host_es_cuda && libera != nullptr) libera(host);
            else                                   std::free(host);
        }
        if (gpu != nullptr) {
            if (auto libera_gpu = g_cuda_free.load()) libera_gpu(gpu);
        }
        return true;
    });
}

}  // namespace

void UcxCommunicator::listener_conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    if (self == nullptr || conn_request == nullptr) return;
    self->enqueue_connection(conn_request);
}

void UcxCommunicator::endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    (void)ep;
    if (self != nullptr) {
        self->endpoint_failed_.store(true);
    }
    std::fprintf(stderr, "UCX endpoint error: %s\n", ucs_status_string(status));
}

// UCX AM receive callback: copy (or rendezvous-receive) the payload into the AM queue.

// ---------------------------------------------------------------------------------------
// Limite de pared para los bucles de progreso.
//
// Nueve de los doce bucles de ucp_worker_progress ya comprueban endpoint_failed_, pero esa
// bandera solo la pone endpoint_error_handler, y UCX solo invoca ese callback de forma fiable
// con UCP_ERR_HANDLING_MODE_PEER -- que, medido el 2026-07-28, hace que UCX descarte los
// transportes que no lo soportan y el plano de datos caiga a un TCP inalcanzable. Es decir:
// las comprobaciones existen y estan bien colocadas, pero estan desactivadas de hecho.
//
// Mientras endpoint_failed_ no tenga otra fuente, este plazo es el UNICO mecanismo de liveness.
// Apagado por defecto (0 = comportamiento original, espera indefinida).
namespace {
inline long gvs_progress_timeout_ns() {
    static const long v = [] {
        const char *e = std::getenv("GVIRTUS_UCX_PROGRESS_TIMEOUT_MS");
        return e ? std::strtol(e, nullptr, 10) * 1000000L : 0L;
    }();
    return v;
}
inline bool gvs_deadline_exceeded(const std::chrono::steady_clock::time_point &t0,
                                  unsigned long &spins) {
    const long lim = gvs_progress_timeout_ns();
    if (lim <= 0) return false;
    if ((++spins & 4095u) != 0) return false;   // consultar el reloj cada 4096 vueltas
    return std::chrono::steady_clock::now() - t0 > std::chrono::nanoseconds(lim);
}
}  // namespace


// --- Trazas de diagnostico del camino RmaSetup -> construccion del pool -------------------
// GVIRTUS_RMA_TRACE=1 las enciende. Apagadas por defecto: son fprintf sincronos y falsearian
// cualquier medida de latencia.
namespace {
inline bool gvs_rma_trace() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_RMA_TRACE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}
}  // namespace
#define GVS_RTRACE(fmt, ...)                                                    \
    do {                                                                        \
        if (gvs_rma_trace()) {                                                  \
            std::fprintf(stderr, "[GVS RTRACE] " fmt "\n", ##__VA_ARGS__);      \
            std::fflush(stderr);                                                \
        }                                                                       \
    } while (0)

ucs_status_t UcxCommunicator::am_recv_handler(void *arg, const void *header,
                                              size_t header_length, void *data,
                                              size_t length,
                                              const ucp_am_recv_param_t *param) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    (void)header;
    (void)header_length;

    if (self == nullptr) {
        return UCS_OK;
    }

    ucx_debug_log("am_recv_handler: self=%p length=%zu rndv=%d",
                  (void *)self, length,
                  (param != nullptr && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) ? 1 : 0);

    if (length == 0) {
        self->enqueue_am_message(PooledMsg{});
        return UCS_OK;
    }

    // Quick peek at the envelope header (small messages only, always eager):
    //   * RmaSetup → handshake message, unpack rkeys inline
    //   * RmaPosted → data already RDMA-put into an RX slot, queue a
    //                 PooledMsg pointing at that slot instead of acquiring
    //                 a fresh one + memcpy'ing
    if (length >= sizeof(gvirtus::communicators::ucxam::EnvelopeHeader) &&
        (param == nullptr || !(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV))) {
        gvirtus::communicators::ucxam::EnvelopeHeader peek;
        std::memcpy(&peek, data, sizeof(peek));
        if (peek.magic == gvirtus::communicators::ucxam::kEnvelopeMagic) {
            using gvirtus::communicators::ucxam::MessageType;
            if (peek.message_type == static_cast<std::uint16_t>(MessageType::RmaSetup)) {
                self->handle_rma_setup_am(data, length);
                return UCS_OK;
            }
            if (peek.message_type == static_cast<std::uint16_t>(MessageType::RmaPosted)) {
                const size_t slot_idx   = static_cast<size_t>(peek.reserved0);
                const size_t total      = static_cast<size_t>(peek.payload_size);
                // GPUDirect Step B3: non-zero routine_size means the peer
                // routed `gpu_size` bytes into slot.gpu_addr (via NIC
                // peer-DMA). status_code carries gpu_offset — the position
                // in the logical message where the GPU data folds in (i.e.
                // where to put the consolidation cudaMemcpy destination).
                const size_t gpu_size   = static_cast<size_t>(peek.routine_size);
                const size_t gpu_offset = static_cast<size_t>(peek.status_code);
                // F16 slow_read: retrasar la LECTURA del slot, no el ack. La carrera que la
                // guarda de epoch existe para evitar es esta: un ack viejo marca el slot Free,
                // el cliente lo reserva de nuevo y escribe encima MIENTRAS el backend todavia
                // lo esta leyendo. Con los tiempos reales el backend gana siempre -- medidos
                // 89 liberaciones prematuras por corrida y CERO corrupciones -- de modo que
                // sin ensanchar la ventana el dano no es observable ni existiendo el estado.
                // Esto solo la ensancha; no cambia ninguna decision del protocolo.
                static const long gusto_slow_read_ms = []() -> long {
                    const char *f = std::getenv("GVS_FAULT_SLOWREAD_MS");
                    const long v = (f != nullptr && f[0] != '\0')
                                       ? std::strtol(f, nullptr, 10) : 0;
                    if (v > 0) {
                        std::fprintf(stderr, "[GVS FAULT] *** slot READ delayed "
                                             "%ld ms (widens the collision window)\n", v);
                        std::fflush(stderr);
                    }
                    return v > 0 ? v : 0;
                }();
                if (gusto_slow_read_ms > 0)
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(gusto_slow_read_ms));
                std::lock_guard<std::mutex> lk(self->rx_pool_->mu);
                if (slot_idx >= self->rx_pool_->slots.size()) {
                    std::fprintf(stderr,
                                 "RmaPosted: invalid slot_idx=%zu (pool=%zu)\n",
                                 slot_idx, self->rx_pool_->slots.size());
                    return UCS_OK;
                }
                // The peer posted against a layout epoch: record it, so slots the
                // superseded advertisement retired can be released at the next safe
                // point instead of waiting for another regrow that may never come.
                {
                    const std::uint32_t posted_epoch =
                        gvirtus::communicators::ucxam::slot_tag_epoch(peek.request_id);
                    std::uint32_t prev =
                        self->rma_epoch_acked_.load(std::memory_order_relaxed);
                    while (posted_epoch > prev &&
                           !self->rma_epoch_acked_.compare_exchange_weak(
                               prev, posted_epoch, std::memory_order_relaxed)) {
                    }
                    if (posted_epoch > prev) {
                        self->rma_reclaim_requested_.store(true, std::memory_order_release);
                    }
                }

                auto &slot = self->rx_pool_->slots[slot_idx];
                // The client should only ever post into a live persistent slot it was
                // advertised. Anything else means it is working from a layout we have
                // replaced -- which the epoch in the tag is there to make survivable,
                // but it should never happen, so say so rather than consume blindly.
                if (!slot.rma_persistent || slot.rma_retired) {
                    std::fprintf(stderr,
                                 "RmaPosted: slot %zu is %s (epoch tag %llu) -- the peer "
                                 "is using a superseded layout\n",
                                 slot_idx,
                                 slot.rma_retired ? "retired" : "not an RMA slot",
                                 (unsigned long long)peek.request_id);
                }
                slot.in_use = true;
                // Remember this slot was filled by a client RMA put so that
                // release_rx_slot sends a SlotConsumed ack (with this
                // generation) back to the client for ABA-safe reuse.
                slot.rma_origin = true;
                slot.rma_generation = peek.request_id;
                // The peer's data is already in this slot. Mark it busy so the server's
                // own view matches reality; release_rx_slot clears it.
                slot.in_use = true;

                PooledMsg msg{slot.addr, total, slot_idx};

                // Step B3 CONSOLIDATION: temporarily cudaMemcpy the GPU
                // portion back into the host slot at offset (total - gpu_size).
                // This preserves the legacy contiguous-host parser path so
                // Buffer/handler dispatch needs no changes. Step B4 removes
                // this copy and teaches Buffer to read GPU directly.
                if (gpu_size > 0) {
                    if (slot.gpu_addr == nullptr ||
                        gpu_offset + gpu_size > total) {
                        std::fprintf(stderr,
                            "RmaPosted B4: gpu_size=%zu offset=%zu but slot %zu has no GPU shadow "
                            "(or offset+size > total=%zu) — protocol mismatch, dropping\n",
                            gpu_size, gpu_offset, slot_idx, total);
                        slot.in_use = false;
                        return UCS_OK;
                    }
                    // Step B4: no consolidation cudaMemcpy. The GPU portion
                    // stays in slot.gpu_addr and we publish it to the
                    // consumer via PooledMsg.gpu_data/gpu_size. Handlers
                    // that recognize the GPU payload (cudaMemcpy H2D) use
                    // cudaMemcpyDeviceToDevice directly from slot.gpu_addr
                    // instead of bouncing through host.
                    msg.gpu_data = slot.gpu_addr;
                    msg.gpu_size = gpu_size;
                    ucx_debug_log("RmaPosted B4: slot=%zu host_bytes=%zu gpu_bytes=%zu offset=%zu (no consolidation)",
                                  slot_idx, total - gpu_size, gpu_size, gpu_offset);
                }

                self->enqueue_am_message(msg);
                return UCS_OK;
            }
            if (peek.message_type ==
                static_cast<std::uint16_t>(MessageType::SlotConsumed)) {
                // Client side: the backend has finished consuming the remote RX
                // slot reserved0 that we filled via ucp_put. Return it to Free
                // (ABA-guarded by request_id = generation) so a WriteIovRma
                // waiter can reuse it. This is the explicit backend-consumption
                // confirmation the slot lifecycle is tied to.
                self->release_remote_slot(static_cast<size_t>(peek.reserved0),
                                          peek.request_id);
                return UCS_OK;
            }
        }
    }

    // Acquire a pinned slot from the RX pool — slot capacity is pre-allocated,
    // no per-message std::vector zero-init.
    // Demand-driven pool: a message at or above the RMA floor proves this connection
    // moves payloads big enough for the RMA path to pay off, so ask for the pool to be
    // built. Deferred rather than done here -- this callback runs under worker_mutex_
    // (ucp_worker_progress holds it) and both the allocation and the advertisement
    // would deadlock or stall progress. Read() picks it up.
    if (length >= ucx_rma_min_bytes()) {
        // Record the largest payload this connection has actually moved. The pool is
        // sized from this rather than from GVIRTUS_RMA_SLOT_CAP_MB, which becomes a
        // ceiling instead of a target.
        size_t prev = self->rma_pool_hint_bytes_.load(std::memory_order_relaxed);
        while (length > prev &&
               !self->rma_pool_hint_bytes_.compare_exchange_weak(
                   prev, length, std::memory_order_relaxed)) {
        }
        // A message at or above the RMA floor arriving EAGERLY means one of two
        // things, and both are answered the same way. Either the pool has not been
        // built yet, or it has been built too small and the sender's WriteIovRma
        // declined the fast path for capacity -- in which case the payload came down
        // this path precisely because no slot could hold it. Ask for (re)build at a
        // capacity derived from the size just observed. No extra control message is
        // needed: the decline is self-reporting.
        if (!self->rma_pool_ready_.load(std::memory_order_acquire) ||
            rma_slot_cap_for(length) >
                self->rma_pool_cap_.load(std::memory_order_acquire)) {
            self->rma_pool_requested_.store(true, std::memory_order_release);
        }
    }

    size_t slot_idx = self->acquire_rx_slot(length);
    PinnedSlot &slot = self->rx_pool_->slots[slot_idx];
    PooledMsg msg{slot.addr, length, slot_idx};

    if ((param != nullptr) && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        ucp_request_param_t recv_param{};
        recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        recv_param.datatype = ucp_dt_make_contig(1);
        if (slot.memh != nullptr) {
            recv_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            recv_param.memh = slot.memh;
        }

        void *request = ucp_am_recv_data_nbx(self->worker_, data, slot.addr, length,
                                             &recv_param);
        if (request == nullptr) {
            self->enqueue_am_message(msg);
            return UCS_OK;
        }
        if (UCS_PTR_IS_ERR(request)) {
            self->release_rx_slot(slot_idx);
            return UCS_PTR_STATUS(request);
        }

        self->enqueue_am_rndv(request, msg);
        return UCS_INPROGRESS;
    }

    std::memcpy(slot.addr, data, length);
    self->enqueue_am_message(msg);

    // For DATA callbacks we copy and return UCS_OK; UCX releases the data.
    return UCS_OK;
}

sockaddr_storage UcxCommunicator::make_sockaddr(const std::string &host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *res = nullptr;
    const std::string port_str = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        throw std::runtime_error("UcxCommunicator: getaddrinfo failed for " + host + ":" +
                                 port_str);
    }

    sockaddr_storage storage{};
    std::memcpy(&storage, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    return storage;
}

void UcxCommunicator::init_ucx() {
    if (initialized_) return;

    // Initialize UCX context/worker and register the AM receive callback.
    am_id_ = kUcxAmId;
    if (!am_state_) {
        am_state_ = std::make_shared<AmState>();
    }

    // Early read of GVIRTUS_GPUDIRECT (BEFORE ucp_init): UCX reads
    // UCX_RCACHE_ENABLE / UCX_MEMTYPE_CACHE at context creation time. The
    // rcache in this container/UCX combo fails on ucp_mem_map(CUDA) with
    // "failed to insert region [0x0..0x0]: Invalid parameter" — same root
    // cause as the production manual memh cache in WriteIovRma. Force-disable
    // rcache + memtype-cache so the CUDA mem_map succeeds. We use overwrite=0
    // so a user-provided value still wins.
    const char *gpudirect_env_early = std::getenv("GVIRTUS_GPUDIRECT");
    const bool gpudirect_env_set = (gpudirect_env_early != nullptr &&
                                    gpudirect_env_early[0] == '1');
    // GPUDirect requires the negotiated UCX transport to support CUDA
    // peer-DMA. UCX-TCP cannot move CUDA memory ("cannot find remote
    // protocol for put from cuda memory to host" error). Even though the
    // backend may have RDMA-class transports listed in UCX_TLS, if a
    // particular client connects over TCP, ucp_put_nbx from GPU mem fails.
    // Guard at process level: if UCX_TLS doesn't include any CUDA-capable
    // transport, do not enable GPUDirect even when GVIRTUS_GPUDIRECT=1.
    // Run a separate backend with UCX_TLS=tcp,self for UCX-TCP benchmarks.
    const bool tls_supports_cuda = []() {
        const char *tls = std::getenv("UCX_TLS");
        if (tls == nullptr) return true;
        std::string s(tls);
        return s.find("rc_mlx5") != std::string::npos ||
               s.find("dc_mlx5") != std::string::npos ||
               s.find("ud_mlx5") != std::string::npos ||
               s.find("ib")      != std::string::npos;
    }();
    const bool gpudirect_requested = gpudirect_env_set && tls_supports_cuda;
    if (gpudirect_requested) {
        setenv("UCX_RCACHE_ENABLE",   "n", /*overwrite=*/0);
        setenv("UCX_MEMTYPE_CACHE",   "n", /*overwrite=*/0);
    }

    ucp_params_t ucp_params{};
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    // AM: small control + legacy data path. RMA: bulk data via ucp_put_nbx
    // into pre-mem_map'd remote slots (avoids per-message rendezvous handshake).
    ucp_params.features = UCP_FEATURE_AM | UCP_FEATURE_RMA;

    ucs_status_t status = ucp_init(&ucp_params, nullptr, &context_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_init failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // GPUDirect probe (Step 1 of GVIRTUS_GPUDIRECT rollout). The early
    // read above already auto-set UCX_RCACHE_ENABLE=n / UCX_MEMTYPE_CACHE=n
    // when requested, so the ucp_mem_map(CUDA) below has a chance to succeed.
    //
    // Side-effect: we also setenv("GVIRTUS_GPUDIRECT_ACTIVE", "1"/"0") so the
    // cudart backend plugin can detect the post-probe state via getenv without
    // needing to link against this UCX library (avoids RTLD_GLOBAL surprises
    // since plugins are dlopen'd separately from libgvirtus-communicators-ucx).
    if (!gpudirect_requested) {
        g_gpudirect_enabled.store(false);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", "0", /*overwrite=*/1);
        if (gpudirect_env_set && !tls_supports_cuda) {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (UCX_TLS=%s has no CUDA-capable transport)\n",
                std::getenv("UCX_TLS") ? std::getenv("UCX_TLS") : "(unset)");
        } else {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (env GVIRTUS_GPUDIRECT not set)\n");
        }
    } else {
        std::string reason;
        const bool ok = probe_gpudirect(context_, reason);
        // The probe's cudaMalloc(4K) runs through the frontend cudart shim
        // during Connect (mpInitialized == false), so it returns the reentrancy-
        // guard init error (cudaErrorInitializationError) and leaves it as the
        // client-side sticky last error. That is an internal probe artifact, not
        // an application error — clear it so a later cudaGetLastError() by the
        // app doesn't spuriously observe it. On the backend this calls the real
        // cudaGetLastError at init time (a harmless no-op before any app work).
        if (auto gle = g_cuda_get_last_error.load()) gle();
        g_gpudirect_enabled.store(ok);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", ok ? "1" : "0", /*overwrite=*/1);
        if (ok) {
            std::fprintf(stderr,
                "[GVS] GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK, "
                "auto-set UCX_RCACHE_ENABLE=n UCX_MEMTYPE_CACHE=n)\n");
        } else {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (GVIRTUS_GPUDIRECT=1 but probe FAILED: %s) "
                "- falling back to host slots, behavior unchanged\n",
                reason.c_str());
        }
    }

    // parameters for ucp_worker
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create(context_, &worker_params, &worker_);
    if (status != UCS_OK) {
        ucp_cleanup(context_);
        context_ = nullptr;
        throw std::runtime_error("UcxCommunicator: ucp_worker_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // Arrancar el liberador AQUI y no en el primer crecimiento del pool: crear un hilo dentro
    // del callback de recepcion de UCX son ~50 us que no hacen ninguna falta en ese camino.
    gvs_arranca_liberador_una_vez();
    registra_contexto(context_);

    ucp_am_handler_param_t am_param{};
    am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                          UCP_AM_HANDLER_PARAM_FIELD_CB |
                          UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am_param.id = am_id_;
    // Copying eager payloads in the callback, so no persistent data is needed.
    am_param.flags = UCP_AM_FLAG_WHOLE_MSG;
    am_param.cb = &UcxCommunicator::am_recv_handler;
    am_param.arg = this;

    status = ucp_worker_set_am_recv_handler(worker_, &am_param);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: failed to set AM handler: " +
                                 std::string(ucs_status_string(status)));
    }

    // Pre-allocate pinned RX pool so the AM handler doesn't have to
    // zero-init a fresh std::vector for every incoming message.
    init_rx_pool();

    initialized_ = true;
    ucx_debug_log("init_ucx completed host=%s port=%u mode=am", hostname_.c_str(), port_);
}

void UcxCommunicator::destroy_ucx() {
    if (!initialized_) return;

    // Tear down UCX resources in reverse order of creation.
    ucx_debug_log("destroy_ucx begin endpoint=%p listener=%p worker=%p context=%p",
                  (void *)endpoint_, (void *)listener_, (void *)worker_, (void *)context_);

    if (endpoint_ != nullptr) {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

        ucp_request_param_t close_params{};
        close_params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        close_params.flags = endpoint_failed_.load() ? UCP_EP_CLOSE_FLAG_FORCE : 0;

        void *close_req = ucp_ep_close_nbx(endpoint_, &close_params);
        if (UCS_PTR_IS_ERR(close_req)) {
            std::fprintf(stderr, "UCX endpoint close failed: %s\n",
                         ucs_status_string(UCS_PTR_STATUS(close_req)));
        } else {
            wait_request_completion(close_req, "ep_close");
        }
        endpoint_ = nullptr;
    }

    // Release pre-registered TX scratch before tearing down the UCP
    // context — ucp_mem_unmap needs context_ still alive.
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        release_tx_scratch_locked();
    }

    if (owns_listener_ && listener_ != nullptr) {
        ucp_listener_destroy(listener_);
        listener_ = nullptr;
    }

    // Cerrar el trabajo diferido ANTES de deshacer nada. Dos pasos y en este orden:
    //   1. subir la generacion invalida las mejoras de slot que sigan en la cola, para que no
    //      reserven memoria fijada de un pool que esta a punto de desaparecer;
    //   2. drenar en ESTE hilo lo que quede, mientras el pool, el contexto y la conexion siguen
    //      vivos. Sin esto el liberador ejecutaba liberaciones DESPUES del cierre, y en el
    //      frontend `cudaFreeHost` es el shim de GVirtuS -- una RPC sobre una conexion muerta.
    //      Medido: d2hpool volcaba core 3 de 3 al salir, 0 de 3 con la lib anterior.
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        ++rx_pool_->generacion;
    }
    gvs_drena_tareas_ahora();

    // Destroy RMA state BEFORE the worker/context teardown — ucp_rkey_destroy
    // needs an alive context, and destroy_rx_pool calls ucp_mem_unmap.
    destroy_rma_state();
    destroy_rx_pool();
    current_frame_ = PooledMsg{};

    if (owns_worker_ && worker_ != nullptr) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }

    if (owns_context_ && context_ != nullptr) {
        // ANTES del cleanup: a partir de aqui ninguna tarea diferida intentara desmapear con el.
        retira_contexto(context_);
        ucp_cleanup(context_);
        context_ = nullptr;
    }

    initialized_ = false;
    endpoint_failed_.store(false);
    gvs_srcreg_report("al cerrar conexion");
    ucx_debug_log("destroy_ucx completed");
}

void UcxCommunicator::enqueue_connection(ucp_conn_request_h conn_request) {
    // Queue incoming connection requests from the listener callback.
    std::lock_guard<std::mutex> lock(conn_mutex_);
    pending_conn_requests_.push(conn_request);
    ucx_debug_log("enqueue_connection request=%p queue_size=%zu", (void *)conn_request,
                  pending_conn_requests_.size());
    conn_cv_.notify_one();
}

ucp_conn_request_h UcxCommunicator::wait_for_connection_request() {
    // Wait for a pending connection request while progressing the worker.
    std::unique_lock<std::mutex> lock(conn_mutex_);
    for (;;) {
        if (!running_) {
            return nullptr;
        }
        if (!pending_conn_requests_.empty()) {
            ucp_conn_request_h req = pending_conn_requests_.front();
            pending_conn_requests_.pop();
            return req;
        }
        conn_cv_.wait_for(lock, std::chrono::milliseconds(5));
        lock.unlock();
        if (worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }
        lock.lock();
    }
}

void UcxCommunicator::wait_request_completion(void *request, const char *op_name) {
    GVS_TRACE("wait_request_completion");
    // Progress the worker until the request completes (no sleep for low latency).
    ucx_debug_log("%s: wait_request_completion request=%p", op_name, request);

    if (request == nullptr) {
        ucx_debug_log("%s: immediate completion (null request)", op_name);
        return;
    }

    if (UCS_PTR_IS_ERR(request)) {
        throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                 " request error: " +
                                 ucs_status_string(UCS_PTR_STATUS(request)));
    }

    bool cancel_issued = false;
    unsigned long spins = 0;
    static const long progress_timeout_ns = [] {
        const char *v = std::getenv("GVIRTUS_UCX_PROGRESS_TIMEOUT_MS");
        return v ? std::strtol(v, nullptr, 10) * 1000000L : 0L;
    }();
    const auto t_start = std::chrono::steady_clock::now();
    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        if (worker_ != nullptr) {
            ucp_worker_progress(worker_);
        }

        progress_am_rndv();

        // If the endpoint has failed (for example, remote peer reset), cancel
        // the in-flight request so callers can unwind instead of hanging.
        if (!cancel_issued && endpoint_failed_.load() && worker_ != nullptr) {
            ucp_request_cancel(worker_, request);
            cancel_issued = true;
        }
        // La condicion del while es UNICAMENTE que la peticion siga UCS_INPROGRESS, de modo
        // que cancelar no basta: si ucp_request_cancel no la completa -- y con un endpoint ya
        // reseteado por el par es justo el caso en que puede no hacerlo -- el bucle gira para
        // siempre. Salir explicitamente tras la cancelacion convierte un cuelgue silencioso
        // que quema CPU en un error que el llamante puede desenrollar.
        if (cancel_issued && endpoint_failed_.load()) {
            ucp_request_free(request);
            throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                     " abortado: el endpoint fallo (par reseteado)");
        }
        // Limite de seguridad, apagado por defecto. No es un arreglo: es una red para que
        // ningun bucle de progreso pueda girar indefinidamente durante las pruebas aunque la
        // causa del bloqueo sea otra distinta del fallo de endpoint.
        if (progress_timeout_ns > 0) {
            if (++spins % 4096 == 0 &&
                std::chrono::steady_clock::now() - t_start >
                    std::chrono::nanoseconds(progress_timeout_ns)) {
                ucp_request_free(request);
                throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                         " abortado: GVIRTUS_UCX_PROGRESS_TIMEOUT_MS agotado");
            }
        }
    }

    const ucs_status_t final_status = ucp_request_check_status(request);
    ucp_request_free(request);
    if (final_status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                 " completion failed: " +
                                 ucs_status_string(final_status));
    }

    ucx_debug_log("%s: completed status=%s", op_name, ucs_status_string(final_status));
}

void UcxCommunicator::enqueue_am_message(PooledMsg message) {
    // Store a completed AM payload for stream-style Read() / TryAcquireFrame().
    {
        std::lock_guard<std::mutex> lock(am_state_->mutex);
        if (gvs_client_trace()) {
            const double t = gvs_cli_ms();
            g_ultimo_aterrizaje.store(t);
            std::fprintf(stderr, "[GVS CLI RX] mensaje ATERRIZA en la cola t=%.1fms (cola=%zu)\n",
                         t, am_state_->queue.size() + 1);
            std::fflush(stderr);
        }
        am_state_->queue.push_back(message);
    }
    am_state_->cv.notify_one();
}

void UcxCommunicator::enqueue_am_rndv(void *request, PooledMsg msg) {
    // Track a rendezvous receive until UCX reports completion.
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    am_state_->rndv.push_back(PendingAmRecv{request, msg});
}

void UcxCommunicator::progress_am_rndv() {
    // Check rendezvous receive requests and move completed payloads into the queue.
    //
    // Nada que entre en UCX se llama con am_state_->mutex tomado. El callback AM de UCX
    // corre CON el cerrojo del worker y ahi dentro llama a enqueue_am_message(), que toma
    // am_state_->mutex: ese orden (worker -> nuestro) lo impone UCX y no se puede cambiar.
    // Si aqui se tomase el nuestro y luego se entrase en UCX -- que es lo que hacian
    // ucp_request_free() y release_rx_slot() dentro del bucle -- los dos ordenes coexisten y
    // dos hilos se bloquean mutuamente. ThreadSanitizer lo marcaba 4 veces por corrida como
    // "lock-order-inversion (potential deadlock)". Ahora dentro del cerrojo solo se toca
    // nuestra estructura, y lo que entra en UCX se aplaza a despues de soltarlo.
    std::vector<void *> por_liberar;
    std::vector<size_t> slots_por_soltar;
    {
        std::lock_guard<std::mutex> lock(am_state_->mutex);
        for (auto it = am_state_->rndv.begin(); it != am_state_->rndv.end();) {
            if (it->request == nullptr) {
                am_state_->queue.push_back(it->msg);
                it = am_state_->rndv.erase(it);
                continue;
            }

            const ucs_status_t status = ucp_request_check_status(it->request);
            if (status == UCS_INPROGRESS) {
                ++it;
                continue;
            }

            por_liberar.push_back(it->request);
            if (status == UCS_OK) {
                am_state_->queue.push_back(it->msg);
                am_state_->cv.notify_one();
            } else {
                std::fprintf(stderr, "UCX AM rendezvous receive failed: %s\n",
                             ucs_status_string(status));
                // Release the slot since the message was never delivered.
                if (it->msg.slot_idx != static_cast<size_t>(-1)) {
                    slots_por_soltar.push_back(it->msg.slot_idx);
                }
            }
            it = am_state_->rndv.erase(it);
        }
    }
    for (void *req : por_liberar) ucp_request_free(req);
    for (size_t idx : slots_por_soltar) release_rx_slot(idx);
}

void UcxCommunicator::Serve() {
    // Start server listener for UCX client connections.
    init_ucx();

    sockaddr_storage ss = make_sockaddr(hostname_, port_);

    ucp_listener_params_t params{};
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&ss);
    params.sockaddr.addrlen = sizeof(sockaddr_in);
    params.conn_handler.cb = &UcxCommunicator::listener_conn_handler;
    params.conn_handler.arg = this;

    ucs_status_t status = ucp_listener_create(worker_, &params, &listener_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_listener_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    running_ = true;
    std::printf("UCX control-plane ready: Serve (%s:%u)\n", hostname_.c_str(), port_);
    ucx_debug_log("listener created listener=%p", (void *)listener_);
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    // Accept a connection and create a UCX endpoint for the new client.
    auto *self = const_cast<UcxCommunicator *>(this);
    if (!self->running_ || self->listener_ == nullptr) {
        return nullptr;
    }

    ucp_conn_request_h req = self->wait_for_connection_request();
    if (req == nullptr) {
        ucx_debug_log("Accept returned null request (shutdown or no request)");
        return nullptr;
    }

    auto *accepted = new UcxCommunicator(self->hostname_, self->port_);
    accepted->context_ = self->context_;
    accepted->initialized_ = true;
    // Ownership split: shares the listener's UCX context (heavy to create
    // and the rkeys we hand out are scoped to it), but owns a DEDICATED
    // worker so error progress on one accepted connection doesn't poison
    // the worker shared by other connections. Listener stays separate.
    accepted->owns_context_ = false;
    accepted->owns_worker_ = true;
    accepted->owns_listener_ = false;
    accepted->running_ = true;
    accepted->endpoint_failed_.store(false);

    // Per-connection worker. We don't reuse self->worker_; that one only
    // services the listener's conn_handler. Each accepted's data path runs
    // on its own worker -> its own ucp_worker_progress -> isolated request
    // state machine.
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    ucs_status_t status = ucp_worker_create(self->context_, &worker_params,
                                            &accepted->worker_);
    if (status != UCS_OK) {
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server ucp_worker_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // Per-connection AM state, RX pool, and worker mutex — no sharing with
    // the listener or with other accepted connections.
    accepted->worker_mutex_ = std::make_shared<std::mutex>();
    accepted->am_state_ = std::make_shared<AmState>();
    accepted->rx_pool_ = std::make_shared<RxPool>();

    // AM handler bound to THIS accepted's worker, with `arg = accepted` so
    // incoming messages land in its own am_state / rx_pool.
    ucp_am_handler_param_t am_param{};
    am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                          UCP_AM_HANDLER_PARAM_FIELD_CB |
                          UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am_param.id = self->am_id_;
    am_param.flags = UCP_AM_FLAG_WHOLE_MSG;
    am_param.cb = &UcxCommunicator::am_recv_handler;
    am_param.arg = accepted;
    status = ucp_worker_set_am_recv_handler(accepted->worker_, &am_param);
    if (status != UCS_OK) {
        ucp_worker_destroy(accepted->worker_);
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server AM handler register failed: " +
                                 std::string(ucs_status_string(status)));
    }

    ucp_ep_params_t ep_params{};
    // ERR_HANDLING_MODE_PEER es obligatorio para que UCX invoque err_handler.cb de forma
    // fiable ante un fallo del par. Sin el, el callback puede no dispararse nunca, con lo que
    // endpoint_failed_ no se pone y la comprobacion del bucle de progreso jamas se evalua:
    // el frontend se queda girando en ucp_worker_progress al 100 % de un nucleo. Medido con
    // cuDF: 48.279 ticks de CPU en un proceso que no habia ejecutado ni una consulta.
    // ERR_HANDLING_MODE_PEER NO se pide: medido el 2026-07-28, hace que UCX descarte los
    // transportes que no lo soportan y el plano de datos cae a tcp hacia un puerto efimero
    // inalcanzable ("Destination is unreachable" + assertion en stubs.c). El manejador de
    // error si se disparaba con el, pero a costa de perder la ruta RoCE entera. La liveness
    // se consigue con la salida explicita del bucle de progreso, que no altera el transporte.
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.conn_request = req;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = accepted;

    status = ucp_ep_create(accepted->worker_, &ep_params, &accepted->endpoint_);
    if (status != UCS_OK) {
        ucp_worker_destroy(accepted->worker_);
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    std::printf("UCX control-plane accepted connection\n");
    ucx_debug_log("server endpoint created endpoint=%p worker=%p from request=%p",
                  (void *)accepted->endpoint_, (void *)accepted->worker_, (void *)req);

    // Parallel setup: init_rx_pool (~150ms: cudaHostAlloc + ucp_mem_map for
    // each slot) and send_rma_setup (~50ms: pack rkeys + ucp_am_send_nbx)
    // run in a detached thread. The listener can return from Accept()
    // immediately and process the next conn_request while this thread
    // finishes setting up the previous one. The lambda thread spawned by
    // Process.cpp will block at its first incoming AM (via worker progress)
    // until the AM handler can acquire a slot from rx_pool — which is
    // exactly when this setup thread has finished init_rx_pool. Mutex on
    // rx_pool_->mu and am_state_->mutex serialises any actual contention.
    //
    // The client's Connect() waits up to 2 s for server's RmaSetup, so
    // even with setup taking ~250ms in the worst case the client doesn't
    // time out. Net effect for N concurrent connects:
    //   sequential: N × 350 ms serialised in the listener
    //   parallel:   ~max(setup_i) wall time (cudaHostAlloc/ucp_mem_map
    //               serialise at the CUDA/UCX driver level, so ~1.5-2x
    //               speedup rather than perfect N×, but still big).
    // Advertise immediately (with no slots yet) so the client's Connect() does not sit
    // waiting for us to allocate a pool it may never use. The pool is built the first
    // time a message large enough to need it actually arrives.
    std::thread([accepted]() {
        accepted->init_rx_pool();   // no-op unless GVIRTUS_RMA_PREALLOC=1
        accepted->send_rma_setup();
    }).detach();

    return accepted;
}

void UcxCommunicator::Connect() {
    // Connect to the UCX server and create a client endpoint.
    init_ucx();

    sockaddr_storage ss = make_sockaddr(hostname_, port_);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                           UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    // Ver la nota del lado Accept sobre ERR_HANDLING_MODE_PEER.
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&ss);
    ep_params.sockaddr.addrlen = sizeof(sockaddr_in);
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = this;

    ucs_status_t status = ucp_ep_create(worker_, &ep_params, &endpoint_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: client ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    running_ = true;
    endpoint_failed_.store(false);
    std::printf("UCX control-plane connected: Connect (%s:%u)\n", hostname_.c_str(), port_);
    ucx_debug_log("client endpoint created endpoint=%p", (void *)endpoint_);

    // Drive worker progress until the server's RmaSetup AM lands. If it
    // doesn't show up within the budget we silently fall back to the AM
    // data path (rma_setup_received_ stays false → WriteIov picks the IOV
    // branch). Useful when talking to an older server build that never
    // sends RmaSetup.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!rma_setup_received_.load() &&
           std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> wl(*worker_mutex_);
            ucp_worker_progress(worker_);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (!rma_setup_received_.load()) {
        ucx_debug_log("Connect: RmaSetup not received within timeout, RMA path disabled");
    } else {
        ucx_debug_log("Connect: RMA path enabled with %zu remote slots (server -> client)",
                      remote_slots_.size());

        // Bidirectional RMA: now that we know the server is RMA-capable
        // (it sent us its rkeys), advertise our own rx_pool's rkeys so it
        // can ucp_put_nbx into our slots for large responses (D2H 64MB
        // etc). Without this the server's WriteIov for the response falls
        // back to the AM-stream path, which is ~1.7s for 64MB without an
        // rcache. With this it becomes a single RDMA write + tiny AM ≈
        // ~10-15ms.
        send_rma_setup();
        ucx_debug_log("Connect: client rkeys advertised (client -> server done)");
    }
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    GVS_TRACE("Read");
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Read called without an active endpoint");
    }
    if (size == 0) {
        return 0;
    }

    // Safe point for the deferred pool build: no worker_mutex_ is held here.
    // Also fires for a REGROW, not just the first build: rma_pool_requested_ is set
    // again whenever a message arrives eagerly that the current slots cannot hold.
    // Mismo punto seguro para las liberaciones que se aplazaron por una captura abierta.
    // Va PRIMERO: reconstruir el pool con la memoria vieja aun retenida es gastar el doble
    // sin motivo, y aqui la ventana ya esta cerrada.
    // Este "punto seguro" corre en el hilo de la conexion, o sea EN EL CAMINO de su proxima RPC.
    // No esta dentro del callback de UCX, asi que no congela a las demas conexiones -- pero
    // materialise_rma_pool sigue haciendo cudaHostAlloc/cudaMalloc, que esperan a la GPU entera.
    // El guardia lo cubre para que, si eso aparece, salga con su pila y no con otra campana.
    {
        GuardiaProgreso _g;
        drena_liberaciones_aplazadas();
        if (rma_pool_requested_.load(std::memory_order_acquire)) {
            GVS_RTRACE("disparo diferido: entrando a materialise_rma_pool");
            materialise_rma_pool();
        }
        // Safe point for releasing superseded slots (see reclaim_retired_slots).
        if (rma_reclaim_requested_.load(std::memory_order_acquire)) {
            reclaim_retired_slots();
        }
    }
    // Mismo punto seguro: muestreo periodico de metricas. No toma cerrojos del worker.
    gusto_metric_maybe_emit();

    // Drain AM queue into the caller buffer, preserving stream semantics.
    // Busy-poll to keep ucp_worker_progress() running continuously.
    size_t copied = 0;
    const auto gvs_t0_read = std::chrono::steady_clock::now();
    unsigned long gvs_spins_read = 0;
    while (copied < size) {
        if (pending_msg_.data != nullptr &&
            pending_read_offset_ < pending_msg_.size) {
            const size_t available = pending_msg_.size - pending_read_offset_;
            const size_t to_copy = std::min(size - copied, available);
            std::memcpy(buffer + copied,
                        pending_msg_.data + pending_read_offset_,
                        to_copy);
            copied += to_copy;
            pending_read_offset_ += to_copy;
            if (pending_read_offset_ == pending_msg_.size) {
                // Fully consumed — return the pool slot.
                if (pending_msg_.slot_idx != static_cast<size_t>(-1)) {
                    release_rx_slot(pending_msg_.slot_idx);
                }
                pending_msg_ = PooledMsg{};
                pending_read_offset_ = 0;
            }
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(am_state_->mutex);
            if (!am_state_->queue.empty()) {
                if (gvs_client_trace()) {
                    const double t = gvs_cli_ms();
                    const double at = g_ultimo_aterrizaje.load();
                    std::fprintf(stderr,
                        "[GVS CLI RX] hilo %d RECOGE t=%.1fms (ultimo aterrizaje %.1fms, "
                        "espera en cola %.1fms)\n",
                        (int)syscall(SYS_gettid), t, at, t - at);
                    std::fflush(stderr);
                }
                pending_msg_ = am_state_->queue.front();
                am_state_->queue.pop_front();
                pending_read_offset_ = 0;
                continue;
            }
        }

        if (worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }

        progress_am_rndv();

        if (gvs_deadline_exceeded(gvs_t0_read, gvs_spins_read)) {
            std::fprintf(stderr, "[GVS] Read: GVIRTUS_UCX_PROGRESS_TIMEOUT_MS agotado "
                                 "(%zu/%zu bytes)\n", copied, size);
            std::fflush(stderr);
            return copied == 0 ? 0 : copied;
        }
        if (endpoint_failed_.load()) {
            return copied == 0 ? 0 : copied;
        }

    }

    return size;
}

bool UcxCommunicator::TryAcquireFrame(const unsigned char *&data, size_t &size) {
    GVS_TRACE("TryAcquireFrame");
    if (endpoint_ == nullptr || worker_ == nullptr) return false;

    // Safe point for the deferred pool build (no worker_mutex_ held yet). The backend
    // takes requests through this path, not Read(), so without this the pool would
    // never materialise on the side that receives the large payloads.
    // Also fires for a REGROW, not just the first build: rma_pool_requested_ is set
    // again whenever a message arrives eagerly that the current slots cannot hold.
    // Mismo punto seguro para las liberaciones que se aplazaron por una captura abierta.
    // Va PRIMERO: reconstruir el pool con la memoria vieja aun retenida es gastar el doble
    // sin motivo, y aqui la ventana ya esta cerrada.
    // Este "punto seguro" corre en el hilo de la conexion, o sea EN EL CAMINO de su proxima RPC.
    // No esta dentro del callback de UCX, asi que no congela a las demas conexiones -- pero
    // materialise_rma_pool sigue haciendo cudaHostAlloc/cudaMalloc, que esperan a la GPU entera.
    // El guardia lo cubre para que, si eso aparece, salga con su pila y no con otra campana.
    {
        GuardiaProgreso _g;
        drena_liberaciones_aplazadas();
        if (rma_pool_requested_.load(std::memory_order_acquire)) {
            GVS_RTRACE("disparo diferido: entrando a materialise_rma_pool");
            materialise_rma_pool();
        }
        // Safe point for releasing superseded slots (see reclaim_retired_slots).
        if (rma_reclaim_requested_.load(std::memory_order_acquire)) {
            reclaim_retired_slots();
        }
    }
    // Mismo punto seguro: muestreo periodico de metricas. No toma cerrojos del worker.
    gusto_metric_maybe_emit();

    // If we already hold a partially-consumed message, give up — the caller
    // mixed stream Read() with frame mode. Conservative: refuse the handoff.
    if (pending_msg_.data != nullptr && pending_read_offset_ > 0) {
        return false;
    }

    // Drain into current_frame_ once a message is available, busy-polling
    // the worker the same way Read() does.
    const auto gvs_t0_taf = std::chrono::steady_clock::now();
    unsigned long gvs_spins_taf = 0;
    for (;;) {
        if (current_frame_.data != nullptr) {
            data = current_frame_.data;
            size = current_frame_.size;
            return true;
        }

        // Inherit any message we may have moved into pending_msg_ already
        // (e.g., partial consumption of zero bytes).
        if (pending_msg_.data != nullptr && pending_read_offset_ == 0) {
            current_frame_ = pending_msg_;
            pending_msg_ = PooledMsg{};
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(am_state_->mutex);
            if (!am_state_->queue.empty()) {
                current_frame_ = am_state_->queue.front();
                am_state_->queue.pop_front();
                continue;
            }
        }

        if (gvs_spin_trace() && worker_ != nullptr) {
            // Un hilo se queda 50,6 s DENTRO de una iteracion de este bucle. Esto dice en cual
            // de las dos llamadas: esperando el mutex del worker, o dentro del progreso mismo.
            const double _t0 = gvs_cli_ms();
            std::unique_lock<std::mutex> wl(*worker_mutex_);
            const double _t1 = gvs_cli_ms();
            { GuardiaProgreso _g; ucp_worker_progress(worker_); }
            const double _t2 = gvs_cli_ms();
            wl.unlock();
            const double _t3 = gvs_cli_ms();
            progress_am_rndv();
            const double _t4 = gvs_cli_ms();
            if (_t1 - _t0 > 1000.0 || _t2 - _t1 > 1000.0 || _t4 - _t3 > 1000.0) {
                std::fprintf(stderr,
                    "[GVS SPIN] tid=%d TRAMO LENTO: mutex=%.1fms progress=%.1fms rndv=%.1fms\n",
                    (int)syscall(SYS_gettid), _t1 - _t0, _t2 - _t1, _t4 - _t3);
                std::fflush(stderr);
            }
        } else {
            if (worker_ != nullptr) {
                std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
                GuardiaProgreso _g;
                ucp_worker_progress(worker_);
            }
            progress_am_rndv();
        }

        // ¿Este hilo GIRA o esta parado? Es la pregunta que separa "el mensaje no llega" de
        // "el hilo no lo busca". Un hilo que espera 50 s deberia acumular millones de vueltas;
        // si acumula pocas, no esta girando y el bloqueo esta ANTES de aqui.
        // Apagado por defecto (GVS_SPIN_TRACE=1) y sin coste: dos comparaciones por vuelta.
        if (gvs_spin_trace()) {
            static thread_local double t_ult = 0.0;
            static thread_local unsigned long g_ult = 0;
            const double t_ahora = gvs_cli_ms();
            if (t_ult == 0.0) { t_ult = t_ahora; g_ult = gvs_spins_taf; }
            else if (t_ahora - t_ult >= 5000.0) {
                std::fprintf(stderr,
                    "[GVS SPIN] tid=%d GIRANDO %lu vueltas en %.1fs (worker=%p, esperando frame)\n",
                    (int)syscall(SYS_gettid), gvs_spins_taf - g_ult,
                    (t_ahora - t_ult) / 1000.0, (void *)worker_);
                std::fflush(stderr);
                t_ult = t_ahora; g_ult = gvs_spins_taf;
            }
        }

        if (gvs_deadline_exceeded(gvs_t0_taf, gvs_spins_taf)) {
            std::fprintf(stderr, "[GVS] TryAcquireFrame: GVIRTUS_UCX_PROGRESS_TIMEOUT_MS agotado\n");
            std::fflush(stderr);
            return false;
        }
        if (endpoint_failed_.load()) return false;
    }
}

void UcxCommunicator::ReleaseFrame() {
    GVS_TRACE("ReleaseFrame");
    if (current_frame_.slot_idx != static_cast<size_t>(-1)) {
        // Wait for any device work the handler left in flight against this frame
        // (the GPUDirect shadow -> destination copy) BEFORE the slot is declared
        // free. The response has already been sent by now, so this costs the client
        // nothing; skipping it would let the peer's next peer-DMA land on top of a
        // buffer the copy engine is still reading.
        RunFrameDrainHook();
        release_rx_slot(current_frame_.slot_idx);
    }
    current_frame_ = PooledMsg{};
}

// Per-connection GPUDirect gate (Option 2). Returns true iff THIS
// endpoint negotiated an RDMA-class transport (rc_mlx5 / dc_mlx5 /
// ud_mlx5 / ib) capable of carrying CUDA memory operations.
//
// This is a property of the TRANSPORT, not of the local process. In
// particular it does NOT depend on g_gpudirect_enabled (= local probe
// of cudaMalloc + ucp_mem_map(CUDA), which requires nvidia-peermem
// loaded on this host). Reason: frontend Variant B (host → remote GPU
// shadow) puts data FROM host memory, so the local NIC doesn't need
// peer-DMA-from-CUDA capability — only RDMA-class transport plus the
// backend's gpu_rkey suffice.
//
// The "process can locally do CUDA peer-DMA" precondition is enforced
// separately in places where the local side IS the CUDA mem source —
// notably gvirtus_gpudirect_enabled() in CudaRtHandler_memory.cpp,
// which AND-s GVIRTUS_GPUDIRECT_ACTIVE (env, set by init_ucx based
// on the probe) with tls_connection_supports_cuda (this method).
//
// Lazy + cached: ucp_ep_query returns the negotiated lanes only after
// wire-up completes (async, after first AM exchange). The first caller
// (WriteIovRma at the first cudaMemcpy >= 4 MB, or Process.cpp's pre-
// Execute set of tls_connection_supports_cuda) happens well after
// wire-up. ucp_ep_query failures don't cache so a later call retries.
bool UcxCommunicator::current_connection_supports_cuda() const {
    int cached = supports_cuda_cached_.load(std::memory_order_acquire);
    if (cached != -1) return cached == 1;

    if (endpoint_ == nullptr) {
        return false;  // don't cache — endpoint may still be assigned later
    }

    // We use ucp_ep_print_info instead of ucp_ep_query(TRANSPORTS) because
    // in UCX 1.20 (this container) ucp_ep_query returns UCS_OK with
    // num_entries>0 but transport_name/device_name as NULL pointers — a
    // quirk likely tied to lane wire-up state or an ABI mismatch between
    // the pinned header and the loaded .so. ucp_ep_print_info renders the
    // lane info as text to a FILE* and is the API used by ucx_info and
    // verbose UCX logs, so its output is well tested across builds.
    //
    // Captured via open_memstream and grep'd for RDMA-class transport
    // tokens. Expected output lines look like:
    //   #     lane[1]: 2:rc_mlx5/mlx5_1:1.0 md[2] -> md[2]/ib/sysdev[3] ... rma_bw#0 am
    // Tokens rc_mlx5 / dc_mlx5 / ud_mlx5 (mlx5 driver) and rc_verbs /
    // dc_verbs / ud_verbs (generic verbs) indicate an RDMA-class lane.
    char *buf = nullptr;
    size_t buf_size = 0;
    FILE *fp = open_memstream(&buf, &buf_size);
    if (fp == nullptr) {
        return false;  // memstream alloc failed — retry next call
    }
    ucp_ep_print_info(endpoint_, fp);
    std::fclose(fp);

    if (buf == nullptr || buf_size == 0) {
        if (buf) std::free(buf);
        return false;
    }

    const bool supports = (std::strstr(buf, "rc_mlx5") != nullptr) ||
                          (std::strstr(buf, "dc_mlx5") != nullptr) ||
                          (std::strstr(buf, "ud_mlx5") != nullptr) ||
                          (std::strstr(buf, "rc_verbs") != nullptr) ||
                          (std::strstr(buf, "dc_verbs") != nullptr) ||
                          (std::strstr(buf, "ud_verbs") != nullptr);
    std::free(buf);

    supports_cuda_cached_.store(supports ? 1 : 0, std::memory_order_release);
    ucx_debug_log("current_connection_supports_cuda: endpoint=%p -> %s",
                  (void *)endpoint_,
                  supports ? "RDMA (CUDA-capable)" : "non-RDMA (TCP-class)");
    return supports;
}

// Register `slot.addr/capacity` (host) AND `slot.gpu_addr/gpu_capacity` (if
// present) with the UCX context so subsequent ucp_am_recv_data_nbx and
// ucp_put_nbx can use the memh hints and skip on-the-fly IB registration.
// Called with rx_pool_->mu held.
void UcxCommunicator::map_slot_to_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx == nullptr) return;
    if (slot.memh == nullptr && slot.addr != nullptr) {
        ucp_mem_map_params_t map_params{};
        map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        map_params.address = slot.addr;
        map_params.length  = slot.capacity;
        gvs_mm_count("map:L1631", true);
        ucs_status_t st = ucp_mem_map(ctx, &map_params, &slot.memh);
        if (st != UCS_OK) {
            slot.memh = nullptr;  // continue without — UCX rcache will register on first use
        }
    }
    // GPUDirect (Step B1): register the GPU shadow if it exists. UCX needs
    // UCS_MEMORY_TYPE_CUDA explicitly here since memtype-cache is disabled.
    if (slot.gpu_memh == nullptr && slot.gpu_addr != nullptr) {
        ucp_mem_map_params_t gpu_params{};
        gpu_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        gpu_params.address     = slot.gpu_addr;
        gpu_params.length      = slot.gpu_capacity;
        gpu_params.memory_type = UCS_MEMORY_TYPE_CUDA;
        gvs_mm_count("map:L1646", true);
        ucs_status_t st = ucp_mem_map(ctx, &gpu_params, &slot.gpu_memh);
        if (st != UCS_OK) {
            slot.gpu_memh = nullptr;
            ucx_debug_log("map_slot_to_ucp: gpu_addr map FAILED (%s) — slot will keep host-only path",
                          ucs_status_string(st));
        }
    }
}

void UcxCommunicator::unmap_slot_from_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx != nullptr && slot.memh != nullptr) {
        gvs_mm_count("unmap:L1657", false);
        ucp_mem_unmap(ctx, slot.memh);
        slot.memh = nullptr;
    }
    if (ctx != nullptr && slot.gpu_memh != nullptr) {
        gvs_mm_count("unmap:L1661", false);
        ucp_mem_unmap(ctx, slot.gpu_memh);
        slot.gpu_memh = nullptr;
    }
}

// Pre-allocate N RX slots of an initial size. Slots grow on demand later if
// a message arrives that's bigger than the current capacity.
// Slot capacity to build for the largest payload observed on this connection.
// GVIRTUS_RMA_SLOT_CAP_MB stops being the size we allocate and becomes the most we
// are ever willing to allocate; GVIRTUS_RMA_SLOT_MIN_MB is the least. The payload is
// rounded up to a power of two so a workload whose sizes creep upward regrows a
// bounded number of times rather than once per distinct size, and the framing slack
// is added on top so a transfer of exactly a power of two still fits (the slot has to
// carry the request header, routine name and marshalled argument Buffer alongside the
// payload; without the slack it overshoots by ~90 bytes and silently falls back to
// eager AM at a 3.2x cost).
size_t UcxCommunicator::rma_slot_cap_for(size_t hint_bytes) {
    static constexpr size_t kSlotFramingSlack = 64u * 1024u;
    auto env_mb = [](const char *k, size_t dflt) -> size_t {
        const char *v = std::getenv(k);
        if (v == nullptr || v[0] == '\0') return dflt;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(v, &end, 10);
        return (parsed > 0) ? static_cast<size_t>(parsed) : dflt;
    };
    const size_t ceiling = env_mb("GVIRTUS_RMA_SLOT_CAP_MB", 1025) * 1024u * 1024u +
                           kSlotFramingSlack;
    // Floor defaults to the RMA threshold itself, not an arbitrary constant: a slot
    // smaller than the smallest message that takes the RMA path can never be used, and
    // anything larger is memory reserved on speculation.
    //
    // The 16 MB it used to be was picked without measurement. Measured (payload 4 MB,
    // 12 transfers, floors 4/8/16/32 MB): first transfer 29.9/30.2/30.0/29.9 ms and
    // steady state 8.7/10.8/8.0/10.8 GB/s -- no trend, the spread is run-to-run noise.
    // The floor costs nothing in time and everything in memory (4x at floor 16 vs 4 for
    // 4 MB traffic), so it should be as small as is useful. A workload whose sizes creep
    // upward pays a regrow instead, which is one staged transfer (~130 ms, measured in
    // growtest) and is bounded to log2 steps by the power-of-two rounding.
    const size_t floor_default = ucx_rma_min_bytes();
    const char *floor_env = std::getenv("GVIRTUS_RMA_SLOT_MIN_MB");
    const size_t floor_bytes =
        (floor_env != nullptr && floor_env[0] != '\0')
            ? env_mb("GVIRTUS_RMA_SLOT_MIN_MB", 4) * 1024u * 1024u + kSlotFramingSlack
            : floor_default + kSlotFramingSlack;
    if (hint_bytes == 0) return ceiling;  // no evidence yet (eager prealloc)

    // Round the PAYLOAD to a power of two, then add the slack -- not the other way
    // round. hint_bytes is a whole wire message, i.e. payload plus ~80 bytes of
    // framing, so rounding it directly would push a 64 MiB transfer (67108942 B) to
    // the next power of two and allocate 128 MiB for a 64 MiB payload. Taking the
    // framing off first lands exactly on 64 MiB + slack, which is what actually has
    // to fit.
    const size_t payload = (hint_bytes > kSlotFramingSlack)
                               ? (hint_bytes - kSlotFramingSlack)
                               : hint_bytes;
    size_t pow2 = 1;
    while (pow2 < payload && pow2 < ceiling) pow2 <<= 1;
    size_t cap = pow2 + kSlotFramingSlack;
    if (cap < floor_bytes) cap = floor_bytes;
    if (cap > ceiling) cap = ceiling;
    return cap;
}

// Free persistent slots retired at least one full epoch ago. Retirement is NOT
// immediate release: when the pool grows, a client that has not yet processed the new
// advertisement may still have a ucp_put in flight to the old address, and the NIC
// would write into freed, unregistered memory. Holding them for one extra epoch means
// every put has by then been addressed against a layout the client demonstrably has
// (it used it). Caller must hold rx_pool_->mu.
void UcxCommunicator::retire_and_free_locked(std::uint32_t now_epoch) {
    const std::uint32_t acked = rma_epoch_acked_.load(std::memory_order_acquire);
    for (auto &sl : rx_pool_->slots) {
        if (!sl.rma_retired || sl.in_use) continue;
        // sl.rma_epoch is the epoch of the advertisement that SUPERSEDED this slot.
        // Two independent reasons it is now safe to release:
        //   (a) the peer has posted against that epoch or later, so it demonstrably
        //       holds the new layout and cannot have a put addressed here; or
        //   (b) a further advertisement has since gone out, the original one-epoch
        //       grace, kept as a backstop for a peer that goes quiet.
        const bool peer_moved_on = acked >= sl.rma_epoch;
        const bool grace_expired = now_epoch > sl.rma_epoch + 1;
        if (!peer_moved_on && !grace_expired) continue;
        // GVIRTUS_RMA_KEEP_RETIRED=1: retencion controlada para el experimento causal.
        //
        // NINGUNA de las dos condiciones de arriba demuestra DRENAJE. Que el par haya
        // reconocido la epoca E prueba que recibio la metadata, no que sus PUT contra E
        // hayan alcanzado completado remoto; y grace_expired libera memoria registrada por
        // el simple paso de dos numeros de epoca, sin prueba ninguna. cuDF dispara varios
        // resizes seguidos y en epoch 3 la conexion se resetea: es el sintoma de un
        // use-after-unmap, y esta variable lo aisla sin cambiar el protocolo.
        //
        // Con ella el slot deja de anunciarse y de aceptar operaciones nuevas, pero NO se
        // desmapea, NO se libera la memoria host, NO se libera el shadow GPU y NO se destruye
        // su metadata: se conserva hasta el teardown de la conexion. Retiene memoria durante
        // la sesion a proposito. Si el fallo desaparece con esto, la causa es el lifetime.
        static const bool keep_retired = [] {
            const char *v = std::getenv("GVIRTUS_RMA_KEEP_RETIRED");
            return v != nullptr && v[0] == '1' && v[1] == '\0';
        }();
        if (keep_retired) {
            static std::atomic<unsigned long> kept{0};
            const unsigned long n = ++kept;
            if (n == 1 || n % 16 == 0) {
                std::fprintf(stderr,
                             "[GVS] rx_pool: KEEP_RETIRED keeps slot (%zu B host%s, "
                             "retired in epoch %u, acked %u) -- %lu kept\n",
                             sl.capacity, sl.gpu_addr ? " + GPU shadow" : "", sl.rma_epoch,
                             acked, n);
                std::fflush(stderr);
            }
            continue;   // sigue retirado: no se anuncia, pero su registro sobrevive
        }
        std::fprintf(stderr,
                     "[GVS] rx_pool: releasing superseded slot (%zu B host%s, retired at "
                     "epoch %u, peer acked %u)\n",
                     sl.capacity, sl.gpu_addr ? " + GPU shadow" : "", sl.rma_epoch, acked);
        std::fflush(stderr);
        unmap_slot_from_ucp(context_, sl);
        free_pinned_host(sl.addr, sl.is_cuda_host);
        free_gpu_slot(sl.gpu_addr);
        sl = PinnedSlot{};  // capacity 0, not persistent, not retired: reusable entry
    }
}

// F4: validacion centralizada del provisioning. Estatica y sin estado salvo el aviso unico.
bool UcxCommunicator::gusto_validate_pool_cfg(size_t slots, size_t cap_bytes,
                                              bool with_shadow) {
    auto env_u64 = [](const char *k, unsigned long long d) -> unsigned long long {
        const char *v = std::getenv(k);
        if (v == nullptr || v[0] == '\0') return d;
        char *end = nullptr;
        unsigned long long p = std::strtoull(v, &end, 10);
        return (end != v) ? p : d;
    };
    const size_t max_slots = (size_t)env_u64("GUSTO_RMA_MAX_SLOTS", 1024);

    if (slots == 0 || slots > max_slots) {
        std::fprintf(stderr, "[GUSTO CFG] REJECTED: GVIRTUS_RMA_SLOTS=%zu outside [1,%zu]. "
                             "The pool is not built; the transport uses the AM path.\n",
                     slots, max_slots);
        std::fflush(stderr);
        return false;
    }
    if (cap_bytes == 0) {
        std::fprintf(stderr, "[GUSTO CFG] REJECTED: slot capacity is 0.\n");
        std::fflush(stderr);
        return false;
    }
    // Desbordamiento ANTES de reservar: por slot se paga host + (shadow ? GPU : 0).
    const size_t per_slot = with_shadow ? (cap_bytes * 2u) : cap_bytes;
    if (with_shadow && per_slot / 2u != cap_bytes) {
        std::fprintf(stderr, "[GUSTO CFG] REJECTED: overflow doubling the capacity "
                             "(%zu B) for the GPU shadow.\n", cap_bytes);
        std::fflush(stderr);
        return false;
    }
    if (per_slot != 0 && slots > (size_t)-1 / per_slot) {
        std::fprintf(stderr, "[GUSTO CFG] REJECTED: %zu slots x %zu B overflows size_t.\n",
                     slots, per_slot);
        std::fflush(stderr);
        return false;
    }
    const size_t total = slots * per_slot;
    const unsigned long long budget = env_u64("GUSTO_RMA_HOST_POOL_BUDGET_BYTES", 0);
    if (budget != 0 && (unsigned long long)total > budget) {
        std::fprintf(stderr,
                     "[GUSTO CFG] REJECTED: the pool asks for %.1f MiB and the budget "
                     "GUSTO_RMA_HOST_POOL_BUDGET_BYTES is %.1f MiB. Not built.\n",
                     total / 1048576.0, budget / 1048576.0);
        std::fflush(stderr);
        return false;
    }
    // Config EFECTIVA. Sin esta linea, "lo puse" y "lo leyo" son indistinguibles desde fuera,
    // que es exactamente como una campana entera acabo midiendo el camino AM creyendo medir
    // el pool.
    std::fprintf(stderr,
                 "[GUSTO CFG] effective pool: slots=%zu cap=%.1f MiB shadow=%s "
                 "total=%.1f MiB%s\n",
                 slots, cap_bytes / 1048576.0, with_shadow ? "yes" : "no",
                 total / 1048576.0,
                 budget ? "" : " (no budget defined)");
    std::fflush(stderr);
    return true;
}

void UcxCommunicator::init_rx_pool() {
    // 2 slots is enough for the current synchronous request/response pattern
    // (the request occupies slot 0 while the response is in flight; once the
    // app receives the response, slot 0 is free for the next request). Was
    // 4 originally but each cudaHostAlloc(64MB) + ucp_mem_map costs ~75ms;
    // halving the count halves per-connection setup time.
    // Slot count and per-slot capacity are configurable for the async
    // dispatcher (Phase 2): more slots let the frontend keep more fire-and-forget
    // large-H2D copies in flight before it must drain. Defaults preserve the
    // original synchronous behaviour (2 x 1025 MB). For async, prefer more,
    // smaller slots to bound memory, e.g. GVIRTUS_RMA_SLOTS=8
    // GVIRTUS_RMA_SLOT_CAP_MB=128. Note total pinned memory is
    // count * cap * (1 + gpudirect_shadow); keep it modest (GPU-leak sensitive).
    auto env_size = [](const char *k, size_t dflt) -> size_t {
        const char *v = std::getenv(k);
        if (v == nullptr || v[0] == '\0') return dflt;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(v, &end, 10);
        return (parsed > 0) ? static_cast<size_t>(parsed) : dflt;
    };
    // F4: el lambda env_size devuelve el DEFAULT ante 0 o ante basura, asi que una erratacomo
    // GVIRTUS_RMA_SLOTS=0 se convertia en 2 en silencio y jamas llegaba a la validacion.
    // Se mira el crudo antes de normalizar.
    {
        const char *raw = std::getenv("GVIRTUS_RMA_SLOTS");
        if (raw != nullptr && raw[0] != '\0') {
            char *end = nullptr;
            unsigned long long p = std::strtoull(raw, &end, 10);
            if (end == raw || *end != '\0' || p == 0) {
                std::fprintf(stderr, "[GUSTO CFG] REJECTED: GVIRTUS_RMA_SLOTS='%s' is not a "
                                     "positive integer. The pool is not built.\n", raw);
                std::fflush(stderr);
                return;
            }
        }
    }
    const size_t kInitialSlotCount = env_size("GVIRTUS_RMA_SLOTS", 2);

    // Lazy by default: a connection that never sends anything at or above the RMA
    // floor can never use these buffers, so it should not pay for them (2.2 s of
    // connect at the stock capacity) nor hold their GPU shadow away from other
    // tenants. materialise_rma_pool() calls this once a large message proves the
    // connection needs it. GVIRTUS_RMA_PREALLOC=1 restores eager provisioning.
    static const bool prealloc = []() {
        const char *e = std::getenv("GVIRTUS_RMA_PREALLOC");
        return e != nullptr && e[0] != '0';
    }();
    if (!prealloc && !rma_pool_requested_.load(std::memory_order_acquire)) return;

    // Size to the evidence, not to the knob. With no evidence (eager prealloc) this
    // returns the ceiling, i.e. the historical behaviour.
    const size_t kInitialSlotCap = rma_slot_cap_for(
        prealloc ? 0 : rma_pool_hint_bytes_.load(std::memory_order_acquire));

    // F4: validar ANTES de reservar. Si no pasa, no se construye nada y no queda un pool a
    // medias: el transporte sigue por AM, que es degradacion, no fallo.
    if (!gusto_validate_pool_cfg(kInitialSlotCount, kInitialSlotCap,
                                 g_gpudirect_enabled.load())) {
        return;
    }

    std::lock_guard<std::mutex> lk(rx_pool_->mu);

    const std::uint32_t new_epoch = rma_pool_epoch_.load(std::memory_order_acquire) + 1;
    // Reclaim anything retired two epochs ago before allocating more.
    retire_and_free_locked(new_epoch);

    // NOT "is the pool empty": the AM receive path appends a message-sized slot for
    // every message that arrives, so by the time a large transfer asks for the pool
    // there are already several tiny slots here. Count the LIVE persistent slots that
    // are actually big enough to serve the RMA path at the capacity we now want.
    // A shadow is allocated only once this connection has PROVEN it moves
    // device-destined bulk data (NoteDeviceDestinedPayload), not merely because the
    // process passed the GPUDirect probe. The shadow doubles the pool's cost in DEVICE
    // memory; a connection that never peer-DMAs into it was pinning that away from
    // every other tenant for nothing. GVIRTUS_RMA_GPU_SHADOW_EAGER=1 restores the old
    // always-allocate behaviour.
    static const bool shadow_eager = []() {
        const char *e = std::getenv("GVIRTUS_RMA_GPU_SHADOW_EAGER");
        return e != nullptr && e[0] != '0';
    }();
    const bool want_shadow =
        shadow_eager || rma_want_gpu_shadow_.load(std::memory_order_acquire);
    const bool need_shadow_now = want_shadow && g_gpudirect_enabled.load();

    // "Already provisioned" now means big enough AND carrying a shadow if one is
    // wanted, so that gaining the shadow triggers a rebuild the same way gaining size
    // does.
    size_t full_slots = 0;
    for (const auto &sl : rx_pool_->slots)
        if (sl.rma_persistent && !sl.rma_retired && sl.capacity >= kInitialSlotCap &&
            (!need_shadow_now || sl.gpu_addr != nullptr))
            ++full_slots;
    if (full_slots >= kInitialSlotCount) {
        rma_pool_cap_.store(kInitialSlotCap, std::memory_order_release);
        rma_pool_has_shadow_.store(need_shadow_now, std::memory_order_release);
        return;  // already provisioned at (at least) this capacity
    }

    // Growing: the live persistent slots are too small for what this connection has
    // turned out to move. Retire them -- stop advertising them and stop handing them
    // out -- but do not free them yet (see retire_and_free_locked).
    for (auto &sl : rx_pool_->slots) {
        if (!sl.rma_persistent || sl.rma_retired) continue;
        const bool too_small = sl.capacity < kInitialSlotCap;
        const bool missing_shadow = need_shadow_now && sl.gpu_addr == nullptr;
        if (!too_small && !missing_shadow) continue;
        sl.rma_retired = true;
        sl.rma_epoch = new_epoch;
        ucx_debug_log("rx_pool: retiring persistent slot (%zu B, shadow=%s) -- regrow to "
                      "%zu B%s",
                      sl.capacity, sl.gpu_addr ? "yes" : "no", kInitialSlotCap,
                      missing_shadow ? " + GPU shadow" : "");
    }

    // GPUDirect (Step B1): when active, each slot ALSO gets a GPU shadow
    // region of the same capacity, mem_map'd as CUDA. The shadow is unused
    // in B1 — purely additive. Step B2 will advertise its rkey to peers;
    // Step B3 will route big H2D payloads here via NIC peer-DMA.
    const bool gpudirect_active = g_gpudirect_enabled.load() && want_shadow;
    size_t gpu_allocated_count = 0;

    // Append: resize() would destroy the small on-demand slots, which may be in use.
    const size_t base = rx_pool_->slots.size();
    rx_pool_->slots.resize(base + (kInitialSlotCount - full_slots));
    for (size_t i = base; i < rx_pool_->slots.size(); ++i) {
        bool is_cuda = false;
        GVS_RTRACE("alloc_pinned_host: pidiendo %zu B (slot %zu)", kInitialSlotCap, i);
        unsigned char *p = alloc_pinned_host(kInitialSlotCap, is_cuda);
        GVS_RTRACE("alloc_pinned_host: VOLVIO (%p)", (void *)p);
        if (p == nullptr) {
            throw std::runtime_error("UcxCommunicator: failed to allocate RX pool slot");
        }
        rx_pool_->slots[i] = PinnedSlot{p, kInitialSlotCap, /*in_use*/false, is_cuda, nullptr};
        rx_pool_->slots[i].rma_persistent = true;  // only these may be ucp_put into
        rx_pool_->slots[i].rma_epoch = new_epoch;

        if (gpudirect_active) {
            GVS_RTRACE("alloc_gpu_slot: pidiendo %zu B", kInitialSlotCap);
            unsigned char *gp = alloc_gpu_slot(kInitialSlotCap);
            GVS_RTRACE("alloc_gpu_slot: VOLVIO (%p)", (void *)gp);
            if (gp != nullptr) {
                rx_pool_->slots[i].gpu_addr = gp;
                rx_pool_->slots[i].gpu_capacity = kInitialSlotCap;
                ++gpu_allocated_count;
            } else {
                ucx_debug_log("rx_pool: slot %zu gpu shadow alloc FAILED — host-only", i);
            }
        }

        map_slot_to_ucp(context_, rx_pool_->slots[i]);
    }
    if (gpudirect_active) {
        std::fprintf(stderr,
            "[GVS] rx_pool: initialized %zu slots x %zu bytes (host) + %zu/%zu GPU shadows x %zu bytes\n",
            kInitialSlotCount, kInitialSlotCap,
            gpu_allocated_count, kInitialSlotCount, kInitialSlotCap);
    }
    ucx_debug_log("rx_pool: initialized %zu slots x %zu bytes (gpu_shadows=%zu)",
                  kInitialSlotCount, kInitialSlotCap, gpu_allocated_count);
    rma_pool_cap_.store(kInitialSlotCap, std::memory_order_release);
    rma_pool_has_shadow_.store(gpu_allocated_count > 0, std::memory_order_release);
}

// A device-destined bulk payload had to be staged through the host slot because this
// connection's pool has no GPU shadow. That is the evidence the shadow is worth its
// device memory HERE, so ask for the pool to be rebuilt with one. The rebuild goes
// through exactly the same materialise/retire/re-advertise path as a size regrow, so
// it inherits the epoch and quiesce handling rather than needing its own.
// Release slots a superseded advertisement retired, once the peer has demonstrably
// moved to the new layout. Called from Read()/TryAcquireFrame(), never from the AM
// callback that observes the epoch: that runs under worker_mutex_ and freeing a slot
// calls cudaFreeHost/cudaFree, which can block and would stall worker progress.
void UcxCommunicator::reclaim_retired_slots() {
    // Igual que materialise_rma_pool: liberar un slot retirado llama a cudaFreeHost/cudaFree.
    // Se comprueba ANTES del exchange para no consumir la peticion; se reclama al cerrar.
    if (captura_abierta()) return;
    if (!rma_reclaim_requested_.exchange(false, std::memory_order_acq_rel)) return;
    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    retire_and_free_locked(rma_pool_epoch_.load(std::memory_order_acquire));
}

void UcxCommunicator::NoteDeviceDestinedPayload(size_t bytes) {
    if (bytes < ucx_rma_min_bytes()) return;
    if (!g_gpudirect_enabled.load()) return;   // no shadow is possible on this backend
    if (rma_pool_has_shadow_.load(std::memory_order_acquire)) return;  // already has one
    if (rma_want_gpu_shadow_.exchange(true, std::memory_order_acq_rel)) {
        return;  // already asked; the rebuild is pending
    }
    std::fprintf(stderr,
                 "[GVS] rx_pool: connection moved %zu B of device-destined payload "
                 "through the host slot -- adding GPU shadows on the next rebuild\n",
                 bytes);
    std::fflush(stderr);
    rma_pool_requested_.store(true, std::memory_order_release);
}

// Build the slot pool and advertise it to the peer. MUST be called from a context that
// does not hold worker_mutex_: send_rma_setup() takes it, and the allocation itself
// would stall ucp_worker_progress for ~150 ms if it ran in the AM callback.
void UcxCommunicator::materialise_rma_pool() {
    GVS_RTRACE("materialise_rma_pool: ENTRADA");
    struct GvsExitTrace { ~GvsExitTrace() { GVS_RTRACE("materialise_rma_pool: SALIDA"); } } gvs_et;
    // Con una captura abierta, construir el pool haria cudaHostAlloc + cudaMalloc + los frees
    // del retiro: cuatro maneras distintas de invalidarla. Se sale SIN consumir la peticion,
    // de modo que el primer TryAcquireFrame posterior al EndCapture lo construye igual. Hasta
    // entonces el transporte sigue por mensajes activos: degradacion, no fallo.
    if (captura_abierta()) {
        GVS_RTRACE("materialise_rma_pool: aplazado (captura abierta)");
        return;
    }
    // Re-entrant by design: the pool is built the first time a large message arrives
    // and REBUILT, larger, if a later message turns out not to fit. The build and the
    // advertisement must be atomic with respect to a second grow request, or two
    // threads could interleave and publish a layout that does not match the pool.
    std::lock_guard<std::mutex> lk(rma_build_mu_);

    const size_t want = rma_slot_cap_for(
        rma_pool_hint_bytes_.load(std::memory_order_acquire));
    // A rebuild is also needed when the shadow requirement changed, not only the size.
    const bool shadow_gap = rma_want_gpu_shadow_.load(std::memory_order_acquire) &&
                            g_gpudirect_enabled.load() &&
                            !rma_pool_has_shadow_.load(std::memory_order_acquire);
    if (!shadow_gap &&
        rma_pool_ready_.load(std::memory_order_acquire) &&
        rma_pool_cap_.load(std::memory_order_acquire) >= want) {
        // Another thread already grew it to at least what we need.
        rma_pool_requested_.store(false, std::memory_order_release);
        return;
    }

    init_rx_pool();
    send_rma_setup();  // bumps the epoch; only live (non-retired) slots are published
    rma_pool_ready_.store(true, std::memory_order_release);
    rma_pool_requested_.store(false, std::memory_order_release);
}

void UcxCommunicator::destroy_rx_pool() {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    for (auto &slot : rx_pool_->slots) {
        unmap_slot_from_ucp(context_, slot);
        free_pinned_host(slot.addr, slot.is_cuda_host);
        free_gpu_slot(slot.gpu_addr);  // no-op if nullptr
        slot.gpu_addr = nullptr;
        slot.gpu_capacity = 0;
    }
    rx_pool_->slots.clear();
    // Invalida cualquier mejora diferida que estuviera a medio camino con un indice de este
    // pool. Sin esto la tarea vuelve, indexa un deque vacio y corrompe el heap.
    ++rx_pool_->generacion;
}

// Find a free slot of at least `needed` bytes. Grows an existing in-use-free
// slot's capacity if the largest is too small, or appends a new slot if all
// are busy. Returns slot index.
size_t UcxCommunicator::acquire_rx_slot(size_t needed) {
    // Este cronometro se queda puesto. Es el sitio exacto donde una llamada CUDA que sincroniza
    // el dispositivo congelaba a un cliente detras del kernel de otro; que vuelva a pasar tiene
    // que salir por un contador y no por una campana de dos dias. Cuesta dos relojes por
    // mensaje, en un camino que ya hace un memcpy de kilobytes.
    const auto t_ini = std::chrono::steady_clock::now();
    struct Cronometro {
        std::chrono::steady_clock::time_point t0;
        ~Cronometro() {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            if (ms < 100.0) return;
            ++g_slot_atascos;
            double peor = g_slot_peor_ms.load(std::memory_order_relaxed);
            while (ms > peor && !g_slot_peor_ms.compare_exchange_weak(peor, ms)) {}
            std::fprintf(stderr,
                "[GVS SLOT] AVISO: acquire_rx_slot tardo %.1f ms dentro del callback de UCX. "
                "Eso congela el bucle de recepcion de ESTA conexion en pleno trafico normal; "
                "busca una llamada CUDA en este camino (ver el bloque EL LIBERADOR).\n", ms);
            std::fflush(stderr);
        }
    } cronometro{t_ini};
    std::lock_guard<std::mutex> lk(rx_pool_->mu);

    // Try to find a free slot big enough. Persistent RMA slots are excluded: a peer
    // can be mid-put into one without the server knowing, so handing it to an incoming
    // eager AM would overwrite the transfer.
    for (size_t i = 0; i < rx_pool_->slots.size(); ++i) {
        if (rx_pool_->slots[i].rma_persistent) continue;
        if (!rx_pool_->slots[i].in_use && rx_pool_->slots[i].capacity >= needed) {
            rx_pool_->slots[i].in_use = true;
            return i;
        }
    }
    // GPUDirect (Step B1): mirror host grow with a GPU shadow grow if active.
    const bool gpudirect_active = g_gpudirect_enabled.load();

    // No free slot big enough — grow the first free one (or append if none free).
    //
    // CON UNA CAPTURA ABIERTA NO SE RECICLA: crecer un slot lo libera primero, y si ese slot
    // vino de cudaHostAlloc o lleva sombra de GPU, ese free invalida la captura. Este era el
    // fallo real -- reproducido: el primer mensaje de un tamano nuevo dispara el crecimiento,
    // y por eso una copia de calentamiento del MISMO tamano fuera de la ventana hacia pasar
    // todos los tamanos (tests/semantic/graphprobe5.cu). Se anade un slot nuevo en vez de
    // reciclar: solo reserva (paginable dentro de la ventana) y mapea host. Los slots de mas
    // se reutilizan por capacidad en cuanto la ventana se cierra; no se pierde ninguno.
    const bool no_reciclar = captura_abierta();
    // Aqui empieza el camino que corria dentro del callback de UCX con cuatro llamadas CUDA que
    // sincronizan el dispositivo. Ver el bloque "EL LIBERADOR" arriba: la reserva pasa a ser
    // paginable y todo lo lento se va a un hilo de fondo. `en_linea` restaura lo viejo para
    // poder medir la ablacion en las dos direcciones con el mismo binario.
    const bool en_linea = gvs_slot_inline_cuda();
    for (size_t i = 0; !no_reciclar && i < rx_pool_->slots.size(); ++i) {
        if (rx_pool_->slots[i].rma_persistent) continue;  // never repurpose an RMA slot
        if (!rx_pool_->slots[i].in_use) {
            if (en_linea) {
                unmap_slot_from_ucp(context_, rx_pool_->slots[i]);
                free_pinned_host(rx_pool_->slots[i].addr, rx_pool_->slots[i].is_cuda_host);
                free_gpu_slot(rx_pool_->slots[i].gpu_addr);
                bool is_cuda = false;
                unsigned char *p = alloc_pinned_host(needed, is_cuda);
                if (p == nullptr) {
                    throw std::runtime_error("UcxCommunicator: rx_pool grow failed");
                }
                rx_pool_->slots[i] = PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr};
                if (gpudirect_active) {
                    unsigned char *gp = alloc_gpu_slot(needed);
                    if (gp != nullptr) {
                        rx_pool_->slots[i].gpu_addr = gp;
                        rx_pool_->slots[i].gpu_capacity = needed;
                    }
                }
                map_slot_to_ucp(context_, rx_pool_->slots[i]);
                ucx_debug_log("rx_pool: grew slot %zu to %zu bytes (gpu=%s)",
                              i, needed, rx_pool_->slots[i].gpu_addr ? "yes" : "no");
                return i;
            }
            // Lo viejo sale del slot SIN tocarse: lo desmapea y lo libera el liberador.
            PinnedSlot viejo = rx_pool_->slots[i];
            unsigned char *p = alloc_host_paginable(needed);
            if (p == nullptr) {
                throw std::runtime_error("UcxCommunicator: rx_pool grow failed");
            }
            rx_pool_->slots[i] = PinnedSlot{p, needed, /*in_use*/true, /*is_cuda*/false, nullptr};
            map_slot_to_ucp(context_, rx_pool_->slots[i]);   // host: ibv_reg_mr, no toca CUDA
            encola_suelta_de_slot(context_, viejo.memh, viejo.gpu_memh,
                                  viejo.addr, viejo.is_cuda_host, viejo.gpu_addr);
            pide_mejora_slot(i, needed, gpudirect_active);
            ucx_debug_log("rx_pool: grew slot %zu to %zu bytes (paginable; mejora encolada)",
                          i, needed);
            return i;
        }
    }
    // All slots in use — append a new one.
    if (en_linea) {
        bool is_cuda = false;
        unsigned char *p = alloc_pinned_host(needed, is_cuda);
        if (p == nullptr) {
            throw std::runtime_error("UcxCommunicator: rx_pool append failed");
        }
        rx_pool_->slots.push_back(PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr});
        size_t idx = rx_pool_->slots.size() - 1;
        if (gpudirect_active) {
            unsigned char *gp = alloc_gpu_slot(needed);
            if (gp != nullptr) {
                rx_pool_->slots[idx].gpu_addr = gp;
                rx_pool_->slots[idx].gpu_capacity = needed;
            }
        }
        map_slot_to_ucp(context_, rx_pool_->slots[idx]);
        ucx_debug_log("rx_pool: appended slot %zu (%zu bytes, gpu=%s), total=%zu",
                      idx, needed, rx_pool_->slots[idx].gpu_addr ? "yes" : "no",
                      rx_pool_->slots.size());
        return idx;
    }
    unsigned char *p = alloc_host_paginable(needed);
    if (p == nullptr) {
        throw std::runtime_error("UcxCommunicator: rx_pool append failed");
    }
    rx_pool_->slots.push_back(PinnedSlot{p, needed, /*in_use*/true, /*is_cuda*/false, nullptr});
    const size_t idx = rx_pool_->slots.size() - 1;
    map_slot_to_ucp(context_, rx_pool_->slots[idx]);
    pide_mejora_slot(idx, needed, gpudirect_active);
    ucx_debug_log("rx_pool: appended slot %zu (%zu bytes, paginable; mejora encolada), total=%zu",
                  idx, needed, rx_pool_->slots.size());
    return idx;
}

// Mejora diferida de un slot recien crecido: paginable -> fijada de CUDA (+ sombra de GPU si
// GPUDirect esta activo). Corre en el liberador, nunca en el hilo de una conexion, que es todo
// el asunto: cudaHostAlloc y cudaMalloc pueden tardar lo que dure el kernel mas largo de
// CUALQUIER cliente, y aqui eso no bloquea ninguna RPC.
//
// Lifetime: el pool es shared_ptr y se captura un weak_ptr, asi que una conexion que se cierre
// mientras la tarea espera deja la tarea sin nada que hacer en vez de escribir en memoria
// liberada. Los indices siguen siendo validos porque RxPool::slots es un deque: crecerlo no
// invalida elementos existentes.
void UcxCommunicator::pide_mejora_slot(size_t idx, size_t needed, bool con_gpu) {
    // Valvula: con GVS_SLOT_UPGRADE=0 los slots crecidos se quedan PAGINABLES para siempre.
    // Es correcto -- solo cuesta que un cudaMemcpy H2D desde ese slot lo escenifique CUDA en vez
    // de ir directo -- y no necesita recompilar. Esta aqui porque esta mejora ya tumbo el
    // backend una vez (indice a un pool destruido), y un mecanismo de rendimiento que puede
    // matar al servidor tiene que poder apagarse en la linea de lanzamiento.
    static const bool mejora_activa = [] {
        const char *e = std::getenv("GVS_SLOT_UPGRADE");
        const bool off = e != nullptr && e[0] == '0';
        if (off)
            std::fprintf(stderr, "[GVS SLOT] mejora a memoria fijada DESACTIVADA: los slots que "
                                 "crezcan se quedan paginables\n");
        return !off;
    }();
    if (!mejora_activa) return;

    std::weak_ptr<RxPool> debil = rx_pool_;
    ucp_context_h ctx = context_;
    const std::uint64_t gen = rx_pool_->generacion;   // se llama con rx_pool_->mu tomado
    // ~10 s de reintentos a 50 ms. Un slot ocupado sin parar durante 10 s enteros es trafico
    // continuo sobre esa conexion, y ahi cambiarle el buffer debajo no es lo que hay que hacer:
    // se abandona y el contador lo dice.
    auto intentos = std::make_shared<int>(200);
    // Direccion del buffer paginable que se reserva. Se instala el fijado SOLO encima de ese
    // buffer exacto: si en la ventana sin cerrojo el slot cambio de manos, la tarea se vuelve
    // un no-op en vez de escribir sobre un pool que ya no es el suyo.
    auto esperado = std::make_shared<unsigned char *>(nullptr);
    encola_tarea([debil, ctx, idx, needed, con_gpu, intentos, esperado, gen]() -> bool {
        auto pool = debil.lock();
        if (!pool) return true;                  // la conexion se fue: nada que mejorar
        // Reservar el slot: in_use lo saca de la circulacion sin que nadie mas lo toque.
        {
            std::lock_guard<std::mutex> lk(pool->mu);
            if (pool->generacion != gen || idx >= pool->slots.size()) {
                ++g_slot_mejoras_saltadas;
                return true;
            }
            PinnedSlot &s = pool->slots[idx];
            // Que el slot siga siendo EL MISMO que se encolo. Si entre medias volvio a crecer
            // (capacity distinta) o ya se mejoro (is_cuda_host), instalar este buffer seria
            // encogerlo o duplicar trabajo. Se descarta la tarea, no el slot.
            if (s.rma_persistent || s.is_cuda_host || s.capacity != needed) {
                ++g_slot_mejoras_saltadas;
                return true;
            }
            // EN USO no es un descarte, es un "ahora no": acaba de entregarse al mensaje que
            // provoco el crecimiento y se liberara en cuanto el handler termine.
            if (s.in_use) {
                if (--(*intentos) <= 0) { ++g_slot_mejoras_saltadas; return true; }
                return false;                    // reintento en el proximo tick
            }
            s.in_use = true;
            *esperado = s.addr;
        }
        // A partir de aqui el cerrojo esta SUELTO y el slot idx queda reservado. Todo camino de
        // salida tiene que devolverlo, y ninguno puede indexar el pool sin revalidar: la
        // conexion puede cerrarse en esta ventana, y destroy_rx_pool vacia el deque.
        // Devuelve true si el slot seguia siendo el nuestro y se pudo desreservar.
        auto desreserva = [&pool, idx, gen, esperado]() -> bool {
            std::lock_guard<std::mutex> lk(pool->mu);
            if (pool->generacion != gen || idx >= pool->slots.size()) return false;
            PinnedSlot &s = pool->slots[idx];
            if (s.addr != *esperado) return false;
            s.in_use = false;
            return true;
        };

        bool es_cuda = false;
        unsigned char *p = alloc_pinned_host(needed, es_cuda);   // lento a proposito, aqui no duele
        unsigned char *gp = (p != nullptr && es_cuda && con_gpu) ? alloc_gpu_slot(needed) : nullptr;
        auto suelta_lo_reservado = [&p, &gp, es_cuda] {
            if (p != nullptr) {
                auto libera = g_cuda_free_host.load();
                if (es_cuda && libera != nullptr) libera(p); else std::free(p);
            }
            if (gp != nullptr) free_gpu_slot(gp);
        };
        if (p == nullptr || !es_cuda) {
            // Sin memoria fijada disponible (o CUDA no cargado): el slot se queda paginable, que
            // es correcto y solo un poco mas lento. Se devuelve a la circulacion.
            suelta_lo_reservado();
            desreserva();
            ++g_slot_mejoras_fallo;
            return true;
        }
        // El contexto UCX pudo morir mientras se reservaba: sin el no se puede mapear.
        if (!contexto_vivo(ctx)) {
            suelta_lo_reservado();
            desreserva();
            ++g_slot_mejoras_saltadas;
            return true;
        }
        // Mapear fuera del cerrojo del pool: ucp_mem_map no necesita el pool y puede tardar.
        PinnedSlot nuevo{p, needed, /*in_use*/true, /*is_cuda*/true, nullptr};
        nuevo.gpu_addr     = gp;
        nuevo.gpu_capacity = (gp != nullptr) ? needed : 0;
        map_slot_to_ucp(ctx, nuevo);

        PinnedSlot viejo;
        {
            std::lock_guard<std::mutex> lk(pool->mu);
            if (pool->generacion != gen || idx >= pool->slots.size() ||
                pool->slots[idx].addr != *esperado) {
                // El pool se destruyo (o el slot cambio) mientras reservabamos. Lo recien
                // reservado se suelta por el hilo de fondo y no se toca el pool.
                encola_suelta_de_slot(ctx, nuevo.memh, nuevo.gpu_memh,
                                      nuevo.addr, /*es_cuda*/true, nuevo.gpu_addr);
                ++g_slot_mejoras_saltadas;
                return true;
            }
            PinnedSlot &s = pool->slots[idx];
            viejo = s;                       // lo paginable que estaba puesto
            s.addr         = nuevo.addr;
            s.capacity     = needed;
            s.is_cuda_host = true;
            s.memh         = nuevo.memh;
            s.gpu_addr     = nuevo.gpu_addr;
            s.gpu_capacity = nuevo.gpu_capacity;
            s.gpu_memh     = nuevo.gpu_memh;
            s.in_use       = false;          // de vuelta a la circulacion, ya fijado
        }
        encola_suelta_de_slot(ctx, viejo.memh, viejo.gpu_memh,
                              viejo.addr, viejo.is_cuda_host, viejo.gpu_addr);
        ++g_slot_mejoras_ok;
        return true;
    });
}

void UcxCommunicator::release_rx_slot(size_t slot_idx) {
    bool send_ack = false;
    std::uint64_t gen = 0;
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        if (slot_idx >= rx_pool_->slots.size()) return;
        auto &s = rx_pool_->slots[slot_idx];
        s.in_use = false;
        // If this slot was filled by a client RMA put, the client is waiting
        // (or will wait) for an explicit consumption ack before reusing it.
        if (s.rma_origin) {
            send_ack = true;
            gen = s.rma_generation;
            s.rma_origin = false;
        }
    }
    // Sent OUTSIDE rx_pool_->mu: send_slot_consumed takes worker_mutex_, and no
    // release_rx_slot caller holds it, so there is no lock-order inversion.
    if (send_ack) send_slot_consumed(slot_idx, gen);
}

// Client side. The backend confirmed (SlotConsumed) that it finished consuming
// the data we RMA-put into remote slot `slot_idx`. Return the slot to Free so a
// WriteIovRma waiter can reuse it — but ONLY if the generation still matches.
// A stale/duplicate ack (ABA: the slot was already freed and re-acquired for a
// newer op) must be ignored, or it would corrupt an unrelated in-flight write.
// Marca de reentrada para el duplicado diferido de la inyeccion de fallos: evita que la
// aplicacion tardia se programe a si misma otra vez y crezca sin fin.
static thread_local bool tls_ack_diferido = false;

// F3. Integral de ocupacion. Se llama en CADA transicion de estado de slot y solo entonces,
// asi que cuesta un reloj monotono por transicion y no por operacion.
void UcxCommunicator::gusto_occupancy_tick_locked(int delta) {
    const auto now = std::chrono::steady_clock::now();
    if (rma_occ_started_) {
        const std::uint64_t dt = (std::uint64_t)
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - rma_occ_last_).count();
        rma_occupancy_ns_ += dt * (std::uint64_t)rma_inflight_now_;
        rma_occupancy_span_ns_ += dt;
    }
    rma_occ_last_ = now;
    rma_occ_started_ = true;
    if (delta > 0) {
        ++rma_inflight_now_;
        if (rma_inflight_now_ > rma_peak_inflight_) rma_peak_inflight_ = rma_inflight_now_;
    } else if (delta < 0 && rma_inflight_now_ > 0) {
        --rma_inflight_now_;
    }
}

void UcxCommunicator::release_remote_slot(size_t server_idx,
                                          std::uint64_t tag) {
    using gvirtus::communicators::ucxam::slot_tag_epoch;
    using gvirtus::communicators::ucxam::slot_tag_generation;

    // Inyeccion `delay_ack`: aplicar este ack normalmente y programar ADEMAS un duplicado
    // que se aplicara dentro de unos milisegundos, cuando el slot ya se habra liberado y
    // muy probablemente reasignado a otra transferencia. Ese duplicado tardio es la
    // condicion ABA exacta que la guarda de generacion existe para rechazar.
    if (gvirtus::communicators::faulting(gvirtus::communicators::Fault::DelayAck) &&
        !tls_ack_diferido) {
        const int ms = gvirtus::communicators::fault_delay_ms();
        std::thread([this, server_idx, tag, ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            tls_ack_diferido = true;
            this->release_remote_slot(server_idx, tag);
            tls_ack_diferido = false;
        }).detach();
    }

    std::lock_guard<std::mutex> lk(rma_state_mu_);

    // F8 hold_ack: retener el PRIMER ack y no aplicarlo. El trafico normal reasignara el slot;
    // WriteIovRma entrega el retenido en cuanto ese slot vuelve a estar InFlight con otra
    // generacion, que es exactamente la condicion ABA.
    static const bool gusto_espera_epoch = []() {
        const char *e = std::getenv("GVS_FAULT");
        return (e != nullptr && (std::strcmp(e, "epoch_ack") == 0 ||
                                 std::strcmp(e, "epoch_ack_idx") == 0));
    }();
    // `epoch_ack`     re-entrega en cuanto cambia el epoch (demuestra que la guarda CORRE).
    // `epoch_ack_idx` re-entrega solo si ADEMAS coincide el server_idx (la unica combinacion
    //                 en la que el ack viejo podria liberar un slot vivo). Si esta segunda
    //                 nunca dispara, la guarda es defensa en profundidad y no necesidad, y
    //                 hay que decirlo asi.
    static const bool gusto_espera_idx = []() {
        const char *e = std::getenv("GVS_FAULT");
        return (e != nullptr && std::strcmp(e, "epoch_ack_idx") == 0);
    }();
    static const bool gusto_hold_enabled = []() {
        const char *e = std::getenv("GVS_FAULT");
        const bool on = (e != nullptr && (std::strcmp(e, "hold_ack") == 0 ||
                                          std::strcmp(e, "epoch_ack") == 0 ||
                                          std::strcmp(e, "epoch_ack_idx") == 0));
        if (on) std::fprintf(stderr, "[GVS FAULT] *** SlotConsumed HELD until the slot is "
                                     "reassigned (deterministic ABA)\n");
        return on;
    }();
    // En que ack se arma. Para `hold_ack` (condicion ABA por reasignacion del MISMO slot) hay
    // que armar tarde: el primer ack viene de una transferencia de arranque contra la layout
    // inicial, y medido "armado slot=0 vs reasignado slot=9", 190 veces, el slot 0 nunca
    // vuelve. De ahi el default de 20.
    //
    // Para `epoch_ack` la condicion es la contraria y 20 la hace INALCANZABLE en la practica.
    // El ack retenido solo hace dano si, tras el swap, existe un slot con el MISMO server_idx
    // y la MISMA generacion -- y las generaciones reinician a 0 en cada layout, asi que el
    // unico ack que colisiona es el de generacion 1, o sea el PRIMERO de la layout estable.
    // Armar en el ack 20 guarda un tag con generacion ~3 que la guarda de generacion atajaria
    // igualmente, y entonces la celda no demuestra que la guarda de epoch haga falta.
    // GVS_FAULT_ARM lo hace elegible sin recompilar.
    static const unsigned long long gusto_arm_at = []() -> unsigned long long {
        const char *e = std::getenv("GVS_FAULT_ARM");
        const long long v = (e != nullptr && e[0] != '\0') ? std::strtoll(e, nullptr, 10) : 20;
        return v > 0 ? (unsigned long long)v : 20ull;
    }();
    ++gusto_ack_seen_;
    if (gusto_hold_enabled && !gusto_hold_spent_ && !gusto_hold_armed_ &&
        (unsigned long long)gusto_ack_seen_ >= gusto_arm_at) {
        gusto_hold_armed_ = true;
        gusto_hold_spent_ = true;
        gusto_hold_slot_  = static_cast<std::uint16_t>(server_idx);
        gusto_hold_tag_   = tag;
        gusto_hold_epoch_ = remote_epoch_;
        gusto_hold_espera_epoch_ = gusto_espera_epoch;
        gusto_hold_espera_idx_ = gusto_espera_idx;
        ++rma_ack_held_count_;
        // epoch y generacion explicitos: sin ellos no se puede saber si la celda del ablation
        // midio la guarda de epoch o la de generacion, que es la confusion que se busca evitar.
        std::fprintf(stderr, "[GVS FAULT] ack marked for REPLAY: slot=%zu tag=%llu "
                             "(epoch=%u gen=%llu, at ack #%llu)\n",
                     server_idx, (unsigned long long)tag,
                     (unsigned)slot_tag_epoch(tag),
                     (unsigned long long)slot_tag_generation(tag),
                     (unsigned long long)gusto_ack_seen_);
        std::fflush(stderr);
        // NO se hace `return`. Retener el ack dejaria el slot InFlight para siempre, con lo
        // que nunca se reasignaria y el ack retenido nunca llegaria a entregarse: la propia
        // inyeccion se bloquea (medido: released=0, reservas paradas en 7).
        // La condicion ABA de verdad es un DUPLICADO TARDIO: el ack se aplica ahora con
        // normalidad, y se guarda una copia que WriteIovRma re-entrega en cuanto ese mismo
        // slot vuelve a estar en vuelo con OTRA generacion.
    }

    const std::uint32_t ack_epoch = slot_tag_epoch(tag);
    const std::uint64_t generation = slot_tag_generation(tag);

    // Epoch guard. An ack minted against a layout we have already replaced must not
    // touch the current one: slot ids are not stable across a regrow, so matching by
    // id alone could mark a NEW slot free while the backend is still consuming it.
    // Epoch 0 means the peer predates this scheme -- accept it, matching the old
    // generation-only behaviour.
    if (ack_epoch != 0 && remote_epoch_ != 0 && ack_epoch != remote_epoch_ &&
        !ablated(Ablation::NoEpoch)) {
        ++rma_ack_dropped_epoch_count_;
        ucx_debug_log("SlotConsumed: dropping ack from epoch %u (current %u)",
                      ack_epoch, remote_epoch_);
        return;
    }

    // Slots are addressed by the SERVER's index, not our position in the vector.
    for (auto &s : remote_slots_) {
        if (s.server_idx != static_cast<std::uint16_t>(server_idx)) continue;
        // Variante `no_generation`: el slot se libera con cualquier ack, sin comprobar
        // que la generacion siga siendo la que se envio. Un ack duplicado o retrasado
        // libera entonces un slot que ya se reasigno a otra transferencia (ABA).
        const bool casa = (s.generation == generation);
        const bool gen_ok = ablated(Ablation::NoGeneration) ? true : casa;
        // La violacion se cuenta ANTES de decidir, para que el contador diga lo mismo con la
        // guarda puesta (cuantas veces hizo falta) y sin ella (cuantas veces se consumo).
        if (s.state == RemoteSlot::State::InFlight && !casa) ++rma_ack_gen_mismatch_count_;
        // Un ack para un slot ya libre es un duplicado o un tardio: el protocolo lo ignora,
        // pero hay que poder DEMOSTRAR que lo ignoro y no que el fallo no se ejercito.
        if (s.state == RemoteSlot::State::Free) ++rma_ack_on_free_count_;
        if (s.state == RemoteSlot::State::InFlight && gen_ok) {
            s.state = RemoteSlot::State::Free;
            ++rma_ack_applied_count_;
            gusto_occupancy_tick_locked(-1);
        }
        // else: stale/duplicate ack — ignore (ABA guard).
        break;
    }

    // A layout parked by handle_rma_setup_am is installed as soon as the last
    // in-flight transfer drains. Doing it here, rather than making the advertisement
    // wait, keeps the AM callback non-blocking.
    if (rma_swap_pending_) {
        bool any_inflight = false;
        for (const auto &rs : remote_slots_)
            if (rs.state == RemoteSlot::State::InFlight) { any_inflight = true; break; }
        if (!any_inflight) apply_pending_slots_locked();
    }
    rma_slot_cv_.notify_all();
}

// Server side. Notify the client that RMA-origin slot `slot_idx` (at the given
// generation) has been fully consumed and may be reused.
void UcxCommunicator::send_slot_consumed(size_t slot_idx,
                                         std::uint64_t generation) {
    GVS_TRACE("send_slot_consumed");
    // F6/F14: `slow_ack` alarga la ocupacion de CADA slot para poder saturar el pool a
    // proposito. Es la unica via a la contrapresion: en todas las cargas reales el pico fue 2
    // y nunca hubo una espera, asi que el sondeo con plazo de 30 s no se ejercitaba jamas.
    static const long gusto_slow_ack_ms = []() -> long {
        const char *f = std::getenv("GVS_FAULT");
        if (f == nullptr || std::strcmp(f, "slow_ack") != 0) return 0;
        const char *e = std::getenv("GVS_FAULT_MS");
        long v = (e && e[0]) ? std::strtol(e, nullptr, 10) : 50;
        std::fprintf(stderr, "[GVS FAULT] *** SlotConsumed DELAYED %ld ms at the server "
                             "(deliberate pool saturation)\n", v);
        std::fflush(stderr);
        return v > 0 ? v : 50;
    }();
    if (gusto_slow_ack_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(gusto_slow_ack_ms));

    // F16 readvertise: reanunciar el pool VIGENTE sin reconstruirlo. send_rma_setup() publica
    // los rx_slots actuales -- mismos server_idx, porque el indice ES la posicion en el vector
    // -- e incrementa el epoch. Ese par (indices reutilizados, epoch nuevo) es el UNICO estado
    // en el que un ack del layout viejo puede nombrar un slot vivo del nuevo, es el que se
    // observo en cuDF ([0-7] en el epoch 1 y en el 2), y NINGUNA reconstruccion lo produce:
    // al reconstruir, los indices renumeran y la colision es imposible por construccion.
    // Sin esta compuerta la demostracion de dano no se puede montar, y la campana entera
    // midio la ausencia del ESTADO creyendo medir la ausencia del DANO.
    static const long gusto_readvert_cada = []() -> long {
        const char *f = std::getenv("GVS_FAULT");
        if (f == nullptr || std::strcmp(f, "readvertise") != 0) return 0;
        const char *e = std::getenv("GVS_FAULT_EVERY");
        long v = (e != nullptr && e[0] != '\0') ? std::strtol(e, nullptr, 10) : 4;
        if (v <= 0) v = 4;
        std::fprintf(stderr,
                     "[GVS FAULT] *** RE-ADVERTISE of the current pool every %ld acks "
                     "(same server_idx, new epoch)\n", v);
        std::fflush(stderr);
        return v;
    }();
    if (gusto_readvert_cada > 0) {
        static std::atomic<long> gusto_acks_vistos{0};
        const long n = gusto_acks_vistos.fetch_add(1, std::memory_order_relaxed) + 1;
        // Fuera del worker_mutex_ que se toma mas abajo: send_rma_setup() lo toma tambien.
        if (n % gusto_readvert_cada == 0) send_rma_setup();
    }
    if (endpoint_ == nullptr || worker_ == nullptr) return;
    gvirtus::communicators::ucxam::EnvelopeHeader ack{};
    ack.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
    ack.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
    ack.message_type = static_cast<std::uint16_t>(
        gvirtus::communicators::ucxam::MessageType::SlotConsumed);
    ack.header_size = sizeof(ack);
    ack.reserved0 = static_cast<std::uint16_t>(slot_idx);
    ack.request_id = generation;

    // Inyeccion de fallos (GVS_FAULT). `generation` es el tag completo que el cliente
    // acuno, (epoch << 32) | gen, y que aqui se echoa verbatim.
    //
    // stale_ack decrementa SOLO la parte de generacion y deja el epoch intacto: asi el ack
    // sigue perteneciendo al layout vigente y lo unico obsoleto es la operacion, que es la
    // condicion ABA exacta. Si se tocara el epoch, lo atajaria la otra guarda y la celda
    // no mediria lo que dice medir.
    if (gvirtus::communicators::faulting(gvirtus::communicators::Fault::StaleAck)) {
        using gvirtus::communicators::ucxam::make_slot_tag;
        using gvirtus::communicators::ucxam::slot_tag_epoch;
        using gvirtus::communicators::ucxam::slot_tag_generation;
        const std::uint64_t g = slot_tag_generation(generation);
        if (g > 0)
            ack.request_id = make_slot_tag(slot_tag_epoch(generation), g - 1);
    }

    const int repeticiones =
        gvirtus::communicators::faulting(gvirtus::communicators::Fault::DupAck) ? 2 : 1;

    ucp_request_param_t sp{};
    sp.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    sp.datatype = ucp_dt_make_contig(1);
    const auto t_ack0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> wl(*worker_mutex_);
    for (int rep = 0; rep < repeticiones; ++rep) {
        void *req = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                    &ack, sizeof(ack), &sp);
        try {
            wait_request_completion(req, "slot_consumed_ack");
        } catch (const std::exception &e) {
            ucx_debug_log("send_slot_consumed: %s", e.what());
        }
    }
    // Se mide DENTRO del lock del worker a proposito: el coste real del ack incluye esperar
    // ese lock, porque es contencion que el propio protocolo introduce.
    rma_ack_send_ns_ += (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t_ack0).count();
    ++rma_ack_send_count_;
}

// Async H2D Phase 3 drain point. The MemcpyAsync handler issues a fire-and-
// forget D2D from a GPU shadow slot without synchronizing and sets
// tls_async_gpu_pending. Consecutive such copies overlap; but before we send a
// response-bearing reply (the frontend's flow control treats every sync reply as
// "all prior RMA slots drained"), we must block until those in-flight D2Ds have
// fully read their source slots — otherwise the frontend could reuse a remote
// slot and the NIC would peer-DMA fresh data over a shadow still being read.
// cudaDeviceSynchronize is the coarse-but-correct drain; it only runs at a sync
// point AND only when a fire-and-forget GPU copy is actually pending, so its cost
// is amortized across the whole in-flight batch (typically the ring depth).
void UcxCommunicator::drain_device_if_async_pending() {
    if (!gvirtus::communicators::tls_async_gpu_pending) return;
    // cudaDeviceSynchronize invalida una captura abierta. NO se consume la bandera: el drenaje
    // se debe seguir haciendo, solo que al cerrar la ventana. Dentro de ella la bandera no
    // deberia llegar a ponerse -- el handler bajo captura sale por el camino de staging antes
    // de marcarla -- asi que esto es la red, no el mecanismo.
    if (captura_abierta()) return;
    gvirtus::communicators::tls_async_gpu_pending = false;
    auto fn = g_cuda_device_sync.load();
    if (fn != nullptr) fn();
}

// True iff the peer advertised RMA slots and at least one has a usable rkey.
// When the peer's RmaSetup rkey failed to unpack (e.g. a native frontend whose
// UCX exposes no RMA-unpackable md), every remote slot has rkey == null and we
// cannot ucp_put into them — the backend must then NOT take the D2H GPU-scratch
// path (its device fragment would fall through to the AM path and error).
bool UcxCommunicator::rma_put_capable() const {
    // Cached at RmaSetup time (handle_rma_setup_am) so this is a single atomic
    // load — Process.cpp queries it before every RPC, so keep it O(1).
    return rma_put_capable_.load();
}

// Server-side: pack rkeys of every rx_slot, build an RmaSetup AM body, and
// send it to the connected client. Called once per accepted connection,
// right after the endpoint is created (so the client receives this before
// any data traffic).
void UcxCommunicator::send_rma_setup() {
    if (endpoint_ == nullptr || context_ == nullptr) return;

    // Snapshot rx slot metadata. With GPUDirect Step B2 each slot may also
    // expose a GPU shadow (gpu_addr / gpu_capacity / gpu_rkey).
    struct PackedSlot {
        std::uint64_t addr;
        std::uint64_t capacity;
        void *rkey_buf{nullptr};
        size_t rkey_len{0};
        // GPU shadow (optional). When gpu_rkey_buf == nullptr the slot
        // advertises host only — matches the pre-B2 wire format byte for byte.
        std::uint64_t gpu_addr{0};
        std::uint64_t gpu_capacity{0};
        bool persistent{false};
        std::uint16_t server_idx{0};
        void *gpu_rkey_buf{nullptr};
        size_t gpu_rkey_len{0};
    };
    std::vector<PackedSlot> packed;
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        packed.reserve(rx_pool_->slots.size());
        for (size_t si = 0; si < rx_pool_->slots.size(); ++si) {
            auto &slot = rx_pool_->slots[si];
            if (slot.memh == nullptr) continue;  // never mapped, or reclaimed
            // Advertise only slots the peer may actually use: the live persistent
            // pool. The message-sized slots the eager AM path appends are ours alone,
            // and a RETIRED slot must not be advertised at all -- a larger pool has
            // superseded it and it is waiting out its grace epoch before being freed.
            //
            // Skipping entries is safe now only because slot identity is explicit:
            // each descriptor carries the server's own index in the top 16 bits of
            // reserved0, and RmaPosted / SlotConsumed travel with it. Under the old
            // positional mapping this loop's `continue` on a rkey_pack failure
            // silently shifted every later slot by one.
            if (!slot.rma_persistent || slot.rma_retired) continue;
            PackedSlot ps{};
            ps.addr = reinterpret_cast<std::uint64_t>(slot.addr);
            ps.capacity = slot.capacity;
            ps.persistent = slot.rma_persistent;
            ps.server_idx = static_cast<std::uint16_t>(si);
            ucs_status_t st = ucp_rkey_pack(context_, slot.memh,
                                            &ps.rkey_buf, &ps.rkey_len);
            if (st != UCS_OK) {
                std::fprintf(stderr,
                             "UCX rma_setup: ucp_rkey_pack failed (%s)\n",
                             ucs_status_string(st));
                continue;
            }
            // Pack GPU shadow rkey if present.
            //
            // I10 / A2: la sombra solo se anuncia si este backend puede GARANTIZAR que una
            // lectura CUDA posterior vera las escrituras del NIC. Si no puede, no se publica
            // el rkey: el cliente no tiene donde hacer peer-DMA y cae al slot de host, que es
            // memoria que la CPU lee y donde A2 ni siquiera se plantea. El repliegue no
            // necesita mensaje nuevo -- reutiliza el bit has_gpu_shadow que el formato ya
            // lleva. Mismo patron que el gate de transporte de GPUDirect.
            if (slot.gpu_memh != nullptr && slot.gpu_addr != nullptr &&
                !gvirtus::communicators::sombra_gpu_permitida()) {
                static std::once_flag gvs_aviso_vis;
                std::call_once(gvs_aviso_vis, [] {
                    std::fprintf(stderr,
                        "[GVS VIS] GPU shadow NOT advertised: NIC->GPU visibility is %s. "
                        "Large transfers will use the host slot\n",
                        gvirtus::communicators::nombre(gvirtus::communicators::modo()));
                    std::fflush(stderr);
                });
            } else if (slot.gpu_memh != nullptr && slot.gpu_addr != nullptr) {
                ucs_status_t gst = ucp_rkey_pack(context_, slot.gpu_memh,
                                                 &ps.gpu_rkey_buf, &ps.gpu_rkey_len);
                if (gst == UCS_OK) {
                    ps.gpu_addr = reinterpret_cast<std::uint64_t>(slot.gpu_addr);
                    ps.gpu_capacity = slot.gpu_capacity;
                } else {
                    ucx_debug_log("rma_setup: gpu rkey_pack FAILED (%s) — advertising host only",
                                  ucs_status_string(gst));
                    ps.gpu_rkey_buf = nullptr;
                }
            }
            packed.push_back(ps);
        }
    }

    // An EMPTY advertisement is meaningful and must still be sent: the peer's Connect()
    // blocks until it arrives, so skipping it costs the full 2 s handshake timeout on
    // every connection. num_slots = 0 tells the peer "no usable slots, use the eager
    // path", which is the correct state until the pool is built on demand.
    if (packed.empty()) {
        ucx_debug_log("rma_setup: advertising an empty pool (slots are built on demand)");
    }

    // Assemble AM body: [EnvelopeHeader] [N * RmaSlotDescriptor] [N * rkey blobs]
    using gvirtus::communicators::ucxam::EnvelopeHeader;
    using gvirtus::communicators::ucxam::MessageType;
    using gvirtus::communicators::ucxam::RmaSlotDescriptor;
    using gvirtus::communicators::ucxam::kEnvelopeMagic;
    using gvirtus::communicators::ucxam::kEnvelopeVersion;

    // Wire format (Step B2 extension):
    //   [EnvelopeHeader]
    //   [N * RmaSlotDescriptor]     ← per-slot header; descriptor.reserved0
    //                                  bit 0 = "has_gpu_shadow" flag
    //   For each slot in order:
    //     [host_rkey_blob (rkey_size bytes)]
    //     If has_gpu_shadow:
    //       [u64 gpu_addr][u64 gpu_capacity][u32 gpu_rkey_size][gpu_rkey_blob]
    //
    // Old peers (pre-B2) see descriptor.reserved0=0 always → no GPU block →
    // identical to pre-B2 layout.
    constexpr std::uint32_t kHasGpuShadow = 1u << 0;
    constexpr std::uint32_t kSlotPersistent = 1u << 1;  // safe for the peer to ucp_put into

    size_t descriptors_bytes = packed.size() * sizeof(RmaSlotDescriptor);
    size_t rkeys_bytes = 0;
    size_t gpu_extension_bytes = 0;
    for (auto &p : packed) {
        rkeys_bytes += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            gpu_extension_bytes += sizeof(std::uint64_t)  // gpu_addr
                                 + sizeof(std::uint64_t)  // gpu_capacity
                                 + sizeof(std::uint32_t)  // gpu_rkey_size
                                 + p.gpu_rkey_len;
        }
    }
    size_t total_bytes = sizeof(EnvelopeHeader) + descriptors_bytes + rkeys_bytes + gpu_extension_bytes;

    std::vector<unsigned char> buf(total_bytes);
    auto *hdr = reinterpret_cast<EnvelopeHeader *>(buf.data());
    hdr->magic = kEnvelopeMagic;
    hdr->version = kEnvelopeVersion;
    hdr->message_type = static_cast<std::uint16_t>(MessageType::RmaSetup);
    hdr->header_size = sizeof(EnvelopeHeader);
    hdr->reserved0 = 0;
    // Epoch of THIS layout. Bumped on every advertisement, delivered to the client,
    // and echoed by it on every RmaPosted so an ack minted against a superseded
    // layout can be recognised and dropped instead of freeing a live slot.
    const std::uint32_t epoch =
        rma_pool_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    hdr->status_code = epoch;
    hdr->request_id = 0;
    hdr->routine_size = 0;
    hdr->payload_size = static_cast<std::uint64_t>(packed.size());

    size_t off = sizeof(EnvelopeHeader);
    for (auto &p : packed) {
        RmaSlotDescriptor d{};
        d.remote_addr = p.addr;
        d.slot_capacity = p.capacity;
        d.rkey_size = static_cast<std::uint32_t>(p.rkey_len);
        d.reserved0 = (p.gpu_rkey_buf != nullptr) ? kHasGpuShadow : 0u;
        if (p.persistent) d.reserved0 |= kSlotPersistent;
        // Top 16 bits: the server's own index for this slot (see RemoteSlot::server_idx).
        d.reserved0 |= (static_cast<std::uint32_t>(p.server_idx) << 16);
        std::memcpy(buf.data() + off, &d, sizeof(d));
        off += sizeof(d);
    }
    // Per-slot rkey blobs, interleaved with optional gpu extension.
    for (auto &p : packed) {
        std::memcpy(buf.data() + off, p.rkey_buf, p.rkey_len);
        off += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            std::memcpy(buf.data() + off, &p.gpu_addr, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::memcpy(buf.data() + off, &p.gpu_capacity, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::uint32_t gsz = static_cast<std::uint32_t>(p.gpu_rkey_len);
            std::memcpy(buf.data() + off, &gsz, sizeof(std::uint32_t));
            off += sizeof(std::uint32_t);
            std::memcpy(buf.data() + off, p.gpu_rkey_buf, p.gpu_rkey_len);
            off += p.gpu_rkey_len;
        }
    }

    // Send as a single AM.
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = ucp_dt_make_contig(1);
        void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                        buf.data(), buf.size(), &request_param);
        try {
            wait_request_completion(request, "rma_setup_send");
        } catch (const std::exception &e) {
            // A peer that reset while we were advertising slots is a disconnect, not an
            // error: there is no one left to advertise to. Report and return -- the
            // connection's own teardown releases the rkey buffers below.
            std::fprintf(stderr, "[GVS] rma_setup_send: %s -- peer gone, abandoning\n",
                         e.what());
            std::fflush(stderr);
            for (auto &pk : packed) {
                if (pk.rkey_buf != nullptr) ucp_rkey_buffer_release(pk.rkey_buf);
                if (pk.gpu_rkey_buf != nullptr) ucp_rkey_buffer_release(pk.gpu_rkey_buf);
            }
            return;
        }
    }

    // Release the packed rkey buffers (host + optional GPU).
    size_t gpu_advertised = 0;
    for (auto &p : packed) {
        if (p.rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.rkey_buf);
        }
        if (p.gpu_rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.gpu_rkey_buf);
            ++gpu_advertised;
        }
    }
    ucx_debug_log("rma_setup: advertised %zu slots (%zu rkey bytes, %zu with gpu shadow)",
                  packed.size(), rkeys_bytes, gpu_advertised);
}

// Client-side: parse an incoming RmaSetup AM body, unpack each rkey, and
// populate remote_slots_. After this returns the data path can use ucp_put.
void UcxCommunicator::handle_rma_setup_am(const void *data, size_t length) {
    GVS_TRACE("handle_rma_setup_am");
    using gvirtus::communicators::ucxam::EnvelopeHeader;
    using gvirtus::communicators::ucxam::RmaSlotDescriptor;

    if (length < sizeof(EnvelopeHeader)) {
        std::fprintf(stderr, "RmaSetup: body too short (%zu)\n", length);
        return;
    }
    EnvelopeHeader hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    const size_t num_slots = static_cast<size_t>(hdr.payload_size);
    const size_t descriptors_bytes = num_slots * sizeof(RmaSlotDescriptor);
    if (length < sizeof(hdr) + descriptors_bytes) {
        std::fprintf(stderr, "RmaSetup: descriptors truncated\n");
        return;
    }

    const auto *base = static_cast<const unsigned char *>(data);
    const auto *desc_ptr = reinterpret_cast<const RmaSlotDescriptor *>(
        base + sizeof(hdr));
    const unsigned char *rkey_cursor = base + sizeof(hdr) + descriptors_bytes;
    const unsigned char *rkey_end = base + length;

    constexpr std::uint32_t kHasGpuShadow = 1u << 0;
    constexpr std::uint32_t kSlotPersistent = 1u << 1;  // safe for the peer to ucp_put into

    std::vector<RemoteSlot> new_slots;
    new_slots.reserve(num_slots);
    size_t gpu_received = 0;
    for (size_t i = 0; i < num_slots; ++i) {
        if (rkey_cursor + desc_ptr[i].rkey_size > rkey_end) {
            std::fprintf(stderr, "RmaSetup: rkey blob %zu truncated\n", i);
            break;
        }
        ucp_rkey_h rkey = nullptr;
        ucs_status_t st = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &rkey);
        if (st != UCS_OK) {
            std::fprintf(stderr,
                         "RmaSetup: ucp_ep_rkey_unpack[%zu] failed (%s)\n",
                         i, ucs_status_string(st));
            rkey = nullptr;
        }
        rkey_cursor += desc_ptr[i].rkey_size;

        RemoteSlot rs{desc_ptr[i].remote_addr,
                      desc_ptr[i].slot_capacity, rkey};
        rs.persistent = (desc_ptr[i].reserved0 & kSlotPersistent) != 0;
        rs.server_idx = static_cast<std::uint16_t>(desc_ptr[i].reserved0 >> 16);

        // GPUDirect Step B2: parse optional GPU extension after the host
        // rkey blob if the descriptor's flag bit is set. Old peers don't
        // set this flag → no extension to read → rs.gpu_rkey stays null.
        if ((desc_ptr[i].reserved0 & kHasGpuShadow) != 0u) {
            const size_t kFixedExt = sizeof(std::uint64_t)  // gpu_addr
                                   + sizeof(std::uint64_t)  // gpu_capacity
                                   + sizeof(std::uint32_t); // gpu_rkey_size
            if (rkey_cursor + kFixedExt > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu extension header %zu truncated\n", i);
                break;
            }
            std::uint64_t gpu_addr = 0, gpu_cap = 0;
            std::uint32_t gpu_rkey_size = 0;
            std::memcpy(&gpu_addr,      rkey_cursor + 0,  sizeof(std::uint64_t));
            std::memcpy(&gpu_cap,       rkey_cursor + 8,  sizeof(std::uint64_t));
            std::memcpy(&gpu_rkey_size, rkey_cursor + 16, sizeof(std::uint32_t));
            rkey_cursor += kFixedExt;
            if (rkey_cursor + gpu_rkey_size > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu rkey blob %zu truncated\n", i);
                break;
            }
            ucp_rkey_h gpu_rkey = nullptr;
            ucs_status_t gst = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &gpu_rkey);
            if (gst == UCS_OK) {
                rs.gpu_addr     = gpu_addr;
                rs.gpu_capacity = gpu_cap;
                rs.gpu_rkey     = gpu_rkey;
                ++gpu_received;
            } else {
                std::fprintf(stderr,
                             "RmaSetup: gpu rkey unpack[%zu] failed (%s), skipping gpu path\n",
                             i, ucs_status_string(gst));
            }
            rkey_cursor += gpu_rkey_size;
        }

        new_slots.push_back(rs);
    }

    // Los server_idx anunciados deciden si un SlotConsumed de un epoch anterior puede
    // llegar a tocar un slot VIVO del epoch actual: si el indice no vuelve a anunciarse
    // nunca, el bucle de release_remote_slot no encuentra a quien aplicarselo y la guarda
    // de epoch no llega a ser necesaria. Eso hay que poder MEDIRLO, no deducirlo del codigo.
    {
        std::string idxs;
        for (const auto &rs : new_slots) {
            char b[16];
            std::snprintf(b, sizeof(b), "%s%u", idxs.empty() ? "" : ",",
                          (unsigned)rs.server_idx);
            idxs += b;
        }
        // hdr.status_code ES el epoch anunciado; `new_epoch` todavia no existe aqui.
        std::fprintf(stderr, "[GVS IDX] epoch %u advertises server_idx=[%s]\n",
                     (unsigned)hdr.status_code, idxs.c_str());
        std::fflush(stderr);
    }

    // Cache RMA-put capability once here (any slot with a usable rkey) so the
    // per-RPC rma_put_capable() query is a single atomic load. Computed from
    // new_slots before the move below.
    bool put_capable = false;
    for (const auto &rs : new_slots)
        if (rs.rkey != nullptr) { put_capable = true; break; }

    const std::uint32_t new_epoch = hdr.status_code;
    bool applied = false;
    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        // Swapping the layout while a slot is InFlight is NOT safe: that slot would
        // come back Free in the new vector although the backend is still consuming
        // it, and the very next WriteIovRma would ucp_put on top of a live transfer.
        // Park the new layout and let release_remote_slot() install it once the last
        // in-flight transfer has been acknowledged. Meanwhile WriteIovRma stops
        // handing out slots, so the drain is guaranteed to finish.
        // F16 no_park: instalar el layout NUEVO aunque haya transferencias vivas.
        // La capa de slots tiene DOS defensas y la campana solo ha ejercitado la primera: el
        // park impide que el estado peligroso llegue a existir, asi que la guarda de epoch
        // nunca tuvo nada que atajar y no se pudo medir que aporta. Esta compuerta desactiva
        // la primera a proposito -- es deliberadamente insegura -- para aislar la segunda.
        // Apagada salvo que se pida; ninguna corrida de medida la lleva.
        static const bool gusto_no_park = []() {
            const char *e = std::getenv("GVS_FAULT_NOPARK");
            const bool on = (e != nullptr && e[0] != '\0' && e[0] != '0');
            if (on)
                std::fprintf(stderr,
                             "[GVS FAULT] *** NO-PARK: the layout is installed WITH transfers "
                             "in flight (deliberately unsafe)\n");
            return on;
        }();

        bool any_inflight = false;
        size_t inflight_n = 0;
        for (const auto &rs : remote_slots_)
            if (rs.state == RemoteSlot::State::InFlight) { any_inflight = true; ++inflight_n; }

        // Unconditional: whether the quiesce path is ever reached is a property of the
        // protocol's timing, not something to be assumed. Report the in-flight count on
        // EVERY advertisement so "it never parks" can be distinguished from "it parks
        // and we never looked".
        std::fprintf(stderr,
                     "[GVS] rma_setup: epoch %u arrived, %zu/%zu slots in flight -> %s\n",
                     hdr.status_code, inflight_n, remote_slots_.size(),
                     (any_inflight && !gusto_no_park)
                         ? "PARK"
                         : (any_inflight ? "INSTALL NOW (NO-PARK forced)"
                                         : "install now"));
        std::fflush(stderr);

        if (any_inflight && !gusto_no_park) {
            // A second advertisement arriving while an earlier one is still parked
            // supersedes it; drop the older parked rkeys rather than leak them.
            if (rma_swap_pending_) {
                for (auto &rs : pending_slots_) {
                    if (rs.rkey != nullptr) ucp_rkey_destroy(rs.rkey);
                    if (rs.gpu_rkey != nullptr) ucp_rkey_destroy(rs.gpu_rkey);
                }
            }
            GVS_RTRACE("handler: instalando epoch %u (%zu slots)", new_epoch, pending_slots_.size());
            pending_slots_ = std::move(new_slots);
            pending_epoch_ = new_epoch;
            rma_swap_pending_ = true;
            // Unconditional (not ucx_debug_log): this is the branch that makes
            // re-advertisement safe under concurrency, and a test needs to be able to
            // prove it actually ran rather than assume it. Turning on full debug
            // logging to see it would perturb the very timing that reaches it.
            ++rma_swap_parked_count_;
            std::fprintf(stderr,
                         "[GVS] rma_setup: epoch %u PARKED (transfers in flight), "
                         "park #%llu\n",
                         new_epoch, (unsigned long long)rma_swap_parked_count_);
            std::fflush(stderr);
        } else {
            pending_slots_ = std::move(new_slots);
            pending_epoch_ = new_epoch;
            rma_swap_pending_ = true;
            apply_pending_slots_locked();
            GVS_RTRACE("handler: apply_pending_slots_locked VOLVIO (epoch %u)", new_epoch);
            applied = true;
        }
        rma_setup_received_.store(true);
    }
    if (applied) rma_put_capable_.store(put_capable);
    rma_setup_cv_.notify_all();
    rma_slot_cv_.notify_all();
    ucx_debug_log("rma_setup: epoch %u, %zu remote slots (%zu with gpu shadow)%s",
                  new_epoch, remote_slots_.size(), gpu_received,
                  applied ? "" : " [deferred]");
}

// Install the parked layout. Caller holds rma_state_mu_ and must have established
// that no slot is InFlight. Destroys the outgoing rkeys -- the previous code moved a
// new vector over the old one and leaked every rkey it held, which was harmless only
// because a second advertisement never happened.
// Destroy the remote keys retired by a layout swap. MUST NOT be called from an AM
// callback: that is the entire reason the retire list exists.
void UcxCommunicator::drain_retired_rkeys() {
    GVS_TRACE("drain_retired_rkeys");
    std::vector<ucp_rkey_h> doomed;
    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        if (retired_rkeys_.empty()) return;
        doomed.swap(retired_rkeys_);
    }
    // Outside the lock as well as outside the callback: ucp_rkey_destroy() may take UCX
    // locks, and holding rma_state_mu_ across it would invert the order the AM path uses.
    // DIAGNOSTIC BUILD: leak instead of destroying. If the double free disappears, the
    // fault is in this destruction; if it persists, it is elsewhere and was merely hidden
    // behind the deadlock. Bounded by epoch count, so a few keys at most.
    if (std::getenv("GVS_LEAK_RETIRED_RKEYS") != nullptr) return;
    for (ucp_rkey_h k : doomed)
        if (k != nullptr) ucp_rkey_destroy(k);
}

void UcxCommunicator::apply_pending_slots_locked() {
    GVS_TRACE("apply_pending_slots_locked");
    if (!rma_swap_pending_) return;
    // RETIRE, do not destroy. Both callers reach here from inside the AM receive callback,
    // and ucp_rkey_destroy() from there re-enters the worker lock UCX already holds.
    // drain_retired_rkeys() finishes the job outside any callback.
    for (auto &rs : remote_slots_) {
        if (rs.rkey != nullptr) retired_rkeys_.push_back(rs.rkey);
        if (rs.gpu_rkey != nullptr) retired_rkeys_.push_back(rs.gpu_rkey);
    }
    remote_slots_ = std::move(pending_slots_);
    pending_slots_.clear();
    remote_epoch_ = pending_epoch_;
    rma_swap_pending_ = false;
    next_remote_slot_idx_ = 0;
    if (rma_swap_parked_count_ > 0) {
        std::fprintf(stderr, "[GVS] rma_setup: epoch %u INSTALLED after park\n",
                     remote_epoch_);
        std::fflush(stderr);
    }

    bool put_capable = false;
    for (const auto &rs : remote_slots_)
        if (rs.rkey != nullptr) { put_capable = true; break; }
    rma_put_capable_.store(put_capable);
    ucx_debug_log("rma_setup: epoch %u installed (%zu slots)", remote_epoch_,
                  remote_slots_.size());
}

void UcxCommunicator::gusto_emit_metric(const char *tag) {
    size_t slots_total = 0;
    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        slots_total = remote_slots_.size();
    }
    const double occ_avg = rma_occupancy_span_ns_
        ? (double)rma_occupancy_ns_ / (double)rma_occupancy_span_ns_ : 0.0;
    const std::uint64_t adm_rma = gusto_admit_rma_.load();
    const std::uint64_t adm_am  = gusto_admit_am_.load();
    const std::uint64_t adm_tot = adm_rma + adm_am;
    std::fprintf(stderr,
        "GUSTO_METRIC tag=%s slots_total=%zu peak_inflight=%u avg_inflight=%.3f "
        "peak_waiters=%u reservations=%llu waited=%llu wait_us_avg=%.2f "
        "admit_rma=%llu admit_am=%llu rma_pct=%.2f rma_fellback=%llu "
        "decline_capacity=%llu decline_timeout=%llu decline_swap=%llu decline_epfail=%llu "
        "ack_sent=%llu ack_gen_mismatch=%llu ack_on_free=%llu ack_applied=%llu "
        "ack_held=%llu ack_released=%llu "
        "ack_epoch_dropped=%llu parked=%llu "
        // Salud del arreglo de la serializacion: si `slot_stalls` deja de ser 0, alguna
        // llamada CUDA volvio al camino de la RPC y hay que ir a buscarla. Se emite
        // periodicamente y no solo en el teardown, que es donde la campana ya se quedo una
        // vez sin poder leer un contador porque el arnes mataba el proceso.
        "slot_deferred=%llu slot_pinned=%llu slot_pin_failed=%llu slot_pin_skipped=%llu "
        "slot_stalls=%llu slot_worst_ms=%.1f\n",
        tag, slots_total, rma_peak_inflight_, occ_avg,
        rma_peak_waiters_,
        (unsigned long long)rma_acquire_count_,
        (unsigned long long)rma_acquire_waited_count_,
        rma_acquire_count_ ? (rma_acquire_ns_ / 1000.0 / rma_acquire_count_) : 0.0,
        (unsigned long long)adm_rma, (unsigned long long)adm_am,
        adm_tot ? 100.0 * (double)adm_rma / (double)adm_tot : 0.0,
        (unsigned long long)gusto_rma_fellback_.load(),
        (unsigned long long)gusto_decline_capacity_.load(),
        (unsigned long long)gusto_decline_timeout_.load(),
        (unsigned long long)gusto_decline_swap_.load(),
        (unsigned long long)gusto_decline_epfail_.load(),
        (unsigned long long)rma_ack_send_count_,
        (unsigned long long)rma_ack_gen_mismatch_count_,
        (unsigned long long)rma_ack_on_free_count_,
        (unsigned long long)rma_ack_applied_count_,
        (unsigned long long)rma_ack_held_count_,
        (unsigned long long)rma_ack_released_count_,
        (unsigned long long)rma_ack_dropped_epoch_count_,
        (unsigned long long)rma_swap_parked_count_,
        (unsigned long long)g_slot_diferidos.load(),
        (unsigned long long)g_slot_mejoras_ok.load(),
        (unsigned long long)g_slot_mejoras_fallo.load(),
        (unsigned long long)g_slot_mejoras_saltadas.load(),
        (unsigned long long)g_slot_atascos.load(),
        g_slot_peor_ms.load());
    std::fflush(stderr);
}

// Muestreo periodico. Se llama desde Read(), punto sin cerrojos sostenidos que cualquier
// carga real recorre constantemente.
void UcxCommunicator::gusto_metric_maybe_emit() {
    static const long ms = []() -> long {
        const char *e = std::getenv("GUSTO_RMA_METRICS_MS");
        if (e == nullptr || e[0] == '\0') return 5000;
        char *end = nullptr;
        long v = std::strtol(e, &end, 10);
        return (end != e && v >= 0) ? v : 5000;
    }();
    if (ms == 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (!gusto_metric_started_) {
        gusto_metric_started_ = true;
        gusto_metric_last_ = now;
        return;
    }
    if (std::chrono::duration_cast<std::chrono::milliseconds>(
            now - gusto_metric_last_).count() < ms)
        return;
    gusto_metric_last_ = now;
    gusto_emit_metric("periodic");
}

void UcxCommunicator::destroy_rma_state() {
    // Resumen de los guards, una sola vez y sin depuracion encendida. Un cero aqui NO
    // significa que el guard funcione: significa que nunca se ejercito, que es
    // exactamente lo que hay que poder distinguir. tests/adversarial/ lo exige.
    if (rma_acquire_count_ > 0 || rma_ack_send_count_ > 0) {
        std::fprintf(stderr,
                     "[GVS] rma cost: reservations=%llu (waited %llu) mean=%.2fus | "
                     "registration cache: hits=%llu misses=%llu (%.1f%%) | "
                     "acks sent=%llu mean=%.2fus | "
                     "acks con generacion desajustada=%llu\n",
                     (unsigned long long)rma_acquire_count_,
                     (unsigned long long)rma_acquire_waited_count_,
                     rma_acquire_count_ ? (rma_acquire_ns_ / 1000.0 / rma_acquire_count_) : 0.0,
                     (unsigned long long)g_reg_cache_hits.load(),
                     (unsigned long long)g_reg_cache_misses.load(),
                     (g_reg_cache_hits.load() + g_reg_cache_misses.load())
                         ? 100.0 * g_reg_cache_hits.load() /
                               (g_reg_cache_hits.load() + g_reg_cache_misses.load())
                         : 0.0,
                     (unsigned long long)rma_ack_send_count_,
                     rma_ack_send_count_ ? (rma_ack_send_ns_ / 1000.0 / rma_ack_send_count_) : 0.0,
                     (unsigned long long)rma_ack_gen_mismatch_count_);
        std::fflush(stderr);
    }
    {
        std::fprintf(stderr,
                     "[GVS] rma teardown: acks dropped by epoch=%llu, "
                     "advertisements parked=%llu\n",
                     (unsigned long long)rma_ack_dropped_epoch_count_,
                     (unsigned long long)rma_swap_parked_count_);
    gvirtus::communicators::informa("teardown");
    if (g_a1_asumidas.load() || g_a1_fences.load() || g_a1_flushes.load() ||
        g_a1_declinadas.load()) {
        std::fprintf(stderr,
            "[GVS VIS] A1 teardown: policy=%s assumed=%llu fences=%llu flushes=%llu "
            "declined=%llu fence_failures=%llu\n",
            gvs_a1_nombre(gvs_a1_politica()),
            (unsigned long long)g_a1_asumidas.load(),
            (unsigned long long)g_a1_fences.load(),
            (unsigned long long)g_a1_flushes.load(),
            (unsigned long long)g_a1_declinadas.load(),
            (unsigned long long)g_a1_fence_fallos.load());
        std::fflush(stderr);
    }
        std::fflush(stderr);
    }
    // F3: se emite siempre (aunque todo sea cero) porque un cero explicito distingue
    // "no se ejercito" de "no se instrumento", que es justo lo que hay que poder separar.
    gusto_emit_metric("teardown");

    // Anything a previous swap retired goes now, before the layout itself.
    drain_retired_rkeys();
    {
    std::lock_guard<std::mutex> lk(rma_state_mu_);
    for (auto &rs : remote_slots_) {
        if (rs.rkey != nullptr) retired_rkeys_.push_back(rs.rkey);
        if (rs.gpu_rkey != nullptr) retired_rkeys_.push_back(rs.gpu_rkey);
    }
    remote_slots_.clear();
    pending_slots_.clear();
    rma_swap_pending_ = false;
    rma_setup_received_.store(false);
    rma_put_capable_.store(false);
    next_remote_slot_idx_ = 0;
    }
    drain_retired_rkeys();
    // Outside the lock: release every ucp_mem_map this context still owns -- the H2D
    // source and D2H destination registrations from the shared registry, and this
    // communicator's own GPU-scratch registrations. Reaching ucp_cleanup with these
    // still mapped is what corrupted the heap at exit ("corrupted size vs. prev_size in
    // fastbins" in the frontend, "double free in tcache" in the backend), because by
    // then the application had already freed the memory they described.
    release_registrations_for_context(context_);
    {
        std::lock_guard<std::mutex> lk(gpu_get_mu_);
        gvs_mm_count("unmap:L2594", false);
        for (auto &kv : gpu_get_regs_) ucp_mem_unmap(context_, kv.second.memh);
        gpu_get_regs_.clear();
    }
}

// D2H-via-GET, server side. Register the backend's GPU scratch [gpu_addr,len)
// for remote RDMA-READ (ucp_mem_map CUDA) and pack its rkey so the client can
// ucp_get_nbx from it directly. The registration is cached per device address
// (the TLS gpu scratch is reused / grows monotonically), so steady state pays
// only a cheap ucp_rkey_pack. Passive-responder side: no active send-from-cuda
// proto is constructed here, so this works under the forced rcache-off config
// that blocks the server ucp_put-from-cuda path. Returns false on any failure;
// the caller then keeps the legacy (host-staged / put) response path.
bool UcxCommunicator::PrepareGpuGet(void *gpu_addr, size_t len,
                                    std::uint64_t &out_remote_addr,
                                    std::vector<char> &out_rkey) {
    if (context_ == nullptr || gpu_addr == nullptr || len == 0) return false;

    // Uses the same registry as the H2D source and the D2H destination, so there is one
    // invalidation path and one teardown sweep instead of three private maps that each
    // leaked. gpu_get_regs_ was the last one still never unmapped: the GPU scratch is
    // cudaFree'd and reallocated whenever it grows, and the registration for the old
    // allocation simply stayed -- the same recycled-address hazard already proven on
    // both application-buffer caches, plus the teardown leak.
    //
    // Cacheable is asserted here rather than asked of the frontend hook: this buffer is
    // the BACKEND's own scratch, its lifetime is ours, and get_tls_gpu_scratch /
    // get_d2h_get_scratch invalidate before freeing it. The frontend's hook is not
    // installed in the backend process anyway (it lives in the frontend half of the
    // cudart plugin), so asking would fail closed and cost a registration per D2H.
    std::lock_guard<std::mutex> lk(gpu_get_mu_);
    const std::uint64_t key = reinterpret_cast<std::uint64_t>(gpu_addr);
    auto it = gpu_get_regs_.find(key);
    if (it != gpu_get_regs_.end() && it->second.len < len) {
        // Same base address, larger transfer than we registered: remap. This is also
        // what covers the scratch growing (cudaFree + cudaMalloc), so no cross-thread
        // invalidation is needed -- and must not be used, see the header.
        gvs_mm_count("unmap:L2631", false);
        ucp_mem_unmap(context_, it->second.memh);
        gpu_get_regs_.erase(it);
        it = gpu_get_regs_.end();
    }
    if (it == gpu_get_regs_.end()) {
        // The scratch may be DEVICE memory (the GPUDirect path) or HOST memory (the
        // RDMA-without-GPUDirect path added alongside this). Detect rather than assume:
        // registering host memory with UCS_MEMORY_TYPE_CUDA fails, and the whole point
        // of the host variant is that it needs no peermem and no CUDA memory type, so
        // it works under the forced rcache-off configuration that blocks a
        // server-active cuda->host put.
        const bool src_is_device = is_gpu_pointer(gpu_addr);
        ucp_mem_map_params_t p{};
        p.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        p.address = gpu_addr;
        p.length = len;
        if (src_is_device) {
            p.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            p.memory_type = UCS_MEMORY_TYPE_CUDA;
        }
        ucp_mem_h m = nullptr;
        gvs_mm_count("map:L2653", true);
        ucs_status_t mst = ucp_mem_map(context_, &p, &m);
        if (mst != UCS_OK) {
            ucx_debug_log("PrepareGpuGet: ucp_mem_map(%s) failed: %s",
                          src_is_device ? "CUDA" : "HOST", ucs_status_string(mst));
            return false;
        }
        it = gpu_get_regs_.emplace(key, GpuGetReg{len, m}).first;
    }
    ucp_mem_h memh = it->second.memh;

    void *rkey_buf = nullptr;
    size_t rkey_size = 0;
    ucs_status_t st = ucp_rkey_pack(context_, memh, &rkey_buf, &rkey_size);
    if (st != UCS_OK || rkey_buf == nullptr) {
        ucx_debug_log("PrepareGpuGet: ucp_rkey_pack failed: %s", ucs_status_string(st));
        return false;
    }
    out_rkey.assign(reinterpret_cast<char *>(rkey_buf),
                    reinterpret_cast<char *>(rkey_buf) + rkey_size);
    ucp_rkey_buffer_release(rkey_buf);
    out_remote_addr = reinterpret_cast<std::uint64_t>(gpu_addr);
    return true;
}

// D2H-via-GET, client side. Unpack the server-supplied rkey against our
// endpoint and RDMA-GET `count` bytes from the server's GPU scratch straight
// into `dst_host` (the caller's pinned host buffer — UCX registers it on the
// fly; the client-side rcache works). Single-threaded per connection (same
// contract as WriteIovRma), so no worker_mutex_ needed. Returns false on error.
// Desglose por fases del GET (GVIRTUS_D2H_PHASES=1).
//
// Las trazas de perfil situan el 94,6 % del coste del D2H dentro de esta funcion: 9,3 ms
// por columna de 62,5 MiB cuando la transferencia pura son 2,7 ms a los 23 GB/s medidos
// con ib_read_bw. Esto reparte esos 9,3 ms entre registrar el destino, transferir y
// desregistrar, para saber cuanto es recuperable y cuanto es fisica.
//
// Se acumula y se imprime cada 64 llamadas: un fprintf por transferencia falsearia justo
// lo que se quiere medir.
namespace {
struct D2hPhases {
    std::atomic<long> n{0}, reg{0}, xfer{0}, dereg{0};
};
D2hPhases &d2h_phases() { static D2hPhases p; return p; }

bool d2h_phases_on() {
    static const bool on = [] {
        const char *v = std::getenv("GVIRTUS_D2H_PHASES");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

inline long d2h_us(const std::chrono::steady_clock::time_point &a,
                   const std::chrono::steady_clock::time_point &b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

void d2h_phases_report(long reg, long xfer, long dereg) {
    auto &p = d2h_phases();
    p.reg.fetch_add(reg); p.xfer.fetch_add(xfer); p.dereg.fetch_add(dereg);
    const long n = p.n.fetch_add(1) + 1;
    if (n % 64 != 0) return;
    std::fprintf(stderr,
                 "[GVS D2H FASES] n=%ld reg=%.0fus xfer=%.0fus dereg=%.0fus total=%.0fus\n",
                 n, (double)p.reg.load()/n, (double)p.xfer.load()/n,
                 (double)p.dereg.load()/n,
                 (double)(p.reg.load()+p.xfer.load()+p.dereg.load())/n);
    std::fflush(stderr);
}
}  // namespace

bool UcxCommunicator::GetFromRemoteGpu(void *dst_host, std::uint64_t remote_addr,
                                       const void *rkey_blob, size_t rkey_len,
                                       size_t count) {
    GVS_TRACE("GetFromRemoteGpu");
    if (endpoint_ == nullptr || dst_host == nullptr || rkey_blob == nullptr ||
        rkey_len == 0 || count == 0) {
        return false;
    }
    ucp_rkey_h rkey = nullptr;
    ucs_status_t st = ucp_ep_rkey_unpack(endpoint_, rkey_blob, &rkey);
    if (st != UCS_OK) {
        ucx_debug_log("GetFromRemoteGpu: ucp_ep_rkey_unpack failed: %s",
                      ucs_status_string(st));
        return false;
    }

    // Register the destination host buffer ourselves and cache the memh, keyed
    // by address (grow-remap like the server scratch). Passed to ucp_get_nbx as
    // a memh hint so UCX does NOT re-register the dst on every call/fragment —
    // the broken rcache can't cache it (rcache=y errors "Bad address"), so
    // without this the per-op registration dominates and D2H collapses at large
    // sizes (64 MB fell to ~1 GB/s). D2H reuses the same dst, so this registers
    // once and every subsequent GET is a pure line-rate RDMA READ.
    // CORRECTNESS FIX (2026-07-25): register PER CALL, do not cache by address.
    //
    // This used to keep a client_dst_regs_ cache keyed by dst_host, invalidated only
    // when a later call asked for MORE bytes at the same address. Nothing invalidated it
    // when the application FREED that buffer -- and an allocator will happily hand the
    // same address back for the next allocation. The cached ucp_mem_h then still
    // described the OLD mapping, so the RDMA GET landed on pages the new buffer no
    // longer maps and the caller silently read whatever its fresh allocation contained.
    //
    // Reproduced by examples/rmatest/dst_realloc.cu: allocate a destination, D2H into
    // it, free it, repeat. Iterations 1 and 2 get distinct addresses and pass; from
    // iteration 3 the allocator recycles one address and every transfer after it fails
    // (16320 of 16385 samples wrong). Identical for pinned (cudaHostAlloc/cudaFreeHost)
    // and pageable (malloc/free) destinations -- it is address recycling, not
    // pinned-ness.
    //
    // An address-keyed registration cache cannot be made safe without invalidation on
    // free. UCX's own rcache would do it via UCM memory hooks, but this deployment
    // forces UCX_RCACHE_ENABLE=n (the CUDA memtype workaround), so there is nothing
    // underneath us either. Registering per call is the only option that is correct for
    // an address we do not control the lifetime of. The cost is measured in the commit
    // message; if it matters, the way back to caching is an explicit invalidation hook
    // driven by cudaFreeHost -- which still would not cover pageable destinations.
    // Registering per call was the first cut of this fix and it cost 3x the D2H
    // bandwidth: simple_matrix N=4096 fell to 8.27 GB/s from ~24, because its h_C is a
    // cudaMallocHost buffer reused every iteration -- exactly the case the cache exists
    // for. Removing the cache was the wrong repair; making it invalidatable is the
    // right one, and is what the H2D source ended up doing. Same registry, same
    // invalidation on cudaFreeHost/cudaFree, same "only cache what we will be told
    // about" rule -- a pageable destination is still registered per call.
    const bool ph = d2h_phases_on();
    const auto t_reg0 = std::chrono::steady_clock::now();
    bool dst_memh_owned = false;
    ucp_mem_h dst_memh = acquire_app_registration(
        context_, dst_host, count,
        gvirtus::communicators::RegistrationCacheable(dst_host, count),
        /*is_cuda*/ false, dst_memh_owned);
    if (dst_memh == nullptr) {
        ucx_debug_log("GetFromRemoteGpu: dst registration failed "
                      "(falling back to on-the-fly reg)");
    }

    const auto t_reg1 = std::chrono::steady_clock::now();

    ucp_request_param_t param{};
    if (dst_memh != nullptr) {
        param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        param.memh = dst_memh;
    } else {
        param.op_attr_mask = 0;
    }
    const auto t_xfer0 = std::chrono::steady_clock::now();
    void *req = ucp_get_nbx(endpoint_, dst_host, count, remote_addr, rkey, &param);
    try {
        wait_request_completion(req, "d2h_get");
    } catch (const std::exception &e) {
        ucx_debug_log("GetFromRemoteGpu: %s", e.what());
        gvs_mm_count("unmap:L2756", false);
        if (dst_memh_owned) ucp_mem_unmap(context_, dst_memh);
        ucp_rkey_destroy(rkey);
        return false;
    }
    // Release the registration with the transfer it was made for. Keeping it would
    // reintroduce exactly the stale-address hazard this call was changed to avoid, and
    // it also stops the process accumulating registrations for buffers it has freed
    // (which is what made teardown corrupt the heap).
    gvs_mm_count("unmap:L2764", false);
    // ONCE. Same defect as in src_reg_invalidate: a timestamp was inserted between an
    // unmap and its duplicate, so every D2H GET that owned its destination registration
    // released it twice. This is the one cuDF hits -- to_pandas frees its destination on
    // every batch, so it aborted before finishing the first.
    const auto t_xfer1 = std::chrono::steady_clock::now();
    if (dst_memh_owned) ucp_mem_unmap(context_, dst_memh);
    const auto t_dereg1 = std::chrono::steady_clock::now();
    ucp_rkey_destroy(rkey);
    if (ph) {
        d2h_phases_report(d2h_us(t_reg0, t_reg1), d2h_us(t_xfer0, t_xfer1),
                          d2h_us(t_xfer1, t_dereg1));
    }
    return true;
}

// RMA-mode send. Two paths, selected by env var GVIRTUS_RMA_ZEROCOPY:
//
//  * "staged" (GVIRTUS_RMA_ZEROCOPY=0): copy ALL iov fragments into the
//     pre-registered local tx_scratch_, then one ucp_put_nbx of the
//     contiguous buffer. Simple, predictable. Pays a host-RAM memcpy of
//     ~2.5ms for a 64MB payload, but no per-call buffer registration.
//
//  * "zerocopy" (default, or GVIRTUS_RMA_ZEROCOPY=1): only the small
//     fragments (header, routine) are staged into tx_scratch_; the large
//     fragment (the user's payload) is ucp_put_nbx'd directly from the
//     caller's buffer. Two puts in flight in parallel. UCX rcache caches
//     the user buffer's registration after first use — steady state is
//     ~2.5ms faster per call. First call against a fresh user buffer pays
//     a one-time registration cost (typically ~10ms for 64MB).
//
// Both paths end with a tiny RmaPosted AM carrying the slot index.
size_t UcxCommunicator::WriteIovRma(const struct iovec *iov, size_t iov_count,
                                    size_t total) {
    GVS_TRACE("WriteIovRma");
    // Retire list first, on the caller's thread. A layout swap runs inside the AM callback
    // and cannot destroy anything there; this is the ordinary path that finishes the job,
    // so the deferral is exercised on every run rather than only at teardown.
    drain_retired_rkeys();

    // Acquire a remote slot with EXPLICIT ownership. The old round-robin assumed
    // a strictly synchronous request/response so the slot was already consumed
    // by the time we wrapped back to it; that invariant breaks under concurrent
    // / async-dispatched prefill (multiple RMA writes in flight), reusing a slot
    // the backend hasn't finished with -> QP error (rma_put_pre EIO crash). Now:
    // wait for a Free slot (backpressure), flip it InFlight, bump its generation.
    // It returns to Free only on the backend's SlotConsumed ack — a local UCX
    // put completion does NOT imply the remote app released the buffer.
    size_t slot_idx;
    std::uint64_t slot_gen;
    std::uint64_t slot_tag;
    RemoteSlot rs;
    bool entregar_retenido = false;      // F8 hold_ack
    std::uint16_t retenido_slot = 0;
    std::uint64_t retenido_tag = 0;
    {
        // I10 / A1, politica `strict`: sin prueba de ordenacion de transporte no se usa el
        // camino directo. Se declina ANTES de reservar nada, de modo que el repliegue es el
        // camino de mensajes activos de siempre -- que no tiene el problema, porque alli el
        // payload viaja DENTRO del mensaje que lo anuncia y no hay dos operaciones que
        // ordenar. Hoy no existe una consulta de UCX que PRUEBE que el AM y el PUT comparten
        // una lane RC ordenada, asi que `strict` declina siempre: es un modo de medir el
        // coste del repliegue y de acotar la afirmacion, no una deteccion.
        if (gvs_a1_politica() == A1Politica::Estricta) {
            g_a1_declinadas.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        std::unique_lock<std::mutex> lk(rma_state_mu_);
        if (remote_slots_.empty()) return 0;
        size_t found = static_cast<size_t>(-1);
        // Pick a free slot that actually FITS. Taking the first free slot and then
        // giving up on capacity means one undersized slot at a low index sends every
        // large transfer down the eager path for the lifetime of the connection --
        // which is exactly what happened once the pool started holding a mix of
        // full-capacity slots and the message-sized ones acquire_rx_slot() appends.
        // Slot selection. Returns true when `found` holds a usable slot, and ALSO when
        // waiting cannot help (nothing in flight, nothing big enough) so the caller
        // falls back to the eager path immediately instead of burning the timeout.
        auto try_pick = [&]() -> bool {
                if (endpoint_failed_.load()) return true;
                // A new layout is parked waiting for the in-flight transfers to
                // drain. Handing out another slot from the outgoing layout would
                // keep the drain from ever completing, so stop issuing until the
                // swap has been installed. Slots still in flight will release and
                // install it; if something goes wrong the 30 s timeout falls back
                // to the eager path rather than wedging.
                if (rma_swap_pending_) return false;
                bool any_free = false;
                // If the peer tags persistent slots, restrict puts to those. If it tags
                // none (an older peer), keep the previous behaviour rather than refusing
                // to use RMA at all.
                bool any_persistent = false;
                for (const auto &r : remote_slots_)
                    if (r.persistent) { any_persistent = true; break; }
                for (size_t i = 0; i < remote_slots_.size(); ++i) {
                    if (remote_slots_[i].state != RemoteSlot::State::Free) continue;
                    any_free = true;
                    if (remote_slots_[i].rkey != nullptr &&
                        (!any_persistent || remote_slots_[i].persistent) &&
                        remote_slots_[i].capacity >= total) {
                        found = i;
                        return true;
                    }
                }
                // Every slot is free and none is big enough: waiting cannot help, so
                // stop instead of burning the 30 s timeout before falling back.
                if (any_free) {
                    bool all_free = true;
                    for (const auto &r : remote_slots_)
                        if (r.state != RemoteSlot::State::Free) { all_free = false; break; }
                    if (all_free) return true;
                }
                return false;
        };

        // POLL, DO NOT BLOCK ON THE CONDITION VARIABLE.
        //
        // The only thing that can make this condition true is processing a SlotConsumed
        // AM, and in a single-threaded client THIS thread is the only one that ever
        // calls ucp_worker_progress. A plain rma_slot_cv_.wait_for() therefore waits
        // for a notification that nobody is left to deliver, and every such wait ran
        // the full 30 s backpressure timeout before falling back to the eager path.
        //
        // Latent since the explicit slot lifecycle went in (3ec4474): with the old
        // fixed 2 x 1025 MB pool a large transfer always fit and never had to wait, so
        // nothing exercised it. Sizing the pool to the traffic makes "no free slot big
        // enough yet" a normal state, which exposed it as a 30 s stall
        // (measured: round 0 of concgrow, 30470 ms vs 41 ms once warm).
        //
        // Unlock rma_state_mu_ before taking worker_mutex_: the AM callback runs under
        // worker_mutex_ and takes rma_state_mu_, so holding both in that order here
        // would invert the lock order.
        const auto t_acq0 = std::chrono::steady_clock::now();
        const auto deadline = t_acq0 + std::chrono::seconds(30);
        bool got = try_pick();
        const bool tuvo_que_esperar = !got;
        if (tuvo_que_esperar) {
            ++rma_waiters_now_;
            if (rma_waiters_now_ > rma_peak_waiters_) rma_peak_waiters_ = rma_waiters_now_;
        }
        while (!got) {
            if (std::chrono::steady_clock::now() >= deadline) break;
            lk.unlock();
            if (worker_ != nullptr) {
                std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
                ucp_worker_progress(worker_);
            }
            std::this_thread::yield();
            lk.lock();
            got = try_pick();
        }
        if (tuvo_que_esperar && rma_waiters_now_ > 0) --rma_waiters_now_;
        rma_acquire_ns_ += (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_acq0).count();
        ++rma_acquire_count_;
        if (tuvo_que_esperar) ++rma_acquire_waited_count_;

        if (!got || endpoint_failed_.load() || found == static_cast<size_t>(-1)) {
            // Say WHY, once. Falling off the RMA fast path for size is a 3.5x cliff
            // (measured: simple_matrix 256 MB 24.32 GB/s vs 324 MB 6.96 GB/s, the
            // staged ceiling) and it used to be announced by a warning further down.
            // Selecting only slots that FIT moved the exit to here, above that warning,
            // and made the cliff silent again. Report it where the decision is now made.
            if (!endpoint_failed_.load() && !remote_slots_.empty()) {
                size_t biggest = 0;
                for (const auto &r : remote_slots_)
                    if (r.rkey != nullptr && r.capacity > biggest) biggest = r.capacity;
                if (biggest > 0 && total > biggest) {
                    static std::atomic<bool> warned{false};
                    bool expected = false;
                    if (warned.compare_exchange_strong(expected, true)) {
                        std::fprintf(stderr,
                            "[GVS] RMA fast path declined: message %zu B exceeds the "
                            "peer's largest slot %zu B -- falling back to eager AM "
                            "(~3.5x slower for large transfers). Raise "
                            "GVIRTUS_RMA_SLOT_CAP_MB above %zu MB ON THE BACKEND (the "
                            "pool lives there; setting it on the frontend has no "
                            "effect).\n",
                            total, biggest,
                            (total + 1024u * 1024u - 1u) / (1024u * 1024u));
                        std::fflush(stderr);
                    }
                }
            }
            // F3: la CAUSA, no solo el hecho. capacity y timeout piden acciones opuestas
            // (subir el tamano de slot vs subir el numero de slots).
            if (endpoint_failed_.load())      ++gusto_decline_epfail_;
            else if (rma_swap_pending_)       ++gusto_decline_swap_;
            else if (!got)                    ++gusto_decline_timeout_;
            else                              ++gusto_decline_capacity_;
            return 0;  // backpressure timeout / endpoint failure -> IOV fallback
        }
        slot_idx = found;
        remote_slots_[slot_idx].state = RemoteSlot::State::InFlight;
        gusto_occupancy_tick_locked(+1);
        slot_gen = ++remote_slots_[slot_idx].generation;
        // F8 hold_ack: si el slot que acabamos de reasignar es el del ack retenido, la
        // condicion ABA ya existe. Se marca para entregarlo en cuanto soltemos el cerrojo.
        if (gusto_hold_armed_) {
            std::fprintf(stderr, "[GVS FAULT] replay? armed slot=%u vs reassigned slot=%u\n",
                         (unsigned)gusto_hold_slot_,
                         (unsigned)remote_slots_[slot_idx].server_idx);
            std::fflush(stderr);
        }
        // Con epoch_ack no basta con reasignar el slot: hay que esperar a que el pool haya
        // cambiado de epoch, que es la condicion que la guarda de epoch existe para atajar.
        const bool mismo_idx = (gusto_hold_slot_ == remote_slots_[slot_idx].server_idx);
        const bool gusto_listo = gusto_hold_espera_idx_
            ? (gusto_hold_epoch_ != remote_epoch_ && mismo_idx)
            : (gusto_hold_espera_epoch_ ? (gusto_hold_epoch_ != remote_epoch_)
                                        : mismo_idx);
        if (gusto_hold_armed_ && gusto_listo) {
            entregar_retenido = true;
            retenido_slot = gusto_hold_slot_;
            retenido_tag  = gusto_hold_tag_;
            gusto_hold_armed_ = false;
            ++rma_ack_released_count_;
        }
        rs = remote_slots_[slot_idx];
        // Tag every RmaPosted with the epoch of the layout it was addressed against,
        // so the returning SlotConsumed can be matched against that layout and not
        // whatever has replaced it.
        slot_tag = gvirtus::communicators::ucxam::make_slot_tag(remote_epoch_, slot_gen);
    }

    // F8 hold_ack: entrega del ack retenido AHORA que el slot esta reasignado. Se hace fuera
    // del cerrojo porque release_remote_slot lo toma. Con la guarda puesta debe rechazarlo
    // (gen_mismatch++ y el slot sigue InFlight); sin ella liberara un slot vivo.
    if (entregar_retenido) {
        std::fprintf(stderr, "[GVS FAULT] delivering held ack: slot=%u tag=%llu "
                             "(the slot has already been reassigned)\n",
                     (unsigned)retenido_slot, (unsigned long long)retenido_tag);
        std::fflush(stderr);
        release_remote_slot(retenido_slot, retenido_tag);
    }

    // RAII: if we bail before success (capacity fallback, or ANY throw during
    // the puts) return the slot to Free so it never leaks (a leak would
    // eventually wedge backpressure). Disarmed on the success path, where the
    // slot is instead freed later by the backend's SlotConsumed ack.
    bool rma_committed = false;
    struct SlotReleaser {
        UcxCommunicator *self; size_t srv_idx; std::uint64_t tag; bool *committed;
        ~SlotReleaser() { if (!*committed) self->release_remote_slot(srv_idx, tag); }
    } slot_releaser{this, rs.server_idx, slot_tag, &rma_committed};

    if (rs.rkey != nullptr && total > rs.capacity) {
        // Falling off the RMA fast path for size is a 3x-class performance cliff, and
        // it used to be completely silent. Say it once, with both numbers, so it is
        // diagnosable from a normal run instead of only from a bandwidth sweep.
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                "[GVS] RMA fast path declined: message %zu B exceeds slot capacity %zu B "
                "-- falling back to eager AM (much slower for large transfers). Raise "
                "GVIRTUS_RMA_SLOT_CAP_MB above %zu MB to keep the fast path.\n",
                total, static_cast<size_t>(rs.capacity),
                (total + 1024u * 1024u - 1u) / (1024u * 1024u));
        }
    }
    if (rs.rkey == nullptr || total > rs.capacity) {
        // Caller will fall back to the IOV path. (slot freed by SlotReleaser)
        return 0;
    }

    // Env-var gated zerocopy: default OFF (set GVIRTUS_RMA_ZEROCOPY=1 to
    // enable). The zerocopy path relies on UCX's registration cache to
    // make repeated ucp_put_nbx(h_user_buf, ...) cheap. In this container
    // build UCX logs "could not create UCP registration cache: Unsupported
    // operation" at init, which means every put re-registers the source
    // buffer (~25ms for 64MB) — regressing write from ~8ms to ~31ms.
    // The staged path uses a pre-mem_map'd tx_scratch and is rcache-
    // independent, so it stays ~8ms warm regardless. Keep zerocopy behind
    // a flag for builds where rcache works (e.g., once nvidia-peermem and
    // UCM event handling are properly set up in the container).
    static const bool zerocopy_enabled = []() {
        const char *v = std::getenv("GVIRTUS_RMA_ZEROCOPY");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();

    // GPUDirect Step B3: set when the big iov fragment is routed to the
    // peer's GPU shadow. Communicated to the peer via RmaPosted:
    //   routine_size = gpu_split_bytes (= big_size routed to GPU)
    //   status_code  = gpu_split_offset (= pre_size in host slot — the
    //                  position where the GPU data belongs in the logical
    //                  message). Allows biggest to be at ANY iov index, not
    //                  just last (Fase 5 puts user_src at index 3 with a
    //                  trailing [count][kind] = 12-byte input_post).
    std::uint64_t gpu_split_bytes  = 0;
    std::uint32_t gpu_split_offset = 0;

    // Find the biggest iov fragment regardless of position. The zerocopy
    // path treats it as the payload to ucp_put directly from caller memory
    // and stages every other fragment through tx_scratch_. With the legacy
    // 3-entry layout [header][routine][payload] the biggest sits at index
    // iov_count-1, matching the prior behavior. With the Fase 5 layout
    // [header][routine][input_pre][user_src][input_post] the biggest sits
    // at an interior index — argmax catches both.
    size_t biggest_idx = 0;
    size_t big_size = 0;
    for (size_t i = 0; i < iov_count; ++i) {
        if (iov[i].iov_len > big_size) {
            big_size = iov[i].iov_len;
            biggest_idx = i;
        }
    }
    const size_t small_size = (big_size <= total) ? (total - big_size) : 0;
    // Bytes from iov fragments that appear BEFORE the biggest one — they
    // get put to rs.addr + 0. Bytes AFTER the biggest go to
    // rs.addr + pre_size + big_size to preserve wire order.
    size_t pre_size = 0;
    for (size_t i = 0; i < biggest_idx; ++i) pre_size += iov[i].iov_len;
    const size_t post_size = small_size - pre_size;
    // Detect GPU mem in the biggest fragment ONCE here so we can both (a)
    // force zerocopy when GPU is present (staged path would memcpy from GPU
    // into tx_scratch → SIGSEGV) and (b) reuse the value for the memh
    // registration further down.
    const bool big_is_gpu = is_gpu_pointer(iov[biggest_idx].iov_base);

    // Control/data-path gate (GPUDirect B3). Consume the per-message hint set
    // by Frontend::Execute::SetNextDeviceFragment (reset it once here so it
    // never leaks into a later message). The big fragment may be peer-DMA'd
    // into the peer GPU shadow ONLY when it is exactly the Fase-5 device-
    // destined direct-input fragment (today: sync cudaMemcpy H2D — the one
    // routine whose backend handler consumes GetGpuPayload()). Control-path
    // buffers (fatbin, cuModuleLoadData, nvrtc, marshaled args) travel in
    // mpInputBuffer, carry no device fragment, and thus stay in the host slot.
    const void  *dev_frag     = next_dev_frag_addr_;
    const size_t dev_frag_len = next_dev_frag_len_;
    next_dev_frag_addr_ = nullptr;
    next_dev_frag_len_  = 0;
    const bool big_is_device_data = (dev_frag != nullptr) &&
                                    (iov[biggest_idx].iov_base == dev_frag) &&
                                    (big_size == dev_frag_len);

    // Only worth splitting when the "big" fragment is genuinely big and the
    // "small" prefix isn't empty (otherwise we'd just be issuing one put).
    // big_is_gpu overrides zerocopy_enabled: with GPU mem we have no choice,
    // the staged fallback can't memcpy device memory through CPU.
    const bool use_zerocopy = (zerocopy_enabled || big_is_gpu) &&
                              iov_count >= 2 &&
                              big_size >= (16u * 1024u) &&
                              small_size > 0;

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    if (use_zerocopy) {
        // Stage all fragments except the biggest into the registered
        // scratch, contiguously in iov order (pre first, then post).
        ensure_tx_scratch_locked(small_size);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                if (i == biggest_idx) continue;
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        ucx_debug_log("WriteIovRma(zerocopy) slot=%zu pre=%zu big=%zu post=%zu biggest_idx=%zu",
                      slot_idx, pre_size, big_size, post_size, biggest_idx);

        // Issue up to three puts non-blocking. UCX progresses them in
        // parallel; their completions are awaited together at the end.
        ucp_request_param_t p_scratch{};
        p_scratch.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        p_scratch.memh = tx_scratch_.memh;

        void *req_pre = nullptr;
        if (pre_size > 0) {
            req_pre = ucp_put_nbx(endpoint_,
                                  tx_scratch_.addr, pre_size,
                                  rs.addr, rs.rkey, &p_scratch);
        }

        void *req_post = nullptr;
        if (post_size > 0) {
            req_post = ucp_put_nbx(endpoint_,
                                   static_cast<char *>(tx_scratch_.addr) + pre_size,
                                   post_size,
                                   rs.addr + pre_size + big_size,
                                   rs.rkey, &p_scratch);
        }

        ucp_request_param_t p_big{};

        // Manual host-buffer registration cache. UCX rcache fails to
        // init in this container ("rcache failed to install UCM event
        // handler: Unsupported operation"), so we mem_map ptr->memh
        // ourselves and pass it explicitly. Saves ~25ms re-register
        // cost per 64MB put on the warm path.
        //
        // SIZE-THRESHOLD GUARD (added for OpenPose-style workloads):
        // The cache keys by virtual address. If the user frees and reallocs
        // at the same address (Caffe blobs, repeated cudaMallocHost cycles)
        // we'd return a stale memh → IB QP Local Protection error. For
        // small buffers (typical of inference frameworks) we skip the cache
        // and pay ucp_mem_map+unmap per call (~ms penalty). Large stable
        // buffers (simple_matrix-style 4 MB+) still cache for big wins.
        //
        // CONC>=8 CRASH FIX (2026-07-23): the 2 MB HOST threshold was too low.
        // llama.cpp's compute pool frees+reallocs a ~2.06 MB host buffer at a
        // fixed virtual address under 8-way concurrent prefill; being >= 2 MB it
        // was cached, and with rcache disabled (UCX_RCACHE_ENABLE=n) the cached
        // addr-keyed memh went stale on the realloc -> ib_mlx5 "Local protection
        // error (synd 0x4)" on the next RDMA_WRITE (observed: identical va+lkey
        // across 48 crash episodes, len 2162688) -> RC QP fatal -> am_send EIO
        // storm -> frontend dies (the intermittent CONC>=8 UNIQUE crash).
        // Registering such a buffer fresh per put (~1 ms/2 MB) always matches the
        // current pages, so it cannot go stale. DEVICE registrations (the GPU
        // shadow / a real device pointer) are STABLE allocations and stay cached
        // at >= 2 MB. HOST buffers are only cached when large enough that the
        // per-put re-registration would actually hurt (>= 16 MB, e.g. the
        // transfer-bench arrays, which are allocated once and do not churn).
        static constexpr size_t kCacheThreshold  = 2u  * 1024u * 1024u;  // 2 MB (device)
        // Env-tunable so the cost of NOT caching can be measured rather than assumed.
        // The comment above claims ~25 ms per 64 MB re-registration; the equivalent
        // claim on the D2H side ("1 GB/s without a memh") turned out to be about a
        // different thing entirely, so this one gets measured before it is trusted.
        // Set GVIRTUS_RMA_SRC_HOST_CACHE_MIN_MB=0 to disable host source caching.
        // Default is the RMA floor: every buffer that takes this path is worth caching.
        //
        // It used to be 16 MB, and that was NOT a performance choice -- it was damage
        // control for an address-keyed cache that could not be invalidated (commit
        // 360d473, after a stale memh caused IB local-protection errors). Registering
        // 4-8 MB sources afresh on every put was the price of not corrupting them.
        //
        // That hazard is gone: the cache is now invalidated from cudaFreeHost/cudaFree
        // and only holds buffers whose free we are told about. Keeping the threshold
        // would just be paying the old price for a bug that no longer exists, and it
        // is expensive -- measured on simple_matrix, per-put registration is what kept
        // H2D at 8 GB/s below 16 MB while D2H, whose destination registration was never
        // size-gated, reached 21 GB/s at the same 4 MB:
        //
        //     4 MB payload:  7.99 -> 21.88 GB/s   (write 1.050 -> 0.383 ms)
        //     8 MB payload:  8.36 -> 22.96 GB/s   (write 2.006 -> 0.730 ms)
        //
        // That asymmetry between the two directions was this threshold, nothing else.
        static const size_t kHostCacheMin = []() -> size_t {
            const size_t dflt = ucx_rma_min_bytes();
            const char *e = std::getenv("GVIRTUS_RMA_SRC_HOST_CACHE_MIN_MB");
            if (e == nullptr || e[0] == '\0') return dflt;
            char *end = nullptr;
            unsigned long long mb = std::strtoull(e, &end, 10);
            if (end == e) return dflt;
            // 0 means "never cache a host source".
            return (mb == 0) ? std::numeric_limits<size_t>::max()
                             : static_cast<size_t>(mb) * 1024u * 1024u;
        }();
        const void *user_addr_probe = iov[biggest_idx].iov_base;
        // Size threshold AND lifetime: only cache a registration for a buffer whose
        // free the frontend will actually report to us. A plain malloc'd host source
        // fails this and is registered per call -- glibc mmaps large allocations at
        // repeatable addresses, so caching one would reintroduce exactly the stale
        // handle that corrupted the D2H destination path.
        const bool size_ok = big_is_gpu ? (big_size >= kCacheThreshold)
                                        : (big_size >= kHostCacheMin);
        const bool use_memh_cache =
            size_ok && gvirtus::communicators::RegistrationCacheable(user_addr_probe,
                                                                    big_size);

        const void *user_addr = iov[biggest_idx].iov_base;
        ucp_mem_h user_memh = nullptr;
        bool memh_owned = false;  // true iff we own this memh and must unmap after the put

        // big_is_gpu was already computed at WriteIovRma entry (used to force
        // zerocopy when GPU mem is present). Reused here for the mem_map hint
        // — when rcache + memtype-cache are disabled (this container), UCX
        // won't auto-detect CUDA memory and we MUST pass UCS_MEMORY_TYPE_CUDA
        // explicitly.

        auto fill_mp = [&](ucp_mem_map_params_t &mp) {
            mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
            mp.address = const_cast<void *>(user_addr);
            mp.length  = big_size;
            if (big_is_gpu) {
                mp.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
                mp.memory_type  = UCS_MEMORY_TYPE_CUDA;
            }
        };

        if (use_memh_cache) {
            // Keyed by (context, address) -- see the registry definition. Connections
            // are threads in one backend process, each with its own context, so an
            // address-only key can hand this connection another one's handle.
            const SrcRegKey key{context_, user_addr};
            std::lock_guard<std::mutex> lk(g_src_reg_mu);
            auto cit = g_src_regs.find(key);
            // A cached entry that is too small for this transfer describes the wrong
            // extent: drop and re-register rather than put past its end.
            if (cit != g_src_regs.end() && cit->second.len < big_size) {
                gvs_mm_count("unmap:L3175", false);
                ucp_mem_unmap(context_, cit->second.memh);
                g_src_regs.erase(cit);
                cit = g_src_regs.end();
            }
            if (cit != g_src_regs.end()) {
                user_memh = cit->second.memh;
            } else {
                ucp_mem_map_params_t mp{};
                fill_mp(mp);

                gvs_mm_count("map:L3185", true);
                if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                    g_src_regs.emplace(key, SrcReg{user_memh, big_size});
                } else {
                    user_memh = nullptr;
                }
            }
        } else {
            // Small buffer path: register fresh each call, unmap after wait.
            ucp_mem_map_params_t mp{};
            fill_mp(mp);

            gvs_mm_count("map:L3196", true);
            if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                memh_owned = true;
            } else {
                user_memh = nullptr;
            }
        }

        if (user_memh != nullptr) {
            p_big.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
            p_big.memh = user_memh;
        } else {
            p_big.op_attr_mask = 0;
        }

        // GPUDirect Step B3: route the big fragment to the peer's GPU shadow
        // when available. Triggers NIC peer-DMA into remote GPU memory via
        // peermem. The biggest fragment can sit at ANY iov index (Fase 5
        // puts user_src at idx 3 with a 12-byte [count][kind] post). We
        // pass the GPU offset (= pre_size) via RmaPosted.status_code so the
        // receiver knows where to fold the GPU bytes back into the host slot.
        //
        // 4 MB threshold (raised from 64 KB after the B4 sweep showed N=256
        // and N=512 regressed): below this size the 3-put orchestration +
        // peer-DMA setup overhead exceeds the savings from skipping the
        // host bounce. 4 MB matches simple_matrix N=1024 (4 MB), the
        // smallest payload where GPUDirect demonstrably wins.
        //
        // Transport gate: ucp_ep_rkey_unpack(gpu_rkey) returns UCS_OK even
        // when the negotiated transport is TCP (the unpack just parses the
        // blob; transport check happens at put time). If we then attempt
        // ucp_put_nbx to GPU memory over a TCP endpoint, UCX errors out or
        // hangs, killing the connection. Defensive check at WriteIovRma init
        // time: query THIS endpoint's negotiated lanes via ucp_ep_query
        // (lazy + cached). Supersedes the previous process-wide UCX_TLS env
        // probe, so a single backend with UCX_TLS=rc_mlx5,ud_mlx5,tcp,self
        // serves mixed RDMA + TCP frontends correctly — GPUDirect activates
        // only on connections that actually negotiated an RDMA lane.
        // CORRECTNESS (2026-07-23): the 4 MB lower bound was a perf heuristic, but
        // this condition is only ever true for DEVICE-source fragments
        // (big_is_device_data). Routing a device-source fragment to the HOST slot
        // (the else branch) asks UCX for a cuda->host RMA put, which cannot be
        // built under the forced rcache-off / memtype-cache-off config -> the
        // ucp_put fails and the RC QP goes fatal (observed: RDMA_WRITE len ~2 MB
        // -> "QP was flushed" -> every subsequent am_send EIO, crashing the
        // frontend under CONC>=8 real prefill). Device data therefore has NO valid
        // host-slot path and MUST go to the GPU shadow (device->device peer-DMA,
        // which works) at ANY size that fits the shadow. Keep only the capacity
        // upper bound; drop the 4 MB lower bound.
        // Lower bound is env-tunable (was a hardcoded 4 MB). Default 0 = route ALL
        // device-source data to the shadow (correctness: the host-slot device path
        // fails under concurrency). Set GVIRTUS_RMA_GPUDIRECT_MIN_BYTES=4194304 to
        // restore the old size-adaptive behavior (host slot below the bound) for
        // A/B measurement of the shadow-vs-host cost curve.
        static const size_t gpudirect_min_bytes = []() {
            const char *v = std::getenv("GVIRTUS_RMA_GPUDIRECT_MIN_BYTES");
            return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : 0u;
        }();
        const bool route_big_to_gpu = (rs.gpu_rkey != nullptr) &&
                                      (rs.gpu_addr != 0) &&
                                      big_is_device_data &&
                                      (big_size >= gpudirect_min_bytes) &&
                                      (big_size <= rs.gpu_capacity) &&
                                      current_connection_supports_cuda();
        std::uint64_t big_target_addr = route_big_to_gpu
                                        ? rs.gpu_addr
                                        : (rs.addr + pre_size);
        ucp_rkey_h    big_target_rkey = route_big_to_gpu ? rs.gpu_rkey : rs.rkey;

        // GUARD DIAGNOSTIC (2026-07-23, unconditional + flushed): a REAL device
        // pointer (big_is_gpu) that did NOT get routed to the GPU shadow is about
        // to be ucp_put into a HOST slot -> cuda->host RMA -> RC QP fatal. This
        // must never happen post-fix; if it fires, print which sub-condition
        // blocked the GPU route. (big_is_device_data alone is NOT a hazard: on
        // the frontend the H2D data-path source is host memory, and host->host-
        // slot is a normal, safe put.)
        if (big_is_gpu && !route_big_to_gpu) {
            std::fprintf(stderr,
                "GVCRASHDIAG dev->host-slot HAZARD big=%zu dev_data=%d is_gpu=%d "
                "gpu_rkey=%d gpu_addr=%d cap=%zu ge_min=%d le_cap=%d supports_cuda=%d\n",
                big_size, (int)big_is_device_data, (int)big_is_gpu,
                (int)(rs.gpu_rkey != nullptr), (int)(rs.gpu_addr != 0), rs.gpu_capacity,
                (int)(big_size >= gpudirect_min_bytes), (int)(big_size <= rs.gpu_capacity),
                (int)current_connection_supports_cuda());
            std::fflush(stderr);
        }

        if (route_big_to_gpu) {
            gpu_split_bytes  = big_size;
            gpu_split_offset = static_cast<std::uint32_t>(pre_size);
            ucx_debug_log("WriteIovRma(B3 gpu-split) slot=%zu pre=%zu big=%zu post=%zu (to gpu_addr=0x%lx)",
                          slot_idx, pre_size, big_size, post_size, big_target_addr);
        }

        void *req_big = ucp_put_nbx(endpoint_,
                                    iov[biggest_idx].iov_base, big_size,
                                    big_target_addr, big_target_rkey, &p_big);

        wait_request_completion(req_pre,  "rma_put_pre");
        wait_request_completion(req_big,  "rma_put_big");
        wait_request_completion(req_post, "rma_put_post");

        // Per-call ownership cleanup: unmap fresh registrations so the next
        // call sees a clean state. Cached memh (>= kCacheThreshold) stays
        // mapped for amortization across calls.
        if (memh_owned && user_memh != nullptr) {
            gvs_mm_count("unmap:L3301", false);
            ucp_mem_unmap(context_, user_memh);
        }
    } else {
        // Staged path: copy everything into the scratch, single put.
        // CRASH-HUNT DIAGNOSTIC: this path CPU-memcpies every fragment. A device
        // pointer here (use_zerocopy was false despite GPU memory: big_size<16KB
        // OR small_size==0 OR iov_count<2) is an imminent SIGSEGV. Flag it first.
        for (size_t i = 0; i < iov_count; ++i) {
            if (is_gpu_pointer(iov[i].iov_base)) {
                std::fprintf(stderr,
                    "GVCRASHDIAG staged-path DEVICE FRAG idx=%zu len=%zu "
                    "big_is_gpu=%d big_is_device_data=%d big_size=%zu small_size=%zu iov_count=%zu "
                    "-> SIGSEGV imminent\n",
                    i, iov[i].iov_len, (int)big_is_gpu, (int)big_is_device_data,
                    big_size, small_size, iov_count);
                std::fflush(stderr);
            }
        }
        ensure_tx_scratch_locked(total);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        ucx_debug_log("WriteIovRma(staged) slot=%zu total=%zu", slot_idx, total);

        ucp_request_param_t put_param{};
        put_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        put_param.memh = tx_scratch_.memh;
        void *put_req = ucp_put_nbx(endpoint_,
                                    tx_scratch_.addr, total,
                                    rs.addr, rs.rkey, &put_param);
        wait_request_completion(put_req, "rma_put");
    }

    // NOTA historica (2026-07-25) -- conservada porque explica por que el flush estuvo
    // desactivado tanto tiempo, y por que ese razonamiento era un non sequitur.
    //
    // ucp_put_nbx completa LOCALMENTE -- el buffer de origen se puede reutilizar, no que los
    // bytes hayan aterrizado -- y el RmaPosted de abajo no esta ordenado contra escrituras
    // RDMA en vuelo. Se probo ucp_ep_flush_nbx aqui para cerrar ese hueco y NO arreglo el
    // fallo de RMA fire-and-forget que se estaba investigando (mismas transferencias, mismos
    // bytes) mientras casi doblaba el tiempo de emision (310 ms vs 166 ms para 6 x 64 MB). De
    // ahi se concluyo que "no hacia falta", que es la falacia: no arreglar OTRO fallo no
    // convierte una obligacion de ordenacion en innecesaria, y el 1,9x era del regimen
    // fire-and-forget, no del peticion/respuesta sincrono en el que corre el sistema
    // desplegado -- donde el mismo flush se mide por debajo del 0,1 %. Hoy el flush SI se
    // aplica y es el defecto; ver el selector de abajo.

    // I10 / A1 -- PUNTO DE DESCARGA DE LA ORDENACION DE TRANSPORTE.
    //
    // Aqui es donde la obligacion se cumple, se declina o se asume explicitamente. El bloque
    // de arriba explica por que el flush no arreglaba OTRO fallo; eso no lo convierte en
    // innecesario para A1, solo en insuficiente para aquello. La diferencia entre "no hace
    // falta" y "no arreglo lo que yo miraba" es justo la que este selector hace visible.
    switch (gvs_a1_politica()) {
        case A1Politica::Fence: {
            // NO descarga I13 -- ver el bloque del enum. Se emite igualmente porque el brazo
            // existe para medir, y el aviso se da una sola vez al arrancar.
            static std::once_flag aviso;
            std::call_once(aviso, [] {
                std::fprintf(stderr,
                    "[GVS VIS] *** A1 policy `fence` DOES NOT discharge I13. UCX documents the "
                    "IB fence as ordering remote READS against remote WRITES, which is not the "
                    "obligation; and with UCX_RC_MLX5_FENCE=none (an explicit no-op) this arm "
                    "measures identically. Use `flush` for a discharged configuration.\n");
                std::fflush(stderr);
            });
            ucs_status_t st = ucp_worker_fence(worker_);
            if (st != UCS_OK) {
                g_a1_fence_fallos.fetch_add(1, std::memory_order_relaxed);
                // Si el fence no se pudo aplicar se cae al comportamiento incondicional en vez
                // de seguir como si estuviera ordenado.
                ucp_request_param_t fp{};
                void *freq = ucp_ep_flush_nbx(endpoint_, &fp);
                wait_request_completion(freq, "a1_fence_fallback_flush");
                g_a1_flushes.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_a1_fences.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        }
        case A1Politica::Flush: {
            // Completado REMOTO: cuando esto vuelve, los bytes estan en el par. Es la unica
            // de las cuatro que no supone nada sobre como se reparten las lanes, y por eso
            // es la desplegada por defecto.
            ucp_request_param_t fp{};
            void *freq = ucp_ep_flush_nbx(endpoint_, &fp);
            wait_request_completion(freq, "a1_flush");
            g_a1_flushes.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        case A1Politica::Estricta:
            // La decision estricta se toma ANTES de reservar (ver el gate de entrada): si se
            // llega aqui con la politica estricta es que A1 SI se pudo establecer.
        case A1Politica::Asumir:
        default:
            g_a1_asumidas.fetch_add(1, std::memory_order_relaxed);
            break;
    }

    // Tiny RmaPosted notification — same protocol bytes regardless of which
    // data path filled the remote slot.
    {
        gvirtus::communicators::ucxam::EnvelopeHeader notif{};
        notif.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
        notif.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
        notif.message_type = static_cast<std::uint16_t>(
            gvirtus::communicators::ucxam::MessageType::RmaPosted);
        notif.header_size = sizeof(notif);
        // The SERVER's slot id, not our vector position (see RemoteSlot::server_idx).
        notif.reserved0 = rs.server_idx;
        // GPUDirect Step B3: status_code carries the gpu_split_offset (=
        // pre_size, the position in the host slot where the GPU data folds in).
        notif.status_code = gpu_split_offset;
        // (epoch << 32) | generation, echoed verbatim in SlotConsumed so the ack can
        // be rejected if it belongs to a layout we have since replaced.
        notif.request_id = slot_tag;
        // GPUDirect Step B3: non-zero routine_size = bytes that landed in
        // slot.gpu_addr (vs slot.host_addr). The receiver uses this together
        // with status_code (offset) to build a dual PooledMsg. Zero = legacy
        // single-region path.
        notif.routine_size = gpu_split_bytes;
        notif.payload_size = static_cast<std::uint64_t>(total);

        ucp_request_param_t send_param{};
        send_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        send_param.datatype = ucp_dt_make_contig(1);
        void *send_req = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                         &notif, sizeof(notif), &send_param);
        wait_request_completion(send_req, "rma_posted_notify");
    }

    rma_committed = true;  // success: slot stays InFlight until backend SlotConsumed ack
    ucx_debug_log("WriteIovRma done slot=%zu total=%zu gen=%lu", slot_idx, total,
                  (unsigned long)slot_gen);
    return total;
}

// Grow the pre-registered TX scratch to at least `needed` bytes. Must be
// called with worker_mutex_ held. Rounds capacity up to a power of two
// (≥4MB) so consecutive WriteIovs of the same size class hit the warm path.
void UcxCommunicator::ensure_tx_scratch_locked(size_t needed) {
    if (tx_scratch_.capacity >= needed) return;

    // Free previous registration + allocation if any.
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        gvs_mm_count("unmap:L3397", false);
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;

    // Round up to power of two, minimum 4MB.
    size_t cap = 4u * 1024u * 1024u;
    while (cap < needed) cap <<= 1;

    void *addr = nullptr;
    if (posix_memalign(&addr, 4096, cap) != 0 || addr == nullptr) {
        throw std::runtime_error("UcxCommunicator: posix_memalign failed for tx scratch");
    }

    ucp_mem_map_params_t map_params{};
    map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    map_params.address = addr;
    map_params.length = cap;

    ucp_mem_h memh = nullptr;
    gvs_mm_count("map:L3422", true);
    ucs_status_t status = ucp_mem_map(context_, &map_params, &memh);
    if (status != UCS_OK) {
        std::free(addr);
        throw std::runtime_error("UcxCommunicator: ucp_mem_map failed: " +
                                 std::string(ucs_status_string(status)));
    }

    tx_scratch_.addr = addr;
    tx_scratch_.capacity = cap;
    tx_scratch_.memh = memh;
    ucx_debug_log("tx_scratch grown capacity=%zu", cap);
}

void UcxCommunicator::release_tx_scratch_locked() {
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        gvs_mm_count("unmap:L3437", false);
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }

    // Send payload as a single UCX Active Message.
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    ucx_debug_log("Write(AM) begin bytes=%zu", size);

    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0, buffer, size,
                                    &request_param);
    wait_request_completion(request, "am_send");
    ucx_debug_log("Write(AM) done bytes=%zu", size);
    return size;
}

// Two-mode gather-send. For small payloads (under kStagingThreshold) the
// fragments are passed straight to UCX via UCP_DATATYPE_IOV — eager AM
// handles short messages efficiently and the iov metadata cost is
// negligible. For large payloads the fragments are concatenated into a
// pre-registered tx_scratch_ buffer and sent as one contiguous chunk with
// the memh hint, which lets UCX bypass its internal RNDV-fragment staging
// (UCX_RNDV_FRAG_SIZE) and DMA directly from the registered memory.
size_t UcxCommunicator::WriteIov(const struct iovec *iov, size_t iov_count) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: WriteIov called without an active endpoint");
    }
    if (iov == nullptr || iov_count == 0) return 0;

    size_t total = 0;
    for (size_t i = 0; i < iov_count; ++i) total += iov[i].iov_len;

    // RMA fast path: if the server advertised its RX slot rkeys (via
    // RmaSetup at connect time) and the payload is large enough to amortise
    // the staging memcpy, push the bytes via ucp_put_nbx directly into the
    // remote slot and notify with a tiny AM. Avoids UCX's per-message
    // rendezvous handshake (which doesn't amortise in our sync pattern).
    // Floor at the measured crossover, not 64 KB: below a few MB the RMA handshake
    // costs a round trip that a small eager AM never pays. Matches the frontend
    // default so both directions agree (commit f0d8c1f).
    static const size_t kRmaMinBytes = ucx_rma_min_bytes();
    static const size_t kRmaMinBytesUnused = []() -> size_t {
        const char *e = std::getenv("GVIRTUS_RMA_MIN_BYTES");
        if (e == nullptr || e[0] == 0) return 4u * 1024u * 1024u;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(e, &end, 10);
        return (end != e) ? static_cast<size_t>(parsed) : 4u * 1024u * 1024u;
    }();

    // Placement policy. The scalar floor is one of three modes; see RmaPolicy.h for why a
    // single threshold cannot be right for both directions (they want values three orders of
    // magnitude apart) and for the measurements behind the quadrant table.
    //
    // The payload is the LARGEST iov fragment: the others are the envelope header and the
    // marshalled arguments, and including them would misclassify a small transfer whose
    // arguments happen to be bulky.
    const void *payload_ptr = nullptr;
    size_t payload_len = 0;
    for (size_t i = 0; i < iov_count; ++i) {
        if (iov[i].iov_len > payload_len) {
            payload_len = iov[i].iov_len;
            payload_ptr = iov[i].iov_base;
        }
    }
    // WriteIov IS the H2D direction: it is the client-side PUT. D2H is GetFromRemoteGpu.
    // So direction is structural here and needs no flag.
    const bool payload_pinned =
        gvirtus::communicators::HostMemoryIsPinned(payload_ptr, payload_len);
    const bool quiere_rma = gvirtus::communicators::prefer_rma(
        /*h2d=*/true, payload_pinned, total, kRmaMinBytes);

    if (quiere_rma && rma_setup_received_.load()) {
        size_t put = WriteIovRma(iov, iov_count, total);
        if (put == total) {
            ++gusto_admit_rma_;
            return put;  // RMA path completed
        }
        // else: fall through to the IOV/AM path (slot too small or no rkey)
        ++gusto_rma_fellback_;
    }
    // Todo lo que llega aqui sale por AM: lo que la politica no admitio y lo que admitio pero
    // el pool no pudo servir. La suma admit_rma+admit_am es el total de WriteIov, lo que
    // permite dar el % de operaciones que entran al camino RMA sin instrumentar al llamante.
    ++gusto_admit_am_;

    // Staging via the local TX scratch (with memh hint) regresses on this
    // UCX 1.20 + RoCE combo because it forces true-rendezvous over AM.
    // Kept disabled — see commit history for the measurement campaign.
    constexpr size_t kStagingThreshold = static_cast<size_t>(-1);

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    if (total < kStagingThreshold) {
        // Fast path: IOV directly, eager AM.
        std::vector<ucp_dt_iov_t> ucx_iov(iov_count);
        for (size_t i = 0; i < iov_count; ++i) {
            ucx_iov[i].buffer = iov[i].iov_base;
            ucx_iov[i].length = iov[i].iov_len;
        }
        ucx_debug_log("WriteIov(AM,iov) begin frags=%zu total=%zu", iov_count, total);

        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = UCP_DATATYPE_IOV;

        void *request = ucp_am_send_nbx(endpoint_, am_id_,
                                        nullptr, 0,
                                        ucx_iov.data(), iov_count,
                                        &request_param);
        wait_request_completion(request, "am_send_iov");
        ucx_debug_log("WriteIov(AM,iov) done total=%zu", total);
        return total;
    }

    // Staging path: gather into the pre-registered TX scratch and send
    // as one contiguous, memh-hinted, AM rendezvous message.
    ensure_tx_scratch_locked(total);

    {
        char *dst = static_cast<char *>(tx_scratch_.addr);
        size_t off = 0;
        for (size_t i = 0; i < iov_count; ++i) {
            std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
            off += iov[i].iov_len;
        }
    }

    ucx_debug_log("WriteIov(AM,pool) begin frags=%zu total=%zu cap=%zu",
                  iov_count, total, tx_scratch_.capacity);

    // No memh hint: that flag pushes UCX into the slow true-rendezvous
    // path (RTS/RTR + fragmented RDMA) which doesn't amortize over a
    // single 64MB sync request. Without the hint UCX still picks up the
    // ucp_mem_map'd registration via its rcache. The win we're after here
    // is the contiguous buffer (vs IOV) — same protocol, fewer iov ops.
    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request = ucp_am_send_nbx(endpoint_, am_id_,
                                    nullptr, 0,
                                    tx_scratch_.addr, total,
                                    &request_param);
    wait_request_completion(request, "am_send_pool");
    ucx_debug_log("WriteIov(AM,pool) done total=%zu", total);
    return total;
}

void UcxCommunicator::Sync() {
    if (worker_ == nullptr) {
        return;
    }

    // Flush worker to complete any in-flight sends/receives.
    ucp_request_param_t request_param{};
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    ucx_debug_log("Sync begin (worker flush)");
    void *request = ucp_worker_flush_nbx(worker_, &request_param);
    wait_request_completion(request, "worker_flush");
    ucx_debug_log("Sync done");

    progress_am_rndv();
}

void UcxCommunicator::Close() {
    // Signal shutdown and release UCX resources.
    ucx_debug_log("Close called");
    running_ = false;
    conn_cv_.notify_all();
    destroy_ucx();
}

void UcxCommunicator::run() {
    // Placeholder for compatibility with Communicator interface.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_endpoint = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_endpoint) {
        throw std::runtime_error("UcxCommunicator: endpoint type mismatch");
    }

    return std::make_shared<UcxCommunicator>(ucx_endpoint->address(), ucx_endpoint->port());
}
