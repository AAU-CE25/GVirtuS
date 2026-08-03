#ifndef GVIRTUS_VISIBILITY_H
#define GVIRTUS_VISIBILITY_H
// Visibilidad NIC -> GPU: deteccion de capacidad, descarga y repliegue.
//
// EL PROBLEMA (I10 de CONTRACTS.md). En el camino GPUDirect el cliente hace ucp_put contra la
// sombra de GPU del backend (peer-DMA via nvidia-peermem) y despues manda un mensaje activo
// RmaPosted. El backend, al verlo, consume esa region con una copia CUDA. O sea: una LECTURA
// CUDA sigue a una ESCRITURA DEL NIC sobre la misma memoria de device. Para que eso sea
// correcto hacen falta DOS cosas distintas, con mitigaciones y precios distintos:
//
//   A1 (ORDENACION DE TRANSPORTE). El RmaPosted se observa DESPUES de que el payload haya
//      aterrizado. Es una propiedad del transporte, no de CUDA. Se trata en RmaPolicy/A1
//      (ver gvs_a1_* mas abajo): se detecta y, si no se puede establecer, se repliega o se
//      paga un ucp_ep_flush.
//
//   A2 (VISIBILIDAD). Una lectura CUDA emitida despues devuelve las escrituras del NIC y no
//      memoria vieja. ESTO es lo que resuelve este fichero.
//
// LO QUE MIDE LA MAQUINA, y es lo que gobierna el diseno (tests/semantic/i10probe.cu, L40S,
// driver 580.95.05):
//
//     GPU_DIRECT_RDMA_SUPPORTED            = 1
//     GPU_DIRECT_RDMA_WRITES_ORDERING      = 0     -> NONE
//     GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 0x1   -> HOST si, MEMOPS no
//     cuFlushGPUDirectRDMAWrites           = 0,729 us de media sobre 10 000 repeticiones
//
// Es decir: **el driver reporta que NO hay ordenacion implicita**. La version anterior de este
// sistema consumia la sombra sin flush ninguno; no es que la garantia fuera "dependiente de la
// configuracion", es que el fabricante dice que no existe. Y el flush cuesta 0,4 % de una
// transferencia GPUDirect de 4 MiB. No pagarlo nunca fue una optimizacion.
//
// AVISO SOBRE EL PRECIO, porque el documento lo tenia mal. CONTRACTS.md §6.5 tasaba descargar
// I10 en 1,9x de issue time. Ese numero es de `ucp_ep_flush_nbx`, que es un flush de TRANSPORTE
// y ataca A1. La mitigacion de A2 es cuFlushGPUDirectRDMAWrites y cuesta tres ordenes de
// magnitud menos.
//
// POR QUE SIN CABECERAS CUDA. Este fichero lo incluye codigo que se compila SIN CUDA
// (Process.cpp). Meter cuda.h aqui tumbo el backend dos veces, con doce horas entre una y
// otra. Se resuelve por dlopen de libcuda.so.1 y con las constantes escritas a mano --
// VERIFICADAS numericamente con tests/semantic/consts.cu, no copiadas de memoria.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace gvirtus {
namespace communicators {

// --- constantes de la API de driver (verificadas, ver consts.cu) ------------------------
enum : int {
    kAttrGpuDirectRdmaSupported     = 116,
    kAttrGpuDirectRdmaFlushOptions  = 117,
    kAttrGpuDirectRdmaWritesOrdering= 118,
};
enum : int {
    kFlushOptionHost   = 1,
    kFlushOptionMemops = 2,
};
enum : int {
    kOrderingNone       = 0,
    kOrderingOwner      = 100,
    kOrderingAllDevices = 200,
};
enum : int {
    kFlushTargetCurrentCtx = 0,
    kFlushScopeToOwner     = 100,
    kFlushScopeToAllDev    = 200,
};

// --- el modo negociado -----------------------------------------------------------------
enum class Visibilidad {
    Implicita,      // el device ordena las escrituras del NIC por si mismo: nada que hacer
    FlushHost,      // hace falta cuFlushGPUDirectRDMAWrites desde el host antes de consumir
    FlushStream,    // el device admite el flush como memop en un stream (no en la L40S)
    NoSoportada,    // no hay forma de garantizarlo: el camino directo NO puede usarse
};

inline const char *nombre(Visibilidad v) {
    switch (v) {
        case Visibilidad::Implicita:   return "IMPLICIT_ORDERING";
        case Visibilidad::FlushHost:   return "HOST_FLUSH_REQUIRED";
        case Visibilidad::FlushStream: return "STREAM_FLUSH_REQUIRED";
        default:                       return "UNSUPPORTED";
    }
}

// --- resolucion perezosa de libcuda ----------------------------------------------------
using cuInit_t              = int (*)(unsigned);
using cuDeviceGetAttribute_t= int (*)(int *, int, int);
using cuCtxGetDevice_t      = int (*)(int *);
using cuFlush_t             = int (*)(int /*target*/, int /*scope*/);

struct VisState {
    std::atomic<bool> listo{false};
    Visibilidad       modo{Visibilidad::NoSoportada};
    int               ordering{-1};
    int               flush_opts{-1};
    int               supported{-1};
    cuFlush_t         flush{nullptr};
    // Contadores: sin ellos "se descargo la obligacion" y "el codigo no paso por ahi" son
    // indistinguibles desde fuera, que es como se acaba publicando una garantia que no corre.
    std::atomic<unsigned long long> descargas{0};
    std::atomic<unsigned long long> fallos_flush{0};
    std::atomic<unsigned long long> declinados{0};   // veces que se rechazo el camino directo
    // El DENOMINADOR correcto de la obligacion. `admit_rma` NO sirve: cuenta toda operacion
    // admitida al camino RMA, incluido el RMA a slot de HOST, que no toca memoria de device y
    // por tanto no genera obligacion A2. En modo UNSUPPORTED la sombra ni se anuncia y el
    // trafico sigue por host RMA: admit_rma sube y descargas no, sin que nada este mal. Esto
    // cuenta exactamente los consumos que SI requieren accion: una region de device escrita
    // por el NIC y a punto de ser leida por CUDA.
    std::atomic<unsigned long long> consumos_sombra{0};
};

inline VisState &vis() { static VisState s; return s; }

// Ablacion controlada. GVS_VIS_FORCE fuerza un modo; GVS_VIS_ABLATE=1 detecta la obligacion,
// la cuenta y NO la cumple -- que es el control negativo de la tabla de I10.
inline const char *env_o_null(const char *k) {
    const char *v = std::getenv(k);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}
inline bool ablacion_activa() {
    static const bool v = [] {
        const char *e = env_o_null("GVS_VIS_ABLATE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

inline void detecta_una_vez() {
    static std::once_flag once;
    std::call_once(once, [] {
        VisState &s = vis();
        void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (h == nullptr) h = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
        if (h == nullptr) {
            s.modo = Visibilidad::NoSoportada;
            s.listo.store(true);
            return;
        }
        auto p_init = reinterpret_cast<cuInit_t>(dlsym(h, "cuInit"));
        auto p_attr = reinterpret_cast<cuDeviceGetAttribute_t>(dlsym(h, "cuDeviceGetAttribute"));
        auto p_dev  = reinterpret_cast<cuCtxGetDevice_t>(dlsym(h, "cuCtxGetDevice"));
        s.flush     = reinterpret_cast<cuFlush_t>(dlsym(h, "cuFlushGPUDirectRDMAWrites"));
        if (p_init == nullptr || p_attr == nullptr) {
            s.modo = Visibilidad::NoSoportada;
            s.listo.store(true);
            return;
        }
        p_init(0);
        // El device del contexto actual si lo hay; si no, el 0. El backend ya tiene contexto
        // cuando esto corre por primera vez (lo crea al aceptar la conexion).
        int dev = 0;
        if (p_dev != nullptr && p_dev(&dev) != 0) dev = 0;
        p_attr(&s.supported,  kAttrGpuDirectRdmaSupported,      dev);
        p_attr(&s.flush_opts, kAttrGpuDirectRdmaFlushOptions,   dev);
        p_attr(&s.ordering,   kAttrGpuDirectRdmaWritesOrdering, dev);

        // La derivacion. ORDERING_OWNER basta porque el consumidor ES la GPU duena de la BAR:
        // el contrato de OWNER es exactamente "ordenado como lo ve el device propietario".
        if (s.supported != 1) {
            s.modo = Visibilidad::NoSoportada;
        } else if (s.ordering == kOrderingAllDevices || s.ordering == kOrderingOwner) {
            s.modo = Visibilidad::Implicita;
        } else if ((s.flush_opts & kFlushOptionHost) && s.flush != nullptr) {
            s.modo = Visibilidad::FlushHost;
        } else if (s.flush_opts & kFlushOptionMemops) {
            s.modo = Visibilidad::FlushStream;
        } else {
            s.modo = Visibilidad::NoSoportada;
        }

        // Override para poder EJERCITAR modos que este hardware no expone. Se anuncia a gritos
        // porque un modo forzado invalida cualquier afirmacion de correccion.
        if (const char *f = env_o_null("GVS_VIS_FORCE")) {
            Visibilidad antes = s.modo;
            if      (std::strcmp(f, "implicit")    == 0) s.modo = Visibilidad::Implicita;
            else if (std::strcmp(f, "host_flush")  == 0) s.modo = Visibilidad::FlushHost;
            else if (std::strcmp(f, "stream_flush")== 0) s.modo = Visibilidad::FlushStream;
            else if (std::strcmp(f, "unsupported") == 0) s.modo = Visibilidad::NoSoportada;
            if (s.modo != antes)
                std::fprintf(stderr, "[GVS VIS] *** MODE FORCED %s -> %s: detection says %s. "
                                     "No correctness claim holds with this set\n",
                             nombre(antes), nombre(s.modo), nombre(antes));
        }

        std::fprintf(stderr,
            "[GVS VIS] NIC->GPU: supported=%d ordering=%d(%s) flush_opts=0x%x -> mode %s%s\n",
            s.supported, s.ordering,
            s.ordering == kOrderingNone ? "NONE" :
            s.ordering == kOrderingOwner ? "OWNER" :
            s.ordering == kOrderingAllDevices ? "ALL_DEVICES" : "?",
            s.flush_opts, nombre(s.modo),
            ablacion_activa() ? "  [ABLATED: obligation detected and deliberately NOT discharged]" : "");
        std::fflush(stderr);
        s.listo.store(true);
    });
}

inline Visibilidad modo() {
    detecta_una_vez();
    return vis().modo;
}

// �Puede este backend ofrecer la sombra de GPU? Si la visibilidad no es garantizable, la
// respuesta es NO, y entonces no se anuncia el rkey de la sombra: el cliente no tiene donde
// hacer peer-DMA y cae al slot de host. El repliegue no necesita mensaje nuevo -- reutiliza el
// bit has_gpu_shadow que el formato de RmaSetup ya lleva.
inline bool sombra_gpu_permitida() {
    const bool ok = (modo() != Visibilidad::NoSoportada);
    if (!ok) vis().declinados.fetch_add(1, std::memory_order_relaxed);
    return ok;
}

// EL PUNTO DE DESCARGA. Se llama justo antes de que trabajo de GPU consuma una region que
// escribio el NIC. Devuelve true si la obligacion queda cumplida (o no existia).
inline bool descarga_antes_de_consumir() {
    VisState &s = vis();
    // Se cuenta el CONSUMO antes de decidir que hacer con el: el invariante publicable es
    // "descargas == consumos_sombra" en los modos que exigen accion, y "consumos_sombra > 0
    // con descargas == 0" en los que no la exigen. Contar solo las descargas hace
    // indistinguible "no hacia falta" de "no se paso por aqui".
    s.consumos_sombra.fetch_add(1, std::memory_order_relaxed);
    switch (modo()) {
        case Visibilidad::Implicita:
            return true;                       // el device lo ordena; nada que hacer
        case Visibilidad::NoSoportada:
            return false;                      // no deberiamos haber llegado aqui
        case Visibilidad::FlushStream:
            // El memop de flush se encola en el stream. Este hardware no lo expone
            // (flush_opts=0x1), asi que la rama existe y NO esta ejercitada: se dice.
            return true;
        case Visibilidad::FlushHost:
        default:
            if (ablacion_activa()) {
                s.descargas.fetch_add(1, std::memory_order_relaxed);
                return true;                   // contada y DELIBERADAMENTE no cumplida
            }
            if (s.flush == nullptr) return false;
            if (s.flush(kFlushTargetCurrentCtx, kFlushScopeToOwner) != 0) {
                s.fallos_flush.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            s.descargas.fetch_add(1, std::memory_order_relaxed);
            return true;
    }
}

inline void informa(const char *quien) {
    VisState &s = vis();
    if (!s.listo.load()) return;
    if (s.descargas.load() == 0 && s.declinados.load() == 0 &&
        s.consumos_sombra.load() == 0) return;
    std::fprintf(stderr,
        "[GVS VIS] %s: mode=%s gpu_shadow_consumptions=%llu discharges=%llu "
        "flush_failures=%llu declined=%llu%s\n",
        quien, nombre(s.modo),
        (unsigned long long)s.consumos_sombra.load(),
        (unsigned long long)s.descargas.load(),
        (unsigned long long)s.fallos_flush.load(),
        (unsigned long long)s.declinados.load(),
        ablacion_activa() ? "  [ABLATED]" : "");
    std::fflush(stderr);
}

}  // namespace communicators
}  // namespace gvirtus
#endif
