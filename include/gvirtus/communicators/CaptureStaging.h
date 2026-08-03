#ifndef GVIRTUS_CAPTURE_STAGING_H
#define GVIRTUS_CAPTURE_STAGING_H
// Staging de captura de grafos (backend).
//
// EL PROBLEMA. Cuando el cliente captura un grafo, `cudaMemcpyAsync` no se ejecuta: se GRABA.
// Pero el origen de esa copia es el slot UCX (o su sombra de GPU), y el backend libera el slot
// en cuanto responde. El nodo del grafo quedaria apuntando a un slot que ya se reutilizo, asi
// que el handler sincronizaba para garantizar que el device habia terminado antes de soltar el
// slot -- y sincronizar un stream EN CAPTURA la invalida.
//
// LO QUE NO SIRVE. Alargar la vida del slot de la captura al lanzamiento. El pool son DOS slots
// por defecto, asi que el primer par de copias capturadas lo agota, y un grafo existe para
// lanzarse muchas veces: convierte un fallo de conformidad en un interbloqueo.
//
// LO QUE SE HACE. Copiar FUERA del slot en el momento de capturar: el payload se lleva a un
// buffer propiedad del backend y el nodo del grafo se graba contra ESE buffer. El slot se
// libera con normalidad porque el grafo ya no lo referencia. La copia extra se paga UNA VEZ al
// capturar, nunca por lanzamiento.
//
// INVARIANTE QUE ANADE (I11 de CONTRACTS.md): ningun nodo de grafo referencia jamas un slot RX.
//
// ---------------------------------------------------------------------------------------
// TABLA DE VERDAD MEDIDA (tests/semantic/capturetruth{,2,3}.cu, L40S, CUDA 12.6). Todo el
// diseno de este fichero se apoya en ella; NO en el manual, que dos veces llevo a un arreglo
// que rompia por otra via.
//
//   INVALIDAN una captura ThreadLocal abierta en el hilo que captura:
//     cudaMalloc, cudaFree, cudaHostAlloc, cudaFreeHost, cudaMemcpy (sincrono),
//     cudaStreamSynchronize(otro), cudaDeviceSynchronize, cudaStreamQuery(otro),
//     cudaEventQuery, cudaEventSynchronize, cudaEventRecord(stream 0).
//   NO invalidan:
//     malloc/free, cudaMemcpyAsync sobre OTRO stream (H2D, D2D y D2H),
//     cudaEventRecord sobre OTRO stream, cudaStreamCreateWithFlags, cudaEventCreateWithFlags,
//     cudaPointerGetAttributes, cudaGetLastError, cudaLaunchHostFunc(otro).
//
// De ahi salen las dos decisiones que gobiernan el staging:
//
//  (1) TODO el staging es HOST y con malloc. La version anterior usaba cudaHostAlloc para el
//      camino de host -- que invalida -- y cudaMalloc + cudaEventRecord + cudaEventQuery para
//      el camino de sombra de GPU. Ese segundo camino invalidaba por DOS motivos a la vez
//      (la asignacion y la consulta del evento), asi que nunca fue viable como estaba escrito.
//
//  (2) La espera del camino de sombra sale gratis y exacta. `cudaMemcpyAsync` D2H con destino
//      PAGINABLE sobre otro stream no invalida la captura Y BLOQUEA hasta completar (medido:
//      8 MiB, 478us en la llamada frente a 2us en el sync posterior). O sea: se lee la sombra
//      con la garantia de que la copia termino, sin una sola llamada de sincronizacion.
//
// ---------------------------------------------------------------------------------------
// SEMANTICA (medida, tests/semantic/capturetruth2.cu y graphsem.cu). Un nodo H2D capturado lee
// su origen EN EL LANZAMIENTO, no en la captura: nativo, cambiar el buffer de host entre
// capturar y lanzar cambia lo que llega al device. Sobre GVirtuS el origen vive en OTRO
// PROCESO, asi que una foto tomada al capturar entrega bytes viejos en cada relanzamiento --
// medido, y en silencio, que es lo peor. Por eso el frontend REFRESCA el staging antes de cada
// cudaGraphLaunch (ver `refresca`) y el indice `k` de `copias` es el contrato entre los dos
// lados: el k-esimo H2D capturado del cliente es `copias[k]`.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace gvs_capture {

inline bool traza_activa() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_CAPTRACE");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}


// Un buffer de staging propiedad del backend. `bytes` no es decorativo: es lo que permite
// rechazar un refresco cuyo indice se ha desalineado en vez de escribir en el buffer que no es.
struct Copia {
    void  *buf   = nullptr;
    size_t bytes = 0;
};

struct Lote {                       // los buffers de UNA captura, en ORDEN DE CAPTURA
    std::vector<Copia> copias;    // ENTRADAS  (H2D): el frontend las refresca antes de lanzar
    std::vector<Copia> salidas;   // SALIDAS   (D2H): el frontend las recoge tras sincronizar
    ~Lote() {
        for (auto &c : copias)  std::free(c.buf);  // malloc, no cudaHostAlloc: ver la tabla
        for (auto &c : salidas) std::free(c.buf);
    }
};
using LotePtr = std::shared_ptr<Lote>;

inline std::mutex &mu() { static std::mutex m; return m; }
inline std::map<unsigned long long, LotePtr> &por_captura() {
    static std::map<unsigned long long, LotePtr> m; return m;
}
inline std::map<cudaGraph_t, LotePtr> &por_grafo() {
    static std::map<cudaGraph_t, LotePtr> m; return m;
}
inline std::map<cudaGraphExec_t, LotePtr> &por_exec() {
    static std::map<cudaGraphExec_t, LotePtr> m; return m;
}

// Contadores: sin ellos "no se capturo nada" y "se capturo y funciono" son indistinguibles
// desde fuera, que es como una campana entera acaba midiendo el camino que no es.
inline unsigned long long &n_staged()    { static unsigned long long n = 0; return n; }
inline unsigned long long &n_bytes()     { static unsigned long long n = 0; return n; }
inline unsigned long long &n_refresh()   { static unsigned long long n = 0; return n; }
inline unsigned long long &n_refresh_b() { static unsigned long long n = 0; return n; }
inline unsigned long long &n_refresh_ko(){ static unsigned long long n = 0; return n; }

// �Esta capturando este stream? Devuelve el id de captura, o 0 si no.
inline unsigned long long id_de_captura(cudaStream_t s) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    if (cudaStreamGetCaptureInfo(s, &st, &id) != cudaSuccess) return 0;
    if (st != cudaStreamCaptureStatusActive) return 0;
    // Un id 0 con captura activa no deberia ocurrir; si ocurriera, tratarlo como "no captura"
    // es lo seguro: se cae al camino de siempre, que falla de forma visible.
    return id;
}

inline LotePtr lote_de(unsigned long long id) {
    auto &m = por_captura();
    auto it = m.find(id);
    if (it != m.end()) return it->second;
    LotePtr l = std::make_shared<Lote>();
    m[id] = l;
    return l;
}

// Reserva un buffer de staging y lo registra en el lote. El indice implicito (la posicion en
// `copias`) es el contrato con el frontend. Devuelve nullptr si no hay memoria.
inline void *reserva_locked(unsigned long long id, size_t count) {
    void *p = std::malloc(count);
    if (p == nullptr) return nullptr;
    LotePtr l = lote_de(id);
    l->copias.push_back(Copia{p, count});
    ++n_staged(); n_bytes() += count;
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] reserva cid=%llu k=%zu bytes=%zu buf=%p\n",
                     id, l->copias.size() - 1, count, p);
    return p;
}

// Staging desde el SLOT DE HOST. Se copia con la CPU a proposito: ninguna llamada CUDA, que es
// lo unico que no puede invalidar nada.
inline void *stage_host(unsigned long long id, const void *src, size_t count) {
    std::lock_guard<std::mutex> lk(mu());
    void *p = reserva_locked(id, count);
    if (p == nullptr) return nullptr;
    std::memcpy(p, src, count);
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] stage_host bytes=%zu primer=0x%02X\n",
                     count, ((const unsigned char *)src)[0]);
    return p;
}

// Stream interno propio para el staging: NO puede ser el que se esta capturando.
// cudaStreamCreateWithFlags no invalida la captura (medido), asi que crearlo perezosamente
// aqui es seguro incluso dentro de la ventana.
inline cudaStream_t stream_staging() {
    static thread_local cudaStream_t s = nullptr;
    if (s == nullptr && cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
        s = nullptr;
    return s;
}

// Staging desde la SOMBRA DE GPU del slot (peer-DMA). El destino es memoria PAGINABLE, y por
// eso esta llamada es a la vez segura para la captura y bloqueante: cuando vuelve, los bytes
// estan. Ver la tabla de verdad de arriba -- no se usa ni evento ni sincronizacion.
inline void *stage_dev(unsigned long long id, const void *src, size_t count,
                       cudaStream_t interno) {
    if (interno == nullptr) interno = stream_staging();
    if (interno == nullptr) return nullptr;
    std::lock_guard<std::mutex> lk(mu());
    void *p = reserva_locked(id, count);
    if (p == nullptr) return nullptr;
    const cudaError_t ec = cudaMemcpyAsync(p, src, count, cudaMemcpyDeviceToHost, interno);
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] stage_dev bytes=%zu src=%p ec=%d primer=0x%02X\n",
                     count, src, (int)ec, ((const unsigned char *)p)[0]);
    if (ec != cudaSuccess) {
        // El buffer se queda en el lote: liberarlo aqui desalinearia los indices, y el lote
        // entero se libera igualmente cuando muera el ultimo ejecutable.
        return nullptr;
    }
    return p;
}

// Staging de SALIDA: el nodo D2H capturado escribe aqui en cada lanzamiento, y el frontend
// lo recoge despues de sincronizar. El indice j (posicion en `salidas`) es el contrato con el
// espejo del frontend, igual que k lo es para las entradas.
inline void *stage_salida(unsigned long long id, size_t count) {
    std::lock_guard<std::mutex> lk(mu());
    void *p = std::malloc(count);
    if (p == nullptr) return nullptr;
    LotePtr l = lote_de(id);
    l->salidas.push_back(Copia{p, count});
    ++n_staged(); n_bytes() += count;
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] salida cid=%llu j=%zu bytes=%zu buf=%p\n",
                     id, l->salidas.size() - 1, count, p);
    return p;
}

// Devuelve el buffer de salida j de un ejecutable, o nullptr si el indice o el tamano no
// cuadran -- rechazar es mejor que devolver el buffer de otro nodo.
inline void *salida_de(cudaGraphExec_t e, size_t j, size_t bytes) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_exec().find(e);
    if (it == por_exec().end() || !it->second) return nullptr;
    auto &ss = it->second->salidas;
    if (j >= ss.size() || ss[j].bytes != bytes) return nullptr;
    return ss[j].buf;
}

// --- Refresco antes de cada lanzamiento (la mitad semantica del arreglo) ----------------
// Devuelve true si el refresco se aplico. Un indice fuera de rango o un tamano que no cuadra
// se RECHAZAN en vez de escribir en otro buffer: si los dos lados se desalinean alguna vez,
// que se vea en el contador y no en los resultados.
inline bool refresca(cudaGraphExec_t e, size_t k, const void *src, size_t bytes) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_exec().find(e);
    if (it == por_exec().end() || !it->second) { ++n_refresh_ko(); return false; }
    auto &cs = it->second->copias;
    if (k >= cs.size() || cs[k].bytes != bytes || cs[k].buf == nullptr) {
        ++n_refresh_ko();
        if (traza_activa())
            std::fprintf(stderr, "[CAPSTG] refresca RECHAZADO exec=%p k=%zu bytes=%zu copias=%zu\n",
                         (void*)e, k, bytes, cs.size());
        return false;
    }
    std::memcpy(cs[k].buf, src, bytes);
    ++n_refresh(); n_refresh_b() += bytes;
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] refresca exec=%p k=%zu bytes=%zu primer=0x%02X\n",
                     (void*)e, k, bytes, ((const unsigned char *)src)[0]);
    return true;
}

// Variante del refresco cuando el payload llego a la SOMBRA DE GPU del slot en vez de al slot
// de host. No deberia ocurrir -- el frontend marca el splice del refresco como host-destined --
// pero un frontend anterior a ese cambio si lo enruta asi, y leer el hueco del slot de host
// refrescaria con ceros en silencio. El destino es paginable, luego esta copia bloquea.
inline bool refresca_desde_device(cudaGraphExec_t e, size_t k, const void *gpu_src,
                                  size_t bytes) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_exec().find(e);
    if (it == por_exec().end() || !it->second) { ++n_refresh_ko(); return false; }
    auto &cs = it->second->copias;
    if (k >= cs.size() || cs[k].bytes != bytes || cs[k].buf == nullptr) {
        ++n_refresh_ko();
        return false;
    }
    cudaStream_t interno = stream_staging();
    if (interno == nullptr) { ++n_refresh_ko(); return false; }
    if (cudaMemcpyAsync(cs[k].buf, gpu_src, bytes, cudaMemcpyDeviceToHost, interno)
            != cudaSuccess) {
        ++n_refresh_ko();
        return false;
    }
    ++n_refresh(); n_refresh_b() += bytes;
    if (traza_activa())
        std::fprintf(stderr, "[CAPSTG] refresca(sombra) exec=%p k=%zu bytes=%zu primer=0x%02X\n",
                     (void*)e, k, bytes, ((const unsigned char *)cs[k].buf)[0]);
    return true;
}

// Transferencias de propiedad a lo largo de la cadena
//     captura (id) --EndCapture--> grafo --Instantiate--> ejecutable(s)
// Se sigue con shared_ptr, asi que el ultimo GraphDestroy/GraphExecDestroy libera. Un grafo
// puede instanciarse varias veces: los ejecutables COMPARTEN el lote, que es justo lo que
// hace falta porque comparten los nodos.
inline void captura_termina(unsigned long long id, cudaGraph_t g) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_captura().find(id);
    if (it == por_captura().end()) return;
    if (g != nullptr) por_grafo()[g] = it->second;
    por_captura().erase(it);
}
inline void grafo_instanciado(cudaGraph_t g, cudaGraphExec_t e) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_grafo().find(g);
    if (it != por_grafo().end() && e != nullptr) por_exec()[e] = it->second;
}
inline void grafo_destruido(cudaGraph_t g) {
    std::lock_guard<std::mutex> lk(mu());
    por_grafo().erase(g);
}
inline void exec_destruido(cudaGraphExec_t e) {
    std::lock_guard<std::mutex> lk(mu());
    por_exec().erase(e);
}

inline void informa(const char *quien) {
    if (n_staged() == 0 && n_refresh() == 0) return;
    std::fprintf(stderr,
                 "[GVS CAPTURA] %s: %llu copias sacadas del slot (%llu B), "
                 "%llu refrescos aplicados (%llu B), %llu rechazados\n",
                 quien, n_staged(), n_bytes(), n_refresh(), n_refresh_b(), n_refresh_ko());
    std::fflush(stderr);
}


// --- Bracketing temporal: nombrar la llamada que invalida la captura -------------------
// Se recuerda el stream sobre el que se abrio la captura y se informa su estado ANTES y
// DESPUES de cada handler. Adivinar fallo dos veces; esto no adivina.
inline cudaStream_t &g_stream_vigilado() {
    static thread_local cudaStream_t s = nullptr; return s;
}

// Capturas abiertas por ESTE hilo. Empareja Begin/End sin depender de la salud de la captura.
inline int &profundidad_local() { static thread_local int d = 0; return d; }

inline int estado_crudo(cudaStream_t s) {
    if (s == nullptr) return -1;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    if (cudaStreamGetCaptureInfo(s, &st, &id) != cudaSuccess) return -2;
    return (int)st;   // 0 None, 1 Active, 2 Invalidated
}

// Solo se vigila DENTRO de la ventana de captura. Sin esta bandera el global queda
// apuntando a un stream ya destruido y cudaStreamGetCaptureInfo sobre el TUMBA el backend
// -- ocurrido, y es la razon de que exista este comentario.
inline bool &g_ventana_abierta() { static thread_local bool v = false; return v; }

inline void bracket(const char *rutina, const char *cuando) {
    if (!traza_activa() || !g_ventana_abierta()) return;
    cudaStream_t s = g_stream_vigilado();
    if (s == nullptr) return;
    std::fprintf(stderr, "[BRACKET] %-28s %-6s st=%d\n", rutina, cuando, estado_crudo(s));
    std::fflush(stderr);
}

}  // namespace gvs_capture
#endif
