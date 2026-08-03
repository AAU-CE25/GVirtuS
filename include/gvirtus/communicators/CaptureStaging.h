#ifndef GVIRTUS_CAPTURE_STAGING_H
#define GVIRTUS_CAPTURE_STAGING_H
// Staging de captura de grafos (backend).
//
// EL PROBLEMA. Cuando el cliente captura un grafo, `cudaMemcpyAsync` no se ejecuta: se GRABA.
// Pero el origen de esa copia es el slot UCX (o su sombra de GPU), y el backend libera el slot
// en cuanto responde. El nodo del grafo quedaria apuntando a un slot que ya se reutilizo, asi
// que el handler sincronizaba para garantizar que el device habia terminado antes de soltar el
// slot -- y sincronizar un stream EN CAPTURA la invalida. De ahi
// cudaErrorStreamCaptureInvalidated: medido, nativo pasa y los dos brazos de Gusto fallan
// (tests/semantic/graphprobe.cu).
//
// LO QUE NO SIRVE. Alargar la vida del slot de la captura al lanzamiento. El pool son DOS slots
// por defecto, asi que el primer par de copias capturadas lo agota, y un grafo existe para
// lanzarse muchas veces: convierte un fallo de conformidad en un interbloqueo.
//
// LO QUE SE HACE. Copiar FUERA del slot en el momento de capturar: el payload se lleva a un
// buffer propiedad del backend y el nodo del grafo se graba contra ESE buffer. El slot se
// libera con normalidad porque el grafo ya no lo referencia. La copia extra se paga UNA VEZ al
// capturar, nunca por lanzamiento -- que es el sitio correcto, porque un grafo se captura una
// vez y se lanza muchas.
//
// INVARIANTE QUE ANADE (I11 de CONTRACTS.md): ningun nodo de grafo referencia jamas un slot RX.
//
// PROPIEDAD. El staging tiene que vivir mientras viva cualquier ejecutable que lo contenga:
//     captura (id) --EndCapture--> grafo --Instantiate--> ejecutable(s)
// Se sigue esa cadena con shared_ptr, asi que el ultimo GraphDestroy/GraphExecDestroy libera.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace gvs_capture {

struct Lote {                       // los buffers de una captura
    std::vector<void *> host;
    std::vector<void *> dev;
    ~Lote() {
        for (void *p : host) std::free(p);   // malloc, no cudaHostAlloc: ver stage_host
        for (void *p : dev) cudaFree(p);
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
inline unsigned long long &n_staged() { static unsigned long long n = 0; return n; }
inline unsigned long long &n_bytes()  { static unsigned long long n = 0; return n; }

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

// Staging en HOST fijado: para el camino en que el payload llega por el slot de host.
// Se copia con la CPU a proposito -- ninguna llamada CUDA de sincronizacion, que es
// justo lo que invalidaria la captura.
// MEDIDO: cudaHostAlloc aqui invalida la captura. Una ASIGNACION CUDA dentro de la ventana
// es en si misma una accion insegura, asi que el staging que existia para no sincronizar
// rompia la captura por otra via. Se usa malloc: memoria paginable, cero llamadas CUDA. El
// nodo del grafo solo necesita que el puntero siga vivo en el lanzamiento, no que este fijado.
inline void *stage_host(unsigned long long id, const void *src, size_t count) {
    std::lock_guard<std::mutex> lk(mu());
    void *p = std::malloc(count);
    if (p == nullptr) return nullptr;
    std::memcpy(p, src, count);
    lote_de(id)->host.push_back(p);
    ++n_staged(); n_bytes() += count;
    return p;
}

// Staging en DEVICE: para el camino de sombra de GPU (peer-DMA). La copia D2D se emite en el
// stream que se pase (debe ser uno interno, NO el capturado) y se espera con un evento en vez
// de cudaStreamSynchronize, que en captura es una llamada potencialmente insegura.
// Stream interno propio: g_shadow_stream es thread_local y puede no estar creado todavia,
// y depender de ese orden de inicializacion seria un fallo intermitente de los caros.
inline cudaStream_t stream_staging() {
    static thread_local cudaStream_t s = nullptr;
    if (s == nullptr && cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
        s = nullptr;
    return s;
}

inline void *stage_dev(unsigned long long id, const void *src, size_t count,
                       cudaStream_t interno) {
    if (interno == nullptr) interno = stream_staging();
    if (interno == nullptr) return nullptr;
    std::lock_guard<std::mutex> lk(mu());
    void *p = nullptr;
    if (cudaMalloc(&p, count) != cudaSuccess) return nullptr;
    if (cudaMemcpyAsync(p, src, count, cudaMemcpyDeviceToDevice, interno) != cudaSuccess) {
        cudaFree(p); return nullptr;
    }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        cudaFree(p); return nullptr;
    }
    if (cudaEventRecord(ev, interno) != cudaSuccess) {
        cudaEventDestroy(ev); cudaFree(p); return nullptr;
    }
    while (cudaEventQuery(ev) == cudaErrorNotReady) { /* espera activa, sin sincronizar */ }
    cudaEventDestroy(ev);
    lote_de(id)->dev.push_back(p);
    ++n_staged(); n_bytes() += count;
    return p;
}

// Transferencias de propiedad a lo largo de la cadena.
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
    if (n_staged() == 0) return;
    std::fprintf(stderr, "[GVS CAPTURA] %s: %llu copias sacadas del slot, %llu bytes\n",
                 quien, n_staged(), n_bytes());
    std::fflush(stderr);
}


// --- Bracketing temporal: nombrar la llamada que invalida la captura -------------------
// Se recuerda el stream sobre el que se abrio la captura y se informa su estado ANTES y
// DESPUES de cada handler. Adivinar fallo dos veces; esto no adivina.
inline cudaStream_t &g_stream_vigilado() { static cudaStream_t s = nullptr; return s; }

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
inline bool &g_ventana_abierta() { static bool v = false; return v; }

inline void bracket(const char *rutina, const char *cuando) {
    if (!g_ventana_abierta()) return;
    cudaStream_t s = g_stream_vigilado();
    if (s == nullptr) return;
    std::fprintf(stderr, "[BRACKET] %-28s %-6s st=%d\n", rutina, cuando, estado_crudo(s));
    std::fflush(stderr);
}

}  // namespace gvs_capture
#endif
