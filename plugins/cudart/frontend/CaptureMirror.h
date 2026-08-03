#ifndef GVIRTUS_CAPTURE_MIRROR_H
#define GVIRTUS_CAPTURE_MIRROR_H
// Espejo de captura en el FRONTEND.
//
// POR QUE EXISTE. Un nodo H2D capturado lee su origen EN EL LANZAMIENTO, no en la captura
// (medido: tests/semantic/capturetruth2.cu, nativo). Sobre GVirtuS ese origen vive en el
// proceso del cliente, asi que el buffer que el backend copio al capturar entrega los MISMOS
// bytes en cada relanzamiento. Medido antes de este espejo (tests/semantic/graphsem.cu):
// capturar 1 KiB, cambiar el origen, relanzar -> llegaban los bytes viejos, con cudaSuccess
// en todas las llamadas. Un resultado mal en silencio, que es peor que el fallo ruidoso que
// se estaba arreglando.
//
// QUE HACE. Recuerda, en orden, cada cudaMemcpyAsync H2D emitido con una captura abierta.
// Ese orden es el CONTRATO con el backend: el k-esimo H2D capturado es `copias[k]` de su lote
// (ver CaptureStaging.h). Antes de cada cudaGraphLaunch el frontend reenvia el contenido
// ACTUAL de cada origen por el mismo camino de datos que cualquier otro H2D -- splice directo
// en el iov, o sea RMA o mensaje activo segun tamano. El refresco no rodea el diseno de slots:
// lo usa.
//
// COSTE. Cero para quien no captura copias H2D. llama.cpp, que es el consumidor real de
// grafos aqui, captura lanzamientos de kernel y ninguna copia H2D: su lista queda vacia y
// cudaGraphLaunch no emite ni un mensaje extra.
//
// ALCANCE. La captura activa se sigue por HILO. Una captura que se bifurca a otros streams y
// vuelve a unirse comparte el mismo id de captura en el backend, asi que anotar por hilo (y
// no por stream) es lo que mantiene alineados los indices de los dos lados.
#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace gvs_capmirror {

struct Entrada {
    const void *src   = nullptr;   // buffer del cliente; se relee en cada lanzamiento
    size_t      bytes = 0;
};

struct Salida {
    void  *dst   = nullptr;        // buffer del cliente; se reescribe en cada lanzamiento
    size_t bytes = 0;
};

inline std::mutex &mu() { static std::mutex m; return m; }
inline std::map<cudaGraph_t, std::vector<Entrada>> &por_grafo() {
    static std::map<cudaGraph_t, std::vector<Entrada>> m; return m;
}
inline std::map<cudaGraphExec_t, std::vector<Entrada>> &por_exec() {
    static std::map<cudaGraphExec_t, std::vector<Entrada>> m; return m;
}

// Captura activa de ESTE hilo.
inline int &profundidad()               { static thread_local int d = 0; return d; }
inline std::vector<Entrada> &en_curso() { static thread_local std::vector<Entrada> v; return v; }

inline bool capturando() { return profundidad() > 0; }

// Salidas: un nodo D2H capturado escribe en el buffer del cliente en CADA lanzamiento. Se
// recuerda (dst, bytes) en orden; el indice j es el contrato con `salidas` del lote.
inline std::vector<Salida> &salidas_en_curso() {
    static thread_local std::vector<Salida> v; return v;
}
inline std::map<cudaGraph_t, std::vector<Salida>> &salidas_por_grafo() {
    static std::map<cudaGraph_t, std::vector<Salida>> m; return m;
}
inline std::map<cudaGraphExec_t, std::vector<Salida>> &salidas_por_exec() {
    static std::map<cudaGraphExec_t, std::vector<Salida>> m; return m;
}
inline void anota_d2h(void *dst, size_t bytes) {
    if (!capturando()) return;
    salidas_en_curso().push_back(Salida{dst, bytes});
}
inline std::vector<Salida> salidas_de(cudaGraphExec_t e) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = salidas_por_exec().find(e);
    return (it == salidas_por_exec().end()) ? std::vector<Salida>{} : it->second;
}

// Ejecutables lanzados y aun no recogidos, por hilo. Se vacia al sincronizar.
struct Lanzado { cudaGraphExec_t exec; cudaStream_t stream; };
inline std::vector<Lanzado> &lanzados() {
    static thread_local std::vector<Lanzado> v; return v;
}

inline void abre() {
    if (profundidad()++ == 0) { en_curso().clear(); salidas_en_curso().clear(); }
}

inline void anota_h2d(const void *src, size_t bytes) {
    if (!capturando()) return;
    en_curso().push_back(Entrada{src, bytes});
}

inline void cierra(cudaGraph_t g) {
    if (profundidad() > 0) --profundidad();
    if (profundidad() != 0) return;
    if (g != nullptr && (!en_curso().empty() || !salidas_en_curso().empty())) {
        std::lock_guard<std::mutex> lk(mu());
        if (!en_curso().empty())         por_grafo()[g]          = en_curso();
        if (!salidas_en_curso().empty()) salidas_por_grafo()[g]  = salidas_en_curso();
    }
    en_curso().clear();
    salidas_en_curso().clear();
}

inline void instancia(cudaGraph_t g, cudaGraphExec_t e) {
    if (g == nullptr || e == nullptr) return;
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_grafo().find(g);
    if (it != por_grafo().end()) por_exec()[e] = it->second;
    auto is = salidas_por_grafo().find(g);
    if (is != salidas_por_grafo().end()) salidas_por_exec()[e] = is->second;
}

inline void destruye_grafo(cudaGraph_t g) {
    std::lock_guard<std::mutex> lk(mu());
    por_grafo().erase(g);
    salidas_por_grafo().erase(g);
}
inline void destruye_exec(cudaGraphExec_t e) {
    std::lock_guard<std::mutex> lk(mu());
    por_exec().erase(e);
    salidas_por_exec().erase(e);
}

// Copia para poder soltar el cerrojo antes de emitir RPCs.
inline std::vector<Entrada> entradas_de(cudaGraphExec_t e) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = por_exec().find(e);
    return (it == por_exec().end()) ? std::vector<Entrada>{} : it->second;
}

}  // namespace gvs_capmirror
#endif
