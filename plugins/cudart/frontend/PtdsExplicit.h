#ifndef GVIRTUS_PTDS_EXPLICIT_H
#define GVIRTUS_PTDS_EXPLICIT_H
// Traduce el sentinel cudaStreamPerThread a un stream EXPLICITO por hilo frontend.
//
// EL PROBLEMA QUE PRUEBA. `cudaStreamPerThread` es el literal 0x2 y es THREAD-LOCAL: designa
// el default stream del hilo HOST que llama. Al remotar se envia el literal y quien lo
// interpreta es el hilo del BACKEND que atiende la RPC. Eso solo conserva la semantica si un
// hilo frontend es atendido siempre por el mismo hilo de backend -- cosa que hoy se cumple
// (comprobado: 0 avisos de cambio de comunicador), pero que descansa en una propiedad del
// despliegue y no en el protocolo.
//
// LA REPRESENTACION ROBUSTA, que es lo que hace esta perilla: cada hilo frontend crea UNA vez
// su propio stream explicito y lo usa en lugar del sentinel. La identidad deja de depender de
// que hilo del servidor atienda la llamada, porque viaja como un handle concreto.
//
// DOBLE USO, y por eso existe:
//   - brazo E de la matriz de ablacion: si el abort desaparece con esto y aparece sin ello,
//     la causa es la resolucion del sentinel;
//   - arreglo candidato, si resulta serlo.
// Apagada por defecto: cambia el stream que ve el backend y eso no se activa en silencio.
#include <cstdlib>
#include <cuda_runtime.h>

namespace gvs_ptds {

inline bool activo() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_PTDS_AS_EXPLICIT");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

// El stream explicito de ESTE hilo. Se crea perezosamente y por el camino normal, asi que el
// backend lo materializa como cualquier otro stream creado por el cliente.
// `creando` corta la recursion: cudaStreamCreateWithFlags no toma stream, pero si alguna capa
// intermedia acabara llamando aqui, la guarda lo impide.
inline cudaStream_t mio() {
    static thread_local cudaStream_t s = nullptr;
    static thread_local bool creando = false;
    if (s == nullptr && !creando) {
        creando = true;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) s = nullptr;
        creando = false;
    }
    return s;
}

inline cudaStream_t traduce(cudaStream_t in) {
    if (!activo() || in != cudaStreamPerThread) return in;
    cudaStream_t s = mio();
    return s ? s : in;
}

}  // namespace gvs_ptds
#endif
