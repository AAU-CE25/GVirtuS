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
#include <atomic>
#include <cstdio>
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

// SENAL DE VITALIDAD, y existe por un error concreto: el brazo E se corrio una noche entera sin
// que nada comprobara que la traduccion estuviera ocurriendo. El arnes vigilaba `admit_rma`, que
// solo dice que el camino RMA esta vivo, y da igual lo que haga esta perilla. Un brazo E sin
// traducciones ES el brazo D, y yo lo habria contado como E.
// Se emite por stderr en la PRIMERA traduccion y cada 100.000: el arnes mata el proceso con
// `docker rm -f`, asi que un contador de teardown no llegaria a imprimirse nunca.
inline std::atomic<unsigned long> &traducciones() {
    static std::atomic<unsigned long> n{0};
    return n;
}

inline std::atomic<unsigned long> &fallos() {
    static std::atomic<unsigned long> n{0};
    return n;
}

inline cudaStream_t traduce(cudaStream_t in) {
    if (!activo() || in != cudaStreamPerThread) return in;
    cudaStream_t s = mio();
    if (s == nullptr) {
        // Se CUENTA. Si la creacion del stream falla, el brazo E degrada en silencio al brazo D
        // y sin este contador se leeria como "esta carga apenas usa PTDS", que es otra cosa.
        const unsigned long f = fallos().fetch_add(1) + 1;
        if (f == 1) std::fprintf(stderr, "[GVS PTDS] WARNING: explicit-stream creation failed; "
                                         "falling back to the sentinel (untranslated=%lu)\n", f);
        return in;
    }
    const unsigned long n = traducciones().fetch_add(1) + 1;
    // Cadencia por potencias de diez: el arnes lee el ULTIMO valor impreso, asi que una cadencia
    // gruesa convierte el conteo en un simple ">=1". Esto lo deja legible en orden de magnitud.
    bool imprime = false;
    for (unsigned long p = 1; p <= n; p *= 10) if (p == n) { imprime = true; break; }
    if (imprime)
        std::fprintf(stderr, "[GVS PTDS] cudaStreamPerThread -> explicit stream %p (translations=%lu untranslated=%lu)\n",
                     (void *)s, n, fallos().load());
    return s;
}

}  // namespace gvs_ptds
#endif
