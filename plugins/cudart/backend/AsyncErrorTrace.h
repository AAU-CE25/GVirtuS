#ifndef GVIRTUS_ASYNC_ERROR_TRACE_H
#define GVIRTUS_ASYNC_ERROR_TRACE_H
// Atribucion de errores asincronos: de que llamada viene el error que aparece en un sync.
//
// EL PROBLEMA QUE RESUELVE. Una llamada de sincronizacion de CUDA devuelve el error del
// trabajo ASINCRONO anterior en ese stream. Cuando llama.cpp aborta con
//
//     ggml_backend_cuda_buffer_get_tensor -> cudaStreamSynchronize((cudaStream_t)0x2)
//     CUDA error: invalid argument -> ggml_abort()
//
// el sync es donde el error SE REPORTA, no donde se produce. Sin registro no hay forma de
// saber que operacion lo dejo ahi, y el suceso es intermitente (1 de 9 corridas), asi que
// tampoco se puede reproducir a voluntad para averiguarlo.
//
// POR QUE VA SIEMPRE ENCENDIDO. Es la decision de diseno de este fichero. Un registro opt-in
// no estara activo el dia que ocurra un fallo de 1 entre 9 -- que es justamente el dia que
// importa. El coste es un punado de escrituras en un anillo thread_local por operacion
// asincrona, sin asignacion y sin E/S; sale I/O UNICAMENTE cuando un sync devuelve error.
// La alternativa (encenderlo cuando sospechemos) equivale a no tenerlo.
//
// SIN CABECERAS DE CUDA a proposito: el stream y los punteros se guardan como `const void *`.
// Meter cuda.h en una cabecera compartida ya tumbo el backend dos veces.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gvs_async {

struct Op {
    const char   *routine;      // literal estatico: no se copia la cadena
    const void   *stream;
    const void   *dst;
    const void   *src;
    std::size_t   count;
    int           kind;         // cudaMemcpyKind, o -1 si no aplica
    int           rc;           // lo que devolvio la llamada en su momento
    int           line;         // sitio exacto: hay diez cudaMemcpyAsync distintos
    std::uint64_t seq;
};

inline constexpr int kAnillo = 32;

struct Estado {
    Op   anillo[kAnillo];
    int  siguiente = 0;
    std::uint64_t seq = 0;
    bool lleno = false;
};
inline Estado &estado() { static thread_local Estado e; return e; }

// Se llama DESPUES de emitir la operacion asincrona, con lo que devolvio.
inline void registra(const char *routine, int line, const void *stream, const void *dst,
                     const void *src, std::size_t count, int kind, int rc) {
    Estado &e = estado();
    Op &o = e.anillo[e.siguiente];
    o.routine = routine; o.stream = stream; o.dst = dst; o.src = src;
    o.count = count; o.kind = kind; o.rc = rc; o.line = line; o.seq = ++e.seq;
    e.siguiente = (e.siguiente + 1) % kAnillo;
    if (e.siguiente == 0) e.lleno = true;
}

inline const char *nombre_kind(int k) {
    switch (k) {
        case 0: return "H2H"; case 1: return "H2D";
        case 2: return "D2H"; case 3: return "D2D";
        case 4: return "default";
        default: return "-";
    }
}

// Se llama cuando una llamada de SINCRONIZACION devuelve algo distinto de exito. Vuelca las
// operaciones asincronas recientes de ESE stream, de la mas nueva a la mas vieja, para que el
// error deje de ser anonimo. Tambien vuelca las de otros streams al final, porque un error
// puede venir de trabajo que el driver serializo con el stream legacy.
inline void informa(const char *sync_routine, const void *stream, int rc) {
    Estado &e = estado();
    const int n = e.lleno ? kAnillo : e.siguiente;
    std::fprintf(stderr,
        "[GVS ASYNC] *** %s(stream=%p) returned rc=%d. The synchronising call REPORTS the "
        "error; it does not cause it. Recent async operations on this connection thread, "
        "newest first:\n", sync_routine, stream, rc);
    int mostradas = 0;
    for (int k = 1; k <= n; ++k) {
        const Op &o = e.anillo[(e.siguiente - k + kAnillo) % kAnillo];
        if (o.routine == nullptr) continue;
        const bool mismo = (o.stream == stream);
        std::fprintf(stderr,
            "[GVS ASYNC]   %c seq=%llu %s:%d stream=%p dst=%p src=%p count=%zu kind=%s rc=%d\n",
            mismo ? '>' : ' ', (unsigned long long)o.seq, o.routine, o.line, o.stream,
            o.dst, o.src, o.count, nombre_kind(o.kind), o.rc);
        if (++mostradas >= kAnillo) break;
    }
    if (mostradas == 0)
        std::fprintf(stderr, "[GVS ASYNC]   (no async operation recorded on this thread -- the "
                             "error did not come from work this backend enqueued)\n");
    std::fprintf(stderr, "[GVS ASYNC]   '>' marks the stream the sync was called on. An op with "
                         "rc!=0 failed AT ISSUE; an op with rc=0 may still have failed later.\n");
    std::fflush(stderr);
}

}  // namespace gvs_async
#endif
