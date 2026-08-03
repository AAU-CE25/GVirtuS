#ifndef GVIRTUS_SCHED_TRACE_H
#define GVIRTUS_SCHED_TRACE_H
// Traza del planificador del backend. SIN dependencia de CUDA a proposito: esta cabecera se
// incluye desde src/backend/Process.cpp, que se compila sin las cabeceras de CUDA. La
// sustitucion de stream, que si las necesita, vive en el fichero del plugin.
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <mutex>
#include <unordered_map>

namespace gvs {

inline bool sched_trace() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_SCHED_TRACE");
        return e && e[0] == '1';
    }();
    return v;
}

inline bool per_conn_stream() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_PER_CONN_STREAM");
        return e && e[0] == '1';
    }();
    return v;
}

inline long long now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Identificador corto y estable por conexion: el hilo que atiende una conexion es el mismo
// durante toda su vida, asi que un thread_local basta.
inline int conn_id() {
    static std::atomic<int> next{0};
    static thread_local int id = -1;
    if (id < 0) id = next.fetch_add(1);
    return id;
}

// --- Compuerta de reparto equitativo (GVS_FAIR_DISPATCH) --------------------------------
// POR QUE. Medido en miniBUDE N=8: el tenant favorecido corre a 1.00x -- su ritmo en solitario,
// como si no compartiera nada -- y el peor a 6.25x. La desigualdad (mas lento / mas rapido) es
// 4.3-4.9x en LOS TRES transportes (tcp 4.62, ucx_rdma 4.31, ucx_gpudirect 4.87). Nativo, con
// ocho contextos CUDA, reparte a 1.03x. Y nativo+MPS, que mete a los ocho clientes en UN SOLO
// contexto, reparte a 1.02x -- luego compartir contexto NO basta para producir la desigualdad,
// y la hipotesis del "contexto unico sin arbitraje" queda refutada por su propio control.
//
// Lo que queda en pie es que el backend no arbitra: sirve cada RPC en cuanto llega, por orden
// de llegada, y como el cliente se auto-reloja (no pide la iteracion siguiente hasta que vuelve
// la anterior) el que va delante vuelve a pedir antes y se queda delante. MPS es justo porque
// SU servidor arbitra; el driver es justo entre contextos porque los multiplexa; el backend de
// Gusto no hace ni una cosa ni la otra.
//
// QUE ES ESTO. El arbitraje que falta, en su forma minima: deficit round-robin sobre los
// lanzamientos. Una conexion no puede ir mas de GVS_FAIR_LEAD lanzamientos por delante de la
// conexion activa que menos ha recibido. Es una PRUEBA CAUSAL, no una propuesta de diseno: si
// el mecanismo es el que se dice, la desigualdad debe desplomarse; si no cambia nada, la
// explicacion es falsa y hay que decirlo. Apagada salvo que se pida.
inline bool fair_dispatch() {
    static const bool v = [] {
        const char *e = std::getenv("GVS_FAIR_DISPATCH");
        const bool on = e && e[0] == '1';
        if (on) std::fprintf(stderr, "[GVS SCHED] *** reparto equitativo ACTIVO "
                                     "(deficit round-robin sobre los lanzamientos)\n");
        return on;
    }();
    return v;
}

inline unsigned long long fair_lead() {
    static const unsigned long long v = []() -> unsigned long long {
        const char *e = std::getenv("GVS_FAIR_LEAD");
        const long long x = (e && e[0]) ? std::strtoll(e, nullptr, 10) : 1;
        return x > 0 ? (unsigned long long)x : 1ull;
    }();
    return v;
}

struct FairState {
    std::mutex mu;
    std::condition_variable cv;
    std::unordered_map<int, unsigned long long> served;
    std::unordered_map<int, long long> last_ns;
};
inline FairState &fair_state() { static FairState s; return s; }

// Se llama justo ANTES de lanzar.
inline void fair_wait() {
    if (!fair_dispatch()) return;
    const int me = conn_id();
    FairState &S = fair_state();
    std::unique_lock<std::mutex> lk(S.mu);
    const long long ahora = now_ns();
    S.last_ns[me] = ahora;
    // Techo de espera: degradar a FCFS antes que colgarse. Una compuerta de medida que puede
    // bloquear el banco indefinidamente no sirve para medir nada.
    const long long tope = ahora + 2000000000LL;
    for (;;) {
        // Activas = vistas en los ultimos 2 s. Sin la poda, un tenant que TERMINA deja su
        // contador congelado en el minimo y bloquea a todos los demas para siempre.
        unsigned long long minimo = ~0ull;
        const long long corte = now_ns() - 2000000000LL;
        for (const auto &kv : S.last_ns) {
            if (kv.second < corte) continue;
            auto it = S.served.find(kv.first);
            const unsigned long long c = (it == S.served.end()) ? 0ull : it->second;
            if (c < minimo) minimo = c;
        }
        if (minimo == ~0ull) break;
        auto itme = S.served.find(me);
        const unsigned long long mio = (itme == S.served.end()) ? 0ull : itme->second;
        if (mio <= minimo + fair_lead()) break;
        if (now_ns() >= tope) break;
        S.cv.wait_for(lk, std::chrono::milliseconds(2));
    }
    ++S.served[me];
}

inline void fair_done() {
    if (!fair_dispatch()) return;
    fair_state().cv.notify_all();
}

inline void trace(const char *routine, long long t_in, long long t_end) {
    if (!sched_trace()) return;
    std::fprintf(stderr, "GVS_SCHED conn=%d routine=%s t_in=%lld t_end=%lld\n",
                 conn_id(), routine, t_in, t_end);
}

}  // namespace gvs
#endif
