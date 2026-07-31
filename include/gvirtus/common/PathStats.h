/*
 * gVirtuS -- contadores de ruta de datos, compartibles entre plugins.
 *
 * POR QUE EXISTE ESTE FICHERO
 * ---------------------------
 * El plugin del runtime API (cudart) ya contaba sus transferencias, pero el del driver
 * API (cudadr) no contaba nada. Con cuDF eso deja un agujero grande: RMM y cuDF hacen sus
 * copias grandes por el driver API, de modo que en `path.csv` se veian 117 GB de
 * `h2d_gpudirect` y un `d2h_gpudirect` a CERO -- no porque el D2H fuera por host, sino
 * porque nadie lo estaba contando.
 *
 * Sin estos contadores no se puede responder a "¿las transferencias grandes usan de
 * verdad memoria GPU remota sin escala en host?", que es precisamente lo que hay que
 * demostrar y no suponer.
 *
 * QUE SE PUEDE AFIRMAR CON ESTO
 * -----------------------------
 *   - bytes con origen/destino en GPU  -> filas *_gpudirect
 *   - bytes que pasaron por un bufer de host -> filas *_host
 *   - "cero escala en host para lo grande" -> columnas >=4 MiB de las filas *_host a 0
 *   - "el trafico esta repartido por toda la ejecucion, no solo al arrancar" -> el
 *     fichero de serie temporal, que anade una instantanea fechada cada segundo en vez
 *     de sobrescribir el total
 *
 * Ese ultimo punto es el que un total acumulado no puede sostener: 117 GB podrian ser
 * todos del primer segundo.
 *
 * COSTE
 * -----
 * Tres incrementos atomicos relajados por copia. El hilo volcador solo nace si la
 * variable de entorno nombra un fichero. Nada lee estos contadores para decidir, asi que
 * no pueden alterar ninguna ruta de datos.
 *
 * Cabecera con estado en estaticas locales de funcion, no variables `inline`, para no
 * depender de C++17.
 */

#ifndef GVIRTUS_COMMON_PATH_STATS_H
#define GVIRTUS_COMMON_PATH_STATS_H

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace gvirtus {
namespace common {
namespace pathstats {

enum Path { kH2dGpu = 0, kH2dHost = 1, kD2hGpu = 2, kD2hHost = 3, kNPaths = 4 };
enum { kNBuckets = 8 };

inline const char *const *path_names() {
    static const char *const n[kNPaths] = {"h2d_gpudirect", "h2d_host", "d2h_gpudirect",
                                          "d2h_host"};
    return n;
}

struct State {
    std::atomic<unsigned long long> calls[kNPaths];
    std::atomic<unsigned long long> bytes[kNPaths];
    std::atomic<unsigned long long> hist[kNPaths][kNBuckets];
    State() {
        for (int p = 0; p < kNPaths; ++p) {
            calls[p].store(0);
            bytes[p].store(0);
            for (int b = 0; b < kNBuckets; ++b) hist[p][b].store(0);
        }
    }
};

inline State &state() {
    static State s;
    return s;
}

inline int bucket(size_t n) {
    if (n < 4ull * 1024) return 0;
    if (n < 64ull * 1024) return 1;
    if (n < 1024ull * 1024) return 2;
    if (n < 4ull * 1024 * 1024) return 3;
    if (n < 16ull * 1024 * 1024) return 4;
    if (n < 64ull * 1024 * 1024) return 5;
    if (n < 256ull * 1024 * 1024) return 6;
    return 7;
}

/* Total acumulado. Se escribe a un temporal y se renombra, para que quien lea nunca vea
 * un fichero a medias. */
inline void dump_totals(const std::string &path) {
    State &s = state();
    const std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) return;
    std::fprintf(f, "path,calls,bytes,lt4K,lt64K,lt1M,lt4M,lt16M,lt64M,lt256M,ge256M\n");
    for (int p = 0; p < kNPaths; ++p) {
        std::fprintf(f, "%s,%llu,%llu", path_names()[p],
                     s.calls[p].load(std::memory_order_relaxed),
                     s.bytes[p].load(std::memory_order_relaxed));
        for (int b = 0; b < kNBuckets; ++b)
            std::fprintf(f, ",%llu", s.hist[p][b].load(std::memory_order_relaxed));
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::rename(tmp.c_str(), path.c_str());
}

/* Serie temporal: una linea por instantanea y ruta, con marca de tiempo absoluta. Es lo
 * unico que puede sostener "el trafico esta repartido durante toda la ejecucion". */
inline void append_series(const std::string &path) {
    State &s = state();
    static bool header_done = false;
    FILE *f = std::fopen(path.c_str(), "a");
    if (f == nullptr) return;
    if (!header_done) {
        std::fseek(f, 0, SEEK_END);
        if (std::ftell(f) == 0) std::fprintf(f, "t_unix,path,calls,bytes\n");
        header_done = true;
    }
    const double now =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    for (int p = 0; p < kNPaths; ++p) {
        std::fprintf(f, "%.3f,%s,%llu,%llu\n", now, path_names()[p],
                     s.calls[p].load(std::memory_order_relaxed),
                     s.bytes[p].load(std::memory_order_relaxed));
    }
    std::fclose(f);
}

/* `env_var` nombra el fichero de totales; la serie temporal se deriva anadiendo
 * ".series". Se arranca un unico hilo volcador por proceso y por variable. */
inline void count(const char *env_var, Path p, size_t n) {
    State &s = state();
    s.calls[p].fetch_add(1, std::memory_order_relaxed);
    s.bytes[p].fetch_add(n, std::memory_order_relaxed);
    s.hist[p][bucket(n)].fetch_add(1, std::memory_order_relaxed);

    static std::once_flag once;
    std::call_once(once, [env_var]() {
        const char *out = std::getenv(env_var);
        if (out == nullptr || out[0] == '\0') return;
        const std::string dst(out);
        std::thread([dst]() {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                dump_totals(dst);
                append_series(dst + ".series");
            }
        }).detach();
    });
}

}  // namespace pathstats
}  // namespace common
}  // namespace gvirtus

#endif  // GVIRTUS_COMMON_PATH_STATS_H
