#ifndef GVIRTUS_COMMUNICATORS_RMAPOLICY_H
#define GVIRTUS_COMMUNICATORS_RMAPOLICY_H

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// Semantics-aware data-path placement policy.
// ---------------------------------------------------------------------------
// The deployment used a single scalar floor (4 MiB) to choose between active messages and
// RDMA. The 2026-08-01 sweep showed that one number cannot be right, because the crossover
// depends on two properties the frontend already knows at the point of decision:
//
//                    | H2D          | D2H
//   -----------------|--------------|-------------
//   pinned host      |    8 KiB     |   1 MiB
//   pageable host    |    1 MiB     |   2 MiB
//
// Measured crossovers (GB/s, pinned H2D): 16 KiB 0.53->0.63 - 512 KiB 4.66->10.98 -
// 1 MiB 5.20->15.21 - 2 MiB 5.47->18.71. The gain grows monotonically and no measured size
// between 8 KiB and 64 MiB favours AM. For D2H the sign is REVERSED below 1 MiB
// (16 KiB x0.38, 64 KiB x0.44): lowering the floor there is actively harmful, because the
// client-initiated GET registers its destination per call and that fixed cost does not
// amortise on small transfers.
//
// So a single threshold is not merely suboptimal, it is the wrong SHAPE: the two directions
// want thresholds three orders of magnitude apart, and no scalar can satisfy both.
//
// Three policies are selectable so they can be compared against each other:
//
//   scalar    the deployment's 4 MiB for everything (the baseline being argued against)
//   quadrant  the table above
//   oracle    per-size choice taken from the sweep's measured winner, i.e. the best any
//             threshold policy could do. It is not deployable -- it needs the answer in
//             advance -- and exists to bound how much the quadrant step function leaves
//             on the table.
//
// Deliberately NOT included: any per-transfer probing of the pointer. `cudaPointerGetAttributes`
// is remoted on the frontend, so calling it per transfer would cost an RPC to save an RPC.
// The pinned/pageable bit comes from the interval map the frontend already maintains for
// cudaHostAlloc/cudaFreeHost, which is a local lookup.

namespace gvirtus {
namespace communicators {

enum class RmaPolicy { Scalar = 0, Quadrant, Oracle };

inline std::size_t quadrant_threshold(bool h2d, bool pinned);

inline RmaPolicy rma_policy() {
    static const RmaPolicy p = []() {
        const char *v = std::getenv("GVIRTUS_RMA_POLICY");
        // QUADRANT ES EL DEFECTO desde 2026-08-04. Antes lo era `scalar` a 4 MiB, mientras el
        // propio paquete demostraba que la forma escalar es incorrecta: quadrant queda a <=2 %
        // del oraculo en las CUATRO combinaciones direccion x tipo de memoria, donde cualquier
        // umbral unico cae al 22-31 % en una de ellas.
        // El unico coste medido en su contra (-1,1 % en llama 7B) resulto ser la trampa de las
        // dos perillas, no la politica: aparecia solo cuando el suelo del pool superaba el
        // umbral MAS PEQUENO de la tabla. Ver policy_min_threshold() abajo, que lo hace
        // imposible por construccion.
        if (v == nullptr || v[0] == '\0' || std::strcmp(v, "quadrant") == 0) {
            // El banner IMPRIME la tabla, no una copia escrita a mano: al re-medir el cruce
            // a 16 KiB cambie la constante y este literal se quedo diciendo 8K, o sea que el
            // log contradecia al binario que lo emitia. Un valor citado a mano se queda atras;
            // uno leido de la fuente, no.
            std::fprintf(stderr,
                         "[GVS POLICY] four-quadrant placement (H2D pinned %zuK / "
                         "H2D pageable %zuK / D2H pinned %zuK / D2H pageable %zuK)\n",
                         quadrant_threshold(true,  true)  >> 10,
                         quadrant_threshold(true,  false) >> 10,
                         quadrant_threshold(false, true)  >> 10,
                         quadrant_threshold(false, false) >> 10);
            return RmaPolicy::Quadrant;
        }
        if (std::strcmp(v, "scalar") == 0) return RmaPolicy::Scalar;
        if (std::strcmp(v, "oracle") == 0) {
            std::fprintf(stderr, "[GVS POLICY] ORACLE per-size placement "
                                 "(not deployable; upper bound only)\n");
            return RmaPolicy::Oracle;
        }
        std::fprintf(stderr, "[GVS POLICY] unknown value '%s'; using scalar\n", v);
        return RmaPolicy::Scalar;   // valor invalido: se degrada al conservador, no al defecto
    }();
    return p;
}

inline std::size_t env_bytes(const char *k, std::size_t dflt) {
    const char *e = std::getenv(k);
    if (e == nullptr || e[0] == '\0') return dflt;
    char *end = nullptr;
    unsigned long long v = std::strtoull(e, &end, 10);
    return (end != e) ? static_cast<std::size_t>(v) : dflt;
}

// The quadrant table. Each entry is overridable so the policy can be re-tuned on another
// fabric without a rebuild -- the values below are measurements of THIS deployment, not
// constants of the design.
inline std::size_t quadrant_threshold(bool h2d, bool pinned) {
    static const std::size_t t[2][2] = {
        // [h2d][pinned]
        { env_bytes("GVIRTUS_RMA_MIN_D2H_PAGEABLE", 2ull << 20),
          env_bytes("GVIRTUS_RMA_MIN_D2H_PINNED",   1ull << 20) },
        { env_bytes("GVIRTUS_RMA_MIN_H2D_PAGEABLE", 1ull << 20),
          // 16 KiB, no 8 KiB. Re-medido 2026-08-03 con 3 corridas de proceso independientes
          // por celda: a 8 KiB AM gana en 3/3 (0,300 frente a 0,288 GB/s) y a 16 KiB pierde
          // en 3/3. El 8 KiB anterior venia de una sola corrida donde la diferencia era x1,01
          // -- un empate leido como cruce. NO lo movio el trabajo de correccion: `assume` y
          // `flush` cruzan en el mismo tamano.
          env_bytes("GVIRTUS_RMA_MIN_H2D_PINNED",   16ull << 10) },
    };
    return t[h2d ? 1 : 0][pinned ? 1 : 0];
}

// Oracle: the measured winner per size class. H2D from the 2026-08-03 re-measurement (3
// independent process runs per cell, under the deployed safe A1 policy); D2H from the
// 2026-08-01 sweep. Sizes are the power-of-two classes actually measured; anything between
// classes takes the lower class's verdict, which is the conservative reading.
//
// Las dos entradas D2H NO se re-midieron con este metodo y se dejan como estaban a
// proposito: `kGpuDirectD2HThreshold` esta compilado a 4 MiB en CudaRtHandler_memory.cpp y no
// lo gobierna esta puerta, asi que por debajo de 4 MiB los dos brazos toman el mismo camino y
// el barrido no puede separarlos. Cambiar una constante con datos confundidos seria peor que
// dejarla: lo medido es que entre 4 KiB y 2 MiB no hay separacion en D2H (todas las celdas
// dentro de +-1,3 %).
//
// This is a LOOKUP OF THE ANSWER. It is included to bound the quadrant policy, and a run
// using it must never be reported as an achievable configuration.
inline bool oracle_prefers_rma(bool h2d, bool pinned, std::size_t bytes) {
    if (h2d && pinned)   return bytes >= (16ull << 10);   // re-medido 2026-08-03, ver arriba
    if (h2d && !pinned)  return bytes >= (1ull << 20);
    if (!h2d && pinned)  return bytes >= (1ull << 20);
    return bytes >= (2ull << 20);
}

// The single entry point the transport calls. `bytes` is the payload the decision is about.
// The scalar floor is read from its OWN variable, not from `ucx_rma_min_bytes()`.
//
// This matters for the comparison and it is not a detail. `GVIRTUS_RMA_MIN_BYTES` does two
// jobs in this codebase: it gates the placement decision AND it sizes the slot pool (see the
// `floor_default` in the slot-capacity computation, and the pool-on-demand trigger in the AM
// receive path, which reads the BACKEND's copy of it). Lowering it so that small transfers
// have slots at all therefore also changes the decision -- so a "scalar vs quadrant"
// comparison driven by that one variable measures nothing: both arms end up with the same
// gate. Measured: with the variable at 8 KiB, scalar and quadrant produce identical numbers
// (1 MiB h2d pinned: 15.21 vs 15.22 GB/s).
//
// Splitting them lets every policy run with the pool built down to the smallest threshold in
// the table, while only the DECISION differs. That is the only way the three arms are
// comparable.
inline std::size_t scalar_floor_bytes() {
    static const std::size_t v = env_bytes("GVIRTUS_RMA_SCALAR_FLOOR", 4ull << 20);
    return v;
}

// EL SUELO DEL POOL NO PUEDE SER INDEPENDIENTE DE LA POLITICA, y esto esta medido, no razonado.
// Si la politica admite a 16 KiB y el pool se provisiona a >=32 KiB, se admite trafico que no se
// puede servir de forma nativa. El coste aparece EXACTAMENTE al cruzar el umbral mas pequeno de
// la tabla:
//
//     GVIRTUS_RMA_MIN_BYTES   8K      16K     32K     64K     128K    256K
//     llama 7B tg16 (t/s)     137,44  137,45  135,96  135,93  135,89  135,87
//                                     ^ sin coste  ^ -1,1 %, y PLANO a partir de aqui
//
// El pool mide 1,0 MiB a los DOS lados del escalon, asi que no es aprovisionamiento: es admitir
// lo que no se puede servir. Derivar el suelo de la politica lo hace imposible por construccion
// y elimina la trampa de las dos perillas que causo tres confusiones distintas en esta campana.
inline std::size_t policy_min_threshold() {
    switch (rma_policy()) {
        case RmaPolicy::Quadrant: {
            std::size_t m = quadrant_threshold(true, true);
            m = std::min(m, quadrant_threshold(true,  false));
            m = std::min(m, quadrant_threshold(false, true));
            m = std::min(m, quadrant_threshold(false, false));
            return m;
        }
        case RmaPolicy::Oracle:   return 16ull << 10;   // el menor cruce del oraculo
        case RmaPolicy::Scalar:
        default:                  return scalar_floor_bytes();
    }
}

inline bool prefer_rma(bool h2d, bool pinned, std::size_t bytes, std::size_t /*pool_floor*/) {
    switch (rma_policy()) {
        case RmaPolicy::Quadrant: return bytes >= quadrant_threshold(h2d, pinned);
        case RmaPolicy::Oracle:   return oracle_prefers_rma(h2d, pinned, bytes);
        case RmaPolicy::Scalar:
        default:                  return bytes >= scalar_floor_bytes();
    }
}

}  // namespace communicators
}  // namespace gvirtus

#endif  // GVIRTUS_COMMUNICATORS_RMAPOLICY_H
