#ifndef GVIRTUS_COMMUNICATORS_RMAPOLICY_H
#define GVIRTUS_COMMUNICATORS_RMAPOLICY_H

#include <cstdint>
#include <cstdlib>
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

inline RmaPolicy rma_policy() {
    static const RmaPolicy p = []() {
        const char *v = std::getenv("GVIRTUS_RMA_POLICY");
        if (v == nullptr || v[0] == '\0' || std::strcmp(v, "scalar") == 0)
            return RmaPolicy::Scalar;
        if (std::strcmp(v, "quadrant") == 0) {
            std::fprintf(stderr, "[GVS POLICY] four-quadrant placement "
                                 "(H2D pinned 8K / H2D pageable 1M / D2H pinned 1M / D2H pageable 2M)\n");
            return RmaPolicy::Quadrant;
        }
        if (std::strcmp(v, "oracle") == 0) {
            std::fprintf(stderr, "[GVS POLICY] ORACLE per-size placement "
                                 "(not deployable; upper bound only)\n");
            return RmaPolicy::Oracle;
        }
        std::fprintf(stderr, "[GVS POLICY] unknown value '%s'; using scalar\n", v);
        return RmaPolicy::Scalar;
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
          env_bytes("GVIRTUS_RMA_MIN_H2D_PINNED",   8ull << 10) },
    };
    return t[h2d ? 1 : 0][pinned ? 1 : 0];
}

// Oracle: the measured winner per size class, from the 2026-08-01 sweep. Sizes are the
// power-of-two classes actually measured; anything between classes takes the lower class's
// verdict, which is the conservative reading.
//
// This is a LOOKUP OF THE ANSWER. It is included to bound the quadrant policy, and a run
// using it must never be reported as an achievable configuration.
inline bool oracle_prefers_rma(bool h2d, bool pinned, std::size_t bytes) {
    if (h2d && pinned)   return bytes >= (8ull << 10);
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
