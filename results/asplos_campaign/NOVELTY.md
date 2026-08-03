---
title: "Novelty defence — what is new, what is conceded, and the evidence for each"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

> **How to use this document.** It is the answer to *"this is an integration paper"*. Every
> claim below is anchored in this project's own code and measurements, with the file or the
> raw data named. **It deliberately makes no factual claim about what Gleam, rCUDA, Cricket or
> any other system does** — those systems are not on this testbed, the brief forbids requiring
> a comparison against them, and asserting their internals from memory is exactly the failure
> this campaign exists to avoid. Where a specific prior-work statement is needed, the text
> marks it **[CITE]** and says what must be verified in the actual paper before the sentence
> can be written.

# 1. Concede first, and concede early

The strongest version of this paper concedes the obvious in the first paragraph rather than
letting a reviewer find it:

- **API remoting is not new.** Intercepting the CUDA API and forwarding it to a remote GPU is
  a decade-old idea, and GVirtuS itself is prior work — this paper builds on it.
- **Using RDMA as the transport is not new.**
- **Consolidating multiple tenants into one CUDA context is not new**, and this paper proves it
  by measuring the alternative: MPS-configured native reaches 97–99% of the remoted
  throughput (`LLAMA-7B_RESULTS.md` §1b). We say so ourselves rather than compare only against
  the weaker default-native baseline.

Conceding these costs nothing, because none of them is the contribution.

# 2. What is new, and the evidence

## 2.1 The data path is chosen from semantics the runtime already knows

The contribution is not *that* there are several data paths — it is that the **choice is made
from information only the CUDA runtime layer has**, at no probing cost.

**The measurement that makes it a contribution rather than a knob.** A single scalar threshold
is not merely suboptimal, it is the **wrong shape**. The measured crossovers are:

| | H2D | D2H |
|---|---|---|
| pinned host | **8 KiB** | 1 MiB |
| pageable host | 1 MiB | 2 MiB |

Three orders of magnitude between the corners, and **the sign reverses below 1 MiB for D2H**
(at 64 KiB, RMA is 0.44× of active messages — actively harmful). No scalar satisfies both
directions. Source: `include/gvirtus/communicators/RmaPolicy.h`, the 2026-08-01 sweep.

**Why the runtime layer is the right place, and the transport is not.** Both bits are free
there and unavailable below:

- **Direction is structural** — `WriteIov` *is* the H2D path, `GetFromRemoteGpu` *is* D2H. The
  transport does not have to be told.
- **Pinned versus pageable comes from an interval map the frontend already maintains** for
  `cudaHostAlloc`/`cudaFreeHost`. Deliberately *not* used: `cudaPointerGetAttributes`, because
  it is itself remoted — calling it per transfer would cost an RPC to save an RPC. That
  sentence is the contribution in miniature.

**And the bound is measured.** An oracle policy — per-size lookup of the measured winner, not
deployable because it needs the answer in advance — is implemented alongside, so the paper can
state how much the deployable step function leaves on the table instead of claiming optimality.

**[CITE] required:** a sentence establishing that prior remoting systems place transfers by a
size threshold or by a single mechanism, *verified in their papers*, not asserted. If that
cannot be verified, weaken the claim to "we are not aware of a prior system that…" and move on
— the measurement stands regardless of who did what.

## 2.2 A slot-lifetime protocol with ablations that show each guard is load-bearing

The reusable-registration problem — a remote slot reused while an old acknowledgement is still
in flight — is a correctness problem, not a performance one. The protocol keys a slot on
**allocation identity, connection, epoch, slot id and generation**, and every component has an
ablation:

| guard | ablation | result |
|---|---|---|
| allocation-lifetime invalidation | `pointer_keyed` — key the registration cache on pointer value | **65 280 corrupted bytes**; zero without the ablation |
| generation checking | `hold_ack` with the guard off | acknowledgement releases a live slot (`ack_on_free=1`); with the guard, 0 |
| epoch checking | `no_epoch` | `ack_epoch_dropped` falls 1 → 0, nothing else changes |

**The honest part, and it strengthens the paper.** The epoch guard is **reachable but not
exercised by any workload**: cuDF re-advertises slot ids `[0-7]` unchanged across epochs, so
the dangerous state occurs — but the *damage* did not, in **0 deliveries across 186
opportunities** over three workloads. The paper says the guard is a necessary invariant, not an
observed save. A reviewer who finds that themselves is a problem; a reviewer who reads us
saying it is not.

## 2.3 GPU-resident RMA, and the completion-versus-visibility question stated rather than assumed

The bulk path lands **in device memory** rather than in a host bounce buffer. The contribution
is not the mechanism but the fact that the paper asks the right question about it: **"PUT
completed" is not the same as "a kernel may consume the data"**, and the guarantee depends on
the concrete driver and UCX version.

**Bounded, 2026-08-03** (`CONTRACTS.md` §6). It is stated as invariant **I10 and explicitly not
discharged**, which is the honest form. Two assumptions carry it -- **A1**, that the `RmaPosted`
active message is observed only after the RDMA WRITE has landed (not a UCX guarantee: PUT
completion is *local*; it holds here because both travel one RC queue pair), and **A2**, the
GPUDirect read-after-write question, for which we perform no flush. What the paper claims is
therefore *"no visibility failure was observed across 2.64 M RMA admissions with end-to-end
checksum validation, on UCX 1.20.0 with a single RC lane"*, together with the statement that
both assumptions are configuration-dependent and neither negotiated nor checked at runtime.
**Not** a guarantee. §6.5 names what would discharge it -- capability negotiation plus a
*conditional* flush -- and prices the flush at the measured **1.9x issue time**.

## 2.4 A negative result about this class of system that we found in our own work

Under equal fixed work, remoting **does not share the GPU fairly**: in miniBUDE at eight
tenants one tenant runs at 1.001× its single-client rate while another runs 4.87× slower, where
native and native+MPS share to within 1.03×. It appears identically over TCP, so it is not the
data path — it is how the backend orders work from concurrent connections. Independently
reproduced in XSBench (5.98× against native's 1.001×).

**This is a contribution, not a weakness**, and it should be presented as one: it is a
measured, controlled, mechanism-level finding about API-remoting backends that the aggregate
numbers hide entirely. Cohort makespan shows 1.15× where the internal runtime shows 6.0×
(`XSBENCH_RESULTS.md` §3). Any prior evaluation reporting only makespan would have missed it.

**Promoted to a full fourth contribution, 2026-08-03, because it is no longer only a negative
result.** When this was written the cause was unknown, and "we report it honestly" was the most
that could be said. `N1_SCHEDULER.md` now closes it end to end:

1. **Four candidate mechanisms tested, three refuted** -- the data path (identical over TCP),
   the shared legacy stream (per-connection streams make it *worse*, 3.46 -> 6.39), transport
   lock contention (each connection owns its worker and mutex), and **a shared CUDA context**,
   refuted by the MPS control: MPS puts eight clients in one context and shares to within
   **1.02x**.
2. **The mechanism identified**: the backend does not arbitrate. FCFS service plus self-clocked
   clients means the tenant that gets marginally ahead re-submits first and keeps its turn. MPS
   is fair because *its* server arbitrates; the driver is fair between contexts because it
   multiplexes them; this backend does neither.
3. **Confirmed by intervention, not by inference**: deficit round-robin at the launch point cuts
   the inequality **5.02 -> 3.09** (CI95 [2.05, 4.34], excluding the baseline) for **+0.6%
   makespan**, and pulls the leading tenant off its solo rate (216.4 -> 75.9 GFLOP/s).

A measured negative result, a mechanism, and a fix that costs 0.6% is a contribution in its own
right -- not a limitations paragraph. The residual is stated too: launch-count arbitration
equalises *submissions*, not *GPU time*.

# 3. The structural answer to "integration paper"

The counting is in `IMPLEMENTATION_COVERAGE.md` and it supports the argument rather than
undermining it:

- **~16 000 lines of plugin marshalling** across cudart and cudadr, at ~50 lines per API. That
  part **is** integration work and the paper should say so.
- **~5 200 lines of transport and policy** — `UcxCommunicator` (4 936), `RmaPolicy.h` (138),
  `AblationGate.h` (140) — which is where every claim in the paper lives.

The ratio is the argument: **the mechanical part is three times the size of the contribution,
and no amount of the former substitutes for the latter.** A system with 100% API coverage and a
scalar threshold would have more integration and none of this paper's results.

Two further facts that a reviewer can check cheaply:

- **The fast path essentially never gives up**: 80 fallbacks in 2 643 921 RMA admissions,
  0.003%, over 14.78 million operations and 620 teardowns. A placement policy that were merely
  a heuristic knob would not behave like that.
- **Coverage is stated honestly, including where it is bad**: 46.6% of the exported CUDA-family
  ABI is real, and cusolver is 1.7% — ABI padding, not an implementation. We say that
  ourselves.

# 4. What must not be claimed

- **Not** that this is the first system to remote CUDA over RDMA.
- **Not** that context consolidation is our mechanism for the multi-tenant throughput win —
  MPS reproduces it, and we show that.
- **Not** that MPS's failure to reproduce the memory saving refutes a context explanation. It
  does not, and the earlier version of this bullet had it backwards. MPS consolidates
  *scheduling*; each MPS client still creates its own CUDA primary context and pays **429 MiB**
  for it, measured. The memory mechanism is therefore **closed**: the saving is the per-process
  context, which a Gusto tenant never creates (`LLAMA-7B_RESULTS.md` §2).
- **Not** any statement about another system's internals that has not been verified in its
  paper. Every such sentence in the submission should be traceable to a citation, and the
  places that need one are marked **[CITE]** above.

# 5. The one-paragraph version

> API remoting for CUDA is prior work, and so is carrying it over RDMA. What this paper
> contributes is a **placement contract**: the interception layer knows the direction and the
> memory kind of every transfer at no cost, and those two bits decide between four data paths
> whose measured crossovers span three orders of magnitude and whose sign reverses between
> directions — so no single threshold can be right. Making that choice safe across connection
> teardown and slot reuse requires a lifetime protocol keyed on allocation identity, epoch and
> generation, and we show by ablation that removing any one of those corrupts data or releases
> live slots. We also report what the mechanism does **not** buy: MPS-configured native matches
> our multi-tenant throughput, our memory advantage has no established mechanism, and under
> equal fixed work our backend shares the GPU markedly less fairly than native does.
