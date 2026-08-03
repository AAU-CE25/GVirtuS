---
title: "The four reviewer items: what was done, what is bounded, and what is still owed"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

Four items were raised, two marked critical. This document answers each with the evidence and
the file it lives in, and is explicit where an item is **bounded rather than closed** or **still
open**. Nothing here is a plan; everything is either done or named as not done.

| # | item | status |
|---|---|---|
| C1 | NIC-to-GPU visibility: negotiate + flush, **or** reduce the claim | **claim reduced**, `CONTRACTS.md` §6 |
| C2 | make the contracts concrete | **done**, `CONTRACTS.md` |
| D1 | 7B sweep: knee, p99, N summary, longer windows | **all four addressed**; the knee sweep found no capacity separation, which is itself the answer -- see §D1 |
| D2 | decide what fairness is | **decided: fourth contribution**, `PAPER_OPENING.md` §3b |

# C1. NIC-to-GPU visibility -- the claim is reduced, deliberately

**We did not implement negotiation and flush. We reduced the claim instead**, and the reason is
in the measurement: the flush was tried on 2026-07-25 and it reproduced the bug under
investigation *identically* while roughly **doubling issue time (310 ms against 166 for
6 x 64 MB)**. Shipping an unconditional 1.9x cost for no demonstrated benefit would have been
the worse choice.

**What the system actually does.** On the GPUDirect path the client PUTs into the server's GPU
shadow and then sends a small `RmaPosted` active message; a handler consumes the region with
`cudaMemcpyDeviceToDevice`. A CUDA read therefore follows a NIC write to the same device memory
with only an AM between them. Two assumptions carry it:

- **A1, ordering.** The AM is observed only after the WRITE has landed. **Not a UCX guarantee**:
  `ucp_put_nbx` completion is *local*, and the AM is not ordered against writes in flight -- the
  code says so at the call site. It holds here because both travel **one RC queue pair** and IB
  delivers in order on a QP. It would break under multi-rail striping, or if AMs took a different
  transport than the PUT, which our own `UCX_TLS` permits.
- **A2, visibility.** The GPUDirect read-after-write question. The documented mitigation is a
  flush read; we perform none.

**The reduced claim, verbatim, for the paper:**

> No visibility failure was observed across **2 643 921 RMA admissions** with end-to-end
> verification where the workload performs it (XSBench: identical checksum 408237 across every
> arm, native included), on ConnectX-7 / RoCEv2, **UCX 1.20.0**, drivers 580.95.05 and 560.35.05,
> **one RC lane per connection**. A1 and A2 are configuration-dependent and **neither negotiated
> nor checked at runtime**. We do not claim the general guarantee.

**It is structurally visible, not buried in prose.** `CONTRACTS.md` §4 lists nine invariants
with their discharge points; this one is **I10 and is deliberately outside that table**, in §6,
so a reader cannot mistake it for something that was proved. `PAPER_OPENING.md` §4 lists it
among the results that must never appear without their qualifier, and its checklist has a line
for it.

**What would discharge it, specified and priced** (`CONTRACTS.md` §6.5): capability negotiation
at the handshake -- the same shape already used to gate GPUDirect on endpoints that actually
negotiated an RDMA lane, and the field fits the existing `RmaSetup` header -- plus a
**conditional** flush applied only when negotiation says ordering is not guaranteed, at the
measured 1.9x.

# C2. Concrete contracts -- done

`CONTRACTS.md`, 266 lines, structured exactly as asked.

| asked for | where | what it contains |
|---|---|---|
| **data structure** | §1 | the four wire messages with their per-type field reuse; the slot tag `(epoch << 32) \| generation` as the identity of one *transfer*; `RemoteSlot` in full; the server pool and why `server_idx` **is** the vector position |
| **selector pseudocode** | §2 | `prefer_rma(h2d, pinned, bytes)` with all three policies, the 2x2 threshold table, why a scalar is the wrong *shape* rather than merely suboptimal, and the two-jobs trap in `GVIRTUS_RMA_MIN_BYTES` that makes a naive scalar-vs-quadrant sweep measure nothing |
| **state machine** | §3 | the slot (`Free`/`InFlight`, returned by `SlotConsumed` and **not** by local put completion) and the layout (install-now vs **PARK**, installed on drain) |
| **invariants + discharge** | §4 | nine invariants, each with the code fragment that discharges it and, where one exists, the counter proving the discharge point is reached |

**The evidence column is the part worth reading.** Several invariants are backed by a measured
count, not an assertion: the epoch guard rejects **126--280 stale acknowledgements per run**
under fault, the generation guard **38--114**, the park fires **81 times per run**,
`decline_capacity` is **74 in 2.64 M admissions**, and the backpressure fix turned a 30 470 ms
stall into 41 ms.

**Two honesty notes are in the document rather than omitted.** I3 and I4 are *layered*, not
redundant -- ablating the epoch guard alone changes nothing observable because the generation
guard catches what it lets through. And line references drifted once while the document was
being written, so every reference now carries its code fragment and the commit it is valid at.

# D1. The 7B sweep

**p99 and the N=2,4,8 summary: done** (`LLAMA-7B_RESULTS.md` §3d), re-derived from per-request
JSONL that was already on disk -- no re-run needed, 36 cells.

**It corrected a framing error rather than merely adding rows.** §3c reads *"remoting costs an
order of magnitude of tail latency at light load"*; that comes from the N=8, lambda=0.5 cell and
is true **only there** (0.10x). At **N=2 and N=4 the same load gives 0.96x** -- the three systems
are indistinguishable. lambda is a *total* rate, so at N=8 each server receives 0.0625 req/s and
idles, which is the one regime where a network round trip has nothing to hide behind. The
penalty belongs to the idle regime, not to the tenant count.

**What strengthened**: the advantage above the knee holds at every N *and* at p99. native/Gusto
p95 at lambda=1.0 is **2.06x / 2.32x / 2.51x** for N=2/4/8; at lambda=1.5, 1.77x / 2.11x / 2.29x.

**p99 is reported with its limit stated**: at 100--190 pooled requests per cell it is the first
or second largest observation, so it is quoted because it **agrees** with p95 everywhere, not
because it is separately reliable.

**Knee and longer windows: a dedicated sweep was run on 2026-08-03** and its result is written
up in `LLAMA-7B_RESULTS.md` §3e. Design: lambda in {0.55, 0.60, 0.65, 0.70} -- the untested gap
between the load that always meets the SLO (0.50) and the one that meets it only sometimes
(0.75) -- with the window scaled as `max(90, 120/lambda)` so **every point sees at least 120
offered requests instead of 40**, which is the same change that makes p99 mean something. Three
systems, three repetitions, N=8 first so the most informative tenant count completes even if the
queue is cut short.

# D2. Fairness is the fourth contribution -- decided

The previous decision was *"its own subsection, presented as a contribution"*, taken when the
**cause was unknown**. It no longer is, and the decision is upgraded rather than repeated
(`PAPER_OPENING.md` §3b, `NOVELTY.md` §2.4).

What can now go in a contributions list:

1. **The effect.** At N=8 in miniBUDE one tenant runs at **1.00x** its single-client rate, as if
   alone on the machine, while another runs **4.87x** slower; native shares to within 1.03x.
   Reproduced in XSBench (5.98x), identical over TCP.
2. **Three mechanisms excluded by controls** -- and the decisive one is MPS: **eight clients in a
   single CUDA context share to within 1.02x**, so a shared context is not sufficient and the
   obvious explanation is wrong.
3. **The mechanism**: the backend performs no arbitration. FCFS plus self-clocked clients lets an
   early lead compound instead of decaying.
4. **A fix, measured**: deficit round-robin at the launch point, **5.02 -> 3.09** inequality
   (CI95 [2.05, 4.34], excluding the baseline) for **+0.6% makespan**, pulling the leader off its
   solo rate (216.4 -> 75.9 GFLOP/s).

**Why a contribution and not a limitation**: it is a property of API-remoting backends that every
aggregate metric hides -- cohort makespan reads 1.15x where per-tenant runtime reads 6.0x -- so
any prior evaluation reporting makespan alone would have missed it. A measured negative result
*with* an identified cause *and* a remedy costing 0.6% is not a limitations paragraph.

**What must travel with it**: it is workload-dependent (1.03x in BabelStream and CloverLeaf at
the same N), and it is about **fixed-work cohorts, not serving** -- in serving, Gusto and native
are statistically equivalent (paired Jain -0.0027, CI95 [-0.0079, +0.0009]).

Figure: `figures/fig6_n1_arbitraje.pdf`.

# What is still owed

Named here so it is not mistaken for complete:

- **I10 is bounded, not discharged.** Negotiation and the conditional flush are specified and
  priced but not implemented.
- **The N7 demonstration reaches the dangerous state but not corrupted bytes.** 114 premature
  frees per run, zero corrupted bytes across five configurations; the missing sixth condition is
  stated in `N7_EPOCH.md` §5.
- **Transport provenance is still absent from the sidecar** (`GAPS.md` §1).
- **Second hardware is only partial**: the cross-driver check covers the *memory mechanism*
  (431 MiB on dpu-01, 429 on dpu-02), not the transport, the placement policy or the lifetime
  protocol.
