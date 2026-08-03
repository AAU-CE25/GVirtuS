---
title: "The first two pages -- one lead claim, and where everything else goes"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

> There is no paper draft in the repository (no `.tex`, no `paper.pdf`), so this is not an edit
> of an opening -- it is the decision of **which result leads and which are demoted**, which the
> audit has just changed. Once the draft exists, this document is the checklist against it.

# 1. The problem, stated as a count

Nine results currently compete for the reader's attention in the first pages:

1. GPU-resident RMA at 92.5--93.0% of the network ceiling
2. GPU RMA 1.78--1.80x host RMA at 1 GiB
3. cuDF 13.6--15.4% over the same protocol with a host bounce
4. The TinyLlama 2x2 (async versus graphs)
5. Multi-tenant throughput 1.37x over default native
6. ~461 MiB/tenant of memory saved
7. The four-quadrant placement policy and its oracle bound
8. The lifetime protocol and its ablations
9. Capacity under an SLO, and the fairness finding (both new)

**Nine is too many, and three of them no longer survive contact with their own controls.**
A reader who meets all nine in two pages remembers none.

# 2. The lead: one claim, one number, one figure

> **A CUDA interception layer knows the direction and the memory kind of every transfer for
> free, and those two bits are enough to choose between four data paths whose measured
> crossovers span three orders of magnitude -- 8 KiB for pinned H2D against 2 MiB for pageable
> D2H -- and whose sign reverses between directions. No single threshold can be right.**

Everything else in the paper is either evidence for this or a bound on it.

**Why this one and not the throughput numbers.** It is the only claim that (a) is a
*mechanism* rather than a measurement, (b) survives every control we ran, and (c) cannot be
obtained by a system that does not sit at the API boundary. The throughput results are
consequences; the fairness result is a mechanism with a measured fix (§3b); the memory result is
closed down to its line item, the 429 MiB per-process CUDA context.

**The one figure:** the crossover table with the sign reversal, plus the oracle bound showing
how much the deployable step function leaves. Nothing else on page 1.

# 3. What gets demoted, and where it goes

| result | current | **proposed** | why |
|---|---|---|---|
| Four-quadrant placement + oracle | scattered | **lead, page 1** | the mechanism; survives every control |
| Lifetime protocol + ablations | §9 | **page 2** -- it is what makes the lead *safe* | `pointer_keyed` corrupts 65 280 bytes: correctness, not performance |
| RMA at 92.5--93% of ceiling | competing | **evaluation, one line** | a ceiling check, not a finding |
| GPU RMA 1.78--1.80x host RMA | competing | **evaluation, folded into the crossover figure** | it *is* the crossover, at one size |
| cuDF 13.6--15.4% | competing | **evaluation, the end-to-end case** | keep: it is the workload that exercises the lead |
| TinyLlama 2x2 | competing | **evaluation** | orthogonality already qualified; not an opening claim |
| Multi-tenant 1.37x | competing | **evaluation, with the MPS parity in the same sentence** | see §4 |
| ~461 MiB/tenant | competing | **evaluation** | see §4 |
| Capacity under an SLO | new | **evaluation** | capacity itself does **not** separate the systems (identical at 0.50 req/s, 51.2-51.7 t/s over n=3); what separates them is the tail and the goodput above the knee (+9.9% at lambda=1.0, +27.5% at 1.5, both CIs excluding zero) |
| Per-tenant fairness | new | **the fourth contribution, page 2, with its own section** | see §3b: it is no longer a negative result, it is a mechanism plus a fix |
| The contracts | absent | **page 2, beside the lifetime protocol** | `CONTRACTS.md`: nine invariants with discharge points, and the tenth stated as a bounded assumption |

## 3b. Fairness is a contribution, not a limitation -- decided 2026-08-03

The earlier line read *"its own subsection, presented as a contribution: a measured bound on
this class of system."* That was the right call when the **cause was unknown**. It no longer is,
and the decision should be upgraded rather than repeated.

What can now be put in a contributions list, in one sentence each:

1. **The effect.** Under equal fixed work, remoting does not share the GPU fairly: at N=8 in
   miniBUDE one tenant runs at **1.00x** its single-client rate -- as if alone on the machine --
   while another runs **4.87x** slower. Native shares to within 1.03x. Reproduced in XSBench
   (5.98x) and identical over TCP.
2. **Three mechanisms excluded by controls**, one of them by MPS: eight clients in a *single*
   CUDA context share to within **1.02x**, so a shared context is not sufficient and the obvious
   explanation is wrong.
3. **The mechanism**: the backend performs no arbitration. FCFS plus self-clocked clients means
   an early lead compounds instead of decaying.
4. **A fix, measured**: deficit round-robin at the launch point, **5.02 -> 3.09** inequality
   (CI95 excluding the baseline) for **+0.6% makespan**.

**Why this earns a contribution slot:** it is a property of API-remoting backends that *every*
aggregate metric hides -- cohort makespan reads 1.15x where the per-tenant runtime reads 6.0x --
so any prior evaluation reporting makespan alone would have missed it. Reporting it as a
limitation would understate work that has an identified cause and a demonstrated remedy.

**What must travel with it**, in the same sentence, because it is workload-dependent: *"4.87x in
miniBUDE and 5.98x in XSBench against 1.03x in BabelStream and CloverLeaf, same N, same
system"*, and *"in serving, Gusto and native are statistically equivalent (paired Jain
-0.0027, CI95 [-0.0079, +0.0009])."* The finding is about fixed-work cohorts, not about serving.

# 4. Three results that must never appear alone

Each of these is true only with its qualifier attached in the **same sentence**. Splitting them
across paragraphs is how a paper becomes indefensible under review.

- **Multi-tenant throughput.** *"1.37x over default native, and 97--99% -- parity -- against
  MPS-configured native."* Quoting only the first is a straw man; quoting only the second
  discards a practical result. **Never x1.42**: that was the maximum of three runs; the mean is
  x1.37.
- **The memory saving.** *"~461 MiB per tenant, because a remoted tenant never creates a CUDA
  primary context -- 429 MiB, measured -- and MPS cannot match it because MPS shares scheduling,
  not per-client context state."* Both the size and the mechanism are closed; quoting the size
  without the mechanism invites the reviewer to supply a wrong one, which is what happened to us.
- **Fairness.** *"Serving fairness is statistically equivalent to native; under equal fixed
  work it is not -- one tenant runs at its single-client rate while another is 4.87x slower."*
  Workload-dependent, and the dependence is the point. Since 2026-08-03 it also travels with its
  cause and its remedy (§3b), so it must never be written as an unexplained defect.
- **GPU-resident RMA.** *"No visibility failure across 2.64 M RMA admissions with end-to-end
  admissions, with the verifying subset bit-exact, on UCX 1.20.0 with a single RC lane -- we do
  not claim the general
  guarantee."* The bound is not optional garnish: without it the sentence asserts an ordering
  property the UCX API does not provide (`CONTRACTS.md` §6).

# 5. What comes out of the opening entirely

- **The saturated multi-tenant goodput as a rate** (277.3 / 302.9 t/s). The window is 55 s and
  the divisor is 30 s; the ratio survives, the absolute number is not a steady-state rate.
  Replaced by the capacity-under-SLO figure.
- **Any Jain index over per-tenant throughput.** Demand imbalance reaches 7.0x from the arrival
  draw alone; normalised, fairness is 1.0000 exactly.
- **The XSBench MPS row** and the obsolete XSBench cohort table -- both withdrawn
  (`XSBENCH_RESULTS.md` §2).
- **XSBench as a performance result.** It is a *null* result and belongs in the section that
  bounds where the data path does not apply, which is its most useful role.

# 6. The two-page skeleton

**Page 1** -- the problem (a transfer's best path depends on two properties only the API layer
knows); the crossover table with the sign reversal; the oracle bound; the one-sentence
contribution.

**Page 2** -- why it needs a lifetime protocol (slot reuse across epochs and generations), the
three ablations in one small table, and the honest register: what the mechanism does *not* buy
(MPS parity on throughput; fixed-work fairness worse than native, with its mechanism and its
0.6%-cost fix; and **NIC-to-GPU visibility bounded rather than guaranteed**, `CONTRACTS.md` §6).
Then the roadmap.

**Nothing else on the first two pages.** Every number in §3 above has a home in the evaluation.

# 7. Checklist against the draft, once it exists

- [ ] Exactly one claim in the abstract that is a mechanism rather than a measurement.
- [ ] No number appears before the section that measures it.
- [ ] The three qualified results of §4 carry their qualifier in the same sentence.
- [ ] No x1.42 anywhere. No saturated goodput quoted as a rate without its window.
- [ ] Every prior-work sentence traceable to a citation (see `NOVELTY.md`, the **[CITE]** marks).
- [ ] The GPUDirect claim carries its bound; nowhere does it read as a visibility guarantee.
- [ ] Fairness is in the contributions list with its mechanism and its fix, not in limitations.
- [ ] The invariant table of `CONTRACTS.md` §4 appears, with **I10 visibly outside it**.
- [ ] The fairness limitation appears in the contributions list, not only in a limitations
      paragraph -- it is a finding, and burying it invites a reviewer to "discover" it.
