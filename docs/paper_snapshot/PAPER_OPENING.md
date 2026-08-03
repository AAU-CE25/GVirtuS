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
consequences; the fairness result is a bound; the memory result has no established mechanism.

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
| Capacity under an SLO | new | **evaluation** | capacity itself does **not** separate the systems (identical at 0.50 req/s, 57.6 t/s); what separates them is the tail and the goodput above the knee |
| Per-tenant fairness (negative) | new | **its own subsection, presented as a contribution** | a measured bound on this class of system |

# 4. Three results that must never appear alone

Each of these is true only with its qualifier attached in the **same sentence**. Splitting them
across paragraphs is how a paper becomes indefensible under review.

- **Multi-tenant throughput.** *"1.37x over default native, and 97--99% -- parity -- against
  MPS-configured native."* Quoting only the first is a straw man; quoting only the second
  discards a practical result. **Never x1.42**: that was the maximum of three runs; the mean is
  x1.37.
- **The memory saving.** *"~461 MiB per tenant, and MPS does not reproduce it, so the mechanism
  is not context consolidation and is not established."* The saving is closed; the explanation
  is open.
- **Fairness.** *"Serving fairness is statistically equivalent to native; under equal fixed
  work it is not -- one tenant runs at its single-client rate while another is 4.87x slower."*
  Workload-dependent, and the dependence is the point.

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
(MPS parity on throughput; no established mechanism for memory; fixed-work fairness worse than
native). Then the roadmap.

**Nothing else on the first two pages.** Every number in §3 above has a home in the evaluation.

# 7. Checklist against the draft, once it exists

- [ ] Exactly one claim in the abstract that is a mechanism rather than a measurement.
- [ ] No number appears before the section that measures it.
- [ ] The three qualified results of §4 carry their qualifier in the same sentence.
- [ ] No x1.42 anywhere. No saturated goodput quoted as a rate without its window.
- [ ] Every prior-work sentence traceable to a citation (see `NOVELTY.md`, the **[CITE]** marks).
- [ ] The fairness limitation appears in the contributions list, not only in a limitations
      paragraph -- it is a finding, and burying it invites a reviewer to "discover" it.
