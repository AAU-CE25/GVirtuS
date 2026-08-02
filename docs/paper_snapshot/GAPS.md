---
title: "Artifact gaps -- what is missing, why it matters, and the minimum experiment"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

A living list of what the package does **not** contain. Each entry says what is missing, what
claim it blocks, whether it is reconstructible from the server or needs re-measuring, and the
minimum experiment that closes it. Nothing here is estimated or filled in.

# 1. Capacity under an SLO per N -- **rep 1 MEASURED, reps 2 and 3 pending**

Swept 2026-08-03: N  en  {2,4,8} x lambda total  en  {0.25 0.5 0.75 1.0 1.5 2.0}, native and Gusto paired
cell by cell, lambda order randomised, **36 points**. Full result in `LLAMA-7B_RESULTS.md` §3 and
`LLAMA_SLO_capacidad.csv`; figure `figures/fig5_slo_capacidad.pdf`.

| N | max lambda under SLO | native goodput | Gusto goodput |
|---:|---:|---:|---:|
| 2 | 0.50 | 51.2 | **55.5** (+8.4%) |
| 4 | 0.50 | 51.2 | **55.5** (+8.4%) |
| 8 | 1.00 | 98.1 | **115.2** (+17.4%) |

**What is still missing, in order of importance:**

1. **Repetitions 2 and 3.** With n=1 there are no confidence intervals. The direction is
   consistent across all three N, but the difference has **no statistical backing** until they
   land. Launched with a backend restart per cell, which rep 1 did not have.
2. **Resolution around the knee.** Capacity measured in lambda comes out **identical** for both
   systems at every N, because the grid steps are 50% (0.5->0.75 and 1.0->1.5) and both cross the
   SLO inside the same interval. That is not a demonstrated tie: it is a resolution limit. A
   fine sweep between those two points is what would let this metric decide.
3. **The native+MPS arm** is not in this grid. For throughput the MPS control already closed
   the gap in llama, so an honest comparison under an SLO should include it before the headline
   is published.
4. **Transport provenance was not recorded**: the sidecar's `transport` field is empty because
   `bench.py` reads `GVIRTUS_CONFIG` from the invoking process and that variable lives inside
   the container. The arm is fixed by the label and the harness, not by the sidecar. To correct
   before the next packaging.

Harnesses: `~/mt_slo_sweep.sh`, `~/run_slo_grid.sh`, `~/analiza_capacidad.py`,
`~/figura_capacidad.py`. `bench.py` already emits the strict-window metrics.

# 2. Native+MPS memory footprint per tenant -- **MEASUREMENT CLOSED, MECHANISM OPEN**

Measured. Native and native+MPS in the same session on the same GPU, N  en  {1,2,4,8}, per-tenant
= (peak - baseline)/N with 25 samples per point after exercising every pod:

| N | native | native+MPS | Gusto | MPS saves | **Gusto saves** |
|---:|---:|---:|---:|---:|---:|
| 2 | 4948.5 | 4942.5 | 4503 | 6.0 | **445.5** |
| 4 | 4948.0 | 4942.8 | 4492 | 5.2 | **456** |
| 8 | 4947.9 | **4942.9** | 4487 | **5.0 (0.1%)** | **461 (9.3%)** |

**The result is decisive and it refutes the mechanism the paper gave.** `RESULTS.md` §8
attributed the saving to N tenants sharing one CUDA context. MPS consolidates contexts by
construction, so if that were the mechanism MPS would reproduce it. It saves 0.1%.

Consequence: the memory result **survives the MPS control**, unlike the throughput result where
MPS closed the whole gap. The advantage belongs to the remoting architecture, not to context
consolidation. What the correct mechanism *is* remains **unestablished**: the hypothesis is that
under remoting the N frontend processes hold no device memory at all, whereas under MPS there
are still N processes making their own allocations with only the scheduling context shared.
Consistent with the numbers, **untested**, and it must not be written as the explanation.
The experiment that would settle it: dump the backend own device allocations per connection
at the N=8 peak and compare the total against a native pod, so the 461 MiB is attributed to a
line item instead of inferred from a difference of totals.

**So this entry is half closed.** The size of the saving is measured and reproducible; the
cause is not. The defensible sentence is: remoting saves ~461 MiB per tenant, and MPS does
not; why is not established.

Data: `mem_footprint.csv`. Harness: `~/mem_footprint.sh`. Written up in
`LLAMA-7B_RESULTS.md` §2.

# 3. Cause of the uneven sharing -- **missing, needs instrumentation**

It is established **that** the sharing is uneven and **by how much** (`FAIRNESS_RESULTS.md`):
at N=8 in miniBUDE, 34% of iterations are served with no wait while others queue behind up to
ten quanta. **Why** is not.

The current data cannot distinguish three hypotheses:

- connection arrival order (is the favoured tenant always the one that connects first?);
- monopolisation of a backend thread or stream;
- head-of-line blocking in the dispatcher.

**Minimum experiment.** Record in the backend, per connection, the queue-entry and dispatch
instant of every RPC; repeat miniBUDE at N=8. Without that trace one cannot separate backend
wait, GPU wait and RPC time.

# 4. Per-tenant serving timeline -- **not reconstructible, needs re-measuring**

The multi-tenant llama campaign is from 2026-07-26 and carries **no per-request timestamp**
(the `tc_rel_s` / `tarr_rel_s` fields were added 2026-08-02). So these are **not**
reconstructible per tenant: completions inside the window against completions during the drain,
first completion, and longest no-progress interval.

What **was** reconstructed and is packaged: demand-normalised service fractions, per-tenant SLO
attainment, and the permutation null (`llama_fairness_por_tenant.csv`,
`llama_fairness_por_corrida.csv`).

Any future re-run already emits the fields, since the patch is applied.

# 5. Data gaps, closed as such

| gap | status |
|---|---|
| **CloverLeaf without its figure of merit** | it writes to `clover.out`, which is not in the artifact. Only per-tenant wall time exists. **Not reconstructible without a re-run.** |
| **XSBench TCP N=8** | all 64 tenants lack a `Runtime:` line, some record `duration_s = 0.0`. 8 cohorts dropped and **counted**. Classified **E**. |
| **The XSBench MPS row** | the values 25.11 / 50.15 / 98.40 have no raw file in any tree or CSV, and their comparison baseline was the obsolete table. **Withdrawn**, see `XSBENCH_RESULTS.md` §2. |
| **XSBench per-tenant fairness** | **closed 2026-08-02.** The per-client stdout was still on the server; reconstructed into `XSBench_fairness_por_tenant.csv` (306 rows). No re-run needed. |
| **XSBench raw tree partly overwritten** | the density experiment at 1.25e9 reused the `seed*/` directories. Filter on `lookups` before re-deriving. The published tables are already filtered. |

# 6. What to package in the next tar

Files produced on 2026-08-02/03 that were **not** in the previous package:

```
FAIRNESS_RESULTS.md / .pdf              the audit: method and cross-workload synthesis
LLAMA-7B_RESULTS.md / .pdf              all llama detail, consolidated
tenants_canonico.csv                    3688 per-tenant rows, four workloads
fairness_trabajo_fijo_por_corrida.csv   493 cohorts
fairness_trabajo_fijo_resumen.csv       median across cohorts
llama_fairness_por_corrida.csv          39 runs, with the permutation null
llama_fairness_por_tenant.csv           169 tenant-run rows
tabla_D_minibude_por_tenant.csv         352 rows, per-iteration timeline
XSBench_fairness_por_tenant.csv         306 per-tenant rows
LLAMA_SLO_capacidad.csv                 37 points of the capacity sweep
mem_footprint.csv                       native and native+MPS memory, N in {1,2,4,8}
figures/fig1..fig5 (.pdf and .png)      the five figures
GAPS.md / .pdf                          this document
```

And in the repository, under version control:
`results/asplos_campaign/{fairness,memoria,llama_slo_sweep,figures}/` with the scripts that
generate them, and `docs/paper_snapshot/`.

**`~/paper` is not under version control.** The durable copies live in the repository; the tar
should be built from there, not from `~/paper`, or it will lose traceability again.
