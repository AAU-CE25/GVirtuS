---
title: "Artifact gaps -- what is missing, why it matters, and the minimum experiment"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

A living list of what the package does **not** contain. Each entry says what is missing, what
claim it blocks, whether it is reconstructible from the server or needs re-measuring, and the
minimum experiment that closes it. Nothing here is estimated or filled in.

# 1. Capacity under an SLO per N -- **CLOSED 2026-08-03**

Swept twice. The first sweep used a fixed 30 s window and produced a +17.4 percent advantage at
n=1 that did NOT survive n=3; it is retracted (`LLAMA-7B_RESULTS.md` section 3b). The second,
with the window scaled as max(30, 40/lambda) so every point sees at least 40 offered requests,
covers 108 points over 3 systems, 3 tenant counts, 4 loads and 3 repetitions, and includes the
native+MPS arm the first lacked.

**Result: capacity under a 1 s TTFT SLO is identical across all three systems** -- lambda 0.50,
51.2 to 51.7 t/s (mean of n=3), at every N. Paired difference Gusto minus native at the knee is
-0.5 t/s with a bootstrap CI95 of [-1.6, +0.0]. The metric does not discriminate, and that is
now a measured result rather than an artefact.

**What does discriminate**, at N=8, as paired differences over the three repetitions:

| lambda | goodput, Gusto minus native | TTFT p95, Gusto minus native |
|---:|---|---|
| 0.50 | -0.5 t/s, CI95 [-1.6, +0.0] -- includes zero | **+273 ms**, CI95 [+100, +561] |
| 0.75 | -0.8 t/s, CI95 [-2.4, +2.5] -- includes zero | -614 ms, CI95 [-2215, +386] -- includes zero |
| 1.00 | **+9.6 t/s (+9.9%)**, CI95 [+3.2, +16.0] | **-1787 ms**, CI95 [-2561, -1172] |
| 1.50 | **+31.3 t/s (+27.5%)**, CI95 [+12.8, +42.6] | **-2491 ms**, CI95 [-4071, -1533] |

So remoting costs tail latency at light load and buys both back above the knee, which sits near
lambda 1.0. At lambda 1.5 native also records 4 timeouts across the three repetitions to Gusto's 0.

> **Corrected 2026-08-03.** This entry previously read *"57.6 t/s"*, *"an order of magnitude of
> tail latency (603 against 43)"* and *"15 to 33 percent more goodput (118.4 against 102.4)"*.
> Every one of those is **repetition 1**, not the n=3 mean -- the same defect this very entry
> retracts for the first sweep. Recomputed from `LLAMA_SLO_capacidad_v2.csv`, the light-load tail
> cost is a factor of 2.1 rather than an order of magnitude, and the goodput advantage is +9.9%
> and +27.5%. TTFT p95 varies more than 10x between repetitions of the same cell, so **no single
> repetition of this sweep may be quoted.**

**And native+MPS matches Gusto on goodput figure for figure** -- paired difference +0.00 t/s at
lambda 0.5, 1.0 and 1.5, and -0.80 t/s at 0.75 -- with a better tail below saturation. That
confirms context consolidation as the mechanism for the third time and removes multi-tenant
goodput from the list of arguments for remoting.

Two things remain open and are small:

1. **A finer grid between 0.50 and 1.00.** Every system meets the SLO in some repetitions and
   not others at 0.75 and 1.00, so the all-repetitions criterion collapses them all to 0.50. A
   finer sweep there would separate them if anything separates them.
2. **Transport provenance is still not recorded** in the sidecar: bench.py reads GVIRTUS_CONFIG
   from the invoking process and that variable lives inside the container. The arm is fixed by
   the label and the harness. To correct before the next packaging.

Data: `LLAMA_SLO_capacidad_v2.csv`, raw in `results/asplos_campaign/llama_slo_sweep_v2/`.
Figure: `figures/fig5_slo_capacidad_v2.pdf`. Harness `~/sweep_v2.sh`, analysis `~/analiza_v2.py`.

# 2. Native+MPS memory footprint per tenant -- **CLOSED, MECHANISM INCLUDED**

Measured. Native and native+MPS in the same session on the same GPU, N  in {1,2,4,8}, per-tenant
= (peak - baseline)/N with 25 samples per point after exercising every pod:

| N | native | native+MPS | Gusto | MPS saves | **Gusto saves** |
|---:|---:|---:|---:|---:|---:|
| 2 | 4948.5 | 4942.5 | 4503 | 6.0 | **445.5** |
| 4 | 4948.0 | 4942.8 | 4492 | 5.2 | **456** |
| 8 | 4947.9 | **4942.9** | 4487 | **5.0 (0.1%)** | **461 (9.3%)** |

**CLOSED 2026-08-03: the mechanism is the per-process CUDA primary context, measured with a
probe.** The clue was the scaling -- the saving is nearly constant in N (445.5 / 456 / 461), not
`C(1 - 1/N)` as *sharing* one context among N tenants would give. A constant per-tenant saving
means a constant per-tenant cost that Gusto simply does not pay.

`ctx_probe.cu` does nothing but `cudaFree(0)`, forcing the primary context, then sleeps:

| host / GPU | MPS | per-process footprint |
|---|---|---:|
| dpu-01 (backend L40S) | no | **431 MiB** |
| dpu-02 (native L40S) | daemon running | **429 MiB** |

Exactly linear over K = 1, 2 and 4 on both. `nvidia-smi`'s per-process accounting shows **424
MiB** per probe independently, with and without a reachable MPS pipe directory.

**This also corrects how the MPS control was read.** The earlier entry treated MPS as a
*refutation* of the context explanation. The premise was wrong: MPS consolidates the
**scheduling** context, but each client still creates and pays for its own primary context --
429 MiB, measured. MPS failing to reproduce the saving is therefore exactly what the context
explanation predicts. The two measurements even agree on the residual (2 MiB from the probe,
5.0 MiB from the llama experiment).

**The defensible sentence is now:** remoting saves ~426-461 MiB per tenant **because the tenant
process never creates a CUDA context**, and MPS cannot match it because MPS shares scheduling,
not per-client context state. Predicted 429 against 426 measured at N=1 -- **99% accounted for**.

Data: `mem_footprint.csv` (arms `bm`, `bmmps`, `ucx_deployed`), `ctx_probe_results.csv`,
`ctx_probe_perproc.txt`, probe `ctx_probe.cu`. Harnesses `~/mem_footprint.sh`, `~/mem_gusto.sh`.
Written up in `LLAMA-7B_RESULTS.md` §2, which also records a **wrong retraction of mine** on the
way here: a re-measurement taken on a backend another experiment had left at `slots=8`
(4x the deployed pool) briefly appeared to cut the saving to ~170 MiB. It is withdrawn; the
deployed-configuration column reproduces the original to within 2 MiB.

# 3. Cause of the uneven sharing -- **CLOSED 2026-08-03, mechanism plus intervention**

It was established **that** the sharing is uneven and **by how much**; **why** is now
established too, by four exclusions and one causal intervention (`N1_SCHEDULER.md`).

**Refuted:** the data path (identical over TCP), the shared legacy stream (per-connection
streams make it *worse*, 3.46 -> 6.39), lock contention in the transport (each connection owns
its worker and mutex), and -- the one that closed it -- **a shared CUDA context**. CUDA MPS puts
eight clients into one context by construction and shares to within **1.02x**, against 4.3--4.9x
for all three remoting transports. Sharing a context is not sufficient.

**The mechanism:** the backend does not arbitrate. It serves RPCs first-come-first-served, and
every client is self-clocked, so the tenant that gets marginally ahead re-submits first and keeps
its turn. MPS is fair because the MPS server arbitrates; the driver is fair between contexts
because it multiplexes them; the backend does neither.

**Confirmed causally.** `GVS_FAIR_DISPATCH=1` adds deficit round-robin at the launch point.
miniBUDE N=8:

| arbitration | reps | inequality (mean) | CI95 | fastest tenant | makespan |
|---|---:|---:|---|---:|---:|
| FCFS (deployed) | 25 | **5.02** | [4.75, 5.27] | **216.4 GFLOP/s** (its solo rate, in all 25) | 24.89 s |
| deficit RR, lead=1 | 8 | **3.09** | [2.05, 4.34] | 75.9 GFLOP/s | 25.05 s |

**-38% inequality for +0.6% makespan**, and the CI of the intervention excludes the baseline.

**What is still open** is only the residual: launch-count arbitration equalises *submissions*,
not *GPU time*, so a tenant with longer kernels still gets more device for the same turns. A
duration-weighted gate would close it and needs per-kernel timing the current tracing does not
collect.

Data: `results/asplos_campaign/sched_n1/minibude_n8_fair_ab.csv` (33 cohorts, both arms),
`results/asplos_campaign/fairness/tabla_D_minibude_por_tenant.csv` (the MPS control).

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
