---
title: "XSBench -- contention scaling, five systems, and per-tenant fairness"
date: "2026-07-27, corrected 2026-08-01, fairness added 2026-08-02"
geometry: margin=2.3cm
fontsize: 10pt
---

**App:** XSBench CUDA (Monte-Carlo neutron transport proxy), `-s large -m event -G nuclide`,
**`-l 6e8` lookups**. **Metric:** wall time for the whole cohort of N concurrent tenants, and
-- since 2026-08-02 -- **per-tenant Mlookups/s and runtime**. **Systems:** baremetal = native
XSBench on dpu-02's local L40S - tcp / rdma / rdma\_zc / ucx = GVirtuS against the dpu-01
backend. `ucx` = GPUDirect on; `rdma_zc` = GPUDirect off **on the backend** (setting it on the
frontend does not disable it -- pool, GPU shadow and D2H-GET all live backend-side). All runs
post-date the 2026-07-26 04:24 frontend rebuild.

> ## XSBench's `exit_code` is meaningless
>
> **0 of 238 runs at 6e8 exit with code 0.** XSBench carries verification checksums hard-coded
> per problem size, valid only for that size's default lookup count; neither 6e8 nor 1.25e9 is
> one, so it prints `WARNING - INAVALID CHECKSUM!` and returns non-zero even when the
> simulation finishes cleanly. **The correct signal is the `Lookups/s` line.**

**Why 6e8 and not 1.25e9.** `GridInit.cu:66` allocates `lookups x sizeof(unsigned long)` for
the verification array -- 1.25e9 lookups is **10 GB per instance**, so 4 tenants fit in the
46 GB card and 8 do not. Every N=8 point at 1.25e9 was a partial cohort while the harness still
printed `done=8/8`; makespan then tracked survivor count with a perfect correlation (4->108 s,
5->135 s, 6->160 s). At 6e8 each instance needs 4.8 GB and all 8 fit. *(Note `-s small` would not
have fixed this: the size preset changes grid dimensions, not the verification array, which
scales only with `-l`.)*

# 1. Cohort wall time

Rebuilt from the per-client raw records. A cohort counts only if all N clients are present
**and** all N printed `Runtime:` -- the harness's own `done=N/N` is not sufficient, as the
1.25e9 experience showed.

| N | baremetal | ucx (GPUDirect) | rdma\_zc | rdma | tcp |
|---|---:|---:|---:|---:|---:|
| 1 | **12.88** | 13.94 | 13.76 | 13.99 | 14.02 |
| 2 | **25.02** | 27.08 | 26.58 | 27.34 | 27.63 |
| 4 | **50.23** | 52.64 | 52.26 | 53.17 | 52.59 |
| 8 | **98.39** | 103.89 | 103.48 | 103.98 | **does not complete** |

Median cohort makespan, defined `max(t_end) - min(t_start)` over the N clients. Valid cohorts
per cell: 4--10. Recomputed on 2026-08-02 from
`~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/client*/status_raw.json`; it matches
the 2026-08-01 correction block in every cell except `ucx` N=8 (103.89 against 103.62, a
different set of repetitions).

> **An earlier table in this document gave baremetal N=8 = 114.92 s and TCP N=8 = 100.89 s. It
> was wrong and has been deleted.** It should not be reconciled: it inverted the conclusion,
> making baremetal lose to the remote arms from N=2 on and making TCP the fastest arm at N=8
> where in fact TCP does not complete. The valid data is the table above.

**Transport is irrelevant for this workload.** The four remote arms land within 2% of each
other at every N, TCP included up to N=4, and `ucx` vs `rdma_zc` differ by under 0.5%. XSBench
moves its grid to the GPU **once** and then runs 6x10^8 lookups, so the transfer is milliseconds
against a ~100 s makespan and GPUDirect has nothing to accelerate. **This is a null result, and
a useful one:** it bounds where the §1 data-path advantage applies.

**Correctness:** all arms report identical verification checksums at every N (408237 at 6e8),
so remoting is bit-exact here.

**The remoting overhead is flat in scale:** 8.2% at N=1, 8.2% at N=2, 4.8% at N=4, 5.6% at N=8
for `ucx` against baremetal. It does not degrade with tenants.

# 2. The MPS row: withdrawn for want of evidence

This document carried a section claiming that **MPS makes native execution 13--15% faster from
N=2 on**, with baremetal+MPS at 25.11 / 50.15 / 98.40 s. **It is withdrawn**, for two
independent reasons:

1. **Its comparison baseline was the obsolete table.** The 13--15% gain was computed against
   baremetal-without-MPS at 29.53 / 58.05 / 114.92. Against the correct values -- 25.02 / 50.23
   / 98.39 -- the margin vanishes: **-0.4% / +0.2% / 0.0%**. For XSBench, MPS buys nothing
   measurable.
2. **No raw file anywhere backs it.** The values 25.11 / 50.15 / 98.40 appear only in this
   document: there is no `baremetal_mps` tree under `~/xsbench_campaign/`, and no row in any
   CSV. They are not reconstructible.

With it falls the premise the section used to justify itself: *"baremetal without MPS loses to
every remote arm from N=2 on, which is the opposite of what a local GPU should do."* With
correct data **baremetal wins at every N without MPS** (25.02 < 26.58 < 27.08 < 27.34 < 27.63).
There was nothing to explain.

**This is XSBench-specific.** The MPS result for **llama** (`RESULTS.md` §8b, n=3) and for
**miniBUDE** (the `run_baremetal_mps_*` tree in `~/mb_campaign/`) does have raw data and stands
on its own. The rule of reporting both baselines still holds; what is withdrawn is the XSBench
row.

# 3. Per-tenant fairness

What the previous package did not carry. Reconstructed from the per-client `stdout.log`, which
**were still on the server**, in
`~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/client*/`. Per-tenant data in
`XSBench_fairness_por_tenant.csv` (**306 rows**).

Normalised progress uses XSBench's internal `Runtime:` -- the simulation, without start-up --
against the same system's N=1 median. Jain is computed **per cohort** and summarised with the
median across cohorts.

| system | N | cohorts | **Jain** | **slowest/fastest** | Mlookups/s worst | median | best |
|---|---:|---:|---:|---:|---:|---:|---:|
| baremetal | 2 | 5 | **1.0000** | **1.000** | 24.9 | 24.9 | 24.9 |
| baremetal | 4 | 5 | **1.0000** | **1.000** | 12.2 | 12.2 | 12.2 |
| baremetal | 8 | 4 | **1.0000** | **1.001** | 6.2 | 6.2 | 6.2 |
| ucx | 2 | 5 | 0.9007 | 1.994 | 24.1 | 36.1 | **48.1** |
| ucx | 4 | 5 | 0.9309 | 1.992 | 12.1 | 16.2 | 24.1 |
| ucx | 8 | 5 | **0.6613** | **5.982** | 8.0 | 12.0 | **47.7** |
| rdma\_zc | 2 | 5 | 0.9010 | 1.992 | 24.5 | 36.7 | **48.8** |
| rdma\_zc | 4 | 5 | 0.8049 | 2.973 | 16.2 | 20.2 | **48.0** |
| rdma\_zc | 8 | 5 | **0.7163** | **4.848** | 9.6 | 12.1 | **46.5** |
| rdma | 8 | 5 | **0.6933** | **4.882** | 9.7 | 14.1 | **48.5** |
| tcp | 2 | 7 | 0.9009 | 1.992 | 24.3 | 36.2 | **48.2** |
| tcp | 4 | 7 | 0.9309 | 1.995 | 12.2 | 16.2 | 24.3 |

**Native shares evenly.** The N tenants get the same throughput to the third decimal. Its solo
rate is 50.7 Mlookups/s, and the observed split -- 24.9 / 12.2 / 6.2 -- sits slightly under the
ideal 50.7/N -- 25.4 / 12.7 / 6.3: there is a ~2% efficiency loss with tenants, but **it is
shared equally among them**.

**The remote arms do not.** At N=8 the best `ucx` tenant reaches **47.7 Mlookups/s, 96% of its
own solo rate (49.6): it runs almost as if alone on the card** -- while the worst gets 8.0, a
sixth. It is the same pattern as miniBUDE, and it appears **on TCP as well**, so it is not the
data path.

**At N=2 the signature is exact:** slowest/fastest = 1.99 in all four remote arms, with the
best tenant at 97% of its solo rate (48.1 of 49.6) and the worst at half. That is not
contention: it is **serialisation**.

**A warning about which metric one looks at.** With **wall clock** the inequality nearly
disappears (slowest/fastest 1.15--1.41 at N=8) because the cohorts start and end together; with
**internal runtime** it is 4.8 to 6.0x. The gap between the two says the tenants divide their
time between start-up and compute differently, and **cohort makespan cannot show it**. A
document reporting only makespan will conclude, wrongly, that the sharing is fair.

Full context, controls, and the per-iteration decomposition of the same phenomenon in miniBUDE:
`FAIRNESS_RESULTS.md`.

# 4. Tenant density under deliberate over-subscription -- incomplete

Run at **1.25e9** on purpose, so 8 tenants ask 80 GB of a 46 GB card and some must die. The
metric is survivors, not makespan.

| arm | survivors of 8 (5 reps) | mean |
|---|---|---:|
| baremetal | 4, 4, 4, 4, 4 | **4.0** |
| ucx (GPUDirect) | 5, 6, 5, 4, 6 | **5.2** |
| rdma\_zc | *arm void* | -- |
| TCP | 4, 4, 4, 0, 0 | *unusable* |

Baremetal is perfectly reproducible at exactly 4; `ucx` fits **30% more tenants**, consistent
with one shared CUDA context leaving more device memory for the tenants themselves -- the same
mechanism as the ~460 MiB/tenant saving in `RESULTS.md` §8.

**Two arms are not usable and the experiment should not be quoted as a four-way comparison.**
The first `rdma_zc` run recorded `GPUDirect=enabled` on the backend, so it was a duplicate of
`ucx` rather than a GPUDirect-off arm; the re-run returned 2/8 survivors once and then zeros.
TCP produced two reps with zero survivors, undiagnosed. **Baremetal-vs-ucx is the only pair
this experiment supports.**

# 5. Baseline caveat

Baremetal runs on **dpu-02's** L40S while the remote arms use **dpu-01's**. It is a valid
correctness reference and a valid *relative* scaling reference, but not an absolute speed
baseline. That baremetal is fastest at N=1 (12.88 vs 13.94) argues against a hardware
difference driving the N>=2 results, but does not eliminate it.

**The raw tree is partly overwritten.** The density experiment was run at 1.25e9 reusing the
same `seed*/` directories, so `~/xsbench_campaign/results/xsbench/<arm>/N<n>/` now holds a
mixture of both sizes. Anyone re-deriving numbers from the tree -- rather than from the tables --
must filter on `lookups` in `seed_raw.json` or they will average two different workloads. The
tables in this document are filtered.

**The harness never arms the backend.** `run_seed_xs.sh` sets client-side flags only
(`GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`); the frontend flag alone does not disable
GPUDirect. Arm the backend per arm and assert the `[GVS] GPUDirect=` line before trusting an
arm label.

Raw data: `~/xsbench_campaign/results/xsbench/<arm>/N<n>/sync/seed*/` (per-client stdout,
`status_raw.json`, `seed_raw.json`) and `~/xsbench_campaign/results/.attempts/` (a snapshot of
every attempt, including rejected ones). Packaged: `XSBench_fairness_por_tenant.csv`,
`XSBench_filtered_v2.csv`.
