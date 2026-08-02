> AVISO: **The four-arm table near the bottom is the ORIGINAL 2026-07-25 run and is superseded.**
>
> Both halves have since been redone, and the current numbers are in the sections
> immediately below this banner:
>
> **GVirtuS arms** were re-measured only IN PART; the table below says which cells.
> `N=4` and `N=8` reproduce the raw data that ships; `N=1` and `N=2` do not, and remain the
> 07-25 measurement. The `libcudart.so.12` rebuild on dpu-02 (07-26 04:24, up to
> then still on the `cudaGraphExecUpdate` stub) moved things by less than 2.5% where redone.
>
> **The baremetal baseline** had no MPS. With MPS enabled, native execution is 13-14%
> faster from N=2 on and beats GVirtuS at every tenant count; the apparent GVirtuS
> advantage in the original table was the missing MPS, not a property of remoting.
>
> Use the sections below. The original table is kept only for provenance.

## Baremetal re-measured 2026-07-26/27 -- MPS changes the conclusion

> ## 2026-08-01 -- the 256 KiB payload, now measured
>
> Per-RPC trace of an N=1 run: **66 RPCs in total**, maximum payload **262,173 bytes =
> 256 KiB**. The cited figure is confirmed and backed by a file. With only 66 RPCs,
> miniBUDE is not control-bound at all, which supports its role as a roofline control.
> At 256 KiB, RDMA would give 5.96 GB/s against 3.82 for AM (**x1.56**). See §E of
> `RESULTS_2026-08-01.md`.


> **Corregida 2026-08-01.** La versión anterior de esta tabla mezclaba fuentes y
> estadísticos, y el pie («mean of 5 reps») no describía ninguna de sus columnas. Ver
> `CORRECTIONS_2026-08-01.md`. What was checked, cell by cell, against the raw data that
> distribuye:
>
> - **La columna de GVirtuS estaba a medias.** `N=4` y `N=8` **sí** reproducen el crudo
>   current raw data (12.748 vs 12.78 published; 24.888 vs 24.91), so those two cells **were**
>   re-measured. `N=1` and `N=2` reproduce **nothing** that ships (3.679 vs 3.79;
>   6.655 vs 6.79) and remain the old measurement. That is why the banner ("the GVirtuS
>   GVirtuS se rehicieron») y esta sección («la columna sigue siendo pre-cutoff») parecían
>   contradict each other: **both were true, for different cells**, and the table did not say so.
> - **Cada columna usaba un estadístico distinto.** `baremetal` sin MPS era la **mediana de
>   15** repetitions; `baremetal` with MPS the **mean of 5**; GVirtuS the mean of 10 (and of
>   25 at N=8). That matters more than usual here: without MPS the spread is
>   **sd = 1.613 s** at N=8, and with MPS **sd = 0.008 s** -- two hundred times smaller. Taking the
>   mediana en la columna dispersa y la media en la estrecha infla el baseline sin MPS,
>   which is precisely the comparison the conclusion rests on.
> - **`baremetal_mps.csv` lleva `sys=baremetal` en sus filas.** Agrupar por esa columna en
>   rather than by file silently merges the two baselines -- the same error this banner once
>   corrected, lying in wait for whoever aggregates this data next.

Cohort wall-clock, seconds. **Median, with `n` declared per cell**, recomputed from
`data/raw/minibude/` with `workflow/fixups/regen_minibude.py`. `tenants_ok` corrected from
doubling that affects the GVirtuS arms (see below).

| N | baremetal, sin MPS | baremetal, **MPS on** | GVirtuS GPUDirect | GVirtuS rdma | GVirtuS vs bm+MPS |
|---|---:|---:|---:|---:|---:|
| 1 | 3.200 (n=15) | **3.110** (n=5) | 3.640 (n=10) | 3.735 (n=10) | +17.0% más lento |
| 2 | 6.800 (n=15) | **6.040** (n=5) | 6.610 (n=10) | 6.745 (n=10) | +9.4% |
| 4 | 13.660 (n=15) | **12.010** (n=5) | 12.730 (n=10) | 12.855 (n=10) | +6.0% |
| 8 | 27.270 (n=15) | **23.970** (n=5) | 24.870 (n=25) | 25.020 (n=25) | +3.8% |

**The conclusion holds and so does the mechanism**: with MPS, native execution beats
GVirtuS at all four tenant counts, and the remoting penalty **narrows as tenants are added**
-- 17.0% -> 9.4% -> 6.0% -> 3.8%. What changes against the previous version are the first two
figures (20.5% and 12.5%), which came from comparing GVirtuS cells that had not been
re-measured against a baseline that had.

**An incomplete cohort the doubled count was hiding**: `ucx_rdma N=8 rep=1` completed
**7 of 8** tenants. It is the only one of the 230 repetitions.

**Enabling MPS makes native execution 13--14% faster from N=2 on**, and with it **baremetal
beats GVirtuS at every tenant count**. The original table's apparent GVirtuS advantage at
N>=4 (12.78 vs 13.70) was the *absence of MPS in the baseline*, not a property of remoting.

Note the trend that survives: the remoting penalty **shrinks as tenants are added** --
20.5% -> 12.5% -> 6.1% -> 3.8%. Even against a properly configured native baseline, remote
execution closes most of the gap under contention, which is consistent with the
context-consolidation effect measured for llama. It just does not overtake it here.

**Prediction check (registered before measuring):** I expected MPS to have a *small* effect
on miniBUDE because its cohort time scales almost linearly with N, which suggested tenants
were already serialised. 13.8% is not small and it flips the ranking, so that reasoning was
wrong: linear scaling did not imply the contexts were staying out of each other's way.

## GVirtuS arms re-measured 2026-07-27 -- no material change

The GVirtuS rows had been taken 07-25, before the frontend rebuild. All three arms were redone
after the cutoff, with the backend reset per arm and its reported GPUDirect state asserted
(`enabled` for ucx_gpudirect, `disabled` for ucx_rdma) rather than merely logged.

| arm | N=1 | N=2 | N=4 | N=8 | change vs 07-25 |
|---|---:|---:|---:|---:|---:|
| TCP | 3.25 | 6.16 | 12.06 | 23.97 | -1.2% ... +0.2% |
| UCX GPUDirect | 3.70 | 6.73 | 12.85 | 24.86 | -0.6% ... +2.4% |
| UCX-RDMA (no GPUDirect) | 3.75 | 6.83 | 12.87 | 24.94 | -2.5% ... +0.1% |

**Everything moved by less than 2.5%, i.e. inside the noise**, so the stale-frontend defect did
not materially affect this workload. That was the registered prediction: miniBUDE is
compute-bound and issues few RPCs per unit of work, so the graph stub had little to bite on --
unlike llama decode, where the same defect was worth 1.5--2.5x. The prediction held here (it did
not for MPS, see below).

**GPUDirect and RDMA are indistinguishable at every N** (24.86 vs 24.94 at N=8), as expected:
miniBUDE's largest transfer is 256 KB, far below the 4 MiB RMA floor, so neither arm ever
touches the peer-DMA path.

## Fairness between tenants (2026-07-28)

Cohort wall time says how long the work takes; it says nothing about **who gets the GPU**.
The per-tenant GFLOP/s were always in the run directories (`run_<sys>_n<N>_r<R>/t<i>.log`)
but had never been aggregated. They separate the arms in a way wall time does not.

| arm | Jain (N=8) | min/max (N=8) | slowest tenant sees | reps |
|---|---:|---:|---:|---:|
| baremetal | 0.9998 | 0.969 | 97% of the fastest | 5 |
| **baremetal + MPS** | **0.9998** | **0.981** | **98%** | 5 |
| GVirtuS tcp | 0.678 | 0.226 | 23% | 5 |
| GVirtuS rdma | 0.735 | 0.231 | 23% | 14 |
| GVirtuS GPUDirect | 0.690 | 0.214 | 21% | 15 |

**Native shares; remoting serialises.** Under GVirtuS the fastest tenant runs at the full
single-client rate -- 216 GFLOP/s at N=8, identical to N=1 -- while the slowest sits at 43--62.
That is not uneven sharing: some tenants run alone while others wait. Native, with or
without MPS, keeps every tenant within 2--3% of every other.

**This splits the mechanism from the policy.** MPS and the GVirtuS backend both funnel N
tenants into one CUDA context, which is why their *cohort times* track each other here and
in llama and BabelStream. They do not share a scheduling policy: MPS interleaves, the
backend serialises. The mechanism explanation used elsewhere in this campaign is about
context consolidation and remains correct -- but it does not carry fairness with it, and
that has to be said separately.

**The three remote arms do not differ from each other.** `rdma` vs `GPUDirect` at n=15,
exact-tail Mann-Whitney: Jain p=0.093, min/max p=0.419. An earlier n=5 sample looked like a
real gap (min/max p=0.087, with 4 of 5 rdma reps above every gpudirect rep); at n=15 that
fell to 2 of 14 and the p-values went the wrong way. **It was a small-sample artefact.**
What survives is the native-vs-remote separation, which has no overlap at all: 0.9998 for
both native arms against 0.65--0.73 for all three remote ones.

### Two things in this data that must not be quoted

**`agg_throughput_*` for miniBUDE is not a throughput.** It sums per-tenant rates, and that
is only meaningful if the tenants overlap in time. Under GVirtuS they do not -- the sum
reaches 742 GFLOP/s at N=8, **3.4x the physical capacity of the card** (216 GFLOP/s from one
tenant, and the GPU is saturated by one). The number is kept in the CSV because the fairness
ratios derive from the same per-tenant values and those *are* informative, but the quotable
performance metric for miniBUDE is `wall_s`, the cohort wall time.

**`tenants_ok` in `~/mb_campaign/*.csv` is double-counted for the GVirtuS arms.**
`mb_campaign.sh` counts with `grep -alc "valid: true" "$OUT"/t*.log`, and that glob also
matches the `tenant_*.log` files those arms write, so N=8 is recorded as 16. It does not
affect the GFLOP/s statistics -- duplicating every value changes neither mean nor min nor max
-- but the column cannot be used as a completeness filter, because `16 >= 8` would pass even
with dead tenants. `results_minibude.csv` is built by indexing `t{i}.log` explicitly and is
unaffected.

### Provenance

N=8 for the two UCX arms was re-measured at n=15 on 2026-07-28 to settle the rdma-vs-GPUDirect
question, with the backend re-armed per arm and its `[GVS] GPUDirect=` line asserted before
measuring -- necessary because `mb_campaign.sh` sends `MTSYS=ucx` for both arms and only the
backend distinguishes them, so an unarmed backend would silently make them the same
experiment. One rdma repetition produced fewer than 8 tenant logs and was dropped (14 of 15).
All other points are n=5.

## Standing conclusion

| N | baremetal + MPS | UCX GPUDirect | remoting penalty |
|---|---:|---:|---:|
| 1 | 3.14 | 3.70 | +17.8% |
| 2 | 6.03 | 6.73 | +11.6% |
| 4 | 12.05 | 12.85 | +6.6% |
| 8 | 23.99 | 24.86 | +3.6% |

Against a properly configured native baseline, remote execution is **3.6--17.8% slower**, and
**the penalty shrinks monotonically as tenants are added**. Against *default* native (no MPS)
remoting is ahead from N=2. Both numbers belong in any write-up.



# miniBUDE over GVirtuS -- contention scaling, four systems (2026-07-26)

**App:** miniBUDE (Bristol docking proxy), CUDA, compute-bound. Deck `data/bm1`, 8 iterations.
**Metric:** wall time for the whole cohort of N concurrent tenants, mean +/- 95 % CI (Student-t, n=5).
**Systems:** baremetal = native `cuda-bude` on dpu-02's local L40S - tcp / ucx_rdma / ucx_gpudirect = `cuda-bude-gvirtus` against the dpu-01 backend. `ucx_rdma` = UCX/RDMA with `GVIRTUS_GPUDIRECT=0` **on the backend** (setting it on the frontend, as `gvrun.sh ucx_nogds` does, does not disable it).

## Cohort wall time (s)
| N tenants | baremetal | TCP | UCX-RDMA | UCX-GPUDirect |
|---|---:|---:|---:|---:|
| 1 | 3.21 +/- 0.02 | 3.21 +/- 0.11 | 3.69 +/- 0.10 | 3.79 +/- 0.16 |
| 2 | 6.80 +/- 0.01 | 6.13 +/- 0.03 | 6.66 +/- 0.14 | 6.79 +/- 0.28 |
| 4 | 13.70 +/- 0.02 | 12.09 +/- 0.05 | 12.87 +/- 0.23 | 12.78 +/- 0.17 |
| 8 | 27.29 +/- 0.05 | 23.98 +/- 0.10 | 24.97 +/- 0.22 | 24.91 +/- 0.15 |

**Scaling is linear in N for every system** (8 tenants ~= 8x one tenant): miniBUDE is compute-bound and all tenants share one GPU, so they serialise on the kernel. Transport is irrelevant to the steady state -- as expected for a workload whose largest transfer is **256 KB**, far below the 4 MB RMA floor, so it never touches the RMA/GPUDirect path at all. **UCX-RDMA and UCX-GPUDirect are indistinguishable at every N** (24.97 vs 24.91 at N=8), which is the control confirming that.

## Effect of the on-demand slot pool
Same binary, same backend, only the pool policy differs (`GVIRTUS_RMA_PREALLOC=1` + `SLOT_CAP_MB=256` reproduces the old eager behaviour):
| N tenants | eager pool (old) | on-demand pool (current) | gain |
|---|---:|---:|---:|
| 1 | 3.94 s | 3.72 s | -5.6 % |
| 2 | 7.14 s | 6.54 s | -8.4 % |
| 4 | 13.48 s | 12.63 s | -6.3 % |
| 8 | 27.11 s | 24.69 s | -8.9 % |

**GPU memory on the backend: 2 x cap host + 2 x cap GPU shadow per connection -> zero.** miniBUDE never reaches the RMA floor, so no pool is materialised at all (verified: no `rx_pool` event during a run). At the campaign's `SLOT_CAP_MB=256` that is ~512 MB of GPU per tenant reclaimed; at the 1025 MB default, ~2 GB per tenant.

## Two honest caveats
**Baremetal is not the fastest arm at N >= 2** (27.29 s vs 23.98--24.97 for the remoted arms at N=8). This is *not* GVirtuS beating local execution: baremetal runs on dpu-02's local L40S while every GVirtuS arm runs on dpu-01's, and the two cards are not performance-matched under sustained load. Cross-arm wall-time comparisons are therefore only valid **among the three GVirtuS arms**; baremetal is a scaling-shape reference, not a speed baseline. Establishing a true baremetal baseline would require running the native binary on dpu-01 itself.

**Per-tenant GFLOP/s is not comparable across arms** and is deliberately omitted from the headline table. Under baremetal, N tenants time-share the GPU concurrently and each reports ~= 216/N GFLOP/s (24.4 at N=8). Under GVirtuS the backend serialises each connection's kernels on its own thread, so a tenant's kernel runs at near-full rate and reports ~= 216 while waiting longer in queue. Both produce the same cohort wall time by different routes; only wall time measures the same thing in both. miniBUDE's own reported GFLOP/s also excludes connection setup, which is exactly where the pool change acts -- a second reason it would show nothing.

## Validity
All runs valid (`max_diff 0.014 %`), every tenant in every rep, all four systems.

## The quantum accounting (2026-08-02) -- the mechanism, measured

The section above says *"MPS interleaves, the backend serialises"*. That was an
interpretation of an aggregate. It is now a measurement, taken from a field that was already
in the logs and had never been read: `raw_iterations`, the ten timed iterations miniBUDE
prints per run.

**Field semantics, verified arithmetically before use.** `sum_ms` equals the sum of
`raw_iterations[2:]` to the millisecond, so miniBUDE discards two iterations as warm-up and
reports over the last eight; `avg_ms = sum_ms / 8`.

**The measure.** The solo iteration time is the service quantum, and it is essentially
identical across all five systems (295.28--295.91 ms), so the comparison is clean:

    multiplier(i,k) = raw_iterations(i,k) / solo_iteration_time(system)

A multiplier of 1 means that iteration was served with no wait at all. A multiplier of *m*
means *m* quanta elapsed before it completed, i.e. the tenant waited *m-1*. With N tenants
and fair sharing every iteration of every tenant should sit near N -- which is exactly what
native does.

| arm | runs | median multiplier | max | **iterations served with no wait** | longest stall |
|---|---:|---:|---:|---:|---:|
| baremetal | 5 | **7.95** | 7.97 | **5 of 400** | 2.06 s |
| baremetal + MPS | 5 | **7.95** | 7.99 | **7 of 400** | 2.06 s |
| GVirtuS GPUDirect | 15 | **3.50** | **16.00** | **405 of 1200 (34%)** | 4.43 s |
| GVirtuS rdma | 14 | 3.00 | 10.00 | 364 of 1120 (33%) | 2.66 s |
| GVirtuS tcp | 5 | 3.99 | 11.98 | 68 of 400 (17%) | 3.25 s |

One cohort, tenant by tenant -- identical deck, starts within 1 s
(`run_ucx_gpudirect_n8_r1`):

| tenant | slowdown | multiplier per iteration |
|---:|---:|---|
| **2** | **1.001** | `1 1 1 1 1 1 1 1 1 1` |
| **3** | **4.874** | `3 4 2 6 5 6 10 6 3 1` |
| 5 | 4.125 | `4 4 1 8 6 6 4 2 3 3` |
| 8 | 1.750 | `2 2 1 2 1 2 1 1 1 5` |

Tenant 2 runs **as if alone, on all ten iterations**. Tenant 3 queues behind up to **ten**
quanta on a single iteration. Same cohort, same work, same second.

The native control, same cut (`run_baremetal_n8_r1`): every tenant sits at `8` on
essentially every iteration. The only sub-8 values are the first one or two columns, and in
them **the join order is legible**: `5 7.9 8 8...`, `2 8 8...`, `1 2 8 8...`.

**What this adds to the conclusion above.** Native is not merely "fair on average" -- it is
round-robin at exactly N, iteration by iteration. Gusto is not merely "less fair" -- a third
of its iterations are served with **zero** wait, which is what produces both the lower median
slowdown (3.50 against 7.95: the median tenant finishes 2.3x sooner) and the inequality. The
aggregate win and the fairness loss are the same mechanism, and TCP shows it too, so it is
not the data path.

**What it still does not establish** is *why* a given tenant is favoured. Connection order,
monopolisation of a backend thread and head-of-line blocking in the dispatcher all predict
this signature. Separating them needs queue-entry and dispatch timestamps per connection in
the backend; see `GAPS.md` §3.

Per-tenant data: `tabla_D_minibude_por_tenant.csv` (352 tenant-run rows, with the full
multiplier vector per tenant). Figure: `figures/fig3_minibude_cuanto.pdf`. Cross-workload
context and the controls: `FAIRNESS_RESULTS.md`.
