---
title: "llama 7B over GVirtuS -- multi-tenant serving, capacity under an SLO, memory and fairness"
date: "2026-07-26 to 2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

Everything the campaign measured on llama 7B, in one place. `RESULTS.md` keeps the
cross-workload summary and points here; the fairness *method* and the cross-workload
comparison live in `FAIRNESS_RESULTS.md`; the provenance of `LLAMA-7B.csv` is in
`LLAMA-7B.README.md`.

**Setup.** Disaggregated GPU serving via GVirtuS API-remoting over 200 Gb RoCE (ConnectX-7),
2x L40S. Frontend = es-dpu-02, backend GPU = es-dpu-01. Baremetal = native CUDA on dpu-02's
local L40S (no GVirtuS). Model mistral-7b-q4, served by llama.cpp `llama-server` with
`--parallel 1` per pod, `-ngl 99 --no-mmap -c 2048`. Goodput = output tokens completed over the
measurement window. Per-run metrics in `~/results/summary.csv`; per-request JSONL beside it.

**Reading order.** §1 and §2 are the original multi-tenant campaign. §3 is the memory result
and its MPS control. §4 is the capacity-under-SLO sweep that the first two could not give. §5
is per-tenant serving fairness.

# 1. Multi-tenant serving

N isolated pods, one `llama-server --parallel 1` each, open-loop Poisson, 30 s window, UNIQUE, `SEED=1`. Backend reset before every GVirtuS point. Baremetal = N native pods on dpu-02's local L40S. Backend GPU footprint sampled every 2 s; GVirtuS per-pod = (peak - 435 MiB idle baseline) / N.

**Correction (2026-07-26): the earlier version of this table described the load as "lambda=1.0 req/s per pod". It is not.** `bench.py` defines `RATE` as **TOTAL req/s across all servers** (line 10), so offered load was a flat 1 req/s at *every* N -- 29 requests offered at N=1 and 29 at N=8. Two conclusions previously drawn from that row do not survive:
- *"Aggregate goodput is flat at 128.0 t/s for every N => the shared GPU is GPU-bound."* **Withdrawn.** Goodput is flat because **offered load** is flat; nothing saturated. **Baremetal, measured 2026-07-26, is flat at exactly the same 128.0 t/s for all four N** -- if this indicated a GPU ceiling, native execution would not sit on the identical number.
- *"Batching beats isolation ~=4.6x"* (128.0 vs 587 t/s). **Withdrawn** -- it compared an offered-load-limited number against a saturated one.

| N pods | bm goodput | UCX goodput | bm TPOT p50 | UCX TPOT p50 | Jain (both) | bm GPU/pod | **UCX GPU/pod** | **saved/pod** |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | 128.0 | 128.0 | 7.0 ms | 7.2 ms | 1.000 | 4 953 MiB | **4 526 MiB** | 427 MiB |
| 2 | 128.0 | 128.0 | 15.4 ms | 7.2 ms | 0.996 | 4 951 MiB | **4 503 MiB** | 448 MiB |
| 4 | 128.0 | 128.0 | 15.5 ms | 7.3 ms | 0.953 | 4 950 MiB | **4 492 MiB** | 458 MiB |
| 8 | 128.0 | 128.0 | 23.7 ms | 12.0 ms | 0.804 | 4 950 MiB | **4 487 MiB** | 463 MiB |

**At this (unsaturated) load both systems deliver the offered rate exactly, so the table says nothing about throughput retention.** A per-pod-constant sweep (lambda = N, so each tenant offers 1 req/s and N=8 demands ~=1024 t/s against a ~=590 t/s batched ceiling) is running; the saturating comparison belongs there.

**The real multi-tenant result here is memory, and it favours remoting.** Per-tenant GPU footprint is **~=4 490 MiB under GVirtuS vs ~=4 950 MiB native -- about 460 MiB less per tenant, 3.7 GB at N=8.** The mechanism is structural: the backend serves connections as **threads inside one process** (`Process.cpp` detaches a thread per connection; `fork` happens per *configured endpoint*, not per connection), so N tenants share **one CUDA context**, whereas N native pods each pay for their own. On a 46 GB L40S that is **10 tenants versus 9** -- and the per-pod figure is flat to 0.9 % across an 8x range in tenant count, four independent measurements on freshly-reset backends.

**The slot pool contributes nothing to per-tenant cost**, because a llama pod never allocates a GPU shadow: shadows are allocated only once a connection proves it moves device-destined bulk data, and serving never sends a transfer >= 4 MiB (measured max 3.6 MB -- see §2).

| configuration | GPU per pod | max pods on 46 GB |
|---|---:|---:|
| native (one CUDA context per tenant) | 4 950 | **9** |
| old default (1025 MB cap, eager pool) | 4 490 + 2 050 shadow = 6 540 | **6** (pod 7 OOM'd) |
| old workaround (`SLOT_CAP_MB=128`) | 4 490 + 256 shadow = 4 746 | **8** |
| **on-demand pool (current)** | **4 490** | **10, measured STABLE** |

**Fairness (Jain 1.000 -> 0.804) is identical in both systems to three decimals** -- it is a property of how the shared Poisson arrival trace (same `SEED=1`) distributes across pods, **not** of the transport. It should not be cited as a GVirtuS multi-tenancy limitation. Tail latency does degrade with N in both.

**Operational caveat -- it invalidated a first attempt at this table.** The backend did not release GPU memory when a pod disconnected; after an 8-pod run the card sat at 45 217 / 46 068 MiB and the following points could not start. Per-connection reclamation now fixes the leak at the source (~9 GB/run -> ~20 MiB), but every point here is still preceded by `reset_backend.sh` so each starts from an identical 435 MiB.


## 1b. Multi-tenant under real load -- per-pod-constant offered rate (lambda = N)

> **Corregida 2026-08-01**, verificado contra `summary_master.csv`. Tres cosas:
>
> 1. **La columna «TPOT p50» son p95.** Los valores publicados (UCX 44.5, nativo 70.1) son
>    `tpot_p95`; the true p50 values are **27.6** and **65.4**. The numbers are right, the
>    header is not. The same applies to §7's footnote, which says p50 where the criterion is p95.
> 2. **The MPS arm is missing, and it changes the conclusion.** It exists in the data
>    (`mtmpson_n*_l*`) and at N=8 gives **302.9 t/s against UCX's 293.0**: against a native
>    baseline configured with MPS, remoting does **not** win. This is the same correction
>    `MINIBUDE_RESULTS.md` already applied to its table. Both baselines must be quoted, as in C7.
>
>    | N | native without MPS | **native with MPS** | UCX |
>    |--:|--:|--:|--:|
>    | 2 | 234.7 | **285.9** | 277.3 |
>    | 4 | 221.9 | **290.1** | 285.9 |
>    | 8 | 213.3 | **302.9** | 293.0 |
>
> 3. **La prosa usa el máximo donde la tabla usa la media.** La tabla da 293.0 (media de
>    285.9/290.1/302.9, correcto) pero el texto dice «rises -> 302.9» y el párrafo de
>    mechanism quotes 1.42x instead of the table's 1.37x. And 302.9 matches the value of
>    MPS-on, which is how a number ends up attributed to the wrong arm.
>
> What **does** survive intact: the latency advantage. UCX gives 27.6 ms TPOT p50 at N=8
> against 48.6 for native with MPS and 65.4 without it.
§8 never saturates (lambda is a flat 1 req/s **total**), so it cannot express a throughput cost. This sweep fixes the offered rate **per tenant** at 1 req/s, so N=8 demands ~=1024 t/s against a ~=590 t/s batched ceiling. One measurement per point (**n=1 -- see the caveat below**), 30 s window, backend reset before every GVirtuS point.

| N | lambda | baremetal | UCX | ratio | bm TPOT p50 | UCX TPOT p50 | both |
|---|---|----------:|----:|:-----:|------------:|-------------:|:----:|
| 1 | 1 | 128.0 | 128.0 | 1.00x | 7.0 ms | 7.3 ms | STABLE |
| 2 | 2 | 234.7 | 277.3 | **1.18x** | 15.7 ms | 12.9 ms | UNSTABLE |
| 4 | 4 | 221.9 | 285.9 | **1.29x** | 35.4 ms | 21.0 ms | UNSTABLE |
| 8 | 8 | 213.3 | **293.0** | **1.37x** | 70.1 ms | **44.5 ms** | UNSTABLE |

**The two systems move in opposite directions.** Native throughput *declines* as tenants are added (234.7 -> 221.9 -> 213.3) while remoted throughput *rises* (277.3 -> 285.9 -> 302.9); decode latency follows (70.3 ms native vs 42.9 ms remoted at N=8). Every N>=2 point is UNSTABLE in both systems -- offered load is genuinely beyond capacity, which is what makes the comparison meaningful.

**Proposed mechanism -- the same one that explains the memory result in §8, which is why it is worth taking seriously.** The backend serves connections as **threads inside a single process sharing one CUDA context** (`Process.cpp` detaches a thread per connection; `fork` is per configured endpoint). N native pods are N processes with N contexts, and the GPU must context-switch between them. API remoting therefore **consolidates** multi-tenant work into one context, which predicts *both* the ~=460 MiB saved per tenant *and* a throughput advantage that grows with tenant count -- which is what the ratio column shows (1.00 -> 1.18 -> 1.29 -> 1.42).

**Replicated at n=3 (2026-07-26), and it holds.** Unlike the C2/C4 case in §2 -- where the *same system* drifted 28% between blocks -- the completion counts are almost perfectly reproducible: baremetal returned **50 / 50 / 50** completions at N=8 and **52 / 52 / 52** at N=4, against **71 / 67 / 68** and **67 / 67 / 67** for UCX. The between-system gap is an order of magnitude larger than the within-system spread.

**Mechanism: CUDA-context consolidation. Confirmed by an MPS control, not assumed.** There is **no MPS daemon on either GPU** and compute mode is Default, so N native pods are N CUDA contexts that the driver *time-slices* -- kernels from different contexts cannot overlap. GVirtuS instead serves all tenants as **threads in one process sharing one context** (`Process.cpp` detaches a thread per connection; `fork` is per configured endpoint). MPS exists precisely to funnel multiple processes into one context, so enabling it on baremetal should reproduce the effect. It does, completely:

| N | baremetal, MPS **off** | baremetal, MPS **on** | GVirtuS UCX | UCX vs MPS-on |
|---|---:|---:|---:|:---:|
| 2 | 234.7 | **285.9** | 277.3 | 97.0% |
| 4 | 221.9 | **290.1** | 285.9 | 98.6% |
| 8 | 213.3 | **302.9** | 293.0 | 96.7% |
| TPOT p50 @ N=8 | 70.2 ms | **49.2 ms** | 44.5 ms | -- |

Both MPS arms are baremetal with identical CPU load, so this **also rules out the competing explanation** -- that disaggregation simply adds a second host's CPU -- by construction rather than by measurement. (n=3 per cell, blocks interleaved off/on/off/on; both arms run `--user` and `--ipc=host`, since an unprivileged MPS daemon rejects uid-0 clients with "Failed to start server using uid 0: Operation not permitted".)

**How to state this.** Against **default** native multi-tenancy -- no MPS, which is how most installations run -- API remoting is **1.18x / 1.29x / 1.37x** faster at N=2/4/8 with ~40% lower decode latency. Against **MPS-configured** native it is at **97--99% parity**. The honest claim is therefore not "remoting beats native", but: **remoting delivers MPS-equivalent context consolidation for free, as a structural property of the backend's threading model, with no MPS daemon to deploy, configure or fail** -- and the same consolidation is what makes each tenant cost ~460 MiB less GPU (§8).

**Baremetal's isolation ceiling is ~=220 t/s**, versus **663.3 t/s** batched on the same native GPU (§2, `--parallel 8`) -- **batching beats isolation ~=3x**. This supersedes the "~=4.6x" previously quoted here, which was arithmetic on an offered-load-limited number.

*(First attempt at this sweep died and was discarded: a leftover `llama-server` container held the backend's UCX listener port -- `bind(25.25.25.2:32222) Address already in use` -> `ucp_listener_create failed: Device is busy` -- so every pod reported "Endpoint is not connected" against a backend that never started. The harness now removes all frontend containers before the reset and polls for a bound listener rather than sleeping a fixed 25 s, which never covered a from-source backend rebuild.)*


# 2. Memory footprint per tenant

### The MPS control for memory (2026-08-02) -- the saving is real, the stated mechanism is not

The table above attributes the ~460 MiB/tenant saving to context consolidation: *"N tenants
share one CUDA context, whereas N native pods each pay for their own."* **That explanation is
now refuted.** MPS consolidates contexts by construction, so if the mechanism were context
consolidation, MPS would reproduce the saving. It does not.

Measured the same session, same GPU, same model, same method -- per-tenant =
(peak - baseline) / N, sampled 25x per point after exercising every pod:

| N | native | **native + MPS** | Gusto | MPS saves | **Gusto saves** |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 950,0 | 4 978,0 | 4 526 | -28,0 (worse) | **424** |
| 2 | 4 948,5 | 4 942,5 | 4 503 | 6,0 | **445,5** |
| 4 | 4 948,0 | 4 942,8 | 4 492 | 5,2 | **456** |
| 8 | 4 947,9 | **4 942,9** | 4 487 | **5,0 (0,1 %)** | **461 (9,3 %)** |

**MPS saves 0,1 % per tenant. Gusto saves 9,3 %.** At N=1 MPS is actually *worse* by 28 MiB --
the daemon's own footprint, which the baseline then absorbs from N=2 on.

**What this changes.** The memory result survives the MPS control, and that is the opposite of
what happened to the throughput result: there, MPS-configured native closed the whole gap, so
the honest framing became *"remoting delivers MPS-equivalent context consolidation for free"*.
For memory there is no such equivalence -- **MPS-configured native does not get the saving**, so
the advantage belongs to the remoting architecture and not to context consolidation.

**What it does not establish** is the correct mechanism. The measurement rules out context
consolidation as a sufficient explanation; it does not identify what replaces it. The obvious
candidate is that under remoting the N frontend processes hold **no device memory at all**
-- the whole device-side state lives in one process -- whereas under MPS there are still N
processes each making their own device allocations, with only the scheduling context shared.
That is consistent with the numbers but **untested**; a per-allocation attribution inside the
backend would settle it.

Data: `results/asplos_campaign/memoria/mem_footprint.csv`, harness
`~/mem_footprint.sh`. The Gusto column is the prior campaign's measurement, unchanged; the
native and MPS columns were taken together on 2026-08-02 so they are directly comparable, and
the native column reproduces the earlier one to within 3 MiB (4 947,9 against 4 950).

**Caveat in the harness, recorded.** Its `pods_up` column is a 0/1 success flag printed as
`flag/N`, so a value of `1/8` means *"all eight came up"*, not *"one of eight"*. Confirmed by
the peaks, which are exact multiples of the single-tenant footprint (39 584 = 8 x 4 948).
Relabelled in the CSV as `arranque_ok`.


# 3. Useful capacity under an SLO, against tenant count (2026-08-03, rep 1)

The gap §8 and §8b left open. The two earlier campaigns measured in disjoint regimes: lambda total
= 1 (stable, but total demand does not grow with N, so goodput is pinned at 128 t/s for every
N) and lambda = N (where all six points come back `UNSTABLE`, SLO attainment is 0--3% and 77% of
requests die at the 25 s deadline). **There was no measured point where Gusto beat native and
also served.** This sweep covers it.

**Method.** N  en  {2, 4, 8} x lambda total  en  {0.25 - 0.5 - 0.75 - 1.0 - 1.5 - 2.0}, native and Gusto
**paired cell by cell**, lambda order randomised within the repetition, 30 s window, 10 s warm-up,
`NPRED=128`, `REQ_TIMEOUT=25`. **36 points.** Goodput figures are **strict-window** (only
completions with `tc <= t_end`), not the ones in `summary.csv`, which count to
`t_end + REQ_TIMEOUT` and divide by `WINDOW`.

Capacity criterion: **maximum lambda with TTFT p95 < 1 s and zero timeouts**.

| N | system | **max lambda under SLO** | goodput at that load | worst tenant meets 1 s SLO |
|---:|---|---:|---:|---:|
| 2 | native | 0.50 | 51.2 t/s | 100% |
| 2 | **Gusto** | 0.50 | **55.5 t/s** (+8.4%) | 100% |
| 4 | native | 0.50 | 51.2 t/s | 100% |
| 4 | **Gusto** | 0.50 | **55.5 t/s** (+8.4%) | 100% |
| 8 | native | 1.00 | 98.1 t/s | 75% |
| 8 | **Gusto** | 1.00 | **115.2 t/s** (+17.4%) | **100%** |

### First, because it is a negative result about the metric

**Capacity measured in lambda does not discriminate: it comes out identical for both systems at
every N.** With this grid both cross the SLO between the same two points. That is a resolution
limit -- the steps are 0.5->0.75 and 1.0->1.5, jumps of 50% -- not a demonstrated tie: the real
knee lies inside those intervals and could separate them. **Refining the grid around the knee
is what would let this metric decide.**

What does discriminate, and grows with tenants, is the **goodput at the maximum admissible
load**: +8.4% at N=2 and N=4, **+17.4% at N=8**.

### How they degrade past the knee

Here the separation is large, and it is the part an operator cares about:

| N | lambda | native: %SLO 1 s / timeouts / goodput | Gusto: %SLO 1 s / timeouts / goodput |
|---:|---:|---|---|
| 2 | 1.50 | 3% / 0 / 128.0 | **20%** / 0 / **149.3** |
| 4 | 1.50 | 14% / 0 / 119.5 | **44%** / 0 / **153.6** |
| 8 | 1.50 | 41% / **1** / 115.2 | **69%** / **0** / **153.6** |
| 8 | 2.00 | 25% / 17 / 119.5 | 16% / **2** / **157.9** |

At N=8 and lambda=1.5, Gusto serves **69% of requests within 1 s with not a single timeout**,
against native's 41% with one timeout. At lambda=2.0 native loses **17** requests to the deadline;
Gusto, **2**.

### Total capacity grows with N, and that is not an anomaly

Capacity in total lambda is higher at N=8 (1.00) than at N=2 and N=4 (0.50). That is expected: each
pod runs `--parallel 1`, so spreading the same total load over more pods reduces per-pod
queueing. At N=2 and lambda=0.75 each tenant offers 0.375 req/s and p95 goes to 1532 ms; at N=8 with
the same total lambda each offers 0.094 and p95 stays at 903 ms. **More tenants give more aggregate
serving capacity**, up to the GPU ceiling.

### Fairness under an SLO, in the regime where it means something

At the N=8 knee, Gusto's **worst tenant** meets the SLO on **100%** of its requests; native's,
on **75%**. At lambda=1.0 with N=4 the comparison is 83% against 62%, and with N=2, 79% against 64%.
Gusto serves the worst-served tenant better in all three configurations.

> **A metric defect fixed here.** The sidecar's `slo_min_tenant_*` field counted a tenant that
> receives **no requests at all** as 0% attainment, so at low load the aggregate read 100% and
> the "worst tenant" 0%. At lambda=0.25 with N=8 only 5 of the 8 tenants receive demand. The figures
> above are recomputed from the per-request JSONL excluding zero-demand tenants; the
> `tenants_con_demanda` column of `LLAMA_SLO_capacidad.csv` shows how many enter each point.

### Caveats, unvarnished

- **n = 1.** One repetition per cell. No confidence intervals. Repetitions 2 and 3 are running;
  until they land these differences have **no statistical backing**, only a direction that is
  consistent across all three N.
- **The backend is not restarted between cells.** It was restarted before the grid, and the
  order was N=8 -> N=4 -> N=2, so any degradation from accumulating connections would penalise
  the *small* cells, not the N=8 headline. Repetitions 2 and 3 restart per cell.
- **Grid resolution**, as above: capacity in lambda does not discriminate at these steps.
- **Transport provenance was not recorded** in these points' sidecar: the `transport` field is
  empty because `bench.py` reads `GVIRTUS_CONFIG` from the process that invokes it, and here
  that variable lives inside the container. The arm is fixed by the label and the harness, not
  by the sidecar. To be corrected before the next packaging.

Data: `LLAMA_SLO_capacidad.csv` (37 rows; 36 from the grid plus the `r0` smoke point, which
carries `rep=0` and is excluded from the analysis). Figure: `figures/fig5_slo_capacidad.pdf`.
Harnesses: `~/mt_slo_sweep.sh`, `~/analiza_capacidad.py`, `~/figura_capacidad.py`.

# 4. Per-tenant serving fairness, normalised by demand (2026-08-02)

The fairness audit's serving half, consolidated here because it is a llama result. Method,
cross-workload comparison and the metric audit live in `FAIRNESS_RESULTS.md`.

**How the repetitions were separated.** `bench.py` opens the per-request JSONL in **append**
mode and the three repetitions of each multi-tenant cell share a `label`, so they were
concatenated with no separator. They were segmented by cumulative `completed+fail` counts from
`summary.csv`; the sum matches the line count **exactly** on all ten labels checked, with
non-uniform segments such as `[310, 306, 307]`.

### Stable regime

| system | N | **demand imbalance** | Jain completion | Jain SLO 5 s | completion worst--best |
|---|---:|---:|---:|---:|---|
| Gusto GPUDirect | 8 | 4.0x | **1.0000** | **1.0000** | 1.00--1.00 |
| Gusto GPUDirect | 10 | **7.0x** | **1.0000** | **1.0000** | 1.00--1.00 |
| native | 8 | 4.0x | 1.0000 | 0.9982 | 1.00--1.00 |

One tenant received **seven times more requests than another** from the arrival draw alone, and
every tenant still completed 100% of its own.

**This settles the §8 reading.** The Jain values of 0.719--0.804 that `summary.csv` reports for
N=6/8/10 are computed over *per-tenant throughput* -- that is, over unequal demand. Normalised
by demand, fairness is **exactly 1.0000**. §8 already said the decline was an arrival artefact
and should not be cited as a GVirtuS limitation; that reading is now demonstrated rather than
asserted.

### Saturation, against a permutation null

| system | N | observed Jain completion | null p50 | **p** | completion worst--best | SLO 5 s |
|---|---:|---:|---:|---:|---|---|
| Gusto | 8 | 0.957 / 0.968 / 0.930 | 0.934 / 0.932 / 0.933 | **0.77 / 0.90 / 0.48** | 0.15--0.34 | 0.00--0.03 |
| native | 8 | 0.960 x3 | 0.898 / 0.899 / 0.901 | **0.93 / 0.94 / 0.93** | 0.13--0.23 | 0.00--0.03 |

The null shuffles the **tenant labels** while preserving the observed demand: it assumes
neither multinomial arrivals nor equal service times, which the closed form `n/(n+k-1)` does
and which is **not verified** here. The observed Jain falls inside the null in **every** cell.

### Paired comparison and equivalence

13 cells matched on (N, lambda, repetition):

| quantity | mean difference (Gusto - native) | bootstrap CI95 | verdict |
|---|---:|---|---|
| Jain of completion fraction | **-0.0027** | **[-0.0079, +0.0009]** | **EQUIVALENT** (TOST, declared margin +/-0.05) |
| mean completion fraction | **+0.0764** | **[+0.0426, +0.1122]** | **excludes zero: Gusto serves more** |

At N=2, lambda=2 the difference is **+17.3 points** (0.9706 against 0.7975).

On power, unvarnished: with 13 pairs the TOST is weak. A CI that **fits** inside the margin is
evidence of equivalence; one that did not fit would **not** be evidence of difference.

**Classification: C -- the difference is explained by demand and experimental noise.** No
evidence of Gusto-introduced unfairness in serving, in either regime.

### Two traps this table exposes

- **The SLO Jain index means nothing under saturation.** Values like 0.25 or 0.50 arise because
  only one or two tenants have any non-zero attainment and the rest are exactly zero. The
  `SLO5s worst--best` column says it without ambiguity: **0.00--0.03**. That is **uniform
  starvation**, and the CSV labels it as such when the index is 0/0.
- **`jain_share_normalizada` is identical to `jain_completion_fraction`** in all 39 runs. Not a
  coincidence: with `NPRED=128` fixed, every completed request delivers exactly 128 tokens, so
  service share is proportional to completions and the index is scale-invariant. It is a
  redundant column, not a second piece of evidence.

### Limit that cannot be filled

This campaign has **no per-request timestamp** (the fields were added on 2026-08-02, after it
ran). So these are **not reconstructible** per tenant: completions inside the window against
completions during the drain, first completion, and longest no-progress interval. A serving
timeline needs a re-run with the patched `bench.py`.

Data: `llama_fairness_por_tenant.csv` (169 tenant-run rows), `llama_fairness_por_corrida.csv`
(39 runs). Figure: `figures/fig4_llama_por_tenant.pdf`.
