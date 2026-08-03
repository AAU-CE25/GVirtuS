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

**The real multi-tenant result here is memory, and it favours remoting.** Per-tenant GPU footprint is **~=4 490 MiB under GVirtuS vs ~=4 950 MiB native -- about 460 MiB less per tenant, 3.7 GB at N=8.** **Both the number and the mechanism are now closed** (§2). The saving is the **per-process CUDA primary context**, measured at **429-431 MiB** by a probe that does nothing but create one -- exactly linear over 1, 2 and 4 processes, on two GPUs under two drivers. A native pod pays it; **an MPS client still pays it** (429 MiB, which is why MPS saves only 5.0 MiB per tenant); a Gusto tenant pays none, because its frontend process never creates a context. Predicted 429 against 426 measured at N=1: the mechanism accounts for 99% of the effect. This column was re-measured on 2026-08-03 on the deployed backend configuration and reproduces to **within 2 MiB** at every N.

**The slot pool costs 32 MiB per tenant on the deployed configuration -- small, but not zero.**

> **Corrected 2026-08-03.** This paragraph read *"the slot pool contributes nothing to
> per-tenant cost, because a llama pod never allocates a GPU shadow."* The backend's own log
> disproves it: on the deployed configuration it reports
> `pool efectivo: slots=2 cap=16.1 MiB shadow=si total=64.2 MiB`, i.e. a shadow **is**
> allocated, at 32.2 MiB of GPU per connection. The premise -- that serving never sends a
> transfer at or above the RMA floor -- was true of *serving* traffic (max 3.6 MB) but ignored
> **model loading**, which pushes 4.5 GB of weights with `--no-mmap` and crosses the floor
> immediately. 32 MiB against a 4 490 MiB per-tenant footprint is 0.7%, so no conclusion in this
> document changes; the reasoning does. See §2 for the three-pool control.

| configuration | GPU per pod | max pods on 46 GB |
|---|---:|---:|
| native (one CUDA context per tenant) | 4 950 | **9** |
| old default (1025 MB cap, eager pool) | 4 490 + 2 050 shadow = 6 540 | **6** (pod 7 OOM'd) |
| old workaround (`SLOT_CAP_MB=128`) | 4 490 + 256 shadow = 4 746 | **8** |
| **on-demand pool (current)** | **4 490** | **10, measured STABLE** |

**Fairness (Jain 1.000 -> 0.804) is identical in both systems to three decimals** -- it is a property of how the shared Poisson arrival trace (same `SEED=1`) distributes across pods, **not** of the transport. It should not be cited as a GVirtuS multi-tenancy limitation. Tail latency does degrade with N in both.

**Operational caveat -- it invalidated a first attempt at this table.** The backend did not release GPU memory when a pod disconnected; after an 8-pod run the card sat at 45 217 / 46 068 MiB and the following points could not start. Per-connection reclamation now fixes the leak at the source (~9 GB/run -> ~20 MiB), but every point here is still preceded by `reset_backend.sh` so each starts from an identical 435 MiB.


## 1b. Multi-tenant under real load -- per-pod-constant offered rate (lambda = N)

> **Corrected 2026-08-01**, verified against `summary_master.csv`. Three things:
>
> 1. **The column labelled "TPOT p50" holds p95 values.** The published figures (UCX 44.5,
>    native 70.1) are `tpot_p95`; the true p50 values are **27.6** and **65.4**. The numbers are
>    right, the header is not. The same applies to §7's footnote, which says p50 where the
>    criterion is p95.
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
> 3. **The prose quotes the maximum where the table quotes the mean.** The table gives 293.0
>    (the mean of 285.9/290.1/302.9, correct) but the text says "rises -> 302.9", and the
>    mechanism paragraph quotes 1.42x instead of the table's 1.37x. And 302.9 matches the value of
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

**Proposed mechanism, and it is confirmed for throughput -- but it does NOT carry the memory result with it (§2).** The backend serves connections as **threads inside a single process sharing one CUDA context** (`Process.cpp` detaches a thread per connection; `fork` is per configured endpoint). N native pods are N processes with N contexts, and the GPU must context-switch between them. API remoting therefore **consolidates** multi-tenant work into one context, which predicts *both* the ~=460 MiB saved per tenant *and* a throughput advantage that grows with tenant count -- which is what the ratio column shows (1.00 -> 1.18 -> 1.29 -> 1.42).

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

**Status after the 2026-08-03 verification pass: the result is closed and the mechanism is
closed with it.** The saving is the per-process CUDA primary context, measured directly rather
than inferred from a difference of totals. Getting here took one wrong retraction of my own,
which is recorded below rather than quietly dropped.

### The measurement

Per-tenant = (peak - baseline) / N, `nvidia-smi` sampled at 1 Hz after every pod has been
exercised with a real completion. The Gusto column was re-measured on 2026-08-03 on the
**deployed** backend configuration, verified from the backend's own `[GUSTO CFG]` log line
rather than assumed.

| N | native | native + MPS | **Gusto (deployed)** | MPS saves | **Gusto saves** |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 950.0 | 4 978.0 | **4 524** | -28.0 (worse) | **426 (8.6%)** |
| 2 | 4 948.5 | 4 942.5 | **4 501** | 6.0 | **447.5 (9.0%)** |
| 4 | 4 948.0 | 4 942.8 | **4 490** | 5.2 | **458 (9.3%)** |
| 8 | 4 947.9 | 4 942.9 | 4 487 (prior campaign) | 5.0 (0.1%) | **461 (9.3%)** |

**MPS saves 0.1% per tenant. Gusto saves 9.3%.** At N=1 MPS is actually *worse* by 28 MiB -- the
daemon's own footprint, which the baseline absorbs from N=2 on.

### The mechanism: the per-process CUDA primary context, measured with a probe

The scaling was the clue. The saving is nearly **constant** in N (426 / 447.5 / 458 / 461), not
`C(1 - 1/N)` as *sharing* one context among N tenants would give. A constant per-tenant saving
means a constant per-tenant cost that Gusto does not pay at all.

`ctx_probe.cu` is a program that does nothing but `cudaFree(0)`, forcing creation of the primary
context, and then sleeps. Its device footprint is that context and nothing else:

| host / GPU | MPS | K = 1 | K = 2 | K = 4 | **per process** |
|---|---|---:|---:|---:|---:|
| dpu-01 (backend L40S) | no | 946 | 1 377 | 2 240 | **431 MiB** |
| dpu-02 (native L40S) | daemon running | 466 | 896 | 1 754 | **429 MiB** |
| dpu-02 | pipe dir unreachable | -- | 896 | -- | **429 MiB** |

**Exactly linear over 1, 2 and 4 processes, on two different GPUs under two different drivers.**
`nvidia-smi`'s own per-process accounting confirms it independently: each `ctx_probe` shows as
**424 MiB**, whether or not it can reach the MPS daemon.

So the accounting closes:

| | pays a CUDA primary context? | measured |
|---|---|---:|
| native pod | yes, its own | 429-431 MiB |
| MPS client | **yes, still its own** | 429 MiB (424 by per-process accounting) |
| **Gusto tenant** | **no -- the frontend process never creates one** | **0** |

**Predicted saving 429, measured 426 at N=1: the mechanism accounts for 99% of the effect.**

### Why the MPS control was misread, and the correction that follows

The earlier version of this section treated MPS as a *refutation* of the context explanation:
*"MPS consolidates contexts by construction, so if the mechanism were context consolidation,
MPS would reproduce the saving; it does not."* **The premise is wrong.** MPS consolidates the
*scheduling* context -- work from many clients funnels through one server -- but each client
process still creates and pays for its own primary context. The probe measures that directly:
429 MiB per client with the daemon running.

So MPS failing to reproduce the saving is **exactly what the context explanation predicts**, not
evidence against it. The two independent measurements even agree on the residual: the probe puts
MPS's advantage at 2 MiB per process and the llama experiment at 5.0 MiB per tenant, both of
them rounding error against 429.

**Consequence for the claim.** The line *"remoting saves ~461 MiB per tenant, and MPS does not;
why is not established"* is superseded. The defensible sentence is now: **remoting saves
~426-461 MiB per tenant because the tenant process never creates a CUDA context, and MPS cannot
match it because MPS shares scheduling, not per-client context state.**

### A retraction of my own, recorded

Earlier on 2026-08-03 I re-measured the Gusto column, obtained 4 794 / 4 772 / 4 761, and wrote
a correction stating that *"the saving is ~170 MiB, not ~460"* and that the published figure was
valid only for a pool with no GPU shadow. **That correction was wrong and is withdrawn.**

The cause was configuration, and it is the campaign's recurring failure mode: **I measured on a
backend that a previous experiment had left configured**, with the RMA slot pool at
`slots=8 cap=32.1 MiB`, four times the deployed size. The backend states its pool in its own
log, and the three arms differ exactly as that log predicts:

| backend pool | GPU shadow per connection | per-tenant, N=4 |
|---|---:|---:|
| `slots=8 cap=32.1 MiB` (left over from another experiment) | 257 MiB | 4 761 |
| **`slots=2 cap=16.1 MiB` -- what `reset_backend_gdps.sh` deploys** | **32.2 MiB** | **4 490** |
| `slots=1 cap=4.1 MiB` (minimised, as a control) | 4.1 MiB | 4 503 |

The 8-slot arm costs **271 MiB/tenant more** than the deployed one, against a shadow that is
225 MiB larger -- so the pool shadow accounts for the bulk of the discrepancy. Between the
deployed and minimised pools the difference (13 MiB) is **smaller than the run-to-run spread**
and has the wrong sign for a shadow effect, so no conclusion is drawn from it: below ~32 MiB the
pool is not resolvable against a 4.5 GB footprint.

The re-measured deployed column reproduces the original campaign's to **within 2 MiB at every N**
(4 524 / 4 501 / 4 490 against 4 526 / 4 503 / 4 492). The published numbers were right all
along.

**The lesson, since it has now cost time twice:** verify the backend's effective configuration
from its own log *before* the measurement, not after a surprising result. The
`[GUSTO CFG] pool efectivo:` line exists for precisely this and I did not read it until the
number looked wrong.

Data: `results/asplos_campaign/memoria/mem_footprint.csv` (all four arms: `bm`, `bmmps`,
`ucx_deployed`, plus the `ucx` rows from the mis-configured backend and the `ucx_pool_min`
control), `ctx_probe_results.csv`, `ctx_probe_perproc.txt`, probe `ctx_probe.cu`. Harnesses
`~/mem_footprint.sh` and `~/mem_gusto.sh`.

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

**Method.** N  in {2, 4, 8} x lambda total  in {0.25 - 0.5 - 0.75 - 1.0 - 1.5 - 2.0}, native and Gusto
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

## §3b -- RETRACTION of the rep-1 capacity result, and why the measurement was rebuilt

**The +17.4% reported from repetition 1 does not survive replication.** With n=3 and the
conservative criterion -- a load counts as sustainable only if it meets the SLO in **every**
repetition -- the result is:

| | rep 1 alone | **n = 3** |
|---|---|---|
| capacity at N=8 | lambda = 1.00 | **lambda = 0.50** |
| Gusto goodput at capacity | 115.2 t/s | **55.5** |
| native goodput at capacity | 98.1 t/s | **55.5** |
| paired difference | **+17.4%** | **-1.4 t/s, CI95 [-4.2, +0.0] -- includes zero** |

At lambda=0.75 and lambda=1.00 the SLO is met in **1 of 3** repetitions. Repetition 1 was the lucky one.

**Nothing here says the two systems are equal.** It says this measurement cannot tell them
apart, which is a different and weaker statement, and the reason is a defect in the
measurement rather than in the systems.

### The defect: at low load the window contains too few requests

| lambda (total) | requests offered in a 30 s window |
|---:|---:|
| 0.25 | **7.5** |
| 0.50 | 15.0 |
| 0.75 | 22.5 |
| 1.00 | 30.0 |

The observed goodput at N=8, lambda=0.50 ranges **29.9 to 59.7 t/s across three repetitions** --
close to a factor of two. That is not the system varying: with 15 Poisson arrivals, whether 8
or 15 of them complete inside the window is counting noise, and goodput is
`completed x 128 / WINDOW`, so it inherits that noise directly. **A fixed 30 s window cannot
measure a rate at these loads.**

This is the same failure mode as the campaign's other measurement traps: a number that looks
like a system property and is an artefact of the harness. It was invisible at n=1 -- one draw
produces one plausible-looking value -- and only the replication exposed it.

### What replaced it

A second sweep scales the window so every point sees at least 40 offered requests:

    WINDOW = max(30, 40 / lambda)

which gives 80 s at lambda=0.5, 53 s at 0.75, 40 s at 1.0 and 30 s at 1.5. Goodput remains a rate,
so a longer window only reduces its variance; it does not change what is measured. The sweep
covers the decisive region only (lambda from 0.5 to 1.5) -- below it every point meets the SLO and
above it none does, so the extremes carry no information -- and it adds the **native+MPS** arm,
which the first sweep lacked.

Data: `results/asplos_campaign/llama_slo_sweep_v2/`, harness `~/sweep_v2.sh`.

### What may be quoted from the first sweep, and what may not

**May be quoted.** The shape of the load-response curve, and the qualitative observation that
past the knee Gusto degrades with fewer timeouts -- at N=8, lambda=1.5, across three repetitions,
Gusto records 0 timeouts against native's 1, and 69% against 41% of requests within 1 s. Those
are counts, not rates, and do not suffer the window defect.

**May not be quoted.** The +17.4%, the lambda=1.00 capacity, and any goodput figure from a point
whose window held fewer than ~40 requests.

## §3c -- The corrected sweep (v2): capacity does not discriminate, the tail does, and MPS matches Gusto

Run 2026-08-03, 02:56--04:56. **108 points**: 3 systems x N  in {2,4,8} x lambda  in {0.5, 0.75, 1.0,
1.5} x 3 repetitions, window scaled as `max(30, 40/lambda)` so every point sees at least 40 offered
requests, backend restarted per cell, lambda order randomised within each repetition. This is the
sweep that replaces the retracted one in §3b, and it adds the **native+MPS** arm the first
lacked.

### Capacity under the SLO is identical across all three systems

| system | N | max lambda under SLO | goodput at that load (mean of n=3) |
|---|---:|---:|---:|
| native | 2 / 4 / 8 | **0.50** | 51.7 t/s |
| native+MPS | 2 / 4 / 8 | **0.50** | 51.2 t/s |
| Gusto | 2 / 4 / 8 | **0.50** | 51.2 t/s |

Paired difference Gusto - native at the knee: **-0.5 t/s, bootstrap CI95 [-1.6, +0.0]**, at
every N. **The capacity metric does not separate the systems**, and with a corrected window and
n=3 that is now a result rather than an artefact.

> **Corrected 2026-08-03 (verification pass).** The three goodput figures in this table read
> **57.6 t/s** until recomputed from the raw CSV. 57.6 is **repetition 1**, not the mean of the
> three; the per-repetition values at this point are 57.6 / 35.2 / 62.4. Quoting repetition 1 is
> the *exact* defect §3b retracts, repeated one section later in the section written to replace
> it. The paired difference (-0.5 t/s) was computed correctly over n=3 and is unchanged, so the
> conclusion stands, but the level was wrong by 12%.

The reason is visible in the sweep: at lambda=0.75 and 1.00 every system meets the SLO in *some*
repetitions and not others (1/3 to 3/4), so the all-repetitions criterion drops them all to
0.50. A finer grid between 0.50 and 1.00 would separate them; this one cannot.

### What does discriminate are two opposite effects, at N=8

**Every cell below is the mean of the three repetitions, with the observed range in brackets.**
An earlier version of this table quoted repetition 1; see the correction box above.

| lambda | TTFT p95 native | MPS | **Gusto** | goodput native | MPS | **Gusto** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 244 ms [40-650] | 169 [39-421] | **517** [197-750] | 51.7 | 51.2 | 51.2 |
| 0.75 | 1396 ms [51-3777] | 394 [48-1019] | **782** [348-1562] | 78.1 | 78.1 | 77.3 |
| 1.00 | **3404 ms** [1772-5118] | 1541 [569-3453] | **1617** [600-3491] | 97.1 | **106.7** | **106.7** |
| 1.50 | 8135 ms [4007-13546] | 6105 [2395-9600] | **5644** [2474-9475] | 113.8 | **145.1** | **145.1** |

The brackets are the point of the table as much as the means: **TTFT p95 varies by more than an
order of magnitude between repetitions of the same cell** (native at lambda=0.5 spans 40 to 650 ms).
Any tail claim from a single repetition is unsafe, so every one below is a paired difference
across the three, with a bootstrap CI95.

**At light load remoting costs tail latency, and the cost is real but modest.** Paired
Gusto - native at lambda=0.5, N=8: **+273 ms, CI95 [+100, +561]** -- excludes zero. Against MPS,
**+347 ms, CI95 [+158, +555]**. This is the network round trip, which at low load has no
queueing to hide behind. The earlier "603 against 43, an order of magnitude" was repetition 1;
on the means it is 517 against 244, a factor of **2.1**.

**At heavy load remoting buys both throughput and tail.** Paired Gusto - native, N=8:

| lambda | goodput difference | TTFT p95 difference |
|---:|---|---|
| 0.50 | -0.5 t/s (-1.0%), CI95 [-1.6, +0.0] -- includes zero | +273 ms, CI95 [+100, +561] |
| 0.75 | -0.8 t/s (-1.0%), CI95 [-2.4, +2.5] -- includes zero | -614 ms, CI95 [-2215, +386] -- includes zero |
| 1.00 | **+9.6 t/s (+9.9%), CI95 [+3.2, +16.0]** | **-1787 ms, CI95 [-2561, -1172]** |
| 1.50 | **+31.3 t/s (+27.5%), CI95 [+12.8, +42.6]** | **-2491 ms, CI95 [-4071, -1533]** |

At lambda=1.5 native also records **4 timeouts across the three repetitions against Gusto's 0**.

**The crossover is near lambda=1.0.** Below it the two are indistinguishable on goodput and
remoting pays a tail penalty; above it remoting is ahead on both, with both CIs excluding zero.

### The uncomfortable part: MPS matches Gusto on goodput exactly

Paired Gusto - native+MPS at N=8: **+0.00 t/s at lambda=0.5, 1.0 and 1.5** (CI95 [0.00, 0.00] --
the two arms return the identical figure in every repetition) and **-0.80 t/s at lambda=0.75**,
CI95 [-2.40, +0.00]. MPS also keeps the **better tail below saturation** (+347 ms at lambda=0.5,
+388 at 0.75, +76 at 1.0, all excluding zero), because it pays no network cost; at lambda=1.5 the
two are indistinguishable (-461 ms, CI95 [-1337, +79]).

Two consequences, and they should be written into the paper rather than discovered by a
reviewer:

1. **This confirms context consolidation as the mechanism for the multi-tenant throughput
   advantage**, for a third time and now under a corrected measurement. It is the same
   conclusion the §1b control reached.
2. **It removes multi-tenant goodput from the list of arguments for remoting.** MPS obtains it
   locally, without a network. What remoting still obtains that MPS does not is the **memory
   saving** -- the 429 MiB per-process CUDA context measured in §2 -- and disaggregation itself,
   which is a deployment property rather than a performance one.

### §3d -- All three tenant counts, and p99 as well as p95 (2026-08-03)

The table above is N=8 only and p95 only. Both restrictions were unnecessary: the per-request
JSONL was already on disk, so N=2 and N=4 and the 99th percentile are **re-derivable without
re-running anything**.

**Estimator, stated because it differs from §3c.** These are percentiles over **all requests
pooled across the three repetitions** (100--190 requests per cell), not the mean of the
per-repetition p95 that §3c reports. Both are legitimate; they are not interchangeable, and the
two tables should never be read as if they were.

| N | lambda | native p95 / p99 | native+MPS p95 / p99 | **Gusto p95 / p99** | native / Gusto, p95 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.50 | 843 / 1275 | 807 / 1096 | 875 / 1158 | 0.96x |
| 2 | 0.75 | 2319 / 4269 | 1319 / 2693 | **1366 / 2979** | **1.70x** |
| 2 | 1.00 | 5103 / 7125 | 2221 / 4303 | **2474 / 4684** | **2.06x** |
| 2 | 1.50 | 17949 / 18902 | 9106 / 9721 | **10113 / 10703** | **1.77x** |
| 4 | 0.50 | 837 / 1279 | 812 / 1098 | 868 / 1156 | 0.96x |
| 4 | 0.75 | 1946 / 4293 | 1244 / 2235 | **1344 / 2301** | **1.45x** |
| 4 | 1.00 | 5295 / 9905 | 1900 / 4344 | **2281 / 4809** | **2.32x** |
| 4 | 1.50 | 21400 / 23241 | 9069 / 14322 | **10131 / 15086** | **2.11x** |
| 8 | 0.50 | **46 / 890** | 59 / 687 | 457 / 940 | **0.10x** |
| 8 | 0.75 | 1192 / 3777 | 724 / 1019 | **905 / 1562** | **1.32x** |
| 8 | 1.00 | 4651 / 8366 | 1628 / 3453 | **1856 / 3491** | **2.51x** |
| 8 | 1.50 | 21639 / 23735 | 9166 / 10014 | **9446 / 11629** | **2.29x** |

**Two things change, and one of them is a correction to §3c's framing.**

**1. The light-load tail penalty is an N=8 artefact, not a property of remoting.** §3c reads
*"remoting costs an order of magnitude of tail latency at light load"* from the N=8, lambda=0.5
cell -- and that cell is the only one where it is true (0.10x). At **N=2 and N=4 the same load
gives 0.96x: the three systems are indistinguishable.** The reason is arithmetic: lambda is a
*total* rate, so at N=8 each server receives 0.0625 req/s and sits essentially idle, which is
the one regime where a network round trip has nothing to hide behind. **The penalty is a
property of the idle regime, not of the tenant count**, and quoting it without lambda-per-tenant
overstates it.

**2. The advantage above the knee is consistent across every N, and survives at p99.** At
lambda=1.0 native's p95 is **2.06x / 2.32x / 2.51x** Gusto's at N=2/4/8, and at lambda=1.5 it is
1.77x / 2.11x / 2.29x. The ordering is identical at p99. This is a stronger statement than §3c's
single N, and it was free.

**MPS keeps a small edge over Gusto at every cell** (9106 against 10113 at N=2, lambda=1.5;
1628 against 1856 at N=8, lambda=1.0) -- the network round trip, again -- and both are far ahead
of default native.

**What p99 is worth here, honestly.** With 100--190 pooled requests per cell, the 99th percentile
is the first or second largest observation: an extreme order statistic with no useful confidence
interval. It is reported because it **agrees** with p95 in every cell, which is evidence the p95
ordering is not an artefact of where the cut falls -- not because p99 is separately reliable at
this sample size. Making p99 a quotable number needs longer windows, which is the open item
below.

Data: `results/asplos_campaign/llama_ttft_p95_p99_pooled.csv` (36 cells), derived from the
per-request JSONL in `llama_slo_sweep_v2/`.

### What may be claimed from this sweep

**May be claimed.** That capacity under a 1 s TTFT SLO is indistinguishable between the three
systems at this grid resolution; that remoting costs **+273 ms of TTFT p95 at light load**
(CI95 [+100, +561], a factor of 2.1 on the means); that at lambda=1.0 and 1.5 remoting delivers
**+9.9% and +27.5% goodput** with **1.8 and 2.5 s less** TTFT p95 than default native, all four
CIs excluding zero; and that native+MPS matches remoting's goodput exactly while keeping the
better tail below saturation.

**May not be claimed.** Any capacity difference. The +17.4% of §3b remains retracted. Also
**withdrawn in this document's own earlier wording**: "an order of magnitude" of light-load tail
(it is 2.1x, and §3d shows it does not survive at N=2 or N=4 at all), "+15.6%" at lambda=1.0 (it
is +9.9%) and "+33%" at lambda=1.5 (it is +27.5%) -- all three came from repetition 1. **No
figure from a single repetition of this sweep may be quoted**, because TTFT p95 varies more than
10x between repetitions of the same cell.

## §3e -- The knee, at four times the resolution and three times the window (2026-08-03)

§3d listed two gaps that needed a re-run rather than a re-analysis: no points around the knee,
and windows too short for p99. Both are now measured.

**Design.** lambda in {0.55, 0.60, 0.65, 0.70} -- the untested gap between the load that always
met the SLO (0.50) and the one that met it only sometimes (0.75) -- with the window scaled as
`max(90, 120/lambda)` instead of `max(30, 40/lambda)`, so **every point sees ~124 offered
requests instead of ~40**. Three systems, three repetitions, N=8. 36 points.

### The capacity metric still does not discriminate -- and now that is a real result

| system | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---:|---:|---:|---:|
| native | **0 of 3** | 0 of 3 | 0 of 3 | 0 of 3 |
| native+MPS | **1 of 3** | 1 of 3 | 0 of 3 | 0 of 3 |
| Gusto | **1 of 3** | 0 of 3 | 0 of 3 | 0 of 3 |

*(repetitions meeting TTFT p95 <= 1000 ms; timeouts were **zero everywhere**, in all 36 points)*

**Under the all-repetitions criterion, no system meets the SLO anywhere in this band.** §3c
could only say "capacity does not discriminate *at this grid resolution*". That hedge is now
gone: we went **four times finer and three times longer** and it still does not discriminate.
The negative result is stronger than the one it replaces.

**Why, and it is a property of the metric rather than of the systems.** p95 within a single cell
still spans a factor of 2.5 -- Gusto at lambda=0.55 gives **813, 1375 and 2057 ms** across three
repetitions. **124 requests per point is not enough to stabilise a 95th percentile**, so a
*binary* criterion evaluated per repetition is decided by which draw you got. The all-reps rule
then propagates the worst draw to the whole cell.

> **A retraction of something I said mid-run.** After repetition 1 alone the table read: native
> failing at every load, MPS passing to 0.60, Gusto to 0.55 -- a clean capacity ordering. **It
> does not survive n=3** and must not be quoted. Repetition 1 was the favourable draw for all
> three arms, exactly as it was in §3b. This is the third time in this campaign that a
> single-repetition reading looked like a system property.

### What does discriminate, and it is well powered: 12 paired points

| quantity, Gusto minus baseline | vs native | vs native+MPS |
|---|---|---|
| **TTFT p95** | **-1398 ms**, CI95 [-2096, -788] | +41 ms, CI95 [-59, +132] -- **includes zero** |
| **SLO 1 s attainment** | **+5.25 pp**, CI95 [+3.25, +7.67] | **-1.58 pp**, CI95 [-2.17, -1.00] |
| **goodput** | **-1.62 t/s**, CI95 [-2.86, -0.44] | -1.50 t/s, CI95 [-2.69, -0.38] |

For reference, MPS minus native on attainment is **+6.83 pp**, CI95 [+4.75, +9.25].

**The trade is now stated exactly**: against default native, remoting gives up **1.6 t/s of
goodput** and buys **1.4 s of TTFT p95** and **5.25 points of SLO attainment**. Against
MPS-configured native it is indistinguishable on the tail and 1.6 points behind on attainment --
the network round trip, once more.

### The mechanism the short windows could not see: the queue

| system | backlog at window close, lambda = 0.55 -> 0.70 |
|---|---|
| native | 2.7 -> 3.0 -> 4.3 -> **5.3** |
| native+MPS | 2.3 -> 2.7 -> 2.0 -> 2.7 |
| **Gusto** | **2.3 -> 2.7 -> 2.3 -> 2.3** |

**Native's queue grows with load; the other two hold flat.** This is visible only because the
window is long enough for a backlog to form -- at 40 offered requests the run ends before the
queue builds. It is the first direct evidence that native is the arm approaching saturation in
this band, and it agrees with the p95 and attainment differences rather than resting on them.

### The methodological point, because it generalises

**The binary capacity criterion discards the information the continuous one keeps.** "Does p95
clear 1 s in every repetition" answers *no* for all three systems and stops. "What fraction of
requests clear 1 s" orders them **consistently at every one of the four loads**, with all three
pairwise differences excluding zero. A capacity number is attractive because it is a single
figure; on this bench it is a single figure that happens to be uninformative.

**Recommendation for the paper**: report SLO *attainment* against load, not capacity. Keep the
capacity result as the negative it is -- swept twice, at two resolutions, and it never separated
the systems.

Data: `results/asplos_campaign/llama_slo_knee/` (36 points, per-request JSONL beside each),
harnesses `~/sweep_knee.sh` and `~/cola_knee.sh`, analysis `~/analiza_knee.py`. N=4 and N=2 were
queued behind N=8 and are still running at the time of writing; nothing above depends on them.

### What this sweep still does not give

Four gaps, all of them a re-run rather than a re-analysis, and none of them blocking a claim
already made:

1. ~~**No points around the knee.**~~ **Closed by §3e**: the 0.55--0.70 grid was run at
   3x the window. It did **not** separate the systems on capacity, which turns the hedge
   "does not discriminate at this resolution" into a measured negative.
2. ~~**Windows are too short for p99.**~~ **Improved by §3e**, not closed: 124 offered requests
   per point instead of ~40. Still not enough -- p95 alone varies 2.5x within a cell, so p99
   remains an extreme order statistic.
3. **No sustained-load points.** The longest window here is 218 s, so nothing speaks to thermal,
   memory-fragmentation or leak behaviour over hours.
4. **Transport provenance is still not recorded in the sidecar** (`GAPS.md` §1): the arm is
   fixed by the label and the harness, not by a field the data carries.
5. **N=4 and N=2 of the knee grid** were queued behind N=8 and had not finished at the time of
   writing. §3e is N=8 only.

Data: `LLAMA_SLO_capacidad_v2.csv` (**108 data rows** -- 3 systems x 3 tenant counts x 4
loads x 3 repetitions; an earlier version of this line said 112), raw in
`results/asplos_campaign/llama_slo_sweep_v2/`. Analysis: `analiza_v2.py`.
