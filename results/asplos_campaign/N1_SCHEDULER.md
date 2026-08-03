---
title: "Why service arrives in bursts -- the mechanism, and the intervention that removes it"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

The fairness audit established **that** the backend shares the GPU unevenly under equal fixed
work, and by how much: at N=8 in miniBUDE one tenant runs at 1.00x its single-client rate while
another runs 4.87x slower, where native and native+MPS share to within 1.03x. It did not
establish **why**.

**It is established now.** Four candidate mechanisms were tested and three refuted -- the last
of them by a control that had been sitting unused in the data. The one that survives is that
**the backend does not arbitrate**: it serves each RPC on arrival, first-come-first-served, and
because every client is self-clocked the tenant that gets ahead re-submits first and stays
ahead. The claim is not left as an inference: §5 adds the missing arbitration behind an
env-gate and the inequality drops from **5.02 to 3.09** (-38%, CI95 excluding the baseline) at
**+0.6% makespan**, with the runaway tenant pulled off its solo rate.

# 1. The finding, and the correction the verification pass forced

Ten cohorts of eight tenants, identical work, two backend configurations. In each cohort, the
tenant with the lowest slowdown:

| arm | cohort | fastest tenant | its slowdown | slowest tenant | its slowdown |
|---|---:|---:|---:|---:|---:|
| stream0 | 1 | **t1** | 1.19 | t5 | 4.01 |
| stream0 | 2 | **t1** | 1.14 | t5 | 4.25 |
| stream0 | 3 | **t1** | 1.09 | t6 | 3.40 |
| stream0 | 4 | **t1** | 1.06 | t6 | 3.68 |
| stream0 | 5 | **t1** | 1.15 | t4 | 4.87 |
| stream1 | 1 | **t1** | 1.05 | t3 | 6.71 |
| stream1 | 2 | t8 | 1.01 | t3 | 2.03 |
| stream1 | 3 | **t1** | 1.05 | t2 | 6.95 |
| stream1 | 4 | **t1** | 1.06 | t2 | 7.20 |
| stream1 | 5 | **t1** | 1.05 | t4 | 6.46 |

**The first tenant to connect is the fastest in 9 of 10 cohorts**, at 1.01--1.19x -- that is, at
its single-client rate -- while the slowest sits at 2.0--7.2x. The containers are started in index order, so `t1` is the first to connect.

> ### That 9-of-10 is inflated by my own launcher, and must not be quoted
>
> The containers here are started in a tight loop, `mb1` through `mb8`, so start order and
> connection order coincide exactly. Checking the **original** miniBUDE campaign, whose
> launcher differs, the first tenant is fastest in only **16 of 34 cohorts (47%)** --
> t2 in 9, t3 in 4, t4 and t5 in 2 each, t8 in 1.
>
> So the effect is **real but weaker than this run suggests**: 47% against the 12.5% that
> random assignment over eight tenants would give is far beyond chance, but it is not the
> 90% measured here. The citable statement is **the 47% from the independent campaign**;
> the 9-of-10 is an artefact of a launcher that serialises the starts.

> ### Precision added 2026-08-03: the 47% is about the tenant *launched* first, not the one that *connected* first
>
> The 16-of-34 statistic was recomputed from `tabla_D_minibude_por_tenant.csv` and reproduces
> exactly -- but only under the key `tenant_id == 1`. It therefore answers **"is tenant t1 the
> fastest?"**, and the box above then reads t1 as "the first to connect". In the *independent*
> campaign that identification is not safe, which is precisely the objection the box raises
> against the 9-of-10.
>
> **Whether the first tenant to *connect* is favoured cannot be decided from this table.**
> `t_start_epoch_s` is recorded at **whole-second resolution**: the spread within a cohort is
> 0-3 s and up to six of the eight tenants share the same second. Keying on that column instead
> gives 24 of 34 (71%), but the value is set by how ties are broken, not by the data. A sample
> cohort:
>
>     id=2  t+0.00s  slowdown=1.001
>     id=1  t+1.00s  slowdown=2.375
>     id=3  t+1.00s  slowdown=4.874     <- five tenants tie at t+1
>     id=5  t+1.00s  slowdown=4.125
>     id=6  t+1.00s  slowdown=3.250
>     id=7  t+1.00s  slowdown=3.624
>     id=4  t+2.00s  slowdown=2.500
>     id=8  t+2.00s  slowdown=1.750
>
> **What may be claimed:** *the tenant launched first (t1) runs fastest in 16 of 34 cohorts,
> 47% against 12.5% by chance* -- a 3.8x enrichment, and the launch-order bias is real.
> **What may not:** that this is *connection* order specifically. Separating launch order from
> connection order needs a sub-second connection timestamp, which the backend can emit
> (`GVS_SCHED_TRACE` already stamps dispatch) but this table does not carry.

This holds in **both** backend configurations, so it is independent of the intervention below.
The defensible statement is weaker than it first appeared: **launch order biases service in
favour of the earliest tenant, in about half of cohorts rather than almost all of them, and the
advantage when it occurs is large** -- the favoured tenant runs at its single-client rate.
Whether the bias is fixed at connection time or earlier is **not yet separated**; the working
hypothesis remains that it is decided at connection and persists for the whole run.

# 2. Excluded: the shared legacy CUDA stream

**Hypothesis.** The backend serves each connection with its own thread but all threads share
one CUDA context. With the default stream, the frontend sends `0` and the backend launches on
`cudaLaunchKernel(..., 0)` -- the legacy stream, common to every connection. A stream is FIFO,
so sharing would decide service by submission order rather than by turn.

**Intervention.** A backend-side gate, `GVS_PER_CONN_STREAM=1`, substitutes a per-connection
non-blocking stream for the incoming `0`. Two arms, otherwise identical.

| arm | quantum | median multiplier | max | no-wait iterations | **slowest/fastest** |
|---|---:|---:|---:|---:|---:|
| shared legacy stream | 295.4 ms | 3.00 | 9.00 | 99 of 400 (25%) | **3.46** |
| per-connection streams | 295.5 ms | 5.95 | 9.96 | 95 of 400 (24%) | **6.39** |

**Refuted, and the intervention made it worse.** Giving each connection its own stream nearly
doubled the inequality, from 3.46 to 6.39. Whatever orders the work, it is not FIFO ordering
inside one shared stream.

The tenant-by-tenant view shows why the intervention backfires: with independent streams the
first tenant runs `1 1 1 1 1 1 1 1 1 1` -- completely unimpeded -- while the rest sit at 4 to 8.
Removing the shared queue let the incumbent pull further ahead.

# 3. Excluded: contention on a shared worker mutex

**Hypothesis.** `std::mutex` on Linux is not fair; the thread that releases it can reacquire
before a woken thread is scheduled. If every connection thread contended on one UCX worker
mutex, the incumbent would barge and keep winning -- which matches the symptom exactly.

**Refuted by the code.** `UcxCommunicator.cpp:1420` gives every accepted connection its own
mutex, and `:1398` sets `owns_worker_ = true`: each connection has its own UCX worker. There is
no shared lock for the incumbent to barge on.

# 4. Excluded: one shared CUDA context -- refuted by the MPS control

**Hypothesis.** With per-connection threads, workers, mutexes and even streams, what the tenants
still share is **one CUDA context**. Between *contexts* the driver time-slices, which would be
why native is fair at exactly N; between *streams of one context* there is no such arbitration,
so an early lead compounds instead of decaying.

**Refuted, by a control that was already in the data and had not been used.** **CUDA MPS puts
every client into a single context by construction.** If a shared context were sufficient to
produce the unfairness, native+MPS would show it. It does not:

| system | contexts | cohorts | **inequality (slowest / fastest)** |
|---|---|---:|---:|
| baremetal | 8 separate | 5 | **1.03** [1.02 - 1.05] |
| **baremetal + MPS** | **1 shared** | 5 | **1.02** [1.00 - 1.05] |
| tcp | 1 shared (backend) | 5 | **4.62** [3.23 - 5.87] |
| ucx_rdma | 1 shared (backend) | 14 | **4.31** [3.25 - 5.74] |
| ucx_gpudirect | 1 shared (backend) | 15 | **4.87** [3.37 - 6.25] |

Eight clients in one context share to within **2%**. Sharing a context is therefore not
sufficient, and the "no arbitration between streams of one context" account is false as stated.

The per-tenant slowdowns say the same thing more sharply. Under native, *every* tenant sits at
7.58--7.96x, which is a GPU divided eight ways and nothing else. Under remoting the distribution
is 1.00x to 6.25x: **one tenant is running as though it were alone on the machine.**

# 5. The mechanism: the backend does not arbitrate -- and adding arbitration removes it

What is left after four exclusions is not a property of CUDA at all. **MPS is fair because the
MPS server arbitrates; the driver is fair between contexts because it multiplexes them; the
Gusto backend does neither.** It serves each RPC as it arrives, first-come-first-served, and
because every client is self-clocked -- it does not submit iteration *k+1* until *k* returns --
the tenant that gets marginally ahead re-submits first and keeps its turn. Nothing in the path
damps the lead.

**This is testable by supplying the missing piece.** `GVS_FAIR_DISPATCH=1` adds deficit
round-robin at the launch point: a connection may not run more than `GVS_FAIR_LEAD` launches
ahead of the least-served active connection. It is a **causal probe, not a design proposal** --
if the account is right the inequality must fall; if it does not move, the account is wrong.

miniBUDE, N=8, same backend build, gate off then on:

| arbitration | reps | **inequality (mean)** | CI95 | median | **fastest tenant** | **makespan** |
|---|---:|---:|---|---:|---:|---:|
| FCFS (deployed) | 25 | **5.02** | [4.75, 5.27] | 5.12 | **216.4 GFLOP/s** | 24.89 s |
| deficit RR, lead=1 | 8 | **3.09** | [2.05, 4.34] | 2.40 | **75.9 GFLOP/s** | 25.05 s |

**The inequality falls 38%, and the CI of the intervention excludes the baseline mean.** The
number that matters most is the third column from the right: under FCFS the leading tenant
returns **216.4 GFLOP/s in every one of 25 repetitions** -- its solo rate, to three figures, as
if the other seven were not there. With arbitration it drops to 75.9. **The gate is doing
exactly what the account predicts: it stops one tenant from monopolising the dispatcher.**

**And it is nearly free.** Makespan moves from 24.89 s to 25.05 s, **+0.6%**. Per-tenant peak
GFLOP/s falls, because no tenant now gets an uncontended window, but the cohort finishes in the
same time.

**The residual, stated rather than hidden.** The gate does not equalise fully -- the ranges still
overlap (one repetition reached 6.67) -- and the reason is visible in what it counts. Deficit
round-robin over **launch counts** equalises *submissions*, not *GPU time*: a tenant whose
kernels run longer still gets more of the device for the same number of turns. Closing the rest
needs a **duration-weighted** gate, which needs per-kernel timing the current tracing does not
collect. That is the next step, not a claim.

# 6. What may be claimed

**May be claimed.**

- That the sharing is uneven, by how much, and that it is reproducible.
- That it is **not the data path** (identical over TCP), **not the shared legacy stream** (the
  per-connection-stream intervention makes it worse, 3.46 -> 6.39), **not lock contention in the
  transport** (each connection owns its worker and mutex), and **not a shared CUDA context**
  (MPS shares one and stays fair at 1.02).
- That the tenant launched first is fastest in **16 of 34 cohorts (47%) against 12.5% by
  chance** -- with the §1 caveat that this is *launch* order, not demonstrably *connection*
  order.
- **That the cause is the absence of arbitration in the backend dispatcher**, on the strength of
  four exclusions plus an intervention that supplies the missing arbitration and removes 38% of
  the inequality at 0.6% makespan cost.

**May not be claimed.**

- That the fix is complete. Launch-count arbitration leaves a residual, and the reason is known
  (it does not weight by kernel duration).
- That `GVS_FAIR_DISPATCH` is a production feature. It is a probe: env-gated, off by default,
  with a 2 s escape hatch that degrades to FCFS rather than risk a stall.

Instrumentation: `include/gvirtus/communicators/SchedTrace.h` (`GVS_SCHED_TRACE`,
`GVS_PER_CONN_STREAM`, `GVS_FAIR_DISPATCH`, `GVS_FAIR_LEAD`), all env-gated and off by default;
the gate is hooked at `plugins/cudart/backend/CudaRtHandler_execution.cpp` around
`cudaLaunchKernel`. Data: `results/asplos_campaign/sched/` (80 per-tenant logs) and
`results/asplos_campaign/sched_n1/minibude_n8_fair_ab.csv` (33 cohorts, both arms). Analysis:
`analiza_sched.py`.
