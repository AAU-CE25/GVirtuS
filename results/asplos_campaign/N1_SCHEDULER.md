---
title: "Why service arrives in bursts -- two mechanisms excluded, one narrowed"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

The fairness audit established **that** the backend shares the GPU unevenly under equal fixed
work, and by how much: at N=8 in miniBUDE one tenant runs at 1.00x its single-client rate while
another runs 4.87x slower, where native and native+MPS share to within 1.03x. It did not
establish **why**. This is the attempt, and it ends with two candidate mechanisms excluded by
evidence and a third narrowed.

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

# 4. What remains, stated as a hypothesis rather than a finding

With per-connection threads, per-connection workers, per-connection mutexes and even
per-connection CUDA streams, what the tenants still share is **one CUDA context and the GPU
work distributor**.

The remaining candidate is a **positive feedback loop that a single context does not damp**:
each client is self-clocked -- it submits its next iteration only after the previous one
returns -- so the tenant that completes first re-submits first, and stays ahead. Between
*contexts* the driver time-slices, which is why native is fair at exactly N. Between *streams
of one context* there is no such arbitration, so an early lead compounds instead of decaying.

This is **consistent with all three observations** -- the connection-order effect, the failure of
per-connection streams to help, and native's exact-N fairness -- but it is **not tested**. The
experiment that would settle it: stagger the client start times deliberately and check whether
the favoured tenant tracks the start order rather than the connection order, and separately
measure GPU-side kernel start timestamps per stream with CUPTI, which the current tracing
cannot see.

# 5. What may be claimed

**May be claimed.** That the sharing is uneven, by how much, that it is reproducible, that it
tracks connection order in 9 of 10 cohorts, that it is not the data path (it appears identically
over TCP), that it is not the shared legacy stream (the intervention worsens it), and that it is
not lock contention in the transport (each connection has its own).

**May not be claimed.** Any specific mechanism. The paper should say the cause is not
established and name the excluded candidates -- that is a stronger position than an unverified
explanation, and the exclusions are themselves results.

Instrumentation: `include/gvirtus/communicators/SchedTrace.h` (`GVS_SCHED_TRACE`,
`GVS_PER_CONN_STREAM`), both env-gated and off by default. Data:
`results/asplos_campaign/sched/`, 80 per-tenant logs. Analysis: `analiza_sched.py`.
