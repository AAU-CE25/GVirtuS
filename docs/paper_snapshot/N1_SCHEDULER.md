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

# 1. The finding that survived: connection order decides

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
its single-client rate -- while the slowest sits at 2.0--7.2x. The containers are started in
index order, so `t1` is the first to connect.

This holds in **both** backend configurations, so it is independent of the intervention below.
It is the strongest statement the data supports: **service order is decided at connection time
and persists for the whole run.**

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
