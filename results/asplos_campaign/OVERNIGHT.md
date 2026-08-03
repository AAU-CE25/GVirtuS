---
title: "Overnight session, 2026-08-03 -- what closed, what was retracted, what is still running"
date: "2026-08-03 03:00"
geometry: margin=2.3cm
fontsize: 10pt
---

# Read this first: one number was retracted

**The +17.4% capacity advantage I reported from repetition 1 does not survive replication.**
With n=3 and a criterion that requires the SLO to be met in *every* repetition, capacity at N=8
falls from lambda=1.00 to lambda=0.50, both systems deliver 55.5 t/s there, and the paired difference is
**-1.4 t/s with a bootstrap CI95 of [-4.2, +0.0] -- it includes zero**.

The cause is a harness defect, not a property of the systems: a fixed 30 s window at lambda=0.25
offers 7.5 requests, so goodput inherits Poisson counting noise directly. Observed goodput at
N=8, lambda=0.50 ranged **29.9 to 59.7 t/s** across three repetitions. At n=1 this is invisible.

A second sweep with the window scaled as `max(30, 40/lambda)` is **running now** and will finish
around 07:10. Full detail in `LLAMA-7B_RESULTS.md` §3b.

# The four "necessary" items

| # | item | status |
|---|---|---|
| 1 | instrument the backend scheduler, explain the burst | **partially closed** -- two mechanisms excluded, one narrowed |
| 2 | defence against "integration paper" | **closed** -- `NOVELTY.md` |
| 3 | quantify coverage, LOC, fallback, extension cost | **closed** -- `IMPLEMENTATION_COVERAGE.md` |
| 4 | simplify the first two pages | **closed** -- `PAPER_OPENING.md` |

## 1 -- Connection order decides service. Two mechanisms excluded.

**Established.** The first tenant to connect is the fastest in **9 of 10 cohorts**, at
1.01--1.19x its single-client rate, while the slowest sits at 2.0--7.2x. It holds under both
backend configurations, so service order is fixed at connection time and persists.

**Excluded by intervention.** The shared legacy CUDA stream. A backend gate giving each
connection its own non-blocking stream **nearly doubled** the inequality: slowest/fastest 3.46
-> **6.39**. Removing the shared queue let the incumbent pull further ahead.

**Excluded by reading the code.** Contention on a shared worker mutex: every accepted
connection gets its own mutex and its own UCX worker (`UcxCommunicator.cpp:1420, :1398`).

**Remaining hypothesis, labelled as one.** Each client is self-clocked, so whoever finishes
first re-submits first and stays ahead; between contexts the driver time-slices, between
streams of one context nothing arbitrates. Consistent with all three observations, untested.

## 2, 3, 4 -- the written items

`NOVELTY.md` makes **no factual claim about Gleam, rCUDA or Cricket** -- they are not on this
testbed and asserting their internals from memory is the failure this campaign exists to
avoid. Sentences that need a verified prior-work citation are marked **[CITE]**.

`IMPLEMENTATION_COVERAGE.md`: **940 of 2023** exported CUDA-family entry points are real
(46.5%), and it says where that is bad -- cusolver is **1.7%**, ABI padding rather than an
implementation. Fallback rate over 620 teardowns and **14.78 million operations**: **80
fallbacks in 2 643 921 RMA admissions, 0.003%**. Extension cost **44--54 lines per API**.

`PAPER_OPENING.md`: nine results currently compete in the first two pages and three no longer
survive their own controls. One lead claim is proposed -- the placement contract -- and every
other result is demoted with an explicit destination.

# The three "very valuable" items

| # | item | status |
|---|---|---|
| 5 | second hardware validation | **role reversal ruled out by the user; partial cross-driver check done** |
| 6 | attribute the ~460 MiB | **CLOSED, with a mechanism, confirmed on both GPUs** |
| 7 | harmful epoch-collision demonstration | **still open; my first explanation of why was wrong and is retracted** |

## 6 -- the memory saving is the per-process CUDA context, measured directly

The scaling was the clue: the saving is nearly **constant** (445.5 / 456 / 461 MiB at N=2/4/8),
not `C-(1-1/N)` as context sharing would give.

A program that does nothing but create its primary context costs **429 MiB**, exactly linear
over 1, 2 and 4 processes. Under MPS it costs **422--423**, and that **6--7 MiB difference matches
the 5.0 MiB per tenant** that the llama experiment measured MPS saving -- two independent
measurements of the same quantity agreeing.

**So:** a native pod pays 429 MiB of context, an MPS client still pays 422, and a remoted tenant
pays **zero**, because its frontend process has no CUDA context at all. That accounts for
93--96% of the saving. The claim now has a line item instead of a hypothesis.

> **Confirmed and tightened 2026-08-03.** The probe was re-run on **both** GPUs: **431 MiB** per
> process on dpu-01 (backend L40S) and **429 MiB** on dpu-02 (native L40S), exactly linear over
> K = 1, 2, 4 in both -- so the figure holds across two drivers, not one. `nvidia-smi`'s own
> per-process accounting gives **424 MiB** per probe independently, and gives the *same* value
> whether or not the client can reach the MPS pipe directory. The Gusto column was also
> re-measured on the deployed backend (4 524 / 4 501 / 4 490 at N=1/2/4), so the saving at N=1 is
> **426 MiB against a predicted 429 -- 99% accounted for**, better than the 93--96% stated above.
> Raw: `memoria/ctx_probe_results.csv`, `memoria/ctx_probe_perproc.txt`.
>
> One detour is recorded in `LLAMA-7B_RESULTS.md` §2: an intermediate re-measurement briefly cut
> the saving to ~170 MiB and I wrote a correction saying so. It was taken on a backend a previous
> experiment had left at `slots=8` (4x the deployed RMA pool, 257 MiB of GPU shadow per
> connection). **That correction is withdrawn.**

## 7 -- the harmful demonstration failed, and the reason is the finding

`epochharm.cu` keeps eight buffers in flight across pool rebuilds and validates every byte.
Both arms -- guard on and `no_epoch` -- report **zero** corrupted bytes.

**That is not evidence the guard is unnecessary -- but the reason I first gave for it was
wrong.**

> **Retracted the same day.** I wrote here that the experiment is inert because *"the epoch
> changes but the indices renumber, `[0-7]` -> `[9-16]`, so no index can collide."* That came
> from a **de-duplicated grep over the backend log**, which collapsed three separate
> announcements into what looked like one renumbering. The full log shows a
> **re-advertisement with identical indices did occur**, so the premise of the explanation is
> false. See `N7_EPOCH.md`, which carries the corrected analysis.

The real reason the demonstration stays benign is a mechanism I had not accounted for: the
backend **parks** the install. `UcxCommunicator.cpp:2966` chooses between `PARK` and
`install now` on `any_inflight`, and a parked layout is installed only once the last in-flight
transfer drains -- so the dangerous state never forms, and the epoch guard sits behind the park
as a second line of defence rather than as the first. `ack_epoch_dropped=0` is consistent with
that: the guard had nothing to fire on.

**So the experiment must defeat the park**, not merely produce the re-advertisement. That needs
a fault gate forcing `install now` regardless of `any_inflight`, or a workload that produces
the residual naturally. Both routes are specified in `N7_EPOCH.md`.

## 5 -- role reversal ruled out by the user; a partial cross-driver check exists instead

The original plan was to reverse the roles: backend on dpu-02, frontend on dpu-01. dpu-02's
L40S runs a **different driver** (560.35.05 against dpu-01's 580.95.05), so that would have been
a genuine second configuration. **The user has since ruled it out** ("no vamos a montar backend
en dpu02 y frontend en dpu01"), so it is closed as a decision, not as a pending task. It was in
any case not attempted overnight, because dpu-02's GPU was occupied by the capacity sweep until
~07:10 and standing up new infrastructure unattended and untested is how a night's data gets
lost.

**What does exist is a narrower cross-driver result**, obtained on 2026-08-03 while closing the
memory mechanism. The per-process CUDA context probe was run on **both** L40S cards:

| host | driver | per-process context |
|---|---|---:|
| dpu-01 | 580.95.05 | **431 MiB** |
| dpu-02 | 560.35.05 | **429 MiB** |

Exactly linear over K = 1, 2, 4 on both. This validates the **memory mechanism** across two
drivers -- and it is the claim that most needed it, being the one argument for remoting that
MPS does not match. It does **not** validate the transport, the placement policy or the
lifetime protocol on second hardware; those remain single-configuration and should be declared
as such in the submission's threats-to-validity.

# Two mistakes of mine tonight, recorded

1. **I broke the backend.** `SchedTrace.h` included `cuda_runtime.h` and is included from
   `src/backend/Process.cpp`, which compiles without CUDA headers. The backend was down for
   about six minutes. Fixed by splitting the header; the CUDA-dependent half now lives in the
   plugin.
2. **I interfered with my own measurement.** The context-footprint probes used dpu-02's GPU and
   the MPS daemon at 02:47--02:52, exactly when the sweep's next cell was `bmmps`. Its pods could
   not come up, the cell stalled, and the sweep had to be cleaned and relaunched at 02:56 --
   losing about fifteen minutes. The rule I already knew and broke: **do not run anything on a
   GPU that a background campaign is using.**

# Running now

`cola_v2` (pid 552312, launched 02:56): 3 systems x 3 N x 3 repetitions x 4 loads, windows
scaled to >=40 requests per point, **including the native+MPS arm** the first sweep lacked.
Expected to finish ~07:10. Output in `results/asplos_campaign/llama_slo_sweep_v2/`; analyse
with `analiza_capacidad_n3.py` pointed at that directory.

# Commits

`ba89efb` coverage, novelty, opening - `3cd48db` retraction and v2 harness -
`8e83865` N1 and the memory attribution - `58c2ac9` N7
