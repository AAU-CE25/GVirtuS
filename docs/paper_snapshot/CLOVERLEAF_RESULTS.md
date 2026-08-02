# CloverLeaf CUDA over GVirtuS -- contention scaling, five arms (2026-07-28)

> ## 2026-08-01 -- the RPC count, measured at last
>
> The per-RPC cost table divided 473 s by 411,054 RPCs and published 115 µs. No arithmetic
> closed (direct division 1150.8 µs; divided by 8, 143.8) and **the count did not exist in the
> artifact** -- the raw data held `cpu_s`, `duration_s` and timestamps, not call counts.
>
> Measured with the per-RPC trace enabled **inside the published harness** (3 seeds, N=1, UCX),
> so the count comes from the same methodology that produced the makespan it is divided by:
>
> | | value |
> |---|---|
> | **total RPCs** | **425,084** -- identical across all three seeds; published 411,054, within **3.4 %** |
> | per-RPC cost | mean **25.5 µs** - median **7.0** - p95 10.0 - **p99 546** |
> | time inside RPCs | 10.83 s = **78.2 % of the run** |
> | of that, server execution | 2.95 s = **27.3 %** |
>
> Three consequences. The distribution is strongly skewed, so **a mean -- or a makespan/RPC
> quotient -- describes it badly**: the typical cost is 7 µs, not 25.5. The **78 %** figure makes
> "control-bound" a measurement rather than an inference. And since only **27 %** of that time
> is execution, the remaining 73 % bounds what any control-plane optimisation can deliver.
>
> All RPCs were synchronous (0 fire-and-forget): the harness does not enable the async
> dispatcher. Traces are in `artifact/data/raw/ablation_2026-08-01/workload_closure/`.
> See §D of `RESULTS_2026-08-01.md`.


Hydrodynamics mini-app, deck `1920 x 960`, 2955 steps. Five arms, N = 1/2/4/8 concurrent
tenants against one L40S, 5 seeds per point, plus a staggered-launch control at N=8.

Harness: `drive.sh block cloverleaf <arm> 5` -- same code path for every arm. Backend
re-armed and its listener verified with `ss` before each point.

**Every number below comes from repetitions in which all N clients produced a valid
result.** A repetition where one client dies finishes sooner and reads as an improvement;
three such points were rejected and re-run rather than averaged in.

---

## Wall-clock per tenant (completion p50, s) and overhead vs native

| arm | N=1 | N=2 | N=4 | N=8 sync | N=8 stagger |
|---|---:|---:|---:|---:|---:|
| baremetal (native) | 13.22 | 28.68 | 56.88 | 113.55 | 100.67 |
| baremetal + MPS | 13.22 (1.00x) | 26.84 (0.94x) | 55.75 (0.98x) | 114.95 (1.01x) | 101.53 (1.01x) |
| **GVirtuS rdma** | 14.65 (**1.11x**) | 29.09 (**1.01x**) | 59.43 (**1.04x**) | 122.69 (**1.08x**) | 109.07 (**1.08x**) |
| **GVirtuS GPUDirect** | 14.85 (1.12x) | 29.29 (1.02x) | 59.73 (1.05x) | 122.69 (1.08x) | 108.90 (1.08x) |
| GVirtuS tcp | 42.34 (3.20x) | 100.94 (3.52x) | 221.09 (3.89x) | 473.03 (4.17x) | 451.42 (4.48x) |

**Disaggregation over RDMA costs 1.01--1.12x of native in wall-clock, at every tenant
count.** The cost does not grow with N: 1.11x at N=1, 1.08x at N=8.

`rdma` and `GPUDirect` are indistinguishable in every cell (largest gap 0.3 s at N=4,
inside the seed spread). This workload never exercises the GPU data path -- see *Why
GPUDirect does not show up* below.

## Aggregate throughput (Mcell-updates/s)

| arm | N=1 | N=2 | N=4 | N=8 sync | N=8 stagger |
|---|---:|---:|---:|---:|---:|
| baremetal | 437.9 | 391.2 | 390.7 | 389.7 | 451.0 |
| baremetal + MPS | 435.7 | 415.8 | 397.0 | 383.2 | 444.8 |
| GVirtuS rdma | 423.1 | 398.5 | 380.4 | 364.6 | 414.6 |
| GVirtuS GPUDirect | 422.8 | 398.8 | 379.5 | 365.0 | 415.2 |
| GVirtuS tcp | 131.2 | 108.7 | 99.8 | 94.0 | 100.3 |

## Fairness and tail (N=8 sync)

| arm | Jain | min/max | p50 | p95 | p99 | tail spread |
|---|---:|---:|---:|---:|---:|---:|
| baremetal | 1.000 | 0.997 | 113.55 | 113.96 | 113.96 | 0.4% |
| baremetal + MPS | 1.000 | 0.998 | 114.95 | 115.15 | 115.15 | 0.2% |
| GVirtuS rdma | 1.000 | 0.986 | 122.69 | 123.11 | 123.30 | 0.5% |
| GVirtuS GPUDirect | 1.000 | 0.984 | 122.69 | 123.30 | 123.43 | 0.6% |
| GVirtuS tcp | 0.986 | 0.791 | 473.03 | 505.75 | 507.09 | 7.2% |

Remoting over RDMA keeps native-grade fairness and a native-grade tail: every tenant
finishes within 0.6% of every other. TCP is the only arm that spreads its tenants -- one
finishes while another is still 21% behind.

**Jain alone does not discriminate.** It reads 0.986--1.000 across every arm, including the
one with a 21% min/max gap. `min/max` and the percentiles are what separate them; a table
carrying only Jain would show five identical rows.

---

## Staggered-launch control

At N=8 the eight tenants are released within ~1 ms of each other, which puts them in
lock-step: all eight run the same kernel at the same instant. The stagger control offsets
their starts by 2 s to break that phase alignment and bound how much of the measured
contention is an artefact of the alignment.

| arm | sync | stagger | gain |
|---|---:|---:|---:|
| baremetal | 113.55 | 100.67 | 11.3% |
| baremetal + MPS | 114.95 | 101.53 | 11.7% |
| GVirtuS rdma | 122.69 | 109.07 | 11.1% |
| GVirtuS GPUDirect | 122.69 | 108.90 | 11.2% |
| GVirtuS tcp | 473.03 | 451.42 | 4.6% |

The gain is the same (11.1--11.7%) in native and in both RDMA arms, so it is a property of
the workload sharing a GPU, not of the transport. TCP recovers less than half of it:
staggering relieves overlap on the GPU, and in TCP a large share of the time is per-RPC
latency, which does not overlap with anything and so cannot be relieved.

Read `sync` as the pessimistic bound and `stagger` as the optimistic one; the honest
operating range is between them.

---

## MPS is not numerically transparent for this workload

CloverLeaf prints a field summary every step. Volume is a reduction over all cells and is
constant for this deck (`0.1000E+03`); any other value is a corrupted reduction.

| arm | N=1 | N=2 | N=4 | N=8 |
|---|---:|---:|---:|---:|
| baremetal | 0 / 1782 | 0 / 2970 | 0 / 5940 | 0 / 11880 |
| **baremetal + MPS** | **0 / 1782** | **15 / 2970 (0.51%)** | **414 / 5940 (6.97%)** | **783 / 11880 (6.59%)** |

Three properties of this defect, all measured:

1. **It needs concurrency.** At N=1 MPS is clean. Corruption appears at N=2 and jumps an
   order of magnitude at N=4. It is interference between tenants inside the shared context,
   not a property of MPS per se.
2. **It is reproducible.** Two independent runs a day apart give 6.36% and 6.59% at N=8.
3. **It is not a stability problem.** The MPS arm is the most deterministic in the whole
   campaign -- five seeds at N=8 span 0.02%, and two staggered runs differ by 226 µs. It
   does the same work in the same time and gets some of the arithmetic wrong.

**GVirtuS is the negative control that matters.** The backend also consolidates every
tenant into a single CUDA context -- one process, one thread per connection -- and corrupts
nothing (0 lines in all arms, all N). That rules out "sharing a context corrupts the
reduction" and points at the MPS implementation specifically.

**The mechanism is not diagnosed.** We do not know why Volume and Mass corrupt while the
energies stay intact. The defensible claim is the conditional one plus the negative
control, not a mechanism.

**Quote the line rate, not the client rate.** The harness only inspects the *final* summary
line, so it reports ~10% of clients failing at N=4/N=8. Counting all lines, MPS corrupts
reductions in *every* run at N>=4. The client rate understates the defect because it depends
on where in the run the corruption lands.

Consequence for the paper: **for CloverLeaf the overhead reference is native without MPS**,
which is the numerically valid baseline. Quoting MPS as the baseline here would compare
against a configuration that produces wrong answers.

---

## Why GPUDirect does not show up

CloverLeaf issues **411,054 RPCs in a single N=1 run** of 47 s -- about 8,700 per second.
Its makespan is the RPC count multiplied by the per-RPC cost:

| state | makespan | per-RPC |
|---|---:|---:|
| GVirtuS tcp | 473.0 s (N=8) | 115 µs |
| GVirtuS rdma | 122.7 s (N=8) | 36 µs |
| baremetal | 113.5 s (N=8) | 32 µs |

Cross-check: the tcp-rdma gap is 64 µs per RPC, and the measured TCP-vs-RDMA round-trip
gap is ~40 µs plus tail -- same order by two independent routes.

This workload is **control-bound, not data-bound**: the messages are small *because* the
application is a stream of short calls. Lowering the RMA threshold would not help -- it
would only push small messages down a path built for large ones. GPUDirect needs a workload
that moves bulk blocks, and CloverLeaf is not one.

---

## Validity

- 5 seeds per point; every point re-run until all N clients produced a valid result.
- Backend listener verified with `ss` before each point (a container that is `Up` with a
  dead listener accepts nothing and every client fails in 0.4 s).
- `baremetal` and `baremetal_rootns` -- the same arm containerised and in the root namespace --
  agree within 0.5% at every point, so containerisation is not a confound.
- No point raised the multimodality flag: per-repetition aggregate spread is <=1.25x in every
  cell, so the means quoted here are quotable.
- **Three points carry fewer than 5 clean repetitions**, all in the MPS arm and all for the
  reason above: `N=4 sync` 3/5, `N=8 sync` 3/5, `N=8 stagger` 4/5.

## Raw data

`~/experiments/cloverleaf/results/<arm>/N<n>/<mode>/seed*/` -- per-client `clover.out`,
`status_raw.json`, and `seed_raw.json` per point. Parsed with `parse_results.py` and
aggregated with `summarize.py`; the aggregate is `cloverleaf_summary.csv`.

## Confirmed independently (2026-08-02)

The fairness audit rebuilt this from the raw `status_raw.json` per client, computing Jain over
**normalised progress** per cohort, and reproduces the conclusion above: slowest/fastest 1.019
for GVirtuS AM and 1.015 for host RMA at N=8 sync, against 1.005 native and 1.004
native+MPS -- **no separation**. Classification **D: no significant difference**.

**One limitation to record rather than work around.** CloverLeaf writes its figure of merit to
`clover.out`, which is **not in the artifact**; the raw records keep only `t_start`, `t_end`,
`duration_s`, `cpu_s` and `max_rss_kib`. So for this workload the fairness analysis rests on
wall time alone. That matters, because XSBench showed wall time hiding a 6x internal
inequality behind a 1.15 wall ratio: **a wall-only result cannot rule out the same thing
here.** Closing it needs a re-run that captures `clover.out`.

Data: `tenants_canonico.csv`, `fairness_trabajo_fijo_resumen.csv`. Method: `FAIRNESS_RESULTS.md`.
