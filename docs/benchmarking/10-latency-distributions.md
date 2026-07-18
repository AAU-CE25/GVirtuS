# 10 — Per-RPC latency distributions (p50/p99/tail) across transports

Status: **first distribution-level result — the INFOCOM headline metric.** Last updated: 2026-07-18.

This is the measurement doc 09 flagged as the #1 gap. We instrumented the frontend to record
**every** RPC's round-trip latency and computed the full distribution per transport, from an
**identical** workload (one `llama-bench -p 8 -n 16 -r 1` run = 44,242 RPCs per transport).

## Method
Env-gated instrumentation in `src/frontend/Frontend.cpp::Execute()` (UCX AM path): when
`GVIRTUS_LATENCY_TRACE=<file>` is set, each RPC appends `{routine, payload_bytes, rt_us,
server_us}` to a per-thread buffer (round-trip = `start_send → response fully read`), flushed as
CSV at process exit. Zero overhead when unset (single bool check); token-gen throughput was
identical with tracing on (tg16 11.62 traced vs 11.64 untraced). Backend GPUDirect state verified
on the backend log per doc-00. Raw traces: `lat_{tcp,rdma,gpudirect}.csv`; percentile table:
`latency_percentiles.csv`; analysis: `analyze_latency.py`.

The **control-plane** subset (payload < 4 KiB) isolates the small request/response RPCs that
dominate LLM decode; for these `server_us ≈ 0`, so `rt_us` is essentially pure
transport + dispatch latency.

## Results — control-plane RPC latency (payload < 4 KiB, n = 43,033 each)

| transport | mean | p50 | p90 | p99 | p99.9 | max | **tail ratio p99/p50** |
|-----------|-----:|----:|----:|----:|------:|----:|----------------------:|
| TCP            | 154.8 µs | 58 µs | 75 µs | **1328 µs** | 15550 µs | 33262 µs | **22.9×** |
| RDMA           |  43.7 µs | 42 µs | 47 µs | **56 µs**   |   353 µs |  6040 µs | **1.33×** |
| RDMA+GPUDirect |  44.3 µs | 43 µs | 47 µs | **55 µs**   |   353 µs |  6276 µs | **1.28×** |

Plots: `plots/latency_cdf_control.png` (CDF), `plots/latency_percentiles_control.png` (grouped bars).

## Findings

### 1. The median barely moves; the TAIL is where RDMA wins (the headline)
At p50, RDMA is only ~1.4× better than TCP (42 vs 58 µs). But:
- **p99: RDMA 56 µs vs TCP 1328 µs — a 24× gap.**
- **p99.9: RDMA 353 µs vs TCP 15,550 µs — a 44× gap.**
- **Tail ratio p99/p50: TCP 22.9× vs RDMA 1.33×.**

RDMA delivers a nearly **flat, deterministic** latency distribution; TCP's distribution has a
catastrophic tail (kernel TCP stack: coalescing, delayed ACKs, scheduler wakeups, retransmit
timers). For a networking venue this is the money result: **RDMA's value here is not lower average
latency, it is the near-elimination of tail-latency variance.**

### 2. Means lie — report distributions
TCP's *mean* (155 µs) is ~2.7× its *median* (58 µs) because the tail drags it up; RDMA's mean
(44 µs) ≈ its median (42 µs). Any evaluation that reported only means (as our earlier docs did)
would understate how much better RDMA's *predictability* is. This validates doc 09's core point.

### 3. This explains the application results mechanistically
LLM decode issues ~3,100 serial RPCs per token (doc 07). With RDMA every RPC is ~42–56 µs →
~124 ms/token. With TCP a non-trivial fraction of those 3,100 serial RPCs hit the 1.3 ms (p99) /
15 ms (p99.9) tail, and because they are **serial with no overlap**, each tail event directly adds
to per-token latency — compounding into the measured ~2.2× slowdown (tg16 3.2 vs 11.6 t/s in these
very runs). Tail latency, not bandwidth, is the LLM-inference killer on TCP.

### 4. GPUDirect ≈ RDMA on the control plane (as expected)
GPUDirect only touches the bulk data path (device-resident payloads); the small control RPCs are
identical to plain RDMA (p50 43 vs 42 µs, p99 55 vs 56 µs). Consistent with doc 04/07: GPUDirect's
win is bulk transfers, not per-launch control chatter.

## Warmup robustness (the reported tail is not a cold-start artifact)
The trace deliberately records **every** RPC, including the one-time registration/weight-load
phase, so we re-computed the control-plane percentiles three ways per transport (see
`warmup_sensitivity.py`): (a) all, (b) excluding setup routines (`cudaRegister*`/`cudaUnregister*`),
(c) last 50 % only (pure steady-state decode).

| transport | subset | p50 | p99 | p99.9 | max | tail p99/p50 |
|-----------|--------|----:|----:|------:|----:|------------:|
| TCP  | all           | 58 | 1328 | 15550 | 33262 | 22.9× |
| TCP  | steady (no setup) | 57 | 1734 | 15595 | 33262 | 30.4× |
| TCP  | last 50 %     | 56 | 2102 | 15747 | 30925 | **37.5×** |
| RDMA | all           | 42 |   56 |   353 |  6040 | 1.33× |
| RDMA | steady (no setup) | 43 | 57 |   837 |  6040 | 1.33× |
| RDMA | last 50 %     | 43 |   50 |    **67** |   457 | **1.16×** |

- **Median is warmup-invariant** (TCP 56–58 µs, RDMA 42–43 µs in every subset) → the p50 result is
  solid regardless of warmup.
- **Discarding warmup makes the finding STRONGER, not weaker.** TCP's tail *grows* in steady state
  (p99/p50 22.9 → 37.5×), proving the tail is an **intrinsic TCP property, not a cold-start
  artifact.** Conversely RDMA's few ms-scale outliers *were* cold-start: steady-state RDMA is
  razor-tight (p99.9 = **67 µs** vs TCP 15,747 µs → **235×**). The all-samples table above is thus
  a conservative statement of the RDMA advantage.

Note on the other workloads: all throughput experiments used warmup — llama-bench (built-in warmup
pass), BabelStream (peak of 100 iterations), miniBUDE (one-time `context_ms` excluded from timed
compute), transfer_bw2 (explicit `warm=5` before each timed loop).

## Control-path (AM) vs data-path (bulk) split during llama — is AM the problem?
Analysis `control_vs_data_split.py` on the RDMA trace (44,241 RPCs) answers the natural question
"is LLM inference slow because the Active-Message control path is slow?" — **no.**

| path | RPCs | % count | % of RPC wall time |
|------|-----:|--------:|-------------------:|
| **control** (small AM, < 4 KiB)         | 43,033 | **97.3 %** | **81.1 %** |
| **data** (bulk RDMA/GPUDirect, ≥ 4 KiB) |  1,208 |    2.7 %  |   18.9 %  |

Of the data-path 18.9 %, **16.1 % is the one-time 636 MB weight load** (249 transfers up to 54 MB,
done once at startup). During steady-state token generation the **data path is essentially idle** —
per-token transfers are tiny and ride the control path. This is exactly why GPUDirect ≡ RDMA for
llama: there is no per-token bulk traffic for the data path to accelerate.

**AM is not slow per message** (cudaGetDevice 44.6 µs, cudaGetLastError 40.9 µs, cudaLaunchKernel
43.7 µs; p99 = 56 µs on RDMA). The cost is **count × synchronous**: ~44 k control RPCs × ~43 µs,
issued **serially with no overlap** = 2.32 s of critical-path latency.

Top control-path offenders (post Push/Pop optimization — note `cudaPush/PopCallConfiguration` are
now absent because they were made local, doc 08):

| routine | count | % of RPCs | % of RPC time |
|---------|------:|----------:|--------------:|
| `cudaGetDevice`     | 14,624 | 33.1 % | 28.1 % |
| `cudaGetLastError`  | 11,163 | 25.2 % | 19.7 % |
| `cudaLaunchKernel`  |  8,266 | 18.7 % | 15.6 % |

**`cudaGetDevice` + `cudaGetLastError` alone are 58 % of RPCs and 48 % of control-path time — and
both are cacheable frontend-side (recommended improvement #2).** Caching them would roughly halve
the remaining RPCs. So the LLM gap vs bare metal is not the AM control path being slow, nor a data
path deficiency — it is the **synchronous, per-call control path under a workload that issues
thousands of tiny serial control calls per token**, addressable by caching (#2) and async/batching
(#3–#4), not by faster transport. The control/data split's value for llama is that it makes this
bottleneck *legible* (81 % control, data path idle); its data-path strength shows for bulk
workloads (the weight load here, BabelStream, big matmuls, doc 04).

## Why this matters for the paper
- Converts "GVirtuS is slower on TCP" into a **quantified tail-latency argument** with CDFs and
  percentiles — the language INFOCOM reviewers expect.
- Directly motivates the RDMA/AM control-path design: it exists to kill tail latency for the
  many-tiny-RPC regime that RPC-bound GPU workloads (LLM decode) live in.

## Next distribution work (from doc 09)
- Latency **decomposition** (marshal/wire/backend-dispatch/exec/return) per transport — timers
  already exist in `Execute()`; aggregate them.
- **Throughput–latency saturation** curve + small-message **RPC/s ceiling**.
- **Multi-tenancy**: tail latency of a latency-sensitive tenant under a noisy (bandwidth-heavy)
  neighbor — this instrumentation makes the isolation story directly measurable.

## Reproduce
Set `-e GVIRTUS_LATENCY_TRACE=/tmp/lat_<t>.csv` on the frontend `docker exec` alongside the usual
transport env, run any workload (we used `llama-bench -p 8 -n 16 -r 1`), then
`python analyze_latency.py`. Restart the backend between GD phases and verify the
`GPUDirect=enabled/disabled` line.
