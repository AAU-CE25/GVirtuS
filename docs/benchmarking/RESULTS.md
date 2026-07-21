# GVirtuS benchmark results

Empirical results of the GVirtuS (GUSTO) benchmark campaign on the AAU two-node L40S + RoCEv2
testbed over the UCX communicator (TCP / RDMA / RDMA+GPUDirect). Raw data + plots live under
[`../../benchmarks/`](../../benchmarks/). For **setup, standing rules, known bugs, and the
measurement roadmap** see [`README.md`](README.md); for the **optimization stack + async dispatcher**
see [`ASYNC-DISPATCHER.md`](ASYNC-DISPATCHER.md).

> **The roofline in one line:** GVirtuS is essentially free for **compute-bound** work (miniBUDE
> <0.2%), ~93–95% of native for **bandwidth-bound** work at large sizes (BabelStream), and was
> **catastrophic for launch-/RPC-bound** work (llama ~79× native) — which is what the optimization
> stack + async dispatcher fix (see ASYNC-DISPATCHER.md).

---

## 1. miniBUDE — compute-bound (overhead ~free)

Molecular-docking proxy: high arithmetic intensity, tiny data footprint, one kernel iterated N×.
Deck `bm1`, 8 iters. Data: `benchmarks/miniBUDE-sync/minibude.csv`.

| config | GFLOP/s | % of bare metal | setup transfer (context_ms) | valid |
|--------|--------:|----------------:|----------------------------:|:-----:|
| bare metal         | 216.63 | 100.00% | 0.27 ms  | ✓ |
| UCX-TCP            | 216.29 | 99.84%  | 10.29 ms | ✓ |
| UCX-RDMA           | 216.45 | 99.92%  | 2.14 ms  | ✓ |
| UCX-RDMA+GPUDirect | 216.44 | 99.91%  | 2.25 ms  | ✓ |

- **Remote virtualization is essentially free for compute-bound HPC** — all configs within **0.2%**
  of native. The kernel is ~2364 ms; the network is a one-time few-ms setup transfer.
- **Transport-insensitive throughput**, but the transport still shows in the one-time setup cost
  (TCP 10.3 ms vs RDMA 2.1 ms, ~5×) — a real but irrelevant <0.5% of runtime.
- **Honest framing:** miniBUDE deliberately shows *no* transport difference; that is the correct
  result, not a missing signal. It proves the "overhead-free compute-bound" point and nothing about
  RDMA/GPUDirect (which need a network-stressing workload — see §3, §5).

## 2. BabelStream — memory-bandwidth / control-path

Five kernels (Copy/Mul/Add/Triad/Dot) over arrays ≫ cache; 9 sizes 2¹⁸…2²⁶; 4 configs; 100 iters =
180 points. Data: `benchmarks/BabelStream-sync/`. BabelStream transfers arrays **once** then loops on
device — so it measures **kernel bandwidth** + **per-launch RPC latency**, and is insensitive to the
*data-path* transport in steady state (RDMA ≈ RDMA+GPUDirect here — expected, not an error).

**Control-path RPC latency floor** (smallest size ⇒ ~pure RPC round-trip per launch):

| kernel | bare metal | UCX-TCP | UCX-RDMA | RDMA vs TCP |
|--------|-----------:|--------:|---------:|------------:|
| Triad | 7.2 µs | 322.7 µs | 237.6 µs | 1.36× |
| Dot   | 28.6 µs | 400.4 µs | 302.3 µs | 1.32× |

**RDMA cuts the per-launch RPC round-trip ~26% (~84 µs/call) vs TCP**, consistently. This is *why*
RDMA beats TCP on the small-size bandwidth curves (bandwidth there = payload / RPC latency).

**Triad bandwidth (GB/s), synchronous path:**

| per-buffer | bare metal | TCP | RDMA | RDMA+GPUDirect |
|-----------:|-----------:|----:|-----:|---------------:|
| 2 MiB   | 925\* | 23  | 30  | 31  |
| 32 MiB  | 710   | 444 | 487 | 492 |
| 512 MiB | 681   | 633 | 646 | 641 |

\* Small-size bare-metal numbers are an **L2-cache artifact** (arrays fit the L40S's ~96 MB L2);
only ≥32 MiB/buffer is a fair native reference (~680 GB/s DRAM). Control-path overhead dominates at
small sizes; as arrays grow, GVirtuS approaches native (~93–95% at 512 MiB). *(The async dispatcher
lifts the small-size, launch-bound points ~+20–31% — see ASYNC-DISPATCHER.md. The current build is
also ~5.6× faster on small BabelStream than these older tables, which predate the optimization stack.)*

## 3. Transfer bandwidth (H2D/D2H) — GPUDirect validated

Raw `cudaMemcpy` bandwidth vs size (exactly-sized buffers), 4 KiB…256 MiB. Data:
`benchmarks/transport-characterization/transfer/`.

| D2H size | bare metal | UCX-TCP | UCX-RDMA | RDMA+GPUDirect |
|---------:|-----------:|--------:|---------:|---------------:|
| 1 MiB  | 23.3 | 0.45 | 3.1 | 2.7 |
| 4 MiB  | 26.0 | 1.89 | 3.3 | **8.1** |
| 256 MiB| 27.1 | 1.64 | 3.9 | **8.9** |

- **RDMA ~3× faster than TCP on the data path** (H2D 1 MiB: 6.1 vs 1.6 GB/s), across all sizes.
- **GPUDirect gives a clean ~2.3× D2H speedup for transfers ≥4 MiB** (~8.9 vs ~3.9 GB/s), sustained
  to 256 MiB — the backend NIC reads GPU memory directly, skipping the host bounce. **Verified on
  the backend log** (GPUDirect=enabled, GPU shadows) per the doc-0 rule.
- **No H2D benefit** (expected): the H2D source is the client's *host* buffer, so there's no
  client-side GPU memory to peer-DMA from. GPUDirect's asymmetry (D2H yes, H2D no) matches its
  mechanism.
- Both stay well below native PCIe (~27 GB/s) because every `cudaMemcpy` is a synchronous RPC over a
  25 GbE RoCE link.

## 4. simple_matrix (cuBLAS SGEMM) vs the project report

Dense `cublasSgemm` on N×N matrices, timing H2D+GEMM+D2H per iter — the cleanest bulk-transfer
workload (N=16000 ⇒ ~1 GB/transfer, where GPUDirect provably engages). N=16000 vs the report's
N=16384. Data: `benchmarks/simple-matrix-sync/matrix_vs_paper.csv`.

| configuration | our `avg_host_ms` | report (ms) | GEMM ms |
|---------------|------------------:|------------:|--------:|
| UCX RDMA + GPUDirect (zerocopy) | **358.4** | 389.5 | 157.9 |
| UCX RDMA (zerocopy)             | **496.9** | 548.4 | 156.8 |
| UCX RDMA (staged, no zerocopy)  | 698.9     | —     | 156.9 |

- **Reproduces the report** within the N=16000-vs-16384 margin (~8–9% lower, as expected).
- **GPUDirect provably works at ~1 GB/transfer: ~1.4× over RDMA** (496.9 → 358.4 ms; report
  548.4 → 389.5, 1.41×). Unlike llama (data path idle), simple_matrix genuinely exercises NIC↔GPU
  DMA. **The win is D2H, and `GVIRTUS_RMA_ZEROCOPY=1` is required to see it** (GEMM ~157 ms is not the
  bottleneck — >92% of the core is data movement).

## 5. llama.cpp — launch-/RPC-bound (the headline)

TinyLlama-1.1B Q4, llama.cpp CUDA backend over GVirtuS. Token generation = thousands of tiny
*sequential* kernel launches, each a synchronous RPC. Data: `benchmarks/llama-{sync,async}/`.

**Baseline (before the optimization stack), token gen tg16 vs native (634.9 t/s):**

| config | tg16 (t/s) | vs bare metal |
|--------|-----------:|--------------:|
| GVirtuS RDMA           | 8.04 | **79× slower** |
| GVirtuS RDMA+GPUDirect | 7.96 | 80× slower |
| GVirtuS TCP            | 3.58 | 177× slower |

- **LLM inference is catastrophically RPC-bound over remoting** — far worse than any other workload,
  because each token is ~500 kernel launches × several blocking RPCs each, with no overlap.
- **Transport matters here (unlike compute-bound apps): RDMA ~2.2× TCP** — decode is dispatch-latency
  bound, so lower per-RPC latency directly speeds generation.
- **GPUDirect ≡ RDMA for llama**: per-token transfers are tiny/host-sourced, so the GPU data path is
  idle (see §6 control/data split). GPUDirect's win is bulk transfer (§3, §4), not per-launch chatter.
- **Correctness preserved**: coherent output ("…the capital of France is Paris").

This 79× gap is what the optimization stack + async dispatcher close to ~3.4× → see
[`ASYNC-DISPATCHER.md`](ASYNC-DISPATCHER.md).

## 6. Per-RPC latency distributions (tail latency — the networking headline)

Env-gated per-RPC trace (`GVIRTUS_LATENCY_TRACE`) over an identical `llama-bench -p 8 -n 16 -r 1`
run (44,242 RPCs/transport). Data: `benchmarks/transport-characterization/latency/`.

**Control-plane RPC latency (payload < 4 KiB, n≈43k each):**

| transport | p50 | p99 | p99.9 | **tail ratio p99/p50** |
|-----------|----:|----:|------:|----------------------:|
| TCP            | 58 µs | **1328 µs** | 15550 µs | **22.9×** |
| RDMA           | 42 µs | **56 µs**   |   353 µs | **1.33×** |
| RDMA+GPUDirect | 43 µs | 55 µs       |   353 µs | 1.28× |

- **The median barely moves; the TAIL is where RDMA wins.** p99: RDMA 56 µs vs TCP 1328 µs (**24×**);
  p99.9: 353 vs 15,550 µs (**44×**). RDMA's distribution is nearly flat/deterministic; TCP's has a
  catastrophic tail (coalescing, delayed ACKs, retransmit timers). **RDMA's value is
  tail-latency elimination, not lower average.**
- **Means lie** — TCP mean (155 µs) ≈ 2.7× its median (58 µs); report distributions, not means.
- **Warmup-robust:** discarding warmup makes it *stronger* — steady-state TCP tail grows (p99/p50
  → 37.5×), steady-state RDMA is razor-tight (p99.9 = 67 µs). The tail is intrinsic to TCP, not a
  cold-start artifact.
- **Control vs data split during llama:** control (small AM, <4 KiB) = **97.3% of RPCs, 81.1% of RPC
  wall time**; the data path is essentially idle in steady state (16.1% is the one-time 636 MB weight
  load). This is *why* GPUDirect ≡ RDMA for llama, and it makes the bottleneck legible: **count ×
  synchronous**, ~44k control RPCs × ~43 µs issued serially = ~2.3 s of critical-path latency — fixed
  by caching + async, not faster transport.

---

## Takeaways
- **Two roofline extremes bracket GVirtuS:** miniBUDE (compute-bound, ~0% overhead) and llama
  (launch-bound, was ~79×). Together they show where remoting is / isn't viable.
- **RDMA's contribution = tail-latency elimination** (p99 24× better than TCP); **GPUDirect's
  contribution = bulk-transfer D2H** (~2.3× micro, ~1.4× simple_matrix). Both are verified, not
  assumed.
- The llama gap is a *synchronous-dispatch* problem — the subject of `ASYNC-DISPATCHER.md`.
