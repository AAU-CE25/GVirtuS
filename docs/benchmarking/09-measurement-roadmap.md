# 09 — Measurement roadmap for an INFOCOM-grade paper

Status: **planning doc.** Last updated: 2026-07-18.

INFOCOM is a **networking/communications** venue, so GUSTO is judged as a *transport system*
(UCX Active-Message control path + RDMA/GPUDirect data path), not just "CUDA that works remotely".
Reviewers will expect the vocabulary of networked-systems evaluation: **latency distributions and
tails, throughput–latency saturation, fairness/isolation under multi-tenancy, efficiency vs
line-rate, and design ablations** — with statistical rigor. What we have so far is mostly *means*
(throughput and average latency). The gap below is what turns this into a competitive submission.

## What we already have (and its limitation)
- Bandwidth vs payload size (transfer_bw2), BabelStream bandwidth sweep, miniBUDE GFLOP/s, llama
  tok/s — all **mean** values.
- BabelStream "RPC latency" (`babelstream_rpc_latency_us.csv`) — **means only**, derived from
  kernel runtimes, one number per transport.
- Overhead breakdown exists in raw form: `Frontend::Execute()` already timestamps
  marshal/write/sync/read_hdr/read_payload per call (the `[GVS PROFILE]` lines) — but the
  **samples are printed/discarded, never aggregated into distributions.**

**The recurring limitation: no distributions, no tails, no load curves, no fairness, single runs.**

---

## Tier 1 — Latency characterization (the headline for INFOCOM)

### 1.1 Per-RPC round-trip latency DISTRIBUTIONS (p50/p90/p99/p99.9/max) + CDFs
The single most important missing metric. For each transport (TCP, RDMA, RDMA+GPUDirect) and each
**call class** — small control AM (e.g. `cudaGetDevice`, `cudaLaunchKernel` header) vs data-path
(`cudaMemcpy` at various sizes) — report the full percentile table and plot CDFs.
- **Why it matters:** a networking paper lives on tail latency. The mean hides that TCP's p99
  likely explodes vs RDMA's tight distribution. This is where the RDMA/AM design *earns* its
  contribution.
- **How (concrete):** extend `Frontend::Execute()` to append each RPC's total round-trip (µs) to a
  thread-local vector, dumped to a file at exit under `GVIRTUS_LATENCY_TRACE=/path` (gate behind an
  env flag so it's zero-cost when off). Tag each sample with routine name + payload size. Compute
  percentiles offline (numpy). ~50–100 k samples per config from a normal llama/BabelStream run.
- **Deliverable:** percentile table per transport×call-class, CDF plot, and a tail-ratio
  (p99/p50) comparison — RDMA should show a far tighter tail than TCP.

### 1.2 Latency decomposition / breakdown
Where does an RPC's time go? Stacked bar per transport: **marshal → send/wire → backend
queue+dispatch → CUDA execution → return path.** We already capture these segments in
`Execute()` and in the backend `[GVS PROFILE]`; aggregate them. Shows *why* GVirtuS adds latency
and which part each transport optimizes (AM cuts control latency; RDMA/GPUDirect cut data
latency).

### 1.3 Application-level serving latency (LLM) — TTFT and ITL
Map the RPC story onto metrics practitioners cite: **time-to-first-token (TTFT)** and **inter-token
latency (ITL)** p50/p99, per transport, ± the Push/Pop optimization. This grounds the low-level
latency work in a real SLO and is very readable for reviewers.

---

## Tier 2 — Throughput–latency & saturation (classic networking curves)

### 2.1 Latency vs offered load (the "hockey stick")
Drive the backend at increasing request rates (or increasing concurrent in-flight RPCs) and plot
achieved throughput and p50/p99 latency vs offered load. Identify the **knee/saturation point**
and max sustainable message rate (ops/s). Apply Little's law as a sanity check. This is the canonical
INFOCOM systems plot and we have none of it yet.

### 2.2 Small-message rate (control-plane ceiling)
Report **RPC/s for tiny messages** (the control-path ceiling), separate from bulk GB/s. LLM decode
is bound by this number, so it's the metric that predicts inference performance.

---

## Tier 3 — Scalability & multi-tenancy (a strong, novel INFOCOM angle)

### 3.1 N frontends → 1 backend
Aggregate throughput and per-client latency as **N = 1,2,4,8…** frontends share one backend/GPU.
Plot throughput scaling and latency degradation. (We've shown 2 frontends work; make it
quantitative.)

### 3.2 Fairness & isolation
- **Fairness:** Jain's fairness index across tenants sharing the backend.
- **Isolation:** does a bandwidth-heavy tenant (BabelStream) inflate a latency-sensitive tenant's
  (llama) tail latency? Measure the victim's p99 with/without a noisy neighbor. Reviewers love
  isolation results.

### 3.3 Concurrency within a frontend
Multiple CUDA streams / host threads per frontend — how does the (currently synchronous) dispatch
scale, and how much does async dispatch (recommended improvement #3) help? Ties the roadmap to the
design proposals.

---

## Tier 4 — Efficiency, robustness, rigor, positioning

### 4.1 Resource efficiency
- **CPU:** backend + frontend CPU utilization; throughput-per-core; CPU cost per RPC. RDMA should
  free CPU vs TCP — quantify it.
- **Memory registration:** count/overhead of NIC registrations (directly ties to the
  `RMA_ZEROCOPY` rcache OOM in doc 05); registration time vs transfer time.
- **GPU utilization / overlap:** how idle the GPU sits under synchronous dispatch (motivates async).
- **Line-rate efficiency:** bandwidth as % of the NIC's 100/200 GbE line rate; the half-power
  point (N/2 payload size where you hit 50% of peak).

### 4.2 Statistical rigor (needed throughout)
- ≥5–10 repetitions, **95% confidence intervals / error bars** on every plot, warmup discarded,
  report coefficient of variation. State methodology (pinning, clocks, isolation) explicitly.

### 4.3 Baselines & related work
- Native CUDA (have it), TCP and RDMA (have them). **Add at least one external baseline** — e.g.
  rCUDA or another CUDA-remoting framework, or position quantitatively vs their published numbers.
  A networking committee will ask "vs the state of the art?".

### 4.4 Design ablations (attribute the wins to GUSTO's choices)
One controlled delta per design decision:
- AM control path vs TCP control path (latency).
- RDMA data path vs staged/TCP (bandwidth + tail).
- GPUDirect vs host-staged RDMA (bulk-transfer latency/bandwidth — we have doc 04).
- Local Push/Pop optimization on/off (we have this — doc 08, 1.45–1.64×).
Each ablation = one figure showing the contribution of that mechanism.

---

## Suggested priority order (highest reviewer value first)
1. **1.1 latency distributions + CDFs** (instrument `Execute()`), all transports — *the* headline.
2. **1.2 latency decomposition** (reuse existing timers).
3. **2.1 throughput–latency saturation curve** + **2.2 small-message RPC/s ceiling**.
4. **3.1–3.2 multi-tenancy scaling + fairness/isolation** (novelty).
5. **1.3 TTFT/ITL** for llama (readable application SLO).
6. **4.2 statistical rigor** retrofit on all existing plots (error bars).
7. **4.3 external baseline** + **4.4 ablation figures**.

## Concrete first step
Add an env-gated latency trace to `src/frontend/Frontend.cpp::Execute()`
(`GVIRTUS_LATENCY_TRACE=<file>`): record per-RPC round-trip µs + routine + payload size to a
thread-local buffer, flush on exit. One normal llama and one BabelStream run per transport then
yield everything needed for 1.1, 1.2, 2.2, and the tail-latency comparisons — from data we're
already generating.
