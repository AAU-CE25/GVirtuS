# GVirtuS GPU-disaggregation over RoCE -- Full Results (for paper review, 2026-07-24)

> ## AVISO: 2026-08-01 -- new results and corrections: see `RESULTS_2026-08-01.md`
>
> Six sections of this document are extended or corrected. Summary of what must **not** be
> cited without reading the new document first:
>
> | affects | what changes |
> |---|---|
> | §4 (graphsxasync ablation) | Superseded by an ablation with 9 metrics and a native arm on the **same GPU**. The effects are **not orthogonal** (residual 2.42x worse than independent) and the cause is measured. Honest parity: **85.7 %**, not ~100 % |
> | §9 (slots) | Correctness ablation of the lifetime protocol: registration invalidation is **demonstrated necessary**; generation and epoch are **not**. Total protocol cost: **~2.6 µs/transfer = 0.1--0.7 %** |
> | §1 and §10 (bandwidth, OSU) | The real AM[left right arrow]RDMA crossover is at **8 KiB** for pinned H2D, not 4 MiB: the current threshold gives up **x3.42**. For D2H, lowering it would instead be harmful (x0.38 at 16 KiB) |
> | §11 (workload spectrum) | CloverLeaf: **425,084 RPCs** measured, median **7 µs**, **78 %** of time inside RPCs. miniBUDE: **256 KiB** payload confirmed with a file |
> | §2, §5, §6 (llama) | `7b_ucxA_gon_uni_c8` measured (**591.6**, n=5): §5 and §6 **hold** with a correct source. §2 C8 reproduces; **§2 C4 is bimodal** and admits no point estimate |
> | XSBench (separate doc) | **`exit_code` is meaningless**: 0 of 238 runs exit 0. Aggregate rebuilt from raw data; every cell now at n>=3 |
>
> **Binary comparability**: the frontend was rebuilt five times on 2026-08-01. Measurements
> from that day are comparable to each other except the 2x2 ablation, taken before the last
> four rebuilds.


Disaggregated GPU serving via GVirtuS API-remoting over 200 Gb RoCE (ConnectX-7), 2x L40S. Frontend = es-dpu-02, backend GPU = es-dpu-01. Baremetal = native CUDA on dpu-02's local L40S (no GVirtuS). Models: mistral-7b-q4 (7B), tinyllama-1.1b-q4 (1B). Serving via llama.cpp `--parallel 8` continuous batching. Goodput = output tokens completed / common window. Metrics captured per run (in `~/results/summary.csv`, 32 cols): goodput, req/s, effBatch, TTFT p50/95/99, TPOT p50/95/99, SLO-goodput@{1,2,5}s, %SLO, backlog, max-inflight, Jain, stable -- plus per-request JSONL. **+/- = 95% confidence interval (Student-t, n-1 df) across independent runs** (c8 uses n=15, other closed-loop points n=5). **Retention (§2) is PAIRED per matched seed** (R[latin subscript small letter i] = system[latin subscript small letter i] / baremetal[latin subscript small letter i], then mean +/- CI of the ratios -- cleaner than dividing two independent means). Matched seed => identical prompt corpus (UNIQUE prefix = deterministic counter), identical arrival trace (open-loop, same Poisson seed), same warm-up/window/NPRED. UNIQUE = real prefill; FIXED = prefix-cache reuse. Companion files: `OSU_RESULTS.md` (fabric ceiling), `CUDF_ETL_RESULTS.md` (the data-path campaign), `ARCHITECTURE_UCX.md` (how the transport is built and why every threshold has the value it has), and the four contention reports `MINIBUDE_`, `BABELSTREAM_`, `CLOVERLEAF_`, `XSBENCH_RESULTS.md`. `DATA_INDEX.md` says which file answers which question.

---

## 1. Data-path bandwidth -- GPU transfer over the fabric (GB/s)
`transfer_bench`, pinned host buffers, size sweep 4 KiB -> 1 GiB, 30 iters / 5 warm-up (TCP: 15/3, max 256 MiB -- it is ~8x slower and the large points cost minutes). Re-measured **2026-07-26** across all four systems. Baremetal = native CUDA on dpu-02's local L40S.
| size | baremetal H2D | baremetal D2H | **GPUDirect H2D** | **GPUDirect D2H** | UCX-RDMA H2D | UCX-RDMA D2H | TCP H2D | TCP D2H |
|-----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 4 KiB | 0.66 | 0.73 | 0.25 | 0.22 | 0.24 | 0.21 | 0.09 | 0.08 |
| 64 KiB | 6.70 | 8.33 | 2.35 | 1.74 | 2.31 | 1.79 | 0.45 | 0.56 |
| 1 MiB | 16.93 | 23.34 | 5.91 | 3.16 | 5.41 | 3.16 | 0.51 | 0.44 |
| 4 MiB | 19.29 | 25.99 | **18.23** | **20.99** | 10.81 | 11.90 | 0.21 | 0.53 |
| 16 MiB | 23.78 | 26.78 | **23.60** | **23.32** | 12.39 | 12.57 | 1.18 | 0.14 |
| 64 MiB | 26.56 | 27.00 | **24.15** | **23.00** | 12.67 | 12.76 | 3.18 | 0.22 |
| 256 MiB | 26.80 | 27.06 | **24.27** | **22.91** | 12.72 | 12.81 | 2.92 | 0.14 |
| 1 GiB | 26.82 | 27.08 | **24.32** | **22.80** | 12.74 | 12.67 | -- | -- |

**Peak (1 GiB, or 256 MiB for TCP):**
| Path | H2D | D2H |
|------|----:|----:|
| Baremetal local PCIe | 26.8 | 27.1 |
| **GVirtuS GPUDirect (remote GPU)** | **24.3** | **22.8** |
| GVirtuS UCX-RDMA, no GPUDirect | 12.7 | **12.8** |
| GVirtuS TCP | 3.2 | 0.2 |
| Raw fabric ceiling (OSU GPU[left right arrow]GPU) | 24.5 | -- |

**Meaning:** byte-movement to/from the GPU. Remote GPU with GPUDirect reaches **~=91% of local PCIe and ~=99% of the fabric ceiling**. The RDMA-vs-GPUDirect split is now measured directly rather than against a "naive host-bounce" estimate: **GPUDirect is worth 1.9x on H2D (24.3 vs 12.7) and 1.8x on D2H (22.8 vs 12.8)** over the same RDMA transport with the peer-DMA route disabled. *(The 4.0x previously reported here was partly an artefact: without GPUDirect the return path was losing **RDMA entirely** and falling back to eager AM, not merely losing peer-DMA. A client-initiated RDMA GET over a pinned host slot restores it to 12.8 GB/s -- symmetric with its own 12.7 H2D -- so the honest peer-DMA gain is ~=1.8x in both directions.)* Against TCP the data path is **7.6x (H2D)** and two orders of magnitude on D2H.

The **knee is at 4 MiB**, which is the RMA floor (`GVIRTUS_RMA_MIN_BYTES`): below it every transfer takes the eager AM path and all three GVirtuS arms coincide, so no transport choice helps. Above it GPUDirect separates immediately (18.2 GB/s already at 4 MiB). The H2D knee used to sit at 16 MiB because host source registrations under 16 MB were re-made on every put; that threshold was damage control for an uninvalidatable registration cache and was removed once the cache became invalidatable, moving the knee down to the RMA floor (4 MiB H2D 8.0 -> 21.9 GB/s, 8 MiB 8.4 -> 22.9).

**Asymmetry note -- withdrawn.** This previously read that the D2H deficit without GPUDirect (5.7 vs 12.7) was *structural*. It was not: the path had simply lost RDMA and fallen back to eager AM. Registering the staged host slot and letting the client RDMA-GET from it is plain host `ucp_mem_map` -- no peermem, no CUDA memory type -- and D2H becomes symmetric with H2D (12.81 vs 12.72). The remaining GPUDirect advantage is the PCIe bounce it removes, ~=1.8x.

*(Caveat: the previous version of this table quoted "staged host-bounce ~=2.5 GB/s" for both directions. That figure is superseded -- it did not distinguish "RDMA into a host slot" from "eager AM", and the two differ by ~5x.)*

## 2. Isolated serving retention -- 7B, graphs ON, UNIQUE, closed-loop

> **2026-08-01**: C8 re-measured = **591.6** (n=5), reproducing the published 589.3
> within 0.4 %. **C4 is bimodal** (204.8-204.8-227.6-238.9-238.9): it admits no point
> estimate, which confirms its withdrawal as a retention point. See §G of
> `RESULTS_2026-08-01.md`.


> ## Audit 2026-08-01 -- against `summary_master.csv` (963 labels, 1381 rows)
>
> **The `baremetal` column reproduces EXACTLY**, all four cells and their `n`:
> `7b_bm_gon_uni_c{1,2,4,8}` -> 139.4 (n=10), 183.5 (n=15), 220.7 (n=10), 663.3 (n=20).
>
> **The `GVirtuS UCX` column reproduces only in part.** The arm is `7b_ucxgd26_gon_uni_c*`
> (UCX with GPUDirect), and its sibling `7b_ucxnogd_gon_uni_c*` is the control without it:
>
> | CONC | published | `ucxgd26` (n=5) | `ucxnogd` (n=5) | |
> |--:|--:|--:|--:|---|
> | 1 | 133.7 | **133.7** | 133.7 | [check mark] exact |
> | 2 | 154.7 | **154.7** | 150.2 | [check mark] exact |
> | 4 | 243.5 | 204.8 | 257.1 | [ballot x] does not reproduce |
> | 8 | 589.3 | 591.6 | 591.6 | [ballot x] did not reproduce -- **see below** |
>
> **Resolved by re-measurement on 2026-08-01** (5 repetitions each, arm `ucxgd26v2`):
>
> - **C8 = 591.6** (all five identical). Against the published 589.3 that is **within 0.4 %**:
>   the cell reproduces.
> - **C4 = 204.8-204.8-227.6-238.9-238.9.** The mean is 223.0, but the population is
>   **bimodal**, with modes at 204.8 and 238.9. No point estimate describes it, which is
>   exactly why C2/C4 were withdrawn as retention points. The published 243.5 is not
>   reproducible because there is nothing to reproduce: it is a distribution, not a value.
>
> **Warning about `7b_ucx_gon_uni_c8`**: it holds 26 rows and is **not a single arm** -- its
> distribution is `0.0 x1 - 250.3 x2 - 318.6 x1 - 546.1 x2 - 568.9 x8 - 591.6 x12`, a mixture
> of configurations plus one failed run. Aggregating it as if it were one population yields
> 521.6, which describes nothing. Do not use it without separating by arm.
>
> What does **not** change: the bimodality of C2/C4 and their withdrawal as retention points.
> The four traces supporting it are clean -- 164 to 192 lines, all unique, zero failures.
All arms are the **same `llama-server` binary and flags** (`-ngl 99 --no-mmap -c 2048 --parallel 8 --metrics`); only the GPU behind it changes (baremetal = dpu-02's local L40S, GVirtuS = remote L40S on dpu-01). Re-measured **2026-07-26 on freshly-reset backends**. Goodput mean +/- 95% CI (Student-t).

| CONC | baremetal | GVirtuS UCX | retention | **overhead** | bm spread | UCX spread |
|------|----------:|------------:|:---------:|:------------:|----------:|-----------:|
| 1 | 139.4 +/- 0.0 (n=10) | 133.7 +/- 0.0 (n=10) | **95.9%** | **4.1%** | 1.00x | 1.00x |
| 2 | 183.5 +/- 17.5 (n=15) | 154.7 +/- 13.7 (n=10) | *(84.3%)* | *not usable* | **1.95x** | 1.35x |
| 4 | 220.7 +/- 28.5 (n=10) | 243.5 +/- 25.3 (n=25) | *(110.3%)* | *not usable* | **1.73x** | **1.94x** |
| 8 | 663.3 +/- 5.2 (n=20) | 589.3 +/- 5.1 (n=10) | **88.8%** | **11.2%** | 1.07x | 1.04x |

**Headline: remoting a 7B serving workload over RDMA costs 4.1% at CONC=1 and 11.2% at CONC=8** (retention 95.9% and 88.8%; C8 interval [88.2, 89.5] => overhead [10.5, 11.8]). Both anchors are deterministic regimes -- zero rep-to-rep variance at C1, ~=1.05x spread at C8 -- so these are the defensible numbers.

Latency tells the same story more favourably: **TPOT p50 12.5 ms vs baremetal 12.0 ms (+4%)** at C8. The 11% goodput cost is mostly *how many requests fit the window*, not per-token decode speed -- consistent with prefill being the communication-sensitive phase.

**C2 and C4 are withdrawn as retention points, and the reason is measured rather than assumed.** Intermediate concurrency is **bimodal in llama.cpp's continuous batching, and it is bimodal on baremetal too**: baremetal C2 spans 125.2--244.6 (**1.95x**) and baremetal C4 spans 170.7--295.8 (**1.73x**) with no GVirtuS anywhere in the path. The C4 point retention evaluates to **110.3%** -- the arithmetic is fine, the quantity is meaningless. At these concurrencies report a distribution or report nothing.

**Same-system noise control (2026-07-26).** 25 C4 repetitions of one identical configuration span **182.0--352.7 (1.94x)**. Two 5-rep blocks of that *same* configuration, measured back-to-back with a backend reset each, returned means of **202.5 and 259.4** -- a 28% apparent "effect" produced by nothing at all. **Any C2/C4 arm-vs-arm claim at n=5 is measuring this, not the transport.** Two earlier "RDMA beats GPUDirect" readings were exactly this artefact.

**GPUDirect vs UCX-RDMA at C2/C4: re-measurement pending.** The arms above pool configurations that differ only in the *frontend* `GVIRTUS_GPUDIRECT` flag, which -- as noted below -- does **not** disable the feature: the slot pool, the GPU shadow and the D2H-GET path all live on the backend, so these are the same system and their agreement is a control, not a comparison. A genuine RDMA arm requires `GVIRTUS_GPUDIRECT=0` **on the backend** (`reset_backend_nogd.sh`); that sweep is queued.

**What is already settled** is that llama serving cannot exercise GPUDirect at all: measured max transfer during serving is **3.6 MB**, below the **4 MiB** RMA floor (`ucx_rma_min_bytes()`), so every request travels the eager AM path in either arm. GPUDirect fires only during model load (3 fatbins, 5.0--5.9 MB) and adds no measurable backend GPU footprint (4 965 vs 4 969 MiB, sampled every 2 s across both configurations). **=> For LLM serving the transport advantage is RDMA-vs-TCP, not GPUDirect-vs-staged**; GPUDirect matters for bulk data-path workloads (§1).

On decode latency: **with CUDA graphs + sufficient batching, TPOT is comparatively transport-insensitive at C1/C8** (UCX ~= baremetal), but **at intermediate concurrency TCP becomes RPC-bound with substantially higher TPOT (94 ms at C4 vs UCX 17 ms)** -- decode is *not* universally transport-invariant. **Prefill/TTFT remain the most consistently communication-sensitive phase.**

**TCP retention** (measured earlier, same caveats apply to its C2/C4 entries): C1 68.2 +/- 1.4%, C8 52.4 +/- 2.5%; the mid-concurrency values (24--36%) sit in the same noise-dominated regime, though TCP's deficit is large enough at C8 to be unambiguous.

## 3. Transport advantage under concurrency -- UCX vs TCP (goodput t/s, C8, graphs ON)
Re-measured **2026-07-27** on build `8504fb6`; both arms in the same pass, backend reset once per transport. n=5, mean +/- 95% CI.

| | UCX | TCP | UCX/TCP |
|---|---:|---:|:---:|
| 7B UNIQUE | 591.6 +/- 0.0 | 482.4 +/- 12.6 | **1.23x** |
| 7B FIXED | 605.3 +/- 15.5 | 514.3 +/- 15.5 | **1.18x** |
| 1B UNIQUE | 1617.1 +/- 75.7 | 995.2 +/- 131.2 | **1.62x** |

**The advantage still grows as the workload gets more RPC-bound** (7B -> 1B, 1.23x -> 1.62x), which is the qualitative claim. **The magnitude, however, is roughly half of what this table previously reported** (1.64x/1.70x/3.06x), and the change is entirely on the TCP side: 7B UNIQUE went 345.9 -> 482.4 (+39%) and 1B 528.3 -> 995.2 (+88%), while UCX barely moved.

**Why the older numbers were lower -- resolved, and it was not the transport.** The 07-24 figures were taken with a **stale frontend library**. The real `cudaGraphExecUpdate` landed in `4316862` on 07-21, but `lib/frontend/libcudart.so.12` on dpu-02 was not rebuilt until 07-26 04:24, so the 07-24 campaign still ran the stub: CUDA graphs did not collapse the launch stream and every token kept costing ~=1,100 RPCs.

The signature is in TPOT, not in batching -- effective batch actually *fell* slightly (7.10 -> 6.61), ruling out a KV-cache or batching explanation:

| | TPOT p50 then | TPOT p50 now | |
|---|---:|---:|---|
| UCX | 18.7 ms | 12.5 ms | 1.5x |
| TCP | 39.3 ms | 16.0 ms | **2.5x** |

**Both transports improved, TCP more** -- which is what a per-RPC cost disappearing looks like: whoever paid most per call gains most when the calls go away. The two TPOTs now converge (12.5 and 16.0) where they were 18.7 and 39.3, which is the same convergence §4 reports for decode.

AVISO: **This affects everything measured on 2026-07-24, not just TCP.** Any figure from that date is a stub-graphs measurement and must not be compared against post-07-26 numbers. Three other explanations were tested and rejected: the GPU memory leak (physically impossible -- with 30 GiB of ballast resident the llama server cannot even start, so those runs were never made under that pressure), async dispatch (already `=1` in the 07-24 launchers), and the dpu-02 CPU governor (the `schedutil` note dates from May; it reads `performance` at 3193 MHz now and was not changed).

## 4. CUDA Graphs + Async ablation -- decode throughput (llama-bench tg16, t/s)

> **Superseded** by §A of `RESULTS_2026-08-01.md`: a 2x2 with RPC/token, control
> bytes, blocked time, TTFT/TPOT/e2e, CPU and GPU utilisation, three raw repetitions and
> a **native arm on the same GPU that serves as backend**. `llama-bench` reported only
> tok/s, and its native arm ran on a different GPU.

Re-measured **2026-07-27**; `-n 16 -r 3`, value +/- llama-bench's own stddev.

| | graphs OFF | graphs ON |
|---|----------:|---------:|
| GVirtuS sync (async OFF) | 166.1 +/- 0.3 | 528.6 +/- 64.8 |
| GVirtuS **async ON** | 374.6 +/- 0.7 | **560.9 +/- 39.7** |
| Baremetal | 545.7 +/- 0.5 | 632.5 +/- 35.3 |

Graphs alone **3.18x**, async alone **2.26x**, together 3.38x -- graphs subsume most of async (+6% on top of graphs). Reproduces the previous measurement within 1.5%.

**Model-size scaling** (best GVirtuS = graphs+async, vs baremetal graphs ON): 1B **88.7%** (560.9/632.5) - 7B **96.7%** (138.6/143.3) - 13B **98.0%** (79.3/80.9).

**Meaning:** in steady-state decode, graphs replace the per-token launch stream -- ~=1,100 kernel launches -- with a single graph-launch RPC, so launch overhead stops mattering and GPU compute becomes the ceiling. Decisive for small launch-bound models, near-irrelevant for large compute-bound ones. Note the graphs-ON runs carry 6--12% stddev versus 0.1% with graphs off; quote the intervals.

## 5. Graphs on/off in *serving* (goodput t/s, UNIQUE, C8, n=5)

> ## Audit 2026-08-01 -- the "7B UCX graphs ON" cell had no UNIQUE data (now measured)
>
> **Nine of the ten cells reproduce exactly** at `n=5`: `7b_tcpA_g{on,off}_uni_c8`
> (482.4 / 209.4), `1b_tcpA_g{on,off}_uni_c8` (995.2 / 313.5), `1b_ucxA_g{on,off}_uni_c8`
> (1617.1 / 1474.6), `1b_bmA_g{on,off}_uni_c8` (2275.6 / 2104.2) and `7b_ucxA_goff_uni_c8`
> (568.9).
>
> **The tenth did not exist.** The `7b_ucxA_*` family, searched on both dpu-02 and the
> workstation, was:
>
> ```
> 7b_ucxA_goff_uni_c8   n=5   568.9          (graphs OFF, UNIQUE)
> 7b_ucxA_gon_fix_c8    n=5   605.3          (graphs ON,  FIXED)   values: 591.6 and 614.4
> 7b_ucxA_gon_uni_c8    DID NOT EXIST on either machine
> ```
>
> The **591.6** used here for "7B UCX graphs ON" was a value from the **FIXED** campaign, so
> the +4 % gain had a FIXED numerator and a UNIQUE denominator, and §6's UCX row compared the
> FIXED campaign against itself.
>
> **Resolved: the cell was measured on 2026-08-01** (5 repetitions, UNIQUE prompts) and yields
> **591.6** -- the same value. Therefore:
>
> - **§5 holds**: 591.6 / 568.9 = **+4.0 %**, now with a matching source.
> - **§6 holds**: 605.3 (FIXED mean) vs 591.6 (UNIQUE, measured) = **1.023x**, and now genuinely
>   compares FIXED against UNIQUE.
>
> The objection is withdrawn; the [dagger] markers can be removed. See §G of `RESULTS_2026-08-01.md`.
**Async dispatch is ON in BOTH arms here**, so this measures the **marginal** gain of graphs given async -- not graphs+async together. The earlier version of this table did not hold async fixed, which is why its 7B UCX entry read +135%.

| | graphs ON | graphs OFF | marginal gain |
|---|---:|---:|:---:|
| 7B UCX | 591.6 +/- 0.0 | 568.9 +/- 0.0 | **+4%** |
| 7B TCP | 482.4 +/- 12.6 | 209.4 +/- 12.7 | **+130%** |
| 1B UCX | 1617.1 +/- 75.7 | 1474.6 +/- 44.8 | +10% |
| 1B TCP | 995.2 +/- 131.2 | 313.5 +/- 12.4 | **+217%** |
| 1B baremetal | 2275.6 +/- 68.9 | 2104.2 +/- 103.3 | +8% |

**Graphs matter in proportion to how much the transport was carrying the RPC stream.** With async already pipelining launches, RDMA has little left to gain (+4% at 7B, +10% at 1B, against baremetal's +8% -- i.e. RDMA behaves like local execution here). TCP, which pays far more per RPC, gains **+130% to +217%**: for TCP, graphs are not an optimisation but the difference between usable and not.

## 6. FIXED vs UNIQUE prompts (7B, graphs ON, C8, n=5)
| | FIXED (prefix cache) | UNIQUE (real prefill) | ratio |
|---|---:|---:|:---:|
| UCX | 605.3 +/- 15.5 | 591.6 +/- 0.0 | 1.02x |
| TCP | 514.3 +/- 15.5 | 482.4 +/- 12.6 | 1.07x |

**Prompt mode barely matters at C8** -- 2% for UCX, 7% for TCP -- because prefill cost amortises under batching. Headline throughput numbers are therefore prompt-mode-robust. (At intermediate concurrency FIXED shows non-monotone points; C1 and C8 are the reliable anchors.)

## 7. Saturation -- max sustainable offered load (open-loop lambda, **10 matched seeds/point**)
Re-measured **2026-07-27**, all three systems on `8504fb6`. Goodput mean +/- 95% CI; **(k/10)** = seeds meeting the strict criterion -- harness-STABLE **and** 0 timeouts **and** TTFT **p95** < 1 s *(corrected 2026-08-01: the footnote said p50; the criterion that reproduces all twelve cells is p95 -- verified against `summary_master.csv`, 12/12)*.

| lambda (req/s) | baremetal | UCX | TCP |
|----------:|----------:|----:|----:|
| 1 | 119 +/- 23 (**10/10**) | 120 +/- 23 (**10/10**) | 130 +/- 28 (6/10) |
| 3 | 387 +/- 42 (**10/10**) | 388 +/- 41 (7/10) | 402 +/- 31 (0/10) |
| 4 | 532 +/- 54 (6/10) | 535 +/- 54 (2/10) | 491 +/- 51 (0/10) |
| 6 | 814 +/- 40 (0/10) | 827 +/- 40 (0/10) | 536 +/- 70 (0/10) |

**Goodputs are near-identical between baremetal and UCX at every load** (119~=120, 387~=388, 532~=535, 814~=827) -- under open-loop arrival the remote GPU delivers the same throughput. What separates them is **stability**, and doubling the seed count changed that picture: **at lambda=3 baremetal is 10/10 while UCX is 7/10**, where the previous 5-seed table showed both at 5/5. So **UCX's fully-stable region ends one load step earlier than baremetal's**, rather than matching it -- the earlier claim was an artefact of too few seeds.

**TCP has no fully-stable operating point at any lambda >= 3** and is already 6/10 at lambda=1, while still reporting goodput in the 400--540 range: it drains backlog with multi-second TTFT rather than failing outright, which is why goodput alone hides the problem and the **stable-seed count is the robust indicator**.

Full per-lambda/per-seed data in `summary.csv` (`7b_*_sat5_l*`).

## 8. Multi-tenant serving -- summary (detail in `LLAMA-7B_RESULTS.md`)

N isolated 7B frontends against one backend GPU. The full campaign -- the offered-load
correction, the per-pod-constant sweep, the memory footprint and its MPS control, the
capacity-under-SLO curve and the per-tenant fairness -- now lives in **`LLAMA-7B_RESULTS.md`**.
What matters at this level:

- **Throughput.** Against *default* native, remoting is ahead by 1.18x / 1.29x / 1.37x at
  N=2/4/8 (n=3, mean; the 1.42x figure was the maximum of three runs and should not be cited).
  Against *MPS-configured* native it is **97-99%, i.e. parity**. Quote both.
- **Memory -- closed, with a mechanism.** Per-tenant GPU footprint is ~4 490 MiB under GVirtuS
  against ~4 948 MiB native, **~461 MiB less per tenant, 3.7 GB at N=8** -- 10 tenants versus 9
  on a 46 GB L40S. The mechanism is the **per-process CUDA primary context**, measured directly:
  a program that does nothing but `cudaFree(0)` costs **429-431 MiB**, exactly linear over 1, 2
  and 4 processes on two GPUs. A native pod pays it, **an MPS client still pays it** (which is
  why MPS saves only 5.0 MiB/tenant, 0.1% -- MPS shares *scheduling*, not per-client context
  state), and a Gusto tenant pays **zero**, because its frontend process never creates a
  context. Predicted 429 against 426 measured: **99% of the effect accounted for**.
- **Capacity under an SLO -- no difference, and the +17.4% is retracted.** With n=3 and a
  window scaled so every point sees at least 40 offered requests, capacity under a 1 s TTFT p95
  SLO is **lambda = 0.50 for all three systems** (native, native+MPS, Gusto), goodput 51.2-51.7
  t/s, paired difference at the knee **-0.5 t/s, CI95 [-1.6, +0.0]**. The earlier **+17.4% came
  from repetition 1 of a sweep whose 30 s window held 7.5-30 requests**, so goodput inherited
  Poisson counting noise (29.9-59.7 t/s across three repetitions at the same point). What *does*
  separate the systems is the tail, in both directions: at N=8 remoting costs **+273 ms of TTFT
  p95 at light load** (CI95 [+100, +561]) and buys it back above lambda~=1.0, where it delivers
  **+9.9% goodput and 1.8 s less p95** at lambda=1.0 and **+27.5% and 2.5 s less** at lambda=1.5,
  all four CIs excluding zero. **native+MPS matches Gusto's goodput figure for figure** (paired
  difference +0.00 t/s at three of the four loads) and keeps the better tail below saturation.
  Every figure here is a mean over n=3: TTFT p95 varies more than 10x between repetitions of the
  same cell, so no single repetition may be quoted. See LLAMA-7B_RESULTS.md §3b-§3c.
- **Fairness in serving -- equivalent to native.** Normalised by demand, serving fairness is
  **statistically equivalent** (paired Jain difference -0.0027, CI95 [-0.0079, +0.0009] inside a
  declared +-0.05 margin) while Gusto completes significantly more of the offered work (+0.0764,
  CI95 excludes zero). The Jain decline of 1.000 -> 0.804 in the raw table is an **arrival
  artefact**: per-tenant demand imbalance reaches 7.0x, and normalised fairness is 1.0000 exactly.
- **Fairness under equal fixed work -- NOT equivalent, and this is the fourth contribution.**
  At N=8 in miniBUDE one tenant runs at **1.00x** its single-client rate, as if alone on the
  machine, while another runs **4.87x** slower; native and native+MPS share to within 1.03x.
  Reproduced in XSBench (5.98x), identical over TCP. **The mechanism is now established**: not
  the data path, not the shared legacy stream, not transport lock contention, and **not a shared
  CUDA context** -- MPS puts eight clients in one context and stays fair at 1.02x. What is left
  is that **the backend does not arbitrate**: FCFS service plus self-clocked clients lets an
  early lead compound. **Confirmed by intervention**: deficit round-robin at the launch point
  cuts the inequality **5.02 -> 3.09** (CI95 [2.05, 4.34], excluding the baseline) for **+0.6%
  makespan**, pulling the leader off its solo rate (216.4 -> 75.9 GFLOP/s). Workload-dependent:
  1.03x in BabelStream and CloverLeaf at the same N. See `N1_SCHEDULER.md`.


## 9. Slot-count sensitivity -- withdrawn, and now explained

> **Extended** by §B of `RESULTS_2026-08-01.md`: correctness ablation of the lifetime
> protocol (4 variants, same binary), 9 stress scenarios and measured cost
> (**~2.6 µs/transfer**). Registration invalidation is demonstrated necessary, with the
> exact signature of the historical defect; **generation and epoch are not**.


The original table reported a 4x goodput collapse with 2-4 backend RMA slots under 8-way
prefill. It was measured on 2026-07-24, one day before commit `f0d8c1f` raised the RMA floor
from 64 KB to the measured 4 MB crossover. With a 64 KB floor every llama transfer went
through the slot pool, so slot count dominated; with the 4 MB floor **llama serving never
touches the pool at all** (measured: zero transfers >= 1 MB during serving).

**What has since been established is stronger than a re-measurement would be.** Under a
synchronous request/response pattern the extra slots are unreachable *by construction*:
`WriteIovRma` scans for the **first** free slot, and slot 0 is always free again by the time
the next request arrives. Slots 1..N-1 are never touched. The pool is also **per connection**,
so concurrency across tenants does not activate them either -- eight tenants are eight pools
of two slots, not one pool of two.

The count becomes live only when several writes are genuinely in flight on one connection,
i.e. under `GVIRTUS_ASYNC_DISPATCH`, which is **not enabled in any measurement in this
campaign** (verified across all twelve backend reset scripts and the running container).

So the value 2 needs no sweep: no sweep could distinguish anything while the pattern is
synchronous, and a flat result would invite the wrong reading ("the count does not matter"
rather than "the extra slots are unreachable"). See `ARCHITECTURE_UCX.md` section 4.2.

## 10. OSU MPI micro-benchmarks -- GPU buffers, baremetal 2-node RoCE

> **Extended** by §C of `RESULTS_2026-08-01.md`: a sweep from 4 KiB to 64 MiB using
> the same binary for all three paths. **The real crossover for pinned H2D is at 8 KiB**,
> and the 4 MiB threshold gives up up to **x3.42**. The first sweep showed all three paths
> identical below 4 MiB: that was an artefact of lowering the floor on the client only
> (the pool is built by the backend, from its own environment).

| Test | Peak |
|------|-----:|
| osu_bw (GPU[left right arrow]GPU, GPUDirect RDMA) | 24.5 GB/s |
| osu_bibw | 48.8 GB/s |
| osu_latency (1 B) | 7.9 µs |
**Meaning:** standard MPI-GPU transport reference over the same fabric; confirms the 24.5 GB/s ceiling that our data-path (§1) reaches ~=99% of. OSU-MPI cannot run *through* GVirtuS (GVirtuS and CUDA-aware MPI both need UCX -> dual-UCX conflict crashes) -- the GVirtuS-side equivalent is §1. GPU collectives (allreduce/alltoall) need a CUDA-aware OpenMPI (system one lacks it); bcast works. Detail in `OSU_RESULTS.md`.

---

## 11. The workload spectrum -- where remoting costs what, and why

> **Extended** by §D and §E of `RESULTS_2026-08-01.md`. CloverLeaf: **425,084 RPCs**
> measured (published 411,054, within 3.4 %), per-RPC cost **median 7 µs / mean 25.5 /
> p99 546**, **78.2 %** of time inside RPCs and only **27.3 %** of that in execution.
> miniBUDE: maximum payload **262,173 B = 256 KiB**, 66 RPCs over the whole execution.


Seven applications, one architecture, one fabric. This is the result that the individual
reports cannot state on their own: **the cost of remoting is a function of how many RPCs a
workload issues and how large its transfers are, not of "virtualising" as such.**

| workload | character | largest transfer | vs default native | vs MPS-configured native |
|---|---|---:|---:|---:|
| OSU data path | pure bulk transfer | 1 GiB | 99% of fabric ceiling, 91% of local PCIe | - |
| miniBUDE | compute-bound | 256 KB | ahead from N=2 | **96.5%** at N=8 (+3.6%) |
| BabelStream | memory-bandwidth | 512 MiB arrays | **105%** at N=8 (0.95x) | 97% (+3%) |
| XSBench | compute-bound, one-shot load | 192.6 MB, once | **111%** at N=8 (0.90x) | 95% (+5.5%) |
| CloverLeaf | control-bound, 411k RPCs/run | small | 92.6% at N=8 (1.08x) | *MPS invalid here* |
| cuDF ETL | transfer-bound | 62.5 MiB x 8 per batch | 89.8% (N=1) -> **100.9%** (N=8) | 72.7-76.7% |
| llama-7B serving | RPC-bound | 3.6 MB | 95.9% (C1), 88.8% (C8) | - |

Read down the "largest transfer" column and the reason for the spread becomes clear.

**GPUDirect only ever fires above the 4 MiB RMA floor.** miniBUDE (256 KB) and llama serving
(3.6 MB) never reach it, so `rdma` and `GPUDirect` are indistinguishable in both -- which is
the control confirming the threshold works, not a disappointment. XSBench moves its grid once
and then computes for 100 s, so its transfer is milliseconds against the makespan. CloverLeaf
issues **411,054 RPCs in a 47 s run** and is dominated by per-RPC latency, which GPUDirect
does not address. **cuDF is the only workload in the set that both moves bulk data and moves
it repeatedly**, and it is the only one where the peer-DMA path is measurable.

**The MPS column is the one that keeps the comparison honest.** The GVirtuS backend serves
every tenant as a thread inside one process, so they share a single CUDA context -- which is
exactly what MPS does for native processes. Against *default* native (no MPS, how most
installations run) remoting is ahead on three of the seven workloads. Against
*MPS-configured* native it is 3-6% behind. Both numbers belong in any write-up; quoting only
the first is a straw man and only the second discards the practical result.

**One exception, measured:** for CloverLeaf, MPS is not a valid baseline. It corrupts the
field-summary reduction in 6.6-7.0% of lines at N>=4 while GVirtuS -- which consolidates
contexts by the same mechanism -- corrupts nothing. See `CLOVERLEAF_RESULTS.md`.

### What the cuDF campaign adds that the others cannot

cuDF is the only workload that exercises the data path, so it is the only place the
architecture's central claim can be tested directly. The ablation is `ucx_host_rma`: same
transport, same slot pool, same everything, except the payload lands in host memory and is
then copied to the GPU.

| | N=1 | N=2 | N=4 | N=8 |
|---|---:|---:|---:|---:|
| GPUDirect - host RMA, pageable | -57.6 ms | -64.2 | -88.2 | -105.8 |
| GPUDirect - host RMA, **pinned** | **-78.0 ms** | - | - | **-127.5 ms** |

Monotone in N, mechanism confirmed by counters (100% of volume by host bounce versus ~0),
and **predicted in advance from PCIe arithmetic** before it was measured. It also survives
the obvious objection: pinning the baseline's memory makes the gap *larger*, not smaller.

The phase decomposition isolates it further. H2D growth from one tenant to eight:

| arm | pageable | pinned |
|---|---:|---:|
| GPUDirect (peer-DMA to GPU) | **x1.91** | **x1.79** |
| host RMA (RDMA + host bounce) | x2.11 | x2.06 |
| native (driver pageable staging) | x2.38 | x2.44 |

Each rung removes one layer of host-side staging and degrades less under concurrency, in
both memory types. Full campaign in `CUDF_ETL_RESULTS.md`.

### And the uncomfortable one

cuDF's parity with native at N>=4 is measured against native **as it is deployed by
default**, on pageable host memory. Pinning the baseline drops retention to **72.7-76.7%**,
measured twice by independent campaigns. The defensible claim is *"we match native as it is
deployed by default, and the mechanism is that our data path is pinned and registered by
construction"* -- not *"we match native"*.

The control that makes the mechanism claim safe is TCP: pinning helps each path in proportion
to how much it DMAs from client host memory, and TCP -- which copies through the kernel into
a socket -- moves **-4.5%** against native's **-48.8%**. If pinning were a generic effect,
TCP would have moved with the rest.

---
## One-line summary

Zero-copy GPUDirect makes the **data path** near-transparent -- 99% of the fabric ceiling,
91% of local PCIe, 1.8x a same-transport RDMA host bounce and 7.6x TCP on H2D. The residual
cost sits in the **control path** (per-operation RPC latency), which CUDA graphs and async
dispatch largely erase (decode 86-98% of baremetal).

**Across seven workloads the cost of remoting tracks RPC count and transfer size, not
virtualisation per se** (section 11): 99% for pure transfer, 96.5% for compute-bound
miniBUDE, 92.6% for control-bound CloverLeaf at N=8, 88.8% for 7B serving at C8, and for
cuDF 89.8% at one tenant rising to parity at eight. Against MPS-configured native the band
is 3-6%; against default native, remoting is ahead on three of the seven, because the backend
delivers MPS-equivalent context consolidation as a structural property of its threading
model -- with no daemon to deploy, configure or fail -- and costs ~460 MiB *less* GPU per
tenant (10 tenants fit on a 46 GB L40S versus 9).

**GPUDirect is measurable on exactly one of these workloads**, and that is a finding rather
than a gap: it fires only above the 4 MiB RMA floor, and only cuDF both moves bulk data and
moves it repeatedly. There the peer-DMA advantage over an otherwise identical host-bounce
path is 57.6 ms at one tenant growing to 105.8 ms at eight, monotone, predicted in advance
from PCIe arithmetic, and *larger* -- not smaller -- when the baseline's memory is pinned.

**The honest qualifier**: cuDF's parity with native at N>=4 holds against native as it is
deployed by default. Against a native baseline with pinned host memory, retention is
72.7-76.7%. Both numbers are reported, because the second is the one a reviewer will compute.

**The correctness qualifier, and it must travel with every GPUDirect number.** On this path the
NIC peer-DMAs into device memory and a CUDA read consumes it with only an active message in
between. **We do not claim that visibility is guaranteed.** The ordering it relies on is not a
UCX guarantee -- `ucp_put_nbx` completion is *local* -- it holds because both operations take one
RC queue pair, and it is not negotiated or checked at runtime. What is claimed: **no visibility
failure across 2.64 M RMA admissions, with the verifying subset (XSBench, identical checksum
408237 across every arm) bit-exact, on UCX 1.20.0 with a
single RC lane**. See `CONTRACTS.md` §6, where this is invariant I10, stated as a bounded
assumption and deliberately kept outside the table of nine that *are* discharged.

**Still open**: why TCP's aggregate throughput *falls* as tenants are added (235 -> 170 -> 83
MB/s) when eight independent round-trip-bound connections should give eight times more.
Bandwidth, packet loss, flow control, socket buffers, backend CPU and the legacy-stream
barrier are all excluded by measurement; the mechanism is not identified.

## Scope: what we are allowed to fix, and what we are not

GVirtuS's claim is **transparency** -- an unmodified CUDA application runs against a remote
GPU. That draws a hard line through every optimisation in this campaign:

**Fixes inside the remoting layer are in scope.** `cudaGraphExecUpdate` was a stub in *our*
frontend for an API llama.cpp already called; implementing it is making our layer support
what the application asked for. Same for the client-GET D2H path, per-connection memory
reclamation, async dispatch of stream-ordered calls, and the (ineffective) cuBLAS async-GEMM
patch. None of these touch the application.

**Changes to the application are out of scope**, even when they would improve our numbers.
The clearest example is the open prefill gap: TTFT is ~2.9x baremetal at C8, and the likely
remedy is extending CUDA-graph capture to prefill with prompt-length bucketing. That is a
llama.cpp change. Making it and then benchmarking llama.cpp would be circular -- we would be
tuning the workload to the system and then measuring the workload. **So the prefill gap is
reported as a measured cost of remoting, not as something we optimise away.**

The consequence for how results should be read: every figure here is what an **unmodified**
application gets. That is a lower number than a tuned one, and it is the only number that
supports the transparency claim.

Worth noting that the cuBLAS ablation, which came back null, strengthens rather than weakens
this: we did look for the optimisation *inside our own layer*, implemented it, and measured
that it does not help this workload (§ see `cublas` labels). Reporting the null is more
credible than not having tried.

## Data provenance / caveats (honest)

### AVISO: Stale-frontend cutoff: 2026-07-26 04:24

`lib/frontend/libcudart.so.12` on dpu-02 was rebuilt at **07-26 04:24**. The real
`cudaGraphExecUpdate` had landed in `4316862` on **07-21**, but until that rebuild the
deployed library still carried the stub, so CUDA graphs did not collapse the launch stream
and every decoded token kept costing ~=1,100 RPCs.

**Consequence: any GVirtuS-side measurement taken before 07-26 04:24 understates decode
throughput, and understates it more the more expensive the transport's per-RPC cost is.**
Measured on the same configuration before and after:

| | TPOT p50 before | TPOT p50 after |
|---|---:|---:|
| UCX 7B UNIQUE C8 | 18.7 ms | 12.5 ms |
| TCP 7B UNIQUE C8 | 39.3 ms | 16.0 ms |

**Baremetal is unaffected** -- it never loads the GVirtuS shim -- so baremetal rows from any
date remain valid, and retention figures computed against them were biased *against*
GVirtuS, not in its favour.

Everything now quoted in §1--§8 has been re-measured after the cutoff. The remaining
pre-cutoff data in `summary.csv` is kept for history under its original labels; the
re-measured points carry an `A`/`26` suffix (`7b_tcpA_*`, `7b_ucxgd26_*`, `1b_ucxA_*`, ...).
Anything else from 07-23/07-24 should be re-measured before use.

**How it was found:** TCP goodput at C8 rose 39% between 07-24 and 07-26 with identical run
parameters. Effective batch *fell* (7.10 -> 6.61), ruling out batching; the whole difference
sat in TPOT. Three candidate causes were tested and rejected first -- the GPU memory leak
(the server cannot even start with 30 GiB of ballast resident, so no run was ever made under
that pressure), async dispatch (already enabled in the 07-24 launchers), and the dpu-02 CPU
governor (that note dates from May; it reads `performance` and was not changed).
- Retention c2/c4 re-measured isolated+drained 2026-07-24 (earlier values were CPU-steal from a co-located baremetal run). TCP c2/c4 low values are REAL (RPC-bound mid-concurrency valley, reproduces).
- Multitenancy 8-pod: fits after backend `GVIRTUS_RMA_SLOT_CAP_MB=128` (root-caused 2026-07-24; default 1025 MB OOMs at pod 7). lambda=2 overloads; stable point lambda~=1.
- §4 uses a single llama-bench campaign (decode ablation); §7 uses **five matched arrival traces per load and system**. Trends robust; CIs shown where n>=2.
- Graphs enabled = env `GGML_CUDA_DISABLE_GRAPHS` UNSET (presence-check, not value).

