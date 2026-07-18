# 03 — BabelStream transport sweep (full-size, all configs)

Status: **complete** — 180 data points saved in the repo.
Last updated: 2026-07-17.

## What was measured

BabelStream (memory-bandwidth proxy) over the two-node L40S + ConnectX testbed, across:

- **4 configurations**
  - `baremetal` — native CUDA on the L40S, no GVirtuS (device-bandwidth ceiling).
  - `gvirtus-tcp` — GVirtuS, UCX communicator, `UCX_TLS=tcp,self`.
  - `gvirtus-rdma` — GVirtuS, UCX RDMA (`rc_mlx5`), **backend `GVIRTUS_GPUDIRECT=0`** (host-staged).
  - `gvirtus-rdma-gpudirect` — GVirtuS, UCX RDMA, **backend `GVIRTUS_GPUDIRECT=1`** (NIC→GPU).
- **9 array sizes** — 2¹⁸…2²⁶ elements (double): 262144 → 67108864 (2 MiB → 512 MiB per buffer).
- **5 kernels** — Copy, Mul, Add, Triad, Dot.
- **100 measured iterations** each (BabelStream reports min/max/avg per kernel).

⇒ 4 × 9 × 5 = **180 data points**, each with min/max/avg runtime + max bandwidth.

## Files (all in `docs/benchmarking/data/babelstream/`)

| File | Contents |
|------|----------|
| `babelstream_sweep.csv` | Raw results. Columns: `config, function, num_times, n_elements, sizeof, max_mbytes_per_sec, min_runtime, max_runtime, average_runtime`. |
| `babelstream_summary_gbps.csv` | Tidy pivot: bandwidth (GB/s) per `function × n_elements × config`. |
| `babelstream_latency_us.csv` | Tidy pivot: per-launch latency (µs) per `function × n_elements × config`. |
| `babelstream_rpc_latency_us.csv` | Control-path RPC latency floor (smallest size) per kernel × config. |
| `plot_sweep.py` | Reproducible plotting script (pandas + matplotlib). |
| `plots/babelstream_<kernel>.png` | Bandwidth vs size, all 4 configs, per kernel. |
| `plots/babelstream_<kernel>_latency.png` | Per-launch **latency** vs size, per kernel. |
| `plots/babelstream_rpc_latency_floor.png` | RPC latency floor bar chart (TCP vs RDMA). |
| `plots/babelstream_triad_gvirtus.png` | Triad, GVirtuS configs only (readable zoom). |
| `plots/babelstream_triad_overhead.png` | Triad GVirtuS bandwidth as a fraction of bare metal. |

Regenerate after editing the CSV: `python docs/benchmarking/data/babelstream/plot_sweep.py`.

## What BabelStream can and cannot show (read this first)

BabelStream transfers its arrays to the GPU **once** at init, then loops kernels on
device-resident data. Therefore:
- It measures **kernel bandwidth** (GPU-bound) + **per-launch RPC latency** (one RPC per
  kernel launch). These are the meaningful axes here.
- It is **insensitive to the data-path transport** for steady state — which is why
  `gvirtus-rdma` and `gvirtus-rdma-gpudirect` are ~identical in every table below. **This is
  expected, not a setup error.** The GPUDirect / bulk-transfer story lives in doc 04
  (transfer bandwidth), where GPUDirect shows a real ~2.3× D2H win. Do not look for GPUDirect
  effects here.

## Control-path RPC latency (the headline networking result)

Every kernel launch over GVirtuS is **one synchronous RPC round-trip**. At the smallest array
size the transfer is negligible, so the per-launch latency is essentially the **pure RPC
floor**:

| kernel | bare metal | UCX-TCP | UCX-RDMA | RDMA speedup vs TCP |
|--------|-----------:|--------:|---------:|--------------------:|
| Copy  | 7.4 µs | 320.0 µs | 237.7 µs | 1.35× |
| Mul   | 7.1 µs | 322.2 µs | 238.0 µs | 1.35× |
| Add   | 7.2 µs | 323.4 µs | 239.7 µs | 1.35× |
| Triad | 7.2 µs | 322.7 µs | 237.6 µs | 1.36× |
| Dot   | 28.6 µs | 400.4 µs | 302.3 µs | 1.32× |

**RDMA cuts the per-launch RPC round-trip by ~26 % (~84 µs/call) vs TCP** (~238 µs vs
~322 µs), consistently across kernels. Bare metal (~7 µs, no RPC) is the floor. Dot is higher
because it also does an explicit D2H copy of the partial sums (extra RPC). This RPC-latency
gap is *why* RDMA beats TCP on the bandwidth curves at small sizes — bandwidth there is simply
`payload / RPC_latency`. Plots: `plots/babelstream_<kernel>_latency.png`,
`plots/babelstream_rpc_latency_floor.png`.

## Key findings — bandwidth (Triad, GB/s)

| per-buffer | bare metal | TCP | RDMA | RDMA+GPUDirect |
|-----------:|-----------:|----:|-----:|---------------:|
| 2 MiB   | 925*  | 23  | 30  | 31  |
| 8 MiB   | 2486* | 92  | 121 | 122 |
| 32 MiB  | 710   | 444 | 487 | 492 |
| 128 MiB | 674   | 519 | 554 | 549 |
| 512 MiB | 681   | 633 | 646 | 641 |

\* Bare-metal small-size numbers are a **cache artifact**, not a real bandwidth. BabelStream
reports the single fastest of 100 iterations, and at small sizes the three arrays fit inside
the L40S's ~96 MB L2 cache, so the kernel reads from cache (up to ~3500 GB/s, 4× the 864 GB/s
DRAM peak). The bare-metal curve peaks right where total size ≈ L2 (≈ 4M elements × 3 × 8 B =
96 MB) and then **drops to the true DRAM rate ~680 GB/s** once the arrays spill out of cache.
This is exactly why BabelStream must be run with arrays ≫ cache; only the **large-size**
points (≥ 32 MiB/buffer) are a fair bare-metal reference. GVirtuS does not show this spike
because per-launch RPC latency dominates at small sizes and masks the cache effect.

**Interpretation:**
1. **Control-path overhead dominates at small sizes.** Each kernel launch is one RPC; for
   small arrays the kernel is tiny so RPC latency caps effective bandwidth (23–30 GB/s at
   2 MiB). As arrays grow, the per-launch cost amortizes and GVirtuS approaches bare metal
   (~633–646 GB/s at 512 MiB ≈ 93–95 % of the ~680 GB/s bare-metal sustained rate).
2. **RDMA clearly beats TCP** at every size (e.g. +30 % at 8 MiB), because the RPC round-trip
   latency is lower on RoCEv2 than on TCP. The RDMA **data path is genuinely used**: backend
   logs show `WriteIovRma(zerocopy) ... big=536870912` (512 MB fragments via `ucp_put`), not
   the AM fallback.
3. **RDMA ≈ RDMA+GPUDirect — and GPUDirect is NOT cleanly validated by this run.** Two reasons:
   (a) **Metric insensitivity** — BabelStream reports steady-state *kernel* bandwidth, which
   GPUDirect cannot change (GPUDirect speeds bulk H2D/D2H *transfer*, done once at init).
   (b) **GPUDirect did not fully engage**: the backend probe succeeds and allocates GPU
   shadows, but the **frontend** probe FAILS (it `dlopen`s the GVirtuS *stub* `libcudart`
   instead of the real one → `cudaMalloc(4K)` fails), so the frontend advertises
   **`0 remote slots with gpu shadow`** (confirmed in backend log:
   `rma_setup: received 2 remote slots (0 with gpu shadow)`). With no GPU shadow on the client
   side, there is no end-to-end NIC↔GPU path, so the two RDMA configs are effectively the same
   data path here. **Do not present this as a validated GPUDirect result.**

## Caveats / methodology notes

- **`GVIRTUS_RMA_ZEROCOPY=1` crashes the backend (OOM, exit 137) at ≥32 MiB.** The GPUDirect
  sweep was run with zerocopy **off** (matching the backend). This is a reproducible GVirtuS
  instability worth a bug note (and consistent with the RDMA instability the report mentions).
- **Frontend-side GPUDirect probe fails** in the frontend container: it `dlopen`s the GVirtuS
  **stub** `libcudart` (first on `LD_LIBRARY_PATH`) instead of the real one, so
  `cudaMalloc(4K)` fails and the frontend logs `GPUDirect=disabled`. This does **not** affect
  BabelStream (its frontend data is host-resident), but it means the **backend** GD flag is
  what distinguishes the two RDMA configs here. Fixing the probe to load the real cudart is a
  separate improvement.
- Each config's semantics are set by transport + backend `GVIRTUS_GPUDIRECT`; TCP never
  engages GPUDirect regardless. The backend was restarted (fresh CUDA context) between the
  GD=0 and GD=1 phases.

## Reproduce

Backend (es-dpu-01): `GVIRTUS_GPUDIRECT={0|1} bash /tmp/gvirtus-backend-run.sh`.
Frontend sweep (es-dpu-02, source-built `gvirtus-fe` container), per config set the transport
env (`UCX_TLS`, `UCX_NET_DEVICES`, `UCX_IB_GID_INDEX`, `GVIRTUS_GPUDIRECT`) then run
`~/benchmarks/harness/sweep_run.sh` with `BIN`, `CONFIG`, `OUT`, `ITERS`. See the plan's
progress log for exact commands.
