# 04 — Transfer bandwidth (H2D / D2H) — GPUDirect validated

Status: **complete — GPUDirect delivers a clean ~2.3× D2H speedup.**
Last updated: 2026-07-17 (revised after correcting a benchmark artifact — see §5).

## Why this benchmark

BabelStream (doc 03) reports steady-state **kernel** bandwidth, insensitive to the transport
(arrays transferred once, then reused on-device). To evaluate the **data path** — where RDMA
and GPUDirect help — we measure raw `cudaMemcpy` **H2D / D2H bandwidth** vs transfer size.
Microbenchmark: `harness/transfer_bw2.cu` — for each size, allocate an exactly-sized device
buffer + pinned host buffer, warmup + 20 timed `cudaMemcpy`, 4 KiB … 256 MiB.

Configs (UCX communicator only): `baremetal` (native PCIe), `gvirtus-tcp` (UCX `tcp,self`),
`gvirtus-rdma` (RoCE, backend `GVIRTUS_GPUDIRECT=0`), `gvirtus-rdma-gpudirect` (RoCE, backend
`GVIRTUS_GPUDIRECT=1`).

> "TCP + GPUDirect" is intentionally NOT tested — GPUDirect requires an RDMA-capable transport
> (NIC peer-DMA to GPU memory), which plain TCP lacks.

## Data / plots (`docs/benchmarking/data/transfer/`)
`transfer_bw2.csv` (raw), `plot_transfer.py`, `plots/transfer_h2d.png`,
`plots/transfer_d2h.png`, `plots/transfer_d2h_rdma_vs_gpudirect.png`.

## Results — bandwidth (GB/s), selected sizes

**H2D (host→device):**
| size | bare metal | UCX-TCP | UCX-RDMA | RDMA+GPUDirect |
|-----:|-----------:|--------:|---------:|---------------:|
| 1 MiB   | 16.8 | 1.6 | 6.1 | 6.0 |
| 8 MiB   | 20.0 | 1.9 | 7.7 | 7.8 |
| 256 MiB | 26.8 | 2.7 | 7.1 | 7.1 |

**D2H (device→host):**
| size | bare metal | UCX-TCP | UCX-RDMA | RDMA+GPUDirect |
|-----:|-----------:|--------:|---------:|---------------:|
| 1 MiB   | 23.3 | 0.45 | 3.1 | 2.7 |
| 4 MiB   | 26.0 | 1.89 | 3.3 | **8.1** |
| 64 MiB  | 27.0 | 1.74 | 3.9 | **8.9** |
| 256 MiB | 27.1 | 1.64 | 3.9 | **8.9** |

## Findings

### 1. RDMA is ~3× faster than TCP on the data path
H2D at 1 MiB: 6.1 vs 1.6 GB/s. The RDMA advantage holds across all sizes and is the concrete
benefit of the UCX+RoCE path over sockets.

### 2. GPUDirect gives a clean ~2.3× D2H speedup for transfers ≥ 4 MiB
`RDMA+GPUDirect` D2H reaches **~8.9 GB/s vs ~3.9 GB/s** for plain RDMA (2.3×), sustained from
4 MiB to 256 MiB. On D2H the backend's NIC reads GPU memory **directly** (GPUDirect RDMA),
skipping the backend host bounce — exactly the path the project's GPUDirect design targets.
Below ~2 MiB, GPUDirect is marginally slower (setup/registration overhead not yet amortized),
so the win is specifically for **bulk** D2H.

### 3. GPUDirect gives no H2D benefit (expected)
H2D is ~7.5 GB/s with or without GPUDirect. The H2D source is the client's **host** buffer, so
there is no client-side GPU memory for the NIC to peer-DMA from; the transfer is transport-
limited on the RC lane either way. GPUDirect's asymmetric benefit (D2H yes, H2D no) is
consistent with its mechanism.

### 4. Both directions remain well below bare-metal PCIe
Bare metal: ~27 GB/s D2H, ~27 GB/s H2D (large). GVirtuS tops out ~8-9 GB/s (GPUDirect D2H) /
~7 GB/s (RDMA H2D) because every `cudaMemcpy` is a synchronous RPC over a 25 GbE RoCE link —
the report's synchronous-dispatch limitation plus the NIC line rate cap achievable bandwidth.

## 5. Correction — earlier "GPUDirect crashes ≥ 8 MiB" was a BENCHMARK ARTIFACT

An earlier version of this doc reported that RDMA+GPUDirect crashed at ≥ 8 MiB. **That was
wrong.** Root cause: the first microbenchmark (`transfer_bw.cu`) allocated **one oversized
256 MiB device buffer** and transferred growing sub-ranges of it, reusing the same buffer for
H2D and D2H. That specific pattern trips a narrow bug in the GVirtuS RMA path and resets the
connection at ≥ 8-16 MiB.

Verification that GPUDirect itself is fine:
- **simple_matrix over GPUDirect passes at N = 2048 / 4096 / 8192 / 16000** — i.e. 16 MiB,
  64 MiB, 256 MiB, and **976 MiB** per `cudaMemcpy` — all `check=pass` (matches the project
  report's large-matrix results).
- The rewritten `transfer_bw2.cu` (exactly-sized buffers, separate H2D/D2H buffers — the
  normal CUDA pattern) runs the **full range to 256 MiB with no crash** and produced the data
  above.

**Remaining real (but narrow) GVirtuS bug:** transferring sub-ranges of a large, registered,
reused device buffer can reset the RDMA connection, and after such a reset the backend's UCX
listener fails to rebind (`ucp_listener_create: Device is busy`) until a container restart.
Worth fixing, but it does **not** affect normal exact-sized allocations. Separately,
`GVIRTUS_RMA_ZEROCOPY=1` still OOM-kills the backend at ≥ 32 MiB (keep it off).

## Takeaways for the paper
- **RDMA vs TCP:** clean ~3× data-path bandwidth win. Solid.
- **GPUDirect:** validated — **~2.3× D2H speedup for bulk transfers** (≥ 4 MiB), no H2D change
  (mechanistically expected). This is a genuine, defensible GPUDirect result.
- Note the buffer-reuse RMA edge case + `RMA_ZEROCOPY` OOM + listener-recovery as known bugs /
  future hardening.
