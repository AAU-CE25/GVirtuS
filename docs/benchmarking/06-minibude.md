# 06 — miniBUDE (compute-bound proxy)

Status: **complete — GVirtuS overhead < 0.2 % across all transports.**
Last updated: 2026-07-17.

## What miniBUDE measures
miniBUDE (UoB-HPC / UK Mini-App Consortium) is the mini-app version of the **Bristol
University Docking Engine** — molecular-docking virtual drug screening. High arithmetic
intensity (many FLOPs/byte), small data footprint, one main compute kernel (`fasten_main`)
iterated N times. Reports **GFLOP/s**. It is the **compute-bound complement to BabelStream**:
it uploads atoms/poses to the GPU **once** (`cudaMalloc`+`cudaMemcpy`, default `MEM=DEFAULT` —
no managed memory), then loops the kernel on device-resident data and copies back a small
energies array.

## Configs & method
Deck `bm1`, 8 iterations, ppwi=1. 4 configs (UCX communicator only): bare metal (native CUDA),
UCX-TCP, UCX-RDMA (backend `GVIRTUS_GPUDIRECT=0`), UCX-RDMA+GPUDirect (backend `=1`).
GPUDirect verified enabled on the backend per the doc-00 checklist for the GD run.

Built for GVirtuS by compiling `src/main.cpp` (`-x cu`) with `--cudart shared` and linking the
GVirtuS frontend stubs; confirmed `ldd` resolves `libcudart.so.12` to
`/usr/local/gvirtus/lib/frontend/`. (nvcc default is *static* cudart, which cannot be
redirected — `--cudart shared` is required to run an existing CUDA app over GVirtuS by
`LD_LIBRARY_PATH`.)

## Results (deck bm1, 8 iters)

| config | GFLOP/s | % of bare metal | giga-interactions/s | setup transfer (context_ms) | valid |
|--------|--------:|----------------:|--------------------:|----------------------------:|:-----:|
| bare metal            | 216.63 | 100.00 % | 5.413 | 0.27 ms  | ✓ |
| UCX-TCP               | 216.29 | 99.84 %  | 5.404 | 10.29 ms | ✓ |
| UCX-RDMA              | 216.45 | 99.92 %  | 5.408 | 2.14 ms  | ✓ |
| UCX-RDMA+GPUDirect    | 216.44 | 99.91 %  | 5.408 | 2.25 ms  | ✓ |

Data: `data/minibude/minibude.csv`. Plots: `plots/minibude_gflops.png`,
`plots/minibude_context_ms.png` (`python data/minibude/plot_minibude.py`).

## Findings

### 1. Remote virtualization is essentially FREE for compute-bound HPC
All GVirtuS configs land within **0.2 %** of bare-metal GFLOP/s. The kernel runs ~295 ms per
iteration (2364 ms total) while the entire network interaction is a one-time setup transfer of
a few ms — so the remoting overhead is amortized into the noise. This is the headline
counterpoint to BabelStream: **when the GPU does real work, GVirtuS adds ~nothing.**

### 2. Transport-insensitive throughput — but transport still shows in setup cost
GFLOP/s is identical across TCP / RDMA / GPUDirect (the kernel dominates). The one place the
transport is visible is the **one-time setup transfer** (`context_ms`): TCP ~10.3 ms vs RDMA
~2.1 ms (~5× — consistent with the transfer-bandwidth doc 04), and GPUDirect ~2.2 ms. For a
compute-bound run this difference is irrelevant (<0.5 % of runtime), but it confirms the
RDMA-vs-TCP data-path gap is real and independent of the workload.

### 3. Correctness preserved
Every config reports `valid: true`, `max_diff_% = 0.014`, identical energies — remote
execution is numerically faithful.

## Notes / limitations
- Larger deck `bm2` (65536 poses) is far heavier; an 8-iteration run exceeded the 300 s
  harness timeout (pure compute time, not a GVirtuS failure). bm1 is sufficient to establish
  the compute-bound overhead result.
- ppwi=1 (miniBUDE default single config). Absolute GFLOP/s would rise with a tuned ppwi, but
  the **GVirtuS-vs-bare-metal ratio** (the quantity of interest) is unaffected.

## ⚠️ Honest framing: miniBUDE deliberately shows NO transport difference

TCP ≈ RDMA ≈ GPUDirect ≈ bare metal here — **and that is the expected, correct result, not a
missing signal.** miniBUDE is ~2364 ms of GPU compute vs a one-time ~2-10 ms transfer, so the
network is <0.5 % of runtime and no transport can move throughput. Even the RPC-latency
advantage (RDMA saves ~84 µs/launch, doc 03) is invisible: 8 launches × 84 µs = 0.67 ms
against 2364 ms.

**What miniBUDE proves:** GVirtuS is overhead-free for compute-bound HPC (a genuine, useful
result). **What it does NOT and CANNOT show:** any transport (RDMA/GPUDirect) benefit — that
requires a network-stressing workload.

**Where the transport story actually lives** (use these for RDMA/GPUDirect claims):
- **Data-path transfer bandwidth** — doc 04: RDMA ~3× TCP, GPUDirect ~2.3× D2H.
- **Control-path RPC latency** — doc 03: RDMA ~26 % lower per-launch than TCP.
- **Call-heavy / streaming apps** (LLM inference, OpenCV-DNN/YOLO from the report): many small
  H2D/D2H + thousands of launches → most transport-sensitive.

Rule of thumb: **on-device compute-bound proxies (miniBUDE; likely XSBench, CloverLeaf) hide
the transport; transfer-heavy and call-heavy workloads expose it.**

## Takeaway for the paper
miniBUDE gives the clean "**overhead-free for compute-bound workloads**" result: <0.2 %
slowdown vs native across every transport. Paired with BabelStream (control-path/RPC-bound) and
the transfer benchmark (data-path RDMA/GPUDirect), it completes the roofline spread:
compute-bound → overhead hidden; bandwidth/latency-bound → transport matters.
