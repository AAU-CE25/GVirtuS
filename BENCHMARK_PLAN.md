# GVirtuS (GUSTO) Benchmark Expansion Plan — InfoComm 2027

## ⚠️ STANDING RULE — ALWAYS VERIFY GPUDIRECT IS WORKING, NEVER ASSUME WITHOUT BACKING
Before reporting ANY GPUDirect/transport result: confirm backend log says
`GPUDirect=enabled`, backend launched with `GVIRTUS_GPUDIRECT=1`, RDMA TLS selected (not tcp),
and the RMA data path shows in the log (`WriteIovRma ... big=`). Cross-check surprises against
simple_matrix at large N. Isolate GD=0 vs GD=1 backends. Full checklist:
docs/benchmarking/00-CRITICAL-verify-gpudirect.md. (We were burned twice by assuming.)


## Goal
Expand the test suite beyond cuBLAS-matmul + OpenCV-DNN/YOLO to recognized HPC proxy
apps + LLM inference, evaluated across bare-metal and UcxCommunicator modes
(RDMA, RDMA+GPUDirect, TCP).

## Confirmed framework capabilities (from plugin/frontend audit)
- User-defined CUDA kernels: YES — `__cudaRegisterFatBinary/End`, `__cudaRegisterFunction`,
  `__cudaRegisterVar`, `__cudaPushCallConfiguration`, `cudaLaunchKernel`, `cudaLaunchKernelExC`.
  => Custom-kernel HPC apps are runnable in principle (SAXPY microbench already proved this).
- Memory/events/streams: cudaMalloc/Free/Memset/MemGetInfo/Memcpy(+Async), events, streams,
  cudaGetDeviceProperties, cudaDeviceGetAttribute, cudaFuncGetAttributes, occupancy — all present.
- cudaMallocManaged: stub present, BUT `cudaMemPrefetchAsync` MISSING => unified-memory apps risky.
- cuBLAS: Sgemm/Dgemm _v2, GemmEx, Sgemm/Dgemm/Hgemm batched + strided-batched, gemv, axpy. YES.
- cublasLt: dedicated frontend/backend (CublasLt.cpp, CublasHandler_Lt.cpp). YES.
- CUDA graphs: partial (9 funcs: Create/Instantiate/Launch/Upload/Destroy/GetNodes...).
- nvrtc, cudnn, cufft, curand, cusolver, cusparse, nvml: present.

## Per-app assessment
| App | CUDA needs | OOB? | Risk | Role in suite |
|-----|-----------|------|------|---------------|
| BabelStream | runtime + 5 trivial kernels, no libs | YES | Low | Memory-bandwidth / data-path stress (RDMA+GPUDirect) |
| miniBUDE | runtime + 1 compute kernel, events | YES | Low | Compute-bound, low-transfer (overhead amortization) |
| XSBench CUDA | runtime + kernel + cudaMallocManaged | MAYBE | Med (managed mem + no prefetch) | Irregular/random access, latency-bound |
| CloverLeaf CUDA | runtime + many kernels + MPI + Fortran | MAYBE | Med-High (build/MPI) | Structured-grid, many-kernel, halo exchange |
| Llama (llama.cpp) | runtime + custom kernels + cuBLAS + cublasLt | MAYBE | Med (call-heavy, graphs) | Application LLM inference (RPC-bound finding) |

Notes:
- XSBench: prefer converting managed mem to explicit cudaMalloc/cudaMemcpy, or use "event-based"
  mode; validate with a spike before committing.
- CloverLeaf: build single-rank (disable MPI) first; Fortran+CUDA build is the main hurdle.
- Llama: recommend llama.cpp (self-contained C++, LD_PRELOAD friendly) with a small GGUF model
  (TinyLlama-1.1B or Llama-3.2-1B). Set GGML_CUDA_DISABLE_GRAPHS=1 initially. PyTorch/HF path
  (like existing qwen3-14b example) is heavier and more RPC-bound.

## Why this suite is good
Covers the roofline spread reviewers expect:
- Bandwidth-bound (BabelStream) -> directly exercises the paper's transport/GPUDirect contribution
- Compute-bound (miniBUDE) -> shows remoting overhead amortized
- Latency/irregular (XSBench) -> stresses control path / small random transfers
- Mixed structured-grid (CloverLeaf) -> many kernels + neighbor exchange
- Application (Llama) -> real workload, exposes known synchronous-RPC bottleneck
All are established mini-apps (UK Mini-App / ExCALIBUR / CORAL), stronger credibility than OpenCV.

## Gaps / caveats to state in paper
- Single-node, single-client only (already a listed limitation).
- Synchronous frontend dispatch => call-heavy apps (Llama, CloverLeaf) RPC-bound (known finding).
- Unified memory not fully supported (prefetch missing) => XSBench caveat.

## Test matrix (per app)
1. Bare metal (native CUDA, no GVirtuS) — baseline
2. GVirtuS TCP (etc/properties.json)
3. GVirtuS UCX — TCP transport preset (ucx.env)
4. GVirtuS UCX — RDMA (no GPUDirect)
5. GVirtuS UCX — RDMA + GPUDirect
=> 5 configs x 5 apps. Reuse steady-state internal-loop methodology (warmup + N measured iters).

## Execution phases
- Phase 0: bring-up harness (per-app examples/<app>/{setup.sh,run.sh,backend cfg}) mirroring
  existing examples/ pattern + LD_PRELOAD of frontend stubs.
- Phase 1: BabelStream + miniBUDE (low risk) — validate all 5 transport configs end-to-end.
- Phase 2: XSBench spike (managed-mem decision) -> port if needed.
- Phase 3: CloverLeaf single-rank build.
- Phase 4: Llama (llama.cpp small model).
- Phase 5: collect metrics (per-iter latency, wall time, #remote calls/iter, bandwidth), plot.

## Testbed access (CONFIRMED working, autonomous non-interactive SSH)
- SSH via ~/.ssh/config aliases: `es-dpu-01`, `es-dpu-02` (ProxyJump `aaugw` = sshgw.aau.dk).
  User `kz08ey@student.aau.dk`, key `~/.ssh/id_ed25519` (no passphrase), authorized on gateway + both nodes.
- Both nodes have an NVIDIA L40S (driver 580.95.05). Both have GVirtuS at
  `~/GVirtuS` on branch `gvs/fix-rdma`. Docker 29.1.3.
- ACTUAL RoCE IPs (swapped vs docs!): es-dpu-01 = 24.24.24.2 / 25.25.25.2 ;
  es-dpu-02 = 24.24.24.1 / 25.25.25.1. Pick backend node explicitly; set
  properties server_address to that node's real 25.25.25.x IP.
- Both nodes having a GPU is required for GPUDirect on the frontend side too.

## Later phase: multitenancy testing
Two (or more) frontends -> one backend already proven to work. Add as a dedicated phase:
run 2 frontends (one per node, or 2 containers) against a single backend concurrently;
measure per-client latency/throughput degradation, fairness, and backend scaling.
Report per-connection isolation and contention. Complements the single-client limitation
noted in the report.

## Metrics to record per run
Steady-state mean/median per-iteration latency, total wall time, remote GVirtuS calls/iteration,
achieved H2D/D2H bandwidth (BabelStream), speedup vs TCP, ratio vs bare metal.

---

## PROGRESS LOG

### Phase 0 — DONE (2026-07-17)
- Testbed up. Backend container `gvirtus-kz08ey` detached on es-dpu-01 (25.25.25.2:32223), GPUDirect enabled.
- server_address fixed to 25.25.25.2 on both nodes (properties_ucx.json, properties_plain_rdma.json).
- simple_matrix smoke test PASS (check=pass, max_abs_err=0).
- Docs: docs/benchmarking/README.md, docs/benchmarking/01-testbed-setup.md.

### Phase 1 — IN PROGRESS
- BabelStream cloned to ~/benchmarks/BabelStream on BOTH nodes (OUTSIDE git repo).
- Harness on es-dpu-02: ~/benchmarks/harness/{nvml_shim.cpp, build_run_babelstream.sh}.
  nvml_shim = no-op NVML (GVirtuS nvml plugin lacks GetHandleByPciBusId_v2 / GetClockInfo;
  cosmetic clock printout only, keeps BabelStream source pristine).
- BARE-METAL baseline DONE on L40S (sm_89, 33.5M doubles):
  Copy 743.7 / Mul 659.0 / Add 668.6 / Triad 683.0 / Dot 669.6 GB/s (peak 864).
- BLOCKER: BabelStream OVER GVirtuS HANGS after fatbinary registration
  (backend log: am_recv_handler length=52200 then no progress). Killed after ~8 min.
  Build+link succeeded (GVirtuS frontend stubs + nvml shim). Hang is at runtime, early
  (kernel registration / first calls). NEEDS DIAGNOSIS next.

### File locations (IMPORTANT for future context)
- Repo docs/plan: LOCAL Windows clone only, branch marcel/ucx-comm/testing, UNCOMMITTED.
- Benchmark apps + harness: on NODES under ~/benchmarks/ (NOT in git).
- Backend detached launcher: es-dpu-01:/tmp/gvirtus-backend-run.sh.
- Results tracked in session SQLite `results` table.

### Next steps
1. Diagnose BabelStream/GVirtuS hang (bump loglevel, isolate which routine stalls;
   likely __cudaRegisterFunction count, cudaGetDeviceProperties, or first kernel launch).
2. Once one transport works, sweep TCP / RDMA / RDMA+GPUDirect at real sizes.
3. miniBUDE next.

### UPDATE 2026-07-17 (Phase 1 blocker found)
- BabelStream over GVirtuS BLOCKED: UCX Active-Message framing bug on ~52KB fatbinary
  (__cudaRegisterFatBinary). Transport-independent (fails on RDMA AND TCP). Root: 52KB
  request is just under the 64KB RMA-path threshold in UcxCommunicator::WriteIov, takes
  eager-AM IOV path which corrupts response framing. Never hit before because prior
  benchmarks used library kernels (tiny fatbin). Blocks ALL 4 HPC proxy apps.
- Full diagnosis: docs/benchmarking/02-babelstream.md.
- FIX needs editing src/communicators/ucx/UcxCommunicator.cpp (in scope, not a plugin)
  AND rebuilding the FRONTEND stubs (currently prebuilt in simple_matrix_gvirtus image).
  Plan: build frontend from source (gvirtus-dev image) so fix can be tested end-to-end.
- Decision pending with user: proceed with communicator fix now.

### UPDATE 2026-07-17 #2 (diagnosis CORRECTED — big progress)
- The "invalid response header" framing bug was a STALE-IMAGE artifact: the prebuilt
  simple_matrix_gvirtus image has an OLDER frontend than the backend source. NOT a real bug.
- Fix: run frontend from SOURCE-built GVirtuS. Set up persistent dev container
  gvirtus-fe-kz08ey on es-dpu-02 (gvirtus-dev image, builds mounted ~/GVirtuS, GPU+IB).
- With matched builds, BabelStream registers fatbin OK, queries device, allocates,
  LAUNCHES kernels (return 0), but faults: cudaStreamSynchronize -> CUDA 700 illegal
  memory access. => REAL blocker is kernel-ARG MARSHALING for custom kernels.
- Suspect: plugins/cudart/frontend/CudaRt_execution.cpp cudaLaunchKernel uses NvInfo
  param layout; BabelStream device-lambda kernels (by-value struct capturing 3 ptrs)
  likely mis-sized -> wrong args -> illegal access. This is a cudart PLUGIN limitation.
- Temp instrumentation [FATBINDBG] left in es-dpu-01 src/backend/Process.cpp — REVERT later.
- Backend confirmed parsing all control msgs correctly (want==frame), incl 52KB fatbin.
- NEXT: (pending user OK to touch plugin) instrument cudaLaunchKernel param dump, confirm
  mismarshaled lambda arg, fix NvInfo parse/marshaling.

### UPDATE 2026-07-17 #3 — BabelStream WORKS over GVirtuS (milestone)
- Root cause of CUDA 700 was NOT arg marshaling (that was verified correct). It was
  BabelStream's dot-reduction `sums` buffer allocated via cudaHostAlloc (pinned host) and
  written DIRECTLY by the kernel (zero-copy). GVirtuS cudaHostAlloc = frontend-local malloc
  (CudaRt_memory.cpp:129) -> backend kernel gets an invalid (frontend) host address.
- GVirtuS LIMITATION (for paper): zero-copy pinned-host memory in kernels unsupported.
- FIX: app adaptation — sums -> cudaMalloc device memory + explicit cudaMemcpy D2H before
  host reduction. Functionally identical, negligible cost. Applied on nodes' BabelStream.
- BabelStream now RUNS TO COMPLETION over GVirtuS (UCX/TCP validated, results pass).
- Debug instrumentation (FATBINDBG in Process.cpp, KARGDBG in CudaRt_execution.cpp) REVERTED
  and both sides rebuilt clean.
- KEY LESSON: always run frontend from SOURCE-built GVirtuS (gvirtus-fe-kz08ey), not the
  stale prebuilt simple_matrix_gvirtus image.

### Remaining Phase 1 work
- Re-baseline bare metal with adapted source (identical for fairness).
- Full-size (33.5M) BabelStream measurements across TCP / RDMA / RDMA+GPUDirect.
- Then miniBUDE (may hit similar patterns; likely cleaner — device mem + explicit copies).

### UPDATE 2026-07-17 #4 — BabelStream added as a first-class GVirtuS example
- Created examples/babelstream/ mirroring simple_matrix:
  setup.sh (clone BabelStream + apply GVirtuS sums->device adaptation via python),
  frontend.sh (nvcc build vs frontend stubs + nvml_shim, run), backend.sh,
  nvml_shim.cpp, Dockerfile (self-contained image), README.md.
- Makefile: added docker-build-babelstream, local-docker-build-babelstream,
  run-babelstream-test, stop-babelstream-test (+ .PHONY).
- VALIDATED end-to-end on the testbed: setup.sh clones+adapts; frontend.sh compiles and
  runs over GVirtuS (UCX/TCP), all 5 kernels report bandwidth, run completes/validates.
- Shell scripts normalized to LF (were CRLF from Windows authoring).
- NOTE: local repo files created on Windows; run-babelstream-test needs the
  babelstream_gvirtus image (make local-docker-build-babelstream) OR reuse the source-built
  frontend container. Example content mounts to $GVIRTUS_HOME/examples at run time.

### UPDATE 2026-07-17 #5 — FULL BabelStream transport sweep DONE
- 180 data points: 4 configs (baremetal, gvirtus-tcp, gvirtus-rdma [backend GD=0],
  gvirtus-rdma-gpudirect [backend GD=1]) x 9 sizes (2^18..2^26) x 5 kernels, 100 iters.
- Data saved in repo: docs/benchmarking/data/babelstream/{babelstream_sweep.csv,
  babelstream_summary_gbps.csv, plot_sweep.py, plots/*.png}. Doc: 03-transport-sweep.md.
- Findings: TCP<RDMA~GPUDirect at small sizes; all GVirtuS -> ~93-95% baremetal at 512MiB
  (control-path amortization). RDMA~GPUDirect because BabelStream measures KERNEL bandwidth
  (insensitive to GPUDirect, which helps bulk transfer not per-launch). Clear TCP vs RDMA gap.
- Caveats found: GVIRTUS_RMA_ZEROCOPY=1 crashes backend (OOM exit137) at >=32MiB -> ran
  GPUDirect with zerocopy OFF. Frontend GPUDirect probe fails (dlopens stub cudart) but
  doesn't affect BabelStream (host-resident frontend data); backend GD flag is the differentiator.
- Sweep harness: es-dpu-02 ~/benchmarks/harness/sweep_run.sh (sizes baked in); backend
  launcher parametrized: GVIRTUS_GPUDIRECT={0|1} bash /tmp/gvirtus-backend-run.sh.

### Still uncommitted (local repo, branch marcel/ucx-comm/testing)
- examples/babelstream/, Makefile targets, docs/benchmarking/*, BENCHMARK_PLAN.md, data+plots.

### Next candidates
- Transfer-bandwidth dataset (H2D via [GVS PROFILE]) to show GPUDirect benefit directly.
- miniBUDE (compute-bound proxy). Then XSBench, CloverLeaf, Llama.
- Fix frontend GPUDirect probe (load real cudart) + RMA_ZEROCOPY OOM (GVirtuS improvements).

### UPDATE 2026-07-17 #6 — Transfer-bandwidth benchmark + honest GPUDirect verdict
- Added harness/transfer_bw.cu (H2D/D2H cudaMemcpy sweep 4KiB..256MiB). UCX communicator only.
- Data in repo: docs/benchmarking/data/transfer/{transfer_bw.csv, plot_transfer.py, plots/}.
  Doc: docs/benchmarking/04-transfer-bandwidth.md.
- KEY: RDMA ~2.6x faster than TCP on the data path (1MiB H2D: 6.1 vs 1.9 GB/s) = the real win.
- GPUDirect VERDICT (verified honestly): NO transfer speedup vs plain RDMA in overlapping
  range; CRASHES at >=8MiB (Connection reset on am_send_iov RMA path). Plain RDMA (GD=0) runs
  FULL range to 256MiB clean => instability is GPUDIRECT-SPECIFIC (NIC<->GPU peer-DMA), not RDMA.
- Also: after GPUDirect crash, backend listener can't rebind ("Device is busy") -> needs full
  container restart. Plus earlier RMA_ZEROCOPY=1 OOM at >=32MiB. Two+ real stability bugs.
- Why GPUDirect shows nothing: frontend source buffer is host malloc; frontend GPUDirect probe
  fails (dlopens stub cudart) so client advertises 0 gpu shadows -> no end-to-end NIC<->GPU path.
- Tried fixing frontend probe (load real cudart by abs path) -> real cudaMalloc works standalone
  but fails in-process (stub libcuda binds driver). REVERTED (too risky, and frontend GD isn't
  the meaningful one - GVirtuS clients are GPU-less by design). backend-side GD is what matters.

### Corrected earlier overclaim
- 03-transport-sweep.md updated: bare-metal small-size SPIKE = L2 CACHE artifact (arrays fit in
  ~96MB L2 -> up to 3500 GB/s, 4x DRAM peak); only >=32MiB points are fair. GVirtuS ~93-95% of
  bare metal at 512MiB. GPUDirect NOT validated (documented, not claimed).

### Next candidates
- miniBUDE (compute-bound). Then XSBench, CloverLeaf, Llama. Multitenancy later.
- Optional GVirtuS fixes: GPUDirect >=8MiB crash + listener recovery + RMA_ZEROCOPY OOM.

### UPDATE 2026-07-17 #7 — GPUDirect VALIDATED (corrected earlier wrong claim)
- User challenged "GPUDirect crashes >=8MiB" (report tested N~16000 matrices fine). They were right.
- PROVED simple_matrix over GPUDirect passes at N=2048/4096/8192/16000 = 16/64/256/976 MiB per
  cudaMemcpy, all check=pass. GPUDirect is NOT broken.
- Root cause of earlier crash: my transfer_bw.cu allocated ONE 256MB device buffer and transferred
  sub-ranges reusing it for H2D+D2H -> trips a narrow GVirtuS RMA edge-case bug. NOT GPUDirect.
- Rewrote transfer_bw2.cu (exact-size buffers, separate H2D/D2H, normal pattern). Runs FULL range
  to 256MiB clean, all 4 configs.
- REAL RESULT (transfer_bw2.csv): GPUDirect gives ~2.3x D2H speedup for >=4MiB (8.9 vs 3.9 GB/s
  plain RDMA), sustained to 256MiB. NO H2D benefit (source is host mem - expected). RDMA ~3x TCP.
  Data: docs/benchmarking/data/transfer/{transfer_bw2.csv, plots/transfer_d2h*.png}.
- Deleted buggy transfer_bw.csv. Rewrote doc 04 with positive validated GPUDirect story + the
  narrow buffer-reuse bug documented as future work (+ listener non-recovery, RMA_ZEROCOPY OOM).
- LESSON: benchmark allocation pattern matters; always cross-check a surprising 'framework bug'
  against a known-good app (simple_matrix) before claiming it.

### Corrected files
- docs/benchmarking/04-transfer-bandwidth.md (rewritten), README.md index, plot_transfer.py (D2H focus).

### UPDATE 2026-07-17 #8 — Enriched BabelStream (user: "feels too little")
- ASSESSMENT: babelstream data was VALID + correct setup (rdma=GD0 backend, gpudirect=GD1), but
  we only PLOTTED bandwidth and under-used it. rdma~=gpudirect is EXPECTED (BabelStream transfers
  arrays once -> transport-insensitive kernel bandwidth; GPUDirect story is in doc 04, not here).
- ENRICHED from SAME data (no re-run): extracted per-launch LATENCY (average_runtime).
- HEADLINE latency result: control-path RPC floor TCP ~322us vs RDMA ~238us per kernel launch
  => RDMA cuts RPC round-trip ~26% (~84us/call), consistent across kernels. baremetal ~7us.
  This explains WHY rdma>tcp on bandwidth (small-size bw = payload/RPC_latency).
- plot_sweep.py rewritten: now 13 plots (bandwidth + latency per kernel + RPC-floor bar) + 3
  summary CSVs (gbps, latency_us, rpc_latency_us). doc 03 updated with latency section + a
  "what BabelStream can/cannot show" note.
- Takeaway: BabelStream = control-path RPC latency + kernel-bandwidth amortization story.
  Transfer bench (doc 04) = data-path RDMA/GPUDirect story. Complementary, both now rich.

### UPDATE 2026-07-17 #9 — miniBUDE (compute-bound) DONE
- miniBUDE (UoB-HPC molecular docking) = compute-bound complement to BabelStream. Default
  MEM=DEFAULT (cudaMalloc+cudaMemcpy, no managed). Custom kernel fasten_main. Reports GFLOP/s.
- Build for GVirtuS: nvcc -x cu ... --cudart shared, link frontend stubs. (nvcc default STATIC
  cudart can't be LD redirected; --cudart shared required.) Needs git in build container +
  CMake-generated meta_build.h/meta_vcs.h (built native first). USE_PPWI=1 (comma-list mangles).
- RESULT (deck bm1, 8 iter, all verified): baremetal 216.63 | tcp 216.29 (99.84%) |
  rdma 216.45 (99.92%) | rdma+gpudirect 216.44 (99.91%) GFLOP/s. valid=true all.
  => GVirtuS overhead <0.2% for compute-bound. Transport only shows in one-time setup
  context_ms (TCP 10.3ms vs RDMA 2.1ms) - amortized over 2364ms compute.
- GPUDirect verified enabled on backend for GD run (per doc-00 rule).
- Data: docs/benchmarking/data/minibude/{minibude.csv, plot_minibude.py, plots/}. Doc: 06-minibude.md.
- bm2 deck too heavy (65536 poses) - 8-iter exceeded 300s timeout (pure compute, not a bug). bm1 sufficient.
- Binaries on node: ~/benchmarks/miniBUDE/build/cuda-bude (native), build/cuda-bude-gvirtus (stubs).

### Suite progress: BabelStream (bandwidth/RPC) + transfer (data-path RDMA/GPUDirect 2.3x D2H) +
### miniBUDE (compute-bound, overhead-free) DONE. Roofline spread taking shape.
### Next: XSBench (irregular/latency-bound), CloverLeaf (structured grid), Llama (LLM inference).

### UPDATE 2026-07-17 #10 — STRATEGIC INSIGHT (user: "no difference tcp vs rdma, why?")
- CORRECT observation. miniBUDE compute-bound (2364ms kernel vs 2-10ms transfer) => transport
  invisible <0.5% runtime. RPC saving 8*84us=0.67ms vs 2364ms = nothing. EXPECTED, not a failure.
- KEY STRATEGY: transport (RDMA/GPUDirect) only shows in NETWORK-STRESSING workloads:
  * REVEALS transport: transfer microbench (RDMA 3x TCP, GPUDirect 2.3x D2H), BabelStream RPC
    latency (RDMA -26%), and CALL-HEAVY/STREAMING apps (LLM inference, OpenCV report 5x).
  * HIDES transport (on-device compute-bound): miniBUDE confirmed; XSBench + CloverLeaf LIKELY
    same (Monte Carlo / stencil keep data on device).
- RECOMMENDATION: deprioritize XSBench/CloverLeaf (likely reproduce "compute-bound=no diff");
  PRIORITIZE Llama LLM inference (token-by-token = many small H2D/D2H + thousands of launches =
  most transport-sensitive real app, matches report's OpenCV finding). Decision pending w/ user.
- miniBUDE doc 06 reframed honestly: it's the "overhead-free/negative" result, NOT a transport result.

### UPDATE 2026-07-17 #11 — Llama LLM inference DONE (transport-revealing extreme)
- llama.cpp CUDA (TinyLlama-1.1B Q4) built with --cudart shared + BUILD_SHARED_LIBS -> links
  GVirtuS frontend stubs. GGML_CUDA_DISABLE_GRAPHS=1.
- FIXED (KEPT): cudaDeviceGetPCIBusId missing backend handler. Frontend stub existed but backend
  rejected routine -> llama crashed at ggml_backend_cuda_reg. Added handler in
  plugins/cudart/backend/{CudaRtHandler.h,.cpp,CudaRtHandler_device.cpp}. Verified: generates
  coherent text over GVirtuS ("Paris"). Documented in doc 05.
- DOCUMENTED bug: intermittent GGML_ASSERT(ptr%256==0 aligned) in llama-bench graph_reserve
  (repeated context create/destroy). llama-cli reliable. GVirtuS cudaMalloc passes real aligned
  ptr, so it's a reserve-buffer path issue. Workaround: llama-cli / -r1 fresh backend.
- RESULT: pp8 (prompt eval) 49.0 t/s vs baremetal 1518.6 = 31x slower; tg8 (token gen) 7.9 vs
  569 = 72x SLOWER. 40-token gen took ~8 min. => LLM inference CATASTROPHICALLY RPC-bound.
- FINDING: this is THE confirmation of the report's synchronous-dispatch bottleneck. Token gen =
  thousands of tiny SEQUENTIAL kernel-launch RPCs; transport (TCP/RDMA/GD) barely matters
  (dispatch-latency-bound, not bandwidth). Only async dispatch/batching/graph capture would help.
- Full TCP-vs-RDMA llama sweep blocked by the flaky alignment crash; single clean GPUDirect
  column + mechanism suffices. Data: docs/benchmarking/data/llama/. Doc: 07-llama.md.
- ROOFLINE COMPLETE: miniBUDE (compute, ~0% overhead) <-> Llama (call-bound, 72x) bracket the extremes.

### Suite status: BabelStream, transfer, miniBUDE, Llama DONE. XSBench/CloverLeaf deprioritized
### (likely compute-bound = no transport diff, like miniBUDE). Multitenancy still pending.
### 2 GVirtuS fixes KEPT: (none reverted). 1 fix = cudaDeviceGetPCIBusId. Bugs doc = 05.

### UPDATE 2026-07-17 #12 — WHICH CALL IS THE KILLER (empirical backend profile)
- Captured backend routine histogram during llama runs. TWO killers:
- STARTUP: cudaRegisterFunction 6665 (76%) + cudaRegisterVar 1789 (20%) + fatbin 282 = 8736 =
  93% of ALL calls, ONE-TIME. Cause: ggml-cuda ~141 .cu fatbinaries x dozens of funcs each,
  every one a SEPARATE synchronous RPC. ~8700 blocking RPCs x 238us ~= 2s just for CUDA init.
- PER-TOKEN: async ops executed SYNCHRONOUSLY. Non-reg calls dominated by cudaMemcpyAsync(197)
  + cudaStreamSynchronize(198) pairing 1:1. cudaMemcpyAsync is SUPPOSED to be non-blocking but
  GVirtuS blocks on it (full RPC). cudaLaunchKernel also blocks. Nothing overlaps.
- QUANTIFIED: 7.88 tok/s = 127ms/token; /238us per RPC = ~530 sequential RPCs/token (22 layers x
  ~24 ops). Native pipelines them (~1.7ms/token); GVirtuS serializes (~127ms) = 72x gap.
- KILLER = the ordinary hot-loop trio cudaLaunchKernel/cudaMemcpyAsync/cudaStreamSynchronize
  forced through blocking req/response ~530x/token, zero overlap. Fix = ASYNC DISPATCH (+ launch
  batching / CUDA-graph capture). Transport can't help (round-trip COUNT x latency, not bandwidth).
- NOTE: llama decode-graph reserve crashes intermittently (alignment assert, doc 07) so full
  steady-gen profile partial; startup profile + per-token derivation are solid. Doc 07 updated.

### UPDATE 2026-07-17 #13 — EVIDENCE-BASED llama profile (CORRECTS #12)
- Captured backend TIMESTAMPED dispatch log (77,345 dispatches), analyzed inter-dispatch timing
  offline. Artifacts: docs/benchmarking/data/llama/{profile_backend_log.py, call_profile_summary.txt}.
- CORRECTED #12 errors:
  * INIT/REGISTRATION is NOT the killer: 8,736 register calls in 0.96s (was 93% of COUNT but <1s).
  * The ~8-min walltime was MOSTLY A HANG: ONE 491.7s stall before the final dispatch -> 500s
    timeout. Actual work finished in ~6.5s. The hang = intermittent decode-graph alignment assert.
  * Per-RPC latency is FAST ~9us median (small AM on RDMA), NOT 238us. Mean was outlier-skewed.
- REAL killer = RPC COUNT per token: ~500 kernel launches/token x 6.2 RPCs/launch = ~3,100
  synchronous serial RPCs/token -> ~250ms/token ~4tok/s vs ~570 native = ~140x.
- 6.2 RPCs/launch: cudaGetDevice(1.8) + cudaGetLastError(1.3) + Push+Pop+LaunchKernel(1 each).
  ~4 of 6.2 are NEEDLESS: Push/PopCallConfiguration are THREAD-LOCAL (should never be RPCs);
  GetDevice/GetLastError are cacheable. Fixing those = ~5x fewer RPCs WITHOUT async.
- Two fixes: (1) cheap/no-async: local Push/Pop + cache GetDevice/GetLastError (~5x). (2) async
  dispatch for LaunchKernel/MemcpyAsync + graph capture. Transport can't help (count x latency).
- doc 07 "which call is the killer" section rewritten evidence-based. results table corrected.
- POTENTIAL HIGH-VALUE FIX for next: make cudaPushCallConfiguration/cudaPopCallConfiguration
  local no-ops in the frontend (store config thread-local, consume in cudaLaunchKernel). Would
  cut 2 RPCs/launch immediately. Worth attempting + measuring.

### UPDATE 2026-07-18 #14 — HANG FIXED + full clean llama transport sweep (KEY milestone)
- FIXED the intermittent EXIT=134 alignment crash that invalidated prior llama end-to-end numbers.
  Root cause: ggml TENSOR_ALIGNMENT=32 asserts on a HOST ptr from cudaHostAlloc/cudaMallocHost,
  which GVirtuS implements as plain malloc() (glibc 16-byte aligned) -> 16%32!=0 ~half the time.
  Fix (KEPT): plugins/cudart/frontend/CudaRt_memory.cpp cudaHostAlloc + cudaMallocHost now use
  posix_memalign(ptr,256,size) (matches real CUDA pinned alignment). cudaFreeHost=free() (compat).
  Deployed to es-dpu-02, frontend cudart rebuilt. VALIDATED: llama-bench 5/5 no crash, EXIT=0.
- Also confirmed the separate "8-min hang" = llama-cli interactive stdin wait (harness, NOT
  GVirtuS); use < /dev/null, or prefer llama-bench (self-exits). Both hang causes resolved.
- FULL CLEAN llama-bench sweep (TinyLlama-1.1B Q4, -p8 -n16 -r3), GPUDirect verified on backend log:
  * bare metal        pp8 2773   | tg16 634.9 t/s   (native /usr/local/cuda libcudart, local L40S)
  * GVirtuS TCP       pp8 25.4   | tg16 3.58 t/s    (177x slower tg)
  * GVirtuS RDMA      pp8 50.6   | tg16 8.04 t/s    (79x;  backend log GPUDirect=disabled)
  * GVirtuS RDMA+GD   pp8 49.1   | tg16 7.96 t/s    (80x;  backend log GPUDirect=enabled)
- KEY FINDINGS (now evidence-based, valid): (1) llama ~79x slower even on best transport = RPC-count
  bound (confirms synchronous-dispatch bottleneck). (2) TRANSPORT MATTERS here (opposite of miniBUDE):
  RDMA 2.25x faster than TCP for token gen (latency-bound, many tiny serial round-trips/token).
  (3) GPUDirect == plain RDMA (no benefit): per-token transfers tiny + host-sourced, nothing for
  GPU-NIC peer-DMA to accelerate. GPUDirect win is bulk device transfers (doc 04), not inference.
- Data: docs/benchmarking/data/llama/llama_bench_transports.csv + plot_llama_transports.py +
  plots/{llama_tg16_transports,llama_pp8_transports,llama_tg16_gvirtus_only}.png.
- Docs updated: 07-llama.md (results table, findings 1-2 rewritten, hang section, reproduce),
  05-gvirtus-bugs-and-knobs.md (added cudaHostAlloc/cudaMallocHost alignment FIX under FIXED BUGS).
- 2 GVirtuS fixes now KEPT total: cudaDeviceGetPCIBusId + host-alloc alignment.
- Backend currently GD=0 (plain RDMA) on es-dpu-01. Nothing committed yet (branch marcel/ucx-comm/testing).
- NEXT: multitenancy (2 frontends -> 1 backend). Optional: local Push/PopCallConfiguration no-op.

### UPDATE 2026-07-18 #15 — PROTOTYPE cheap fix (local Push/Pop) implemented, KEPT, validated across suite
- Root-caused RPC amplification: each kernel launch = ~6.2 blocking RPCs. __cudaPushCallConfiguration
  + __cudaPopCallConfiguration were 2 needless RPCs (they only carry launch config from <<<>>> to
  cudaLaunchKernel via a thread-local stack in native CUDA; GVirtuS frontend cudaLaunchKernel
  already re-serializes grid/block/shmem/stream to backend, so backend never needs them).
- FIX (KEPT): plugins/cudart/frontend/CudaRt_internal.cpp — Push/Pop now use a
  thread_local std::stack<GvirtusCallConfig>, no RPC. 100% behaviour-preserving. Deployed es-dpu-02,
  frontend cudart rebuilt.
- VALIDATED across the suite (correctness preserved: llama "...is Paris", miniBUDE valid:true):
  * llama tg16 RDMA      8.04 -> 11.64 t/s  (1.45x)
  * llama tg16 GPUDirect 7.96 -> 11.53 t/s  (1.45x)  [backend GPUDirect=enabled verified]
  * llama tg16 TCP       3.58 -> 5.35  t/s  (1.49x, noisy)
  * llama pp8 RDMA       50.62 -> 69.48     (1.37x)
  * BabelStream 262K Triad (small)  30217 -> 49404 MB/s (1.64x)  [3-run stable]
  * BabelStream 33M Triad (large)   614358 -> 648637     (1.06x)
  * miniBUDE bm1                    216.0 -> 216.5 GFLOP/s (1.00x, no change/no regression)
- Speedup tracks launch-boundedness perfectly (compute 1.00x < bandwidth 1.06x < LLM 1.45x <
  launch-overhead 1.64x). Matches 6.2/4.2 = 1.48x prediction. Strong paper result.
- 3 GVirtuS fixes now KEPT total: cudaDeviceGetPCIBusId, host-alloc alignment, local Push/Pop.
- DOCS: NEW docs/benchmarking/08-recommended-improvements.md (prototype #1 validated + prioritized
  recs #2 cache GetDevice/GetLastError, #3 async dispatch, #4 launch batching/graph, #5 RMA rcache).
  Updated 07-llama.md, README index. Data: data/llama/llama_bench_proto.csv,
  data/prototype_pushpop_summary.csv, data/plots/prototype_pushpop_speedup.png.
- OPERATIONAL: hit backend listener non-recovery (stray gvirtus-backend PIDs holding RoCE port
  after restart -> Address already in use / Device is busy); needed docker rm -f + retry. Also
  GPUDirect first-connection cold-start stall (>4min) on fresh backend -> retry cleared it. Both
  logged in doc 08 "Also worth hardening".
- Backend currently GD=0 (RDMA) on es-dpu-01. Nothing committed (branch marcel/ucx-comm/testing).
- NEXT: multitenancy (2 frontends -> 1 backend). Optional: implement rec #2 (cache GetDevice) for
  compounding speedup.

### UPDATE 2026-07-18 #16 — INFOCOM measurement roadmap (doc 09) — what to measure next
- Investigated the gap for an INFOCOM-grade (networking-venue) paper. Key finding: we have MEANS
  only (throughput + avg latency); missing the networked-systems evaluation vocabulary.
- Confirmed per-RPC timing hooks already exist in src/frontend/Frontend.cpp::Execute()
  (marshal/write/sync/read_hdr/read_payload = the [GVS PROFILE] path) but samples are discarded.
  => latency distributions are BUILDABLE now via an env-gated trace, no redesign.
- Wrote docs/benchmarking/09-measurement-roadmap.md (added to README index). Priority order:
  1. Per-RPC latency DISTRIBUTIONS p50/p90/p99/p99.9 + CDFs + tail-ratio (p99/p50), per transport
     x call-class (control AM vs data-path). THE headline for INFOCOM (RDMA tail << TCP tail).
  2. Latency decomposition (reuse existing timers): marshal->wire->backend dispatch->CUDA->return.
  3. Throughput-latency saturation curve (knee) + small-message RPC/s control-plane ceiling.
  4. Multi-tenancy: N frontends->1 backend scaling, Jain fairness, noisy-neighbor ISOLATION (p99).
  5. LLM serving SLOs: TTFT + inter-token latency p50/p99.
  6. Statistical rigor retrofit (>=5 runs, 95% CIs/error bars) on all plots.
  7. External baseline (rCUDA/other) + design ablations (AM vs TCP control, RDMA vs staged,
     GPUDirect, Push/Pop on/off).
- CONCRETE FIRST STEP: add GVIRTUS_LATENCY_TRACE=<file> to Frontend::Execute() — record per-RPC
  round-trip us + routine + payload size to thread-local buffer, flush at exit. One llama + one
  BabelStream run per transport yields tiers 1.1/1.2/2.2.
- NEXT (choose): implement the latency-trace instrumentation and produce p50/p99/CDFs; and/or
  multitenancy scaling+fairness; and/or rec #2 (cache cudaGetDevice) for compounding speedup.

### UPDATE 2026-07-18 #17 — Latency-trace instrumentation (KEPT) + first p50/p99/tail results
- Implemented GVIRTUS_LATENCY_TRACE=<file> in src/frontend/Frontend.cpp::Execute() (UCX AM path):
  per-thread buffer records {routine,payload_bytes,rt_us,server_us} per RPC, flushed CSV at exit.
  Env-gated, zero overhead when off (tg16 11.62 traced == 11.64 untraced). KEPT in tree.
  namespace gvirtus_lattrace{Tracer singleton + thread_local buffer}. Deployed es-dpu-02, rebuilt.
- Captured 44,242 RPCs/transport from identical llama-bench -p8 -n16 -r1 (TCP, RDMA, GPUDirect;
  backend GD verified). Data in docs/benchmarking/data/latency/lat_{tcp,rdma,gpudirect}.csv.
- HEADLINE RESULT (control-plane RPCs, payload<4KiB):
  * p50:  TCP 58us  RDMA 42us  GPUDirect 43us   (median ~1.4x, modest)
  * p99:  TCP 1328us RDMA 56us GPUDirect 55us   (24x!)
  * p99.9:TCP 15550us RDMA 353us               (44x!)
  * TAIL RATIO p99/p50: TCP 22.9x vs RDMA 1.33x -> RDMA nearly flat/deterministic, TCP catastrophic tail.
  RDMA's value = near-elimination of tail-latency VARIANCE, not lower median. Means lie (TCP mean
  155us = 2.7x its median). This mechanistically explains llama TCP 2.2x slowdown (3100 serial
  RPCs/token, TCP tail events compound). GPUDirect==RDMA on control plane (expected).
- DOCS: NEW 10-latency-distributions.md + README index. data/latency/{lat_*.csv,
  latency_percentiles.csv, analyze_latency.py, plots/latency_cdf_control.png,
  plots/latency_percentiles_control.png}.
- 4 GVirtuS changes now KEPT: cudaDeviceGetPCIBusId, host-alloc alignment, local Push/Pop,
  latency-trace instrumentation.
- Backend currently GD=1 on es-dpu-01. Nothing committed (branch marcel/ucx-comm/testing).
- NEXT (doc 09 order): latency decomposition (reuse Execute timers) -> throughput-latency
  saturation + RPC/s ceiling -> multitenancy scaling + fairness(Jain) + noisy-neighbor isolation
  (tail now directly measurable). Optional: rec #2 cache cudaGetDevice.

### UPDATE 2026-07-18 #18 — Warmup audit (all experiments verified warmed)
- User asked to confirm warmups. Audited every experiment:
  * llama-bench: built-in warmup pass before timed runs. OK
  * BabelStream: peak of 100 iters (cold first iter excluded by construction). OK
  * miniBUDE: one-time context_ms excluded from timed compute. OK
  * transfer_bw2: explicit warm=5 before each H2D/D2H timed loop (verified in source). OK
  * latency trace: records ALL RPCs incl one-time registration -> ran warmup-sensitivity
    (data/latency/warmup_sensitivity.py): recomputed control-plane percentiles all vs
    steady(no-setup) vs last-50%.
- RESULT: median warmup-INVARIANT (TCP 56-58us, RDMA 42-43us in every subset). Discarding warmup
  makes the tail finding STRONGER: TCP tail p99/p50 22.9x -> 37.5x steady (intrinsic TCP property,
  not cold-start); RDMA steady p99.9 = 67us vs TCP 15747us = 235x. All-samples table is conservative.
- Added "Warmup robustness" section to 10-latency-distributions.md + note that throughput expts warmed.
- Nothing committed (branch marcel/ucx-comm/testing). Backend GD=1 on es-dpu-01.

### UPDATE 2026-07-18 #19 — Control-path (AM) vs data-path split during llama (answers "is AM the problem?")
- Analyzed lat_rdma.csv payload sizes (data/latency/control_vs_data_split.py). Answer: AM is NOT slow.
- SPLIT IN PRACTICE (44,241 RPCs): control path (AM, <4KiB) = 97.3% of RPCs, 81.1% of RPC time;
  data path (bulk >=4KiB) = 2.7% RPCs, 18.9% time — and 16.1% of that is the ONE-TIME 636MB weight
  load (249 transfers up to 54MB). During token gen the DATA PATH IS IDLE. -> explains GPUDirect==RDMA.
- Per-AM latency is FAST (~41-45us; p99 56us RDMA). Bottleneck = count x synchronous: 44k serial
  control RPCs x ~43us, no overlap = 2.32s critical path.
- Top offenders (Push/Pop already gone/local): cudaGetDevice 33.1%cnt/28.1%time, cudaGetLastError
  25.2%/19.7%, cudaLaunchKernel 18.7%/15.6%. GetDevice+GetLastError = 58% of RPCs, 48% of time,
  BOTH cacheable -> rec #2 would ~halve remaining RPCs. Trace validates #2 as next big win.
- Takeaway: LLM gap is synchronous control-path chatter, NOT AM speed nor a data-path deficiency;
  split makes bottleneck legible (81% control). Added section to 10-latency-distributions.md.
- NEXT candidate: implement rec #2 (cache cudaGetDevice + safe cudaGetLastError) and re-measure.

### UPDATE 2026-07-18 #20 — Recommended improvement #2 IMPLEMENTED + KEPT: ~5x faster LLM inference
- What cudaGetDevice/cudaGetLastError DO (why they should be local):
  * cudaGetDevice(int*): returns calling thread's current device ordinal (thread-local state, only
    changes via cudaSetDevice; does NOT touch GPU). Frontend already knows it.
  * cudaGetLastError(): returns sticky last error AND resets to cudaSuccess. cudaPeekAtLastError:
    returns without clearing. "Last error" = most recent non-success exit code = frontend already
    sees it on every remoted call.
  * Old GVirtuS remoted BOTH as blocking RPCs -> per doc 10 they were 58% of control RPCs, 48% of time.
- FIX (KEPT, behaviour-preserving): added thread_local cudart_state (current_device + sticky
  last_error) in CudaRt_error.cpp; CudaRtFrontend::Execute() now records every exit code via
  note_exit_code (faithful sticky semantics); cudaGetDevice/cudaGetLastError/cudaPeekAtLastError
  answer LOCALLY (no RPC); cudaSetDevice updates cache on success. Files: CudaRt_error.cpp,
  CudaRt_device.cpp, CudaRtFrontend.h, cuda_internals/CudaRt_internal.h. Deployed es-dpu-02, rebuilt.
- VALIDATED (correctness OK "...is Paris"; llama-bench -p8 -n16 -r3):
  * RDMA tg16:  8.04 (base) -> 11.64 (rec#1) -> 40.42 (rec#1+2)  = 5.03x TOTAL
  * TCP  tg16:  3.58 -> 5.35 -> 10.63 = 2.97x
  * RDMA pp8:  50.6 -> 69.5 -> 249.95
  * GPUDirect+rec2 NOT measured this session (persistent GPUDirect cold-start stall, 3 attempts
    >280s); expected ~=RDMA (control-plane workload, GPUDirect==RDMA established).
- MECHANISM confirmed via trace: cudaGetDevice 14624->0, cudaGetLastError 11163->0. Per-launch RPCs
  6.2 -> 4.2 (rec#1) -> ~1.1 (rec#2, just cudaLaunchKernel). RDMA-vs-TCP gap WIDENED 2.2x->3.8x
  (fewer RPCs => per-RPC tail matters more; TCP tail from doc 10 hurts more).
- 5 GVirtuS changes now KEPT: PCIBusId, host-alloc align, local Push/Pop, latency-trace, GetDevice/
  GetLastError cache. Total LLM speedup from cheap frontend changes = ~5x RDMA, no async yet.
- DOCS updated: 08-recommended-improvements.md (#2 now IMPLEMENTED+VALIDATED), data/llama/
  llama_optimizations_progression.csv, SQL llama_bench_opt2. TODO: refresh 07-llama results table.
- Backend restored GD=0 (RDMA) on es-dpu-01. Nothing committed (branch marcel/ucx-comm/testing).
- NEXT: refresh 07-llama table; then structural rec #3 (async dispatch) or multitenancy; synthetic
  RPC ping-pong microbenchmark for clean per-size latency.

### UPDATE 2026-07-18 #21 — GPUDirect+rec#2 RESOLVED (42.77 t/s) + found frontend-GVIRTUS_GPUDIRECT bug
- Chasing the GPUDirect number, hit persistent GPUDirect crashes (exit 134 core dump) in
  ggml_cuda_kernel_launch: "CUDA error: initialization error". RDMA/TCP+rec#2 worked fine.
- Diagnosis journey: (1) orphaned backend container gvirtus-ll33pq (up 10h, 12.7GB GPU leak) from
  killed backends -> user approved removal, freed GPU. (2) Still crashed with free GPU + fresh backend.
  (3) Added cudaErrorNotReady exclusion to note_exit_code (correct CUDA semantics, KEPT) - not the cause.
  (4) ROOT CAUSE: frontend GVIRTUS_GPUDIRECT=1 triggers a real-libcuda cudaMalloc(4K) probe on the
  GPU-less-ish frontend; when it fails it POISONS the process CUDA context -> ggml init error crash.
  Frontend never needs the flag (GPUDirect negotiated from BACKEND). Removing it from frontend env
  -> GPUDirect works.
- RESULT (frontend GVIRTUS_GPUDIRECT UNSET, backend GD=1, verified enabled; correctness "...is Paris"):
  * GPUDirect rec#2: pp8 300.98, tg16 42.77 t/s  (5.37x vs baseline 7.96; ~= RDMA 40.42 as expected)
- rec#2 now VALIDATED on ALL THREE transports: RDMA 40.42, GPUDirect 42.77, TCP 10.63.
- NEW BUG documented (doc 05): frontend GVIRTUS_GPUDIRECT=1 failing probe poisons CUDA context ->
  crash; workaround set it backend-only; harden by skipping frontend probe / not poisoning on fail.
- KEPT changes now 6: PCIBusId, host-align, Push/Pop-local, latency-trace, GetDevice/GetLastError
  cache, cudaErrorNotReady exclusion.
- DOCS updated: 05 (new bug), 07 (table+GPUDirect+op note), 08 (rec#2 table 3 transports +
  NotReady refinement note), data/llama/llama_optimizations_progression.csv (+GPUDirect),
  plot_optimization_progression.py (+GPUDirect line) + regenerated PNG. SQL llama_bench_opt2 +GPUDirect.
- Backend GD=1 on es-dpu-01 (working). Frontend runs must NOT set GVIRTUS_GPUDIRECT. Nothing committed.
- NEXT: structural rec #3 (async dispatch) / multitenancy / synthetic RPC ping-pong microbench.

### UPDATE 2026-07-18 #22 — INTEGRITY CORRECTION: GPUDirect was ENABLED but NOT proven DMAing into GPU for llama
- User challenged the GPUDirect claim (standing rule: never assume). Checked backend log evidence:
  * GPUDirect=enabled + gpu_shadows=2 allocated (backend CAPABLE) -- YES.
  * BUT rma_setup: "received 2 remote slots (0 with gpu shadow)" -> FRONTEND advertised 0 GPU
    shadows (because I unset GVIRTUS_GPUDIRECT on frontend to avoid the crash) -> client host-staged.
  * ZERO GPU-routing events (gpu_split_offset / route-to-gpu-shadow) logged during llama run (DEBUG on).
- HONEST CONCLUSION: for llama, GPUDirect was enabled/capable on the backend but the DATA PATH was
  NOT meaningfully exercised (frontend host-staged + 0 routing events + llama is control-plane-bound
  with tiny per-token transfers below the GPU-path threshold). That's WHY tg16 GPUDirect 42.77 ==
  RDMA 40.42. The GPUDirect column = "backend-GD-enabled config", NOT proof of NIC->GPU DMA.
- Actual NIC->GPU DMA proof remains doc 04 (bulk transfer benchmark, ~2.3x D2H) -- NOT llama.
- CORRECTED docs to not overclaim: 07-llama.md (GPUDirect caveat), 08 (honesty note),
  data/llama/llama_optimizations_progression.csv (GPUDirect row note). rec#2 speedup (5x) unaffected
  and still valid (it's a control-plane optimization, transport-independent).
- Reinforces prior finding (doc 10 control/data split): llama exercises AM control path ~exclusively.
- Nothing committed. Backend GD=1 on es-dpu-01. Frontend must NOT set GVIRTUS_GPUDIRECT.

### UPDATE 2026-07-18 #23 — CRITICAL LESSON (no regression) + SimpleMatrix vs paper + STANDING CLEANUP RULE
- User: "should take seconds, takes minutes -- did you regress?" Investigation: NO code regression.
  My changes are cudart-frontend + Frontend.cpp latency hook only (verified via git diff --stat);
  branch UCX communicator is AHEAD of feat/ucx-gpudirect. The slowness was OPERATIONAL:
  * Accumulated ZOMBIE frontend processes from my timeout-killed runs (4+ stuck llama/simple_matrix
    holding backend connections + GPU mem).
  * An ORPHANED backend container (gvirtus-ll33pq, 10h, 12.7GB GPU leak) starving GPUDirect.
  After killing zombies + fresh backend: identical N=16000 run went from stall -> ~6-8s.
- ADDED STANDING CROSS-SESSION RULE "ALWAYS CLEAN UP STALLED PROCS/ZOMBIES/CONTEXTS/CONTAINERS" to
  CLAUDE.md AND .github/copilot-instructions.md (both auto-load every session). Also reflected in doc 11.
- KEY REALIZATION: es-dpu-02 (frontend) HAS its own L40S; it's only LOGICALLY GPU-less. The frontend
  GPUDirect probe failing earlier ("cudaMalloc(4K) failed") was the orphaned container's GPU
  EXHAUSTION, not a design flaw. GPUDirect needs GVIRTUS_RMA_ZEROCOPY=1 (user-confirmed). Made
  backend launcher honour GVIRTUS_RMA_ZEROCOPY via env (was hardcoded =0).
- SIMPLEMATRIX N=16000 (avg_host_ms = H2D+GEMM+D2H; check=pass) vs paper N=16384 (Tables 7.5/7.6):
  * UCX RDMA+GPUDirect (zerocopy): 358.4 ms  vs paper 389.5  (ours ~8% lower; smaller N) MATCH
  * UCX RDMA (zerocopy):            496.9 ms  vs paper 548.4  MATCH
  * UCX RDMA (no zerocopy/staged):  698.9 ms  (not a paper config; shows staged D2H penalty)
  * GEMM ~157 ms vs paper ~168 ms MATCH (compute not the bottleneck)
  * GPUDirect ~1.39x over UCX RDMA (paper 1.41x) -- GPUDirect PROVABLY engages at ~1GB/transfer
    (D2H is the win). This IS the workload that demonstrates NIC->GPU DMA (unlike llama).
- DOCS: NEW 11-matrix-vs-paper.md + README index. data/matrix/matrix_vs_paper.csv. SQL matrix_bench.
- Backend currently GD=0 zerocopy=1 on es-dpu-01 (listener up). Frontend GPU-less; do NOT set
  GVIRTUS_GPUDIRECT on frontend for llama, but for matrix it falls back gracefully. Nothing committed.
- NEXT: optionally baremetal simple_matrix; structural async (rec #3); multitenancy; latency ping-pong.

### UPDATE 2026-07-18 #24 — Created benchmark-dev SKILL (docs/benchmark-dev/SKILL.md)
- Comprehensive operating manual with YAML frontmatter (name: benchmark-dev). Sections:
  0. Five standing rules: clean up; back up w/ evidence; don't overclaim; challenge assumptions;
     always verify GPUDirect.
  1. Testbed: es-dpu-01 backend (L40S, RoCE 25.25.25.2, mlx5_1/ens1f1np1, GID3), es-dpu-02 frontend
     (L40S, 25.25.25.1, LOGICALLY GPU-less but has a physical GPU). SSH config (ProxyJump aaugw).
  2. Backend container gvirtus-kz08ey + /tmp/gvirtus-backend-run.sh (GVIRTUS_GPUDIRECT +
     GVIRTUS_RMA_ZEROCOPY via env; rebuilds ~90s; wait for "listener created"). Frontend container
     gvirtus-fe-kz08ey + run env template + LD_LIBRARY_PATH stubs-first.
  3. Test suite: simple_matrix, BabelStream, miniBUDE, llama.cpp, transfer_bw2 (+ XSBench/CloverLeaf
     deprioritized; ping-pong microbench TODO).
  4. Where docs/results live (docs/benchmarking/ + data/, _pdftxt report, plan mirror, SQL tables).
  5. Status + 6 KEPT fixes + headline findings.
  6. Known problems (zombie/orphan contamination, GPUDirect cold-start, listener non-recovery,
     frontend probe, RMA_ZEROCOPY OOM, stale images, backend rebuild).
  7. Standard workflow. 8. INFOCOM metrics.
- Linked from docs/benchmarking/README.md (START HERE banner).
- Nothing committed (branch marcel/ucx-comm/testing).
