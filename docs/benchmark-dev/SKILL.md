---
name: benchmark-dev
description: >
  Operating manual for developing, running, and documenting GVirtuS (project GUSTO)
  performance benchmarks on the two-node AAU testbed (es-dpu-01 / es-dpu-02) over the
  UCX communicator (TCP / RDMA / RDMA+GPUDirect). Use this whenever the task involves
  benchmarking GVirtuS, running an HPC proxy app or LLM over the remote GPU, measuring
  transport latency/throughput, reproducing the report's numbers, or touching the
  frontend/backend on the SSH nodes. Encodes the testbed layout, how to run the existing
  test suite, known problems, where docs/results live, and the mandatory testing rules
  (clean up, back up with evidence, don't overclaim, challenge assumptions, verify GPUDirect).
---

# GVirtuS (GUSTO) Benchmark Development Skill

This skill is the single source of truth for running GVirtuS benchmarks on the AAU testbed.
Read it fully before running anything. It is written to survive across sessions.

---

## 0. THE FIVE STANDING RULES (non-negotiable)

1. **ALWAYS CLEAN UP after testing.** Kill stalled/zombie frontend processes, restart the backend
   fresh between phases, verify GPU memory is freed on BOTH nodes. Leftover state silently corrupts
   results and fakes "slowness"/"hangs". A multi-minute "hang" has repeatedly been *just* zombies +
   a poisoned backend context; cleanup returned the same run to seconds. **Clean first, measure second.**
2. **BACK UP EVERYTHING WITH EVIDENCE.** Every number must be reproducible from a captured artifact
   (CSV, log, backend `docker logs`, profile line). Save raw data under `benchmarks/`,
   record the exact command + env, and log the run. No number without a source.
3. **DON'T OVERCLAIM.** State only what the evidence supports. If GPUDirect was *enabled* but the
   data path wasn't exercised, say exactly that — do not claim "NIC→GPU DMA happened" without a
   backend-log/profile proof. Distinguish "capable/configured" from "demonstrated".
4. **CHALLENGE YOUR ASSUMPTIONS.** When a result is surprising, suspicious, or convenient, re-derive
   it. Cross-check against the report, against a second config, against warmup-excluded subsets.
   Prefer to disprove your own hypothesis before reporting it.
5. **ALWAYS VERIFY GPUDIRECT — NEVER ASSUME.** Before reporting any GPUDirect/transport claim, confirm
   the backend log says `GPUDirect=enabled`, the backend was launched with `GVIRTUS_GPUDIRECT=1`
   (+ `GVIRTUS_RMA_ZEROCOPY=1`), an RDMA TLS was negotiated (not tcp), and the data path actually
   ran (GPU-shadow slots advertised / bulk transfer profiled). Frontend probe is NOT proof.

**Bonus rule — KEEP LOGS LEAN.** Do not benchmark at verbose log levels. TRACE/DEBUG logging (and
`UCX_LOG_LEVEL=debug`, `[GVS PROFILE]`/`[UCX DEBUG]` spew) emits a line per byte/RPC and produces
**multi-hundred-MB → GB log files** that also *slow the run* (I/O + serialization on the hot path)
and skew timings. For measurement runs use `GVIRTUS_LOGLEVEL=40000` (ERROR) or `30000` (WARN) and
`UCX_LOG_LEVEL=error`; only drop to DEBUG/TRACE for a short, targeted diagnostic and then turn it
back up. Never leave a detached backend running at DEBUG/TRACE. If you must capture a big trace,
bound it (`| head`, small `-n`/iters, `timeout`) and delete it afterward. Prefer the purpose-built,
low-overhead `GVIRTUS_LATENCY_TRACE` (fixed-size per-RPC samples) over raw DEBUG logs.

---

## 1. Testbed layout

Two wired nodes at AAU, reached via an SSH gateway. **This machine (Windows) cannot run GVirtuS** —
all builds/runs happen on the nodes over SSH.

| Role | Host | GPU | RoCE IP | NIC / device | Notes |
|------|------|-----|---------|--------------|-------|
| **Backend (server)** | `es-dpu-01` | L40S (46 GB) | `25.25.25.2` | `mlx5_1` / `ens1f1np1`, GID idx 3, RoCEv2 | Runs the GVirtuS backend that owns the real GPU. |
| **Frontend (client)** | `es-dpu-02` | L40S (46 GB) | `25.25.25.1` | same fabric | Runs the CUDA app linked against GVirtuS stubs. **Logically GPU-less** (the app uses the *remote* GPU) but the node physically HAS an L40S. |

- **GVirtuS model:** backend = the GPU server; frontend = the GPU-less client. The frontend's local
  GPU is only used by GVirtuS's *own* GPUDirect probe (see problems), not by the remoted app.
- RoCE IPs are **swapped vs the old docs**: backend=25.25.25.2, frontend=25.25.25.1. Ping ~0.12 ms.
- ⚠️ The report/docs sometimes say the frontend is "GPU-less machine" — remember it still has a
  physical GPU; that distinction caused a wrong "bug" diagnosis before (see §6).

### SSH access (`~/.ssh/config` already set up)
```
Host aaugw        → sshgw.aau.dk (user kz08ey@student.aau.dk)
Host es-dpu-01 es-dpu-02 → %h.srv.aau.dk, ProxyJump aaugw
```
Just `ssh es-dpu-01` / `ssh es-dpu-02`. Key: `~/.ssh/id_ed25519` (installed on gateway + both nodes).

**PowerShell→SSH→bash quoting is painful.** Single-quote literal commands; escape `$` as `\$` inside
double-quoted remote commands; avoid `(` `)` unquoted; strip CRLF (`sed -i "s/\r$//"`); no heredocs.
For multi-line scripts, write the script to a here-string, convert CRLF→LF, and pipe over SSH into
`cat > file` on the node, then run it there.

---

## 2. Backend & frontend containers

### Backend (es-dpu-01) — `gvirtus-kz08ey`
Launcher: **`/tmp/gvirtus-backend-run.sh`** on es-dpu-01. Runs a detached container that
**rebuilds from mounted source on start (cmake+make, ~90 s)** then launches the backend.
```bash
# from es-dpu-01:
GVIRTUS_GPUDIRECT={0|1} GVIRTUS_RMA_ZEROCOPY={0|1} bash /tmp/gvirtus-backend-run.sh
# then WAIT for readiness (do not connect early — that stalls):
docker logs gvirtus-kz08ey 2>&1 | grep -a "listener created"
docker logs gvirtus-kz08ey 2>&1 | grep -aE "GPUDirect=(enabled|disabled)" | tail -1
```
- `GVIRTUS_GPUDIRECT=1` → GPUDirect path; `=0` → plain RDMA. **Restart the backend between GD phases**
  (a crashed/served CUDA context poisons the persistent backend).
- `GVIRTUS_RMA_ZEROCOPY=1` is **required for GPUDirect** and for the fast bulk-transfer (D2H) path.
  It was hardcoded 0; the launcher now honours it via env.
- Backend env baked in: `GVIRTUS_UCX_DATAPATH=am`, `UCX_TLS=rc_mlx5,ud_mlx5,tcp,self`,
  `UCX_NET_DEVICES=mlx5_1:1,ens1f1np1`, `UCX_IB_GID_INDEX=3`, `UCX_SOCKADDR_TLS_PRIORITY=tcp`.
- ⚠️ The launcher bakes in `GVIRTUS_LOGLEVEL=10000` (DEBUG) + `UCX_LOG_LEVEL=info`, which is **too
  verbose for measurement** (container logs grow to hundreds of MB / GB and slow the run). For real
  sweeps override to `GVIRTUS_LOGLEVEL=40000` and `UCX_LOG_LEVEL=error`; only use DEBUG/TRACE briefly
  for diagnostics, and prune `docker logs` between phases (restarting the backend resets them).

### Frontend (es-dpu-02) — `gvirtus-fe-kz08ey`
Persistent source-built container (`aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04`), GVirtuS
installed at `/usr/local/gvirtus`, mounts `~/benchmarks → /benchmarks` and the repo at `/gvirtus`.
```bash
# rebuild a plugin/frontend after editing source:
docker exec gvirtus-fe-kz08ey bash -c "cd /gvirtus/build && make -j8 <target> && make install"
#   targets: cudart, gvirtus-frontend, cublas, cudnn, ...
```
**Run a CUDA app over GVirtuS** = `docker exec` with the transport env + `LD_LIBRARY_PATH` putting the
GVirtuS frontend stubs FIRST:
```
-e GVIRTUS_HOME=/usr/local/gvirtus -e GVIRTUS_CONFIG=/gvirtus/etc/properties_ucx.json
-e GVIRTUS_LOGLEVEL=40000 -e GVIRTUS_UCX_DATAPATH=am
-e GVIRTUS_GPUDIRECT={0|1} -e GVIRTUS_RMA_ZEROCOPY={0|1}
-e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self  (TCP mode: UCX_TLS=tcp,self)
-e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_IB_GID_INDEX=3 -e UCX_SOCKADDR_TLS_PRIORITY=tcp
-e LD_LIBRARY_PATH=/usr/local/gvirtus/lib:/usr/local/gvirtus/lib/frontend:<app libs>
```
Any app built with `--cudart shared` links these stubs (verify with `ldd <bin> | grep cudart` →
resolves to `/usr/local/gvirtus/lib/frontend/`). Static cudart cannot be LD-redirected.

### ⚠️ 2.1 THREE COPIES OF THE SOURCE — KEEP THEM IN SYNC
There are **three independent working trees** and nothing is committed, so they drift silently:

| Copy | Path | Role | How a change gets there / takes effect |
|------|------|------|----------------------------------------|
| **Local (Windows)** | `C:\...\GVirtuS` | where you EDIT | source of truth for edits; `scp` to the node(s). |
| **dpu1 backend** | `es-dpu-01:~/GVirtuS` | backend build | `scp` the file, then **`docker rm -f` + relaunch** (entrypoint rebuilds from mounted source). |
| **dpu2 frontend** | `es-dpu-02:~/GVirtuS` (mounted `/gvirtus`) | frontend build | `scp` the file, then rebuild in-container: `docker exec gvirtus-fe-kz08ey bash -c "cd /gvirtus/build && make -j8 <target> && make install"`. |

**Rules to avoid out-of-sync bugs (this HAS caused wasted hours):**
- Editing a file **locally does nothing on the nodes** until you `scp` it. Always deploy after editing.
- Know **which side a file affects and deploy to the right node(s):**
  - `plugins/<lib>/frontend/*`, `src/frontend/*` → **frontend (dpu2)**, rebuild in-container.
  - `plugins/<lib>/backend/*`, `src/backend/*`, `src/communicators/*` → **backend (dpu1)**, rm+relaunch.
  - **Shared headers** (`include/**`, `plugins/*/cuda_internals/*.h`, `common/*`) affect **BOTH** →
    `scp` to **both** nodes and rebuild both. Forgetting one side = mismatched wire format / silent
    corruption / crashes.
- After deploying, **confirm the rebuild actually happened**: watch for the `.cpp.o` compile +
  `Linking ... .so` lines, check the installed lib timestamp, or `ldd` the app. A silent no-op build
  means you're testing stale binaries.
- If results look impossible or inconsistent, **suspect a sync gap first** (stale node copy) before
  blaming logic.
- When done, decide deliberately what to commit; until then treat local as canonical and re-`scp` if
  a node was rebuilt/reset.

---

## 3. Existing test suite (all on the nodes under `~/benchmarks/` on es-dpu-02)

| Workload | What it stresses | Where | Key result |
|----------|------------------|-------|-----------|
| **simple_matrix** (cuBLAS SGEMM) | **bulk transfer** (H2D+GEMM+D2H, ~1 GB/matrix at N≈16k) | `/gvirtus/examples/simple_matrix/` | Reproduces report; **GPUDirect ~1.4× RDMA** (D2H). The workload that PROVES NIC→GPU DMA. |
| **BabelStream** | memory bandwidth (5 kernels) | `~/benchmarks/BabelStream`, `cuda-stream-fe` | ~93–95% native at large sizes; small sizes launch-overhead-bound. |
| **miniBUDE** | compute-bound (molecular docking) | `~/benchmarks/miniBUDE` (`build/cuda-bude-gvirtus`) | <0.2% overhead — transport-insensitive. |
| **llama.cpp** | control-plane / RPC-bound LLM decode | `~/benchmarks/llama.cpp/build_cuda/bin/` (TinyLlama-1.1B Q4 at `~/benchmarks/models/`) | ~79× slower baseline → **~5× faster after frontend RPC opts**; RDMA 2.2× TCP (tail-latency). |
| **transfer_bw2** | raw H2D/D2H `cudaMemcpy` bandwidth | `~/benchmarks/transfer_bw2`, harness `transfer_bw2.cu` | RDMA ~3× TCP; GPUDirect ~2.3× D2H (≥4 MiB). |

- Use **`llama-bench`** (self-exits) for clean llama numbers, not `llama-cli` (blocks on stdin →
  append `< /dev/null`). Both need `GGML_CUDA_DISABLE_GRAPHS=1`.
- Harness scripts: `~/benchmarks/harness/*.sh` (sweep_run, minibude_run, llama_run, transfer_run, …).
- **Deprioritized:** XSBench, CloverLeaf (expected compute-bound like miniBUDE). A synthetic RPC
  **ping-pong latency microbenchmark** (fixed payload sizes, unloaded+loaded) is still TODO for
  clean per-size latency/CDFs.

---

## 4. Where docs & results live

- **Docs (committed):** `docs/benchmarking/` — 3 consolidated files:
  - `README.md` — testbed setup, standing rules, GPUDirect verification, known bugs/knobs, roadmap.
  - `RESULTS.md` — all workload results (miniBUDE, BabelStream, transfer, simple_matrix, llama, latency).
  - `ASYNC-DISPATCHER.md` — the optimization stack + async dispatcher design + speedups.
- **Result data (committed):** `benchmarks/` — one folder per benchmark × mode:
  `<bench>-{async,sync,baseline}/` for `{llama, miniBUDE, BabelStream, simple-matrix}`, plus
  `_summary/` (cross-benchmark) and `transport-characterization/` (latency CDFs, transfer bandwidth).
  CSVs + plot scripts + PNGs. See `benchmarks/README.md`.
- **Runnable examples (committed):** `examples/{babelstream,llama,minibude,simple_matrix}/`
  (`setup.sh`/`frontend.sh`/`backend.sh`/`Dockerfile`/`README.md`).
- **Raw benchmark inputs/binaries (NOT in git):** on the nodes under `~/benchmarks/` and
  `/gvirtus/examples/`.
- **Session plan / progress log:** `BENCHMARK_PLAN.md` (repo root); append-only historical log (its
  references to the old numbered docs are kept as-is for provenance).
- **The report (ground truth to compare against):** `_pdftxt/page_*.txt` (102-page project report;
  matrix tables 7.2/7.3/7.5/7.6 on pages 78–79, 84).
- **Working branch:** `marcel/ucx-comm/testing` (local + both nodes); changes deployed to nodes via
  `scp` + in-container rebuild.

---

## 5. Status of benchmarking (as of 2026-07-18)

**Done & documented:** testbed bring-up; BabelStream (sweep + latency); transfer bandwidth;
miniBUDE; llama (runs + evidence-based RPC profile + transport sweep); per-RPC latency distributions
(p50/p99/tail); SimpleMatrix vs report.

**GVirtuS fixes made & KEPT (all in `plugins/cudart/frontend/*` + one backend handler + `Frontend.cpp`):**
1. `cudaDeviceGetPCIBusId` — missing backend handler (llama init crash).
2. `cudaHostAlloc`/`cudaMallocHost` — 256-byte `posix_memalign` (ggml alignment crash).
3. `__cudaPush/PopCallConfiguration` — local thread-local stack (no RPC).
4. `cudaGetDevice` / `cudaGetLastError` / `cudaPeekAtLastError` — frontend-local cache (no RPC).
5. `cudaErrorNotReady` excluded from sticky last-error (correct CUDA semantics).
6. `GVIRTUS_LATENCY_TRACE=<file>` — env-gated per-RPC latency tracer in `Frontend::Execute()`.
→ Combined: **~5× faster LLM inference** from cheap, behaviour-preserving frontend changes.

**Headline findings:** llama is control-plane/RPC-bound (data path idle → GPUDirect≡RDMA there);
RDMA's value is **tail-latency** (p99 24× better than TCP), not median; GPUDirect's value is **bulk
transfer** (SimpleMatrix ~1.4×, transfer_bw ~2.3× D2H).

**Next candidates:** structural async dispatch (rec #3) / launch batching / graph capture;
multitenancy (2 frontends → 1 backend: scaling + Jain fairness + noisy-neighbor isolation);
synthetic RPC ping-pong latency microbenchmark; throughput–latency saturation curves.

---

## 6. Known problems / gotchas (check here before diagnosing anything new)

1. **Zombie/orphan contamination (the #1 cause of fake "slowness").** `timeout`-killed frontends leave
   the process AND its backend connection alive; orphaned backend containers leak many GB of GPU
   memory and starve GPUDirect registration (symptoms: `cudaMalloc(4K) failed`, stalls, exit-134).
   **Fix:** kill frontend procs (`docker exec ... pkill -9 -f "simple_matrix|llama|<bin>"`), `docker rm -f`
   the backend, verify `nvidia-smi` on both nodes. Non-root `kill` cannot reap root-owned container procs.
2. **GPUDirect first-connection cold-start stall.** The first bulk run after a fresh GD=1 backend can
   stall for minutes (cold NIC registration of the large GPU rx-pool). Retry / warm up with a small run.
3. **Backend listener non-recovery.** After a client crash/unclean restart, a stray `gvirtus-backend`
   keeps the RoCE port bound (`Address already in use` / `ucp_listener_create failed: Device is busy`).
   Fix: `docker rm -f` + relaunch; ensure no stray backend PIDs remain.
4. **Frontend GPUDirect probe.** Frontend runs a real `cudaMalloc(4K)` probe on its *local* GPU when
   `GVIRTUS_GPUDIRECT=1` is set on the frontend. If that GPU is exhausted (e.g. by a leak) the probe
   fails and can poison the CUDA context → app crash (`initialization error`). For control-plane
   workloads (llama) do NOT set `GVIRTUS_GPUDIRECT` on the frontend. GPUDirect is negotiated from the
   BACKEND; the frontend gracefully falls back to host slots. (This once looked like a code bug — it
   was GPU exhaustion. Challenge assumptions.)
5. **`GVIRTUS_RMA_ZEROCOPY`.** `=1` is required for GPUDirect and the fast D2H path, BUT historically
   OOM'd because UCX couldn't create a registration cache in-container (`rcache ... Unsupported
   operation`) → registration leak at large payloads. Confirm it's stable in the current build before
   trusting large sweeps; for pure control-plane work `=0` is fine.
6. **Stale prebuilt frontend images** give `invalid response header` — build the frontend from source
   (the `gvirtus-fe-kz08ey` container already does).
7. **Backend rebuilds on every launch (~90 s).** Wait for `listener created` before connecting, or the
   frontend stalls against a not-ready backend.
8. **Out-of-sync source trees (local vs dpu1 vs dpu2).** Nothing is committed; the three copies drift.
   Impossible/inconsistent results are often a **stale node copy** (you edited locally but didn't
   `scp`, or rebuilt only one side, or changed a shared header and synced only one node). See §2.1 —
   suspect this FIRST when behaviour doesn't match your code.

---

## 7. Standard workflow for a new benchmark run

1. **Pre-flight cleanup:** on es-dpu-02 kill leftover app procs; `docker ps`/`pgrep` clean; `nvidia-smi`
   free on both nodes.
2. **Launch backend** for the target transport (`GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`); wait for
   `listener created`; capture the `GPUDirect=enabled/disabled` line as evidence.
3. **Warm up** (small run) to establish the connection / registrations, then run the **measured** sweep.
4. **Capture evidence:** stdout CSV, backend `docker logs` (GD state, GPU-shadow/routing for GPUDirect),
   `[GVS PROFILE]` lines for transfers, and — for latency work — `GVIRTUS_LATENCY_TRACE`.
5. **Post-run cleanup:** kill the just-run procs; if switching GD phase, `docker rm -f` + relaunch backend.
6. **Verify GPUDirect** per §0.5 before writing any GPUDirect claim.
7. **Persist:** save raw data under `benchmarks/<bench>-<mode>/`, update the relevant consolidated doc
   (`docs/benchmarking/{RESULTS,ASYNC-DISPATCHER,README}.md`), append a dated `UPDATE #N` to
   `BENCHMARK_PLAN.md`, record in the session SQL table.
8. **Compare to the report** (`_pdftxt/`) when a matching config exists; note the config/N differences.

---

## 8. Metrics worth reporting (for the INFOCOM-grade paper)
Means alone are insufficient. Prefer: latency **distributions** (p50/p90/p99/p99.9 + CDFs + tail
ratio), latency **decomposition** (marshal→wire→dispatch→exec→return), **throughput–latency**
saturation + small-message RPC/s ceiling, **multi-tenancy** scaling + fairness (Jain) + isolation,
LLM TTFT/inter-token latency, CPU/registration efficiency, ≥5 reps with 95% CIs, an external baseline,
and per-mechanism **ablations**. See `docs/benchmarking/README.md` §5 (measurement roadmap).
