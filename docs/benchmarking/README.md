# GVirtuS benchmarking

Benchmark campaign for GVirtuS (project GUSTO) on the AAU two-node testbed, over the UCX communicator
(TCP / RDMA / RDMA+GPUDirect). This folder is the **write-up**; the raw data + plots live under
[`../../benchmarks/`](../../benchmarks/), and the operating manual is the **benchmark-dev skill**
([`../benchmark-dev/SKILL.md`](../benchmark-dev/SKILL.md)).

| Doc | Contents |
|-----|----------|
| **[RESULTS.md](RESULTS.md)** | All workload results: miniBUDE, BabelStream, transfer bandwidth, simple_matrix, llama, per-RPC latency distributions. |
| **[ASYNC-DISPATCHER.md](ASYNC-DISPATCHER.md)** | The optimization stack: frontend RPC reductions + the asynchronous dispatcher (incl. `cudaMemcpyAsync`), design + measured speedups. |
| **README.md** (this file) | Testbed setup, standing rules, GPUDirect verification, known bugs/knobs, measurement roadmap. |

**Headline:** GVirtuS is ~free for compute-bound work (miniBUDE <0.2%) and ~93–95% of native for
bandwidth-bound work, but was catastrophic for launch-/RPC-bound LLM decode (~79× native). The
optimization stack + async dispatcher close that to **~3.4×** (token gen 8 → ~187 t/s over RDMA),
correctness preserved.

---

## 0. Standing rules (non-negotiable — see the skill for the full text)

1. **Clean up after testing.** Kill zombie frontend procs, restart the backend fresh between phases,
   verify GPU freed on both nodes. Leftover state fakes "slowness"/"hangs".
2. **Back up every number with evidence** (CSV, log, backend `docker logs`). Save under
   `benchmarks/`. No number without a source.
3. **Don't overclaim.** Distinguish "capable/configured" from "demonstrated".
4. **Challenge surprising results** — re-derive against a second config / warmup-excluded subset.
5. **Verify GPUDirect — never assume** (see §3).
- **Keep logs lean:** measure at `GVIRTUS_LOGLEVEL=40000` (ERROR) and `UCX_LOG_LEVEL=error`. DEBUG
  per-RPC logging ~halves throughput and produces GB-scale logs. Use the low-overhead
  `GVIRTUS_LATENCY_TRACE` for latency work, not raw DEBUG.

## 1. Testbed

Two directly-wired nodes at AAU, each with an NVIDIA **L40S** (46 GB):

| Node | Role | RoCE IP (`mlx5_1`/`ens1f1np1`, GID idx 3) |
|------|------|-------------------------------------------|
| `es-dpu-01` | **Backend** (owns the GPU) | `25.25.25.2/24` |
| `es-dpu-02` | **Frontend** (client) | `25.25.25.1/24` |

> ⚠️ IPs are **swapped vs older docs**: backend = `25.25.25.2` (set `server_address` to this).
> Both nodes physically have a GPU (GPUDirect on the frontend needs the client NIC to DMA into client
> GPU memory), but the remoted app uses the *remote* GPU.

Access from the Windows dev box: `ssh es-dpu-01` / `ssh es-dpu-02` (ProxyJump through the AAU gateway;
key `~/.ssh/id_ed25519`). Each shell call is a fresh process — long-running services (the backend) run
**detached**.

**Config** (`etc/`): `properties_ucx.json` (UCX, main, port 32223), plus legacy `properties.json`
(TCP) / `properties_plain_rdma.json` / `properties_hybrid.json`. Transport preset in `etc/ucx.env`
(`UCX_TLS=rc_mlx5,ud_mlx5,tcp,self`, `UCX_NET_DEVICES=mlx5_1:1,ens1f1np1`, `UCX_IB_GID_INDEX=3`,
`GVIRTUS_UCX_DATAPATH=am`).

**Backend launch** (es-dpu-01): a detached container (`gvirtus-kz08ey`) whose entrypoint rebuilds from
mounted source (`cmake && make -j && make install`) then runs `gvirtus-backend properties_ucx.json`.
Env knobs: `GVIRTUS_GPUDIRECT={0|1}`, `GVIRTUS_RMA_ZEROCOPY={0|1}`, `GVIRTUS_RMA_SLOTS`,
`GVIRTUS_RMA_SLOT_CAP_MB`, `GVIRTUS_LOGLEVEL`. Healthy log signature: `GPUDirect=enabled ...` →
`rx_pool: initialized N slots ...` → `listener created`. Restart (`docker rm -f` + relaunch) between
GPUDirect phases — a poisoned CUDA context silently invalidates later runs.

**Examples** ([`../../examples/`](../../examples/)): first-class GVirtuS runs of the app benchmarks —
`babelstream/`, `llama/`, `minibude/`, `simple_matrix/` — each with `setup.sh` / `frontend.sh` /
`backend.sh` / `Dockerfile` / `README.md`.

## 2. Data layout ([`../../benchmarks/`](../../benchmarks/))

Per-benchmark × mode folders: `<bench>-async/` (dispatcher on), `<bench>-sync/` (optimized, async
off — includes the older transport/optimization campaign), `<bench>-baseline/` (GVirtuS over the
**legacy TCP communicator**, `tcp/ip` suite / `properties.json` — i.e. without UCX; placeholders to
run later). Plus `_summary/` (cross-benchmark) and `transport-characterization/` (per-RPC latency
CDFs, raw transfer bandwidth). See `benchmarks/README.md`.

## 3. ⚠️ Verify GPUDirect — never assume

We were burned twice by *assuming* GPUDirect was engaged/broken. Before writing "GPUDirect" for a run:

1. Backend log says `GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK ...)` — `disabled` ⇒
   not GPUDirect.
2. Backend launched with `GVIRTUS_GPUDIRECT=1` (`=0`/unset = plain RDMA).
3. An **RDMA** TLS was negotiated (`rc_mlx5`, not `tcp,self`) — GPUDirect is impossible on TCP.
4. Confirm the data path ran: `WriteIovRma(zerocopy) ... big=<bytes>` + `rx_pool: ... + N/N GPU
   shadows`.
5. Cross-check surprises against `simple_matrix` at large N (976 MB/transfer) before claiming a bug.
6. Plain-RDMA vs GPUDirect configs must differ **only** in the backend `GVIRTUS_GPUDIRECT` flag;
   restart the backend (fresh context) between phases.

The **frontend** GPUDirect probe currently fails (it `dlopen`s the GVirtuS stub `libcudart`), so the
client advertises `0 slots with gpu shadow` — fine for host-source transfers; **the backend flag +
backend log is the source of truth.** If you can't point to a backend log line proving GPUDirect
engaged for *this* run, write "unverified".

## 4. Known bugs & knobs (candidates for hardening / paper "future work")

- **Frontend `GVIRTUS_GPUDIRECT=1` can poison the app's CUDA context** — the frontend probe does a
  real `cudaMalloc(4K)` that, on a degraded client GPU, fails and leaves CUDA state poisoned
  (llama `initialization error`, exit 134). **Do not set `GVIRTUS_GPUDIRECT` on the frontend** — it's
  the backend's concern. (The persistent "cold-start stall" was this crash in disguise.)
- **`GVIRTUS_RMA_ZEROCOPY=1` OOM-kills the backend at ≥32 MiB** — UCX can't create a registration
  cache in-container ("could not create UCP registration cache"), so every zero-copy `ucp_put`
  re-registers and leaks pinned NIC registrations. Keep it `=0` for sweeps (GPUDirect *needs* `=1` for
  device buffers — that path works for simple_matrix but watch memory). Fix: working rcache or a
  GVirtuS-managed registration pool.
- **Backend GPU memory leaks across client connections** — after many sequential runs the backend GPU
  climbs (observed to 45/46 GB) and `cudaMalloc` starts failing ("malloc fail"). A fresh backend
  restart clears it; restart between heavy sweeps.
- **Backend listener non-recovery** — after a connection reset the next `ucp_listener_create` fails
  ("Device is busy", port 32223 stuck) → `docker rm -f` + relaunch.
- **Buffer-reuse RMA edge case (narrow)** — transferring growing sub-ranges of one large *reused*
  registered device buffer resets the RDMA connection at ≥8–16 MiB. Normal exactly-sized allocations
  are fine to ~976 MB.
- **BabelStream at exactly 2²² (4,194,304) elements fails** on a fresh backend while 2²¹/2²³ succeed —
  a size-specific bug worth a follow-up.
- **`cudaHostAlloc` zero-copy in kernels unsupported** — GVirtuS `cudaHostAlloc` = frontend-local
  `malloc`, so a kernel writing that pointer faults (CUDA 700). Needs a device-buffer + explicit copy
  adaptation (see the BabelStream example's `setup.sh`).

**Fixes landed (kept in tree):** `cudaDeviceGetPCIBusId` backend handler (llama CUDA-init crash);
`cudaHostAlloc`/`cudaMallocHost` 256-byte `posix_memalign` (ggml alignment crash); the Stage-1 RPC
reductions and the async dispatcher (see ASYNC-DISPATCHER.md); `GVIRTUS_LATENCY_TRACE` instrumentation.

## 5. Measurement roadmap (for an INFOCOM-grade paper)

INFOCOM judges GUSTO as a *transport system*, so means alone are insufficient. Priority order:

1. **Latency distributions + CDFs** (done — RESULTS.md §6; `GVIRTUS_LATENCY_TRACE`). Extend to more
   call-classes.
2. **Latency decomposition** (marshal → wire → backend dispatch → exec → return) — timers already
   exist in `Execute()`; aggregate them.
3. **Throughput–latency saturation** curve (the "hockey stick") + small-message **RPC/s ceiling**
   (the control-plane number that predicts LLM decode).
4. **Multi-tenancy** — N frontends → 1 backend scaling, **Jain fairness**, and **isolation** (does a
   bandwidth-heavy tenant inflate a latency-sensitive tenant's p99?).
5. **LLM SLOs** — TTFT / inter-token latency p50/p99.
6. **Statistical rigor** — ≥5 reps, 95% CIs / error bars, warmup discarded, on every plot.
7. **External baseline** (e.g. rCUDA) + per-mechanism **ablation figures** (AM vs TCP control path;
   RDMA vs staged; GPUDirect vs host-staged; Stage-1 opts on/off; async on/off).
