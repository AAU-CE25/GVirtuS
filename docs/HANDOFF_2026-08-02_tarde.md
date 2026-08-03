---
title: "HANDOFF — 2026-08-02, afternoon session"
subtitle: "What went right, what went wrong, and the exact DPU configuration"
date: "2026-08-02"
geometry: margin=2.5cm
fontsize: 10pt
---

# 0. What this session was

Continuation of `HANDOFF_2026-08-02.md` §7: the six items the morning session left open. Five
are closed, one is closed *with its statistical limit written down instead of rounded away*.
Along the way, four real concurrency defects were found and fixed — none of which was on the
list.

Results: `docs/GUSTO_CONTINUACION_2026-08-02.md`. Reviewer-facing summary: `~/paper/SLOTS.md`
(+ `.pdf`). Raw: `docs/gusto_raw_2026-08-02/`. Three commits, unpushed, on `exp/lazy-pool-v2`.

| # | item | outcome |
|---|---|---|
| 1 | epoch guard, deterministic | closed, **with a retraction of my own** |
| 2 | sustained saturation | closed, 4 cells |
| 3 | ThreadSanitizer | closed — the build was never broken |
| 4 | llama abort cause | closed as **partial**, 26 runs, 0 aborts, not conclusive |
| 5 | cuDF | closed |
| 6 | normalise paper headline | closed — publish ×310, not ×263 |

\newpage

# 1. What went right

**Ran the control every time, and the control is what saved the results.** Three cases where the
first arm alone would have produced a wrong published claim:

- The epoch guard fired (`ack_epoch_dropped=1`). Ablating it changed *nothing else*. It would
  have been easy to stop there and write "the guard is unnecessary". The control that mattered
  was a *third* workload — cuDF — which shows slot ids being re-advertised unchanged across
  epochs. The dangerous state is reachable. §4 below.
- The saturation cells looked like a pool story until S4 (same pool, no injected delay) showed
  *the same 150 waits* at 0.84 µs instead of 23.5 ms. The contention belongs to the workload;
  its cost belongs to the consumer. Without S4 the write-up would have blamed the pool.
- The `rma_checksum` "regression" of −58 % evaporated against the counters in its own log.

**Checked whether an experiment could work before running it.** The library A/B was going to be
done with `LD_LIBRARY_PATH`; the communicator is `dlopen`ed from an absolute path built from
`GVIRTUS_HOME`, so both arms would have loaded the same `.so`. That is the unpaired-build trap
that has already cost this project twice. Caught by reading `CommunicatorFactory.h:72` first.

**Refused to contaminate a running measurement.** No library was rebuilt or reinstalled while
the llama campaign was running, even though there was idle time and an obvious edit queued.

**Extended a campaign because the statistics demanded it, not because it was asked.** 12 runs
with 0 aborts is compatible with the defect being untouched (P = 24 %). Ran 14 more to reach 26,
where the number starts to mean something — and then reported that even 26 clears the original
rate by only 0.23 points.

**Fixed defects in the harness that made previous cells inert**: `GVS_FAULT_MS` was never
propagated to the backend, so `slow_ack` always ran at its 50 ms default; and a banner announced
"no injection" during runs *with* injection.

**Preserved everything**: a `.bak_*` per edit, raw kept including the failed versions,
`lib/` installs verified by md5 against `build/`, no library mutated for an A/B.

\newpage

# 2. What went wrong

Nine, in three families. All were caught, but four of them had already been written into a
document or a memory note before being caught.

## 2.1 Claims that outran the evidence (3)

1. **"Advertised `server_idx` values are strictly increasing, so the epoch guard is
   unreachable."** False. The experiment behind it (95 evaluations, 0 deliveries) only covered
   re-advertisements *with* renumbering. cuDF advertises `[0-7]` in epoch 1 and again in epoch 2.
   Nearly published as "defence in depth, not necessity".
2. **"llama goodput has zero variance, so run-to-run spread is under 0.48 %."** True over 12
   runs, false over 26: one run lands at 568.9 t/s instead of 591.6, a −3.8 % outlier. Zero
   variance at n=12 was a property of the sample, not of the system. Already written into the
   results doc and a memory note before the second batch refuted it.
3. **Miscounted the crash rate**: wrote "6 crashes in 7 runs", the raw says **9 in 10**. I had
   added the two A/B arms as if they were different binaries — at that point the fix was not in
   yet, so both arms were the *same* build. Corrected in the doc and in two memory notes.

## 2.2 Experiments that did not measure what they claimed (2)

4. **`rma_checksum` at 64 MiB against a 4 MiB control.** 64 MiB does not fit a 32 MiB slot, so
   all 60 transfers fell back to the eager path: 6.85 GB/s against 16.47. It looked like a −58 %
   regression and it was RMA-versus-AM. The discriminating counter was in the same log
   (`decline_capacity=60` against the control's `admit_rma=60`) — reading the counter before the
   ratio would have caught it instantly.
5. **miniBUDE run with `cuda-bude` instead of `cuda-bude-gvirtus`**: it aborted with
   `cudaErrorNotSupported` and the extractor wrote `NA` **without the harness complaining**. A
   metric that silently becomes `NA` is worse than one that fails loudly.

## 2.3 Tooling mistakes (4)

6. **`git stash push --staged` on git 2.34, which does not have that flag.** The `||` fallback
   ran `--keep-index`, and the `--amend` that followed swallowed all 24 files into a commit whose
   message described only 7. Caught by inspecting the commit contents; recovered with
   `stash pop` + `reset --soft` and rebuilt as three clean commits. Nothing was lost, but the
   command was careless: a fallback chained with `||` hides exactly this kind of failure.
7. **`pkill -f "syssample.py"` matched its own command line** and killed the ssh session
   (exit 255). Fixed with a self-non-matching pattern (`syssampl[e].py`).
8. **Left a monitor armed** after the campaign it was watching had already finished and been
   reported; it later fired as a timeout. Harmless, noisy.
9. **Assumed a file had been deleted** when `paper/README.md` disappeared, instead of checking
   first: it had been renamed to `SLOTS.md`, same size and timestamp.

## 2.4 The one thing I could not do

**44 orphan sampler processes on dpu-01** (`syssample.py`, ~12 % of a CPU, from campaigns that
ended 40 h earlier) could not be killed: the permission classifier blocked it. Today's numbers
and the morning's share that background, so the comparisons hold; a clean bench needs them gone.

\newpage

# 3. DPU configuration

## 3.1 Topology

| host | role | what runs there |
|---|---|---|
| `es-dpu-01` | **backend** | `gvirtus-ll33pq` container, owns the L40S, listens on `25.25.25.2:32222` |
| `es-dpu-02` | **frontend** | application containers, mount `~/GVirtuS` read-only at `/opt/GVirtuS` |

Fabric `25.25.25.0/24` on `mlx5_1` / `ens1f1np1`, RoCEv2. Both hosts have an L40S and 2×
ConnectX-7 200 Gb. Both are reachable directly with `secure-machine-access`.

## 3.2 Which side reads which knob — the table that keeps being needed

This is the single most expensive class of error in this project: a knob set on the side that
does not read it produces a silent no-op, and the sweep measures nothing.

| variable | read by | effect |
|---|---|---|
| `GVIRTUS_RMA_SLOTS` | **backend** | number of slots in the pool the peer will PUT into |
| `GVIRTUS_RMA_SLOT_CAP_MB` | **backend** | per-slot ceiling; with `PREALLOC=1` this *is* the allocation |
| `GVIRTUS_RMA_SLOT_MIN_MB` | **backend** | per-slot floor |
| `GVIRTUS_RMA_PREALLOC` | **backend** | build the pool at connect instead of on demand |
| `GVIRTUS_RMA_MIN_BYTES` | **both** | RMA floor; the backend's copy also sizes the pool |
| `GUSTO_RMA_HOST_POOL_BUDGET_BYTES` | **backend** | rejects a provisioning that exceeds it |
| `GVS_FAULT` = `slow_ack` | **backend** | `send_slot_consumed` is server-side |
| `GVS_FAULT_MS` | **backend** | delay for `slow_ack` — *was not propagated until today* |
| `GVS_FAULT` = `hold_ack`/`epoch_ack`/`epoch_ack_idx`/`delay_ack` | **frontend** | they live in `release_remote_slot` / `WriteIovRma` |
| `GVS_FAULT_ARM` | **frontend** | which ack arms the replay (default 20; use **1** for `epoch_ack`) |
| `GVS_ABLATE` | side that owns the mechanism | `no_epoch`/`no_generation` are client-side; `pointer_keyed` is registry-side |
| `GVIRTUS_RMA_POLICY`, `GVIRTUS_RMA_SCALAR_FLOOR` | **frontend** | the placement decision |
| `GVIRTUS_ASYNC_DISPATCH` | **frontend** | the only way to get >1 transfer in flight on one connection |

`reset_backend_pool.sh` **prints the effective environment inside the container** before
returning, and `gusto_validate_pool_cfg()` prints the effective pool. Use them; do not assume.

## 3.3 Backend (run on dpu-01)

```bash
GVIRTUS_RMA_MIN_BYTES=8192 GVS_SLOTS=8 GVS_SLOT_MIN_MB=4 GVS_SLOT_CAP_MB=32 \
  GVS_PREALLOC=1 bash ~/reset_backend_pool.sh
```

Current effective environment (the configuration all of today's numbers were taken under):

```
GVIRTUS_GPUDIRECT=1              GVIRTUS_RMA_MIN_BYTES=8192
GVIRTUS_RMA_SLOTS=8              GVIRTUS_RMA_SLOT_CAP_MB=32
GVIRTUS_RMA_SLOT_MIN_MB=4        GVIRTUS_RMA_PREALLOC=1
GVIRTUS_RMA_ZEROCOPY=1           GVIRTUS_UCX_DATAPATH=am
GVIRTUS_LOGLEVEL=40000           GVIRTUS_UCX_PROGRESS_TIMEOUT_MS=0
GVS_FAULT=none                   GVS_FAULT_MS=50
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self UCX_NET_DEVICES=mlx5_1:1,ens1f1np1
UCX_IB_GID_INDEX=3               UCX_SOCKADDR_TLS_PRIORITY=tcp
UCX_LOG_LEVEL=error
```

Image `ll33pq/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04`. It mounts `src/`, `include/`,
`plugins/`, `examples/`, `etc/`, `CMakeLists.txt` and **rebuilds from source on every launch** —
so a source change on dpu-01 only takes effect after a reset, and a source change on dpu-02 has
to be synced across first.

The script serialises with `flock` (fd closed in every child with `9>&-`, learned the hard way),
waits for the container to *disappear* rather than for the port to leave LISTEN, retries once on
a bind race, and distinguishes a bind race from a compile failure.

## 3.4 Frontend (run on dpu-02)

```bash
docker run --rm --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash -e LD_PRELOAD= \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e UCX_MEMTYPE_CACHE=n -e UCX_RCACHE_ENABLE=n \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
  -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=error \
  -e GVIRTUS_HOME=/opt/GVirtuS -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_LOGLEVEL=30000 -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e GVIRTUS_RMA_MIN_BYTES=8192 \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/lib:/usr/local/cuda/lib64 \
  -v "$PWD":/opt/GVirtuS:ro ll33pq/cudf_gvirtus_dyncudf:cuda12.6 \
  -c "ulimit -c 0; LD_PRELOAD=libcuda.so.1:libcudart.so.12 <command>"
```

Four things that are not obvious:

- **`GVIRTUS_HOME` decides which communicator is loaded.** It is `dlopen`ed from
  `$GVIRTUS_HOME/lib/libgvirtus-communicators-<protocol>.so`, an absolute path.
  `LD_LIBRARY_PATH` does **not** override it — so a library A/B done that way silently compares
  a build against itself.
- **`make` leaves the `.so` in `build/`, not in `lib/`.** Copy by hand and verify with `md5sum`.
  Build recipe: `cd ~/GVirtuS/build && CPATH=$HOME/lz4inc LIBRARY_PATH=$HOME/lz4shim make <target> -j8`.
- **`ulimit -c 0` always.** Core dumps fill `/var` and block every container on the host.
- The frontend container has no GPU, so `GPUDirect probe FAILED` in its log is expected and
  harmless; GPUDirect is a backend-side property.

## 3.5 ThreadSanitizer (two non-obvious requirements)

The build was never broken — configure `build_tsan` with the `build_asan` recipe, swapping
`address` for `thread`, and everything compiles. The *run* is what fails, for two reasons:

1. **UCX installs memory hooks that kill the sanitizer**: SIGSEGV before the first transfer.
   Disable with `UCX_MEM_EVENTS=n UCX_MEM_MMAP_RELOC=n UCX_MEM_MALLOC_HOOKS=n UCX_MEM_MALLOC_RELOC=n`.
2. **TSan does not support `LD_PRELOAD` over an uninstrumented executable** the way ASan does.
   The bench itself must be built with it:
   `nvcc --cudart shared -O1 -g -Xcompiler -fsanitize=thread -Xlinker --no-as-needed -o x x.cu -ltsan`,
   and `libtsan.so.0` still has to come **first** in `LD_PRELOAD`.

## 3.6 Harnesses written this session (all on dpu-02, in `~`)

| script | what it drives |
|---|---|
| `gusto_epoch_run.sh` | `growtest` with fault/ablation/arming index selectable |
| `gusto_epochgrow_run.sh` | `epochgrow`, N ascending phases → one regrow per boundary |
| `gusto_sat_run.sh` | `concgrow` with async dispatch → sustained pool saturation |
| `gusto_tsan_run.sh` | any bench under ThreadSanitizer, with the UCX hooks disabled |
| `gusto_firsttouch_ab.sh` | `firsttouch` A/B, two library directories, no mutation of `lib/` |
| `gusto_noregresion.sh` | rpclat + rma_checksum + miniBUDE against the morning control |
| `llama_abort_repro.sh` | N × (llama-server + CONC=8 load), counts aborts |
| `analiza_llama.py` | goodput across the campaign's meta files |

New benches in `examples/rmatest/`: `epochgrow.cu`, `firsttouch.cu` (+ `stress4_tsan`).

\newpage

# 4. State on exit

- **Bench clean**: no frontend containers, backend up and listening on dpu-01 with the §3.3
  configuration.
- **Code**: three commits on `exp/lazy-pool-v2`, **not pushed**. dpu-01 has the same sources
  synced but **uncommitted**.
- **Not committed by policy**: the `.log` and `.csv` raw files, excluded by the project's own
  `.gitignore` (a deliberate 2026-07-21 cleanup). The docs cite them by path. Force-adding them
  is a one-line change if that is wanted.
- **Auxiliary directories left on disk**: `ab_pre/` (pre-fix libraries for the A/B) and
  `build_tsan/`. Neither is on any load path.

# 5. Open

1. **The llama abort has no root cause.** 26 runs, 0 aborts, but that rejects the original 1/9
   only by 0.23 points, and 1/9 was a single event.
2. **`Frontend::Prepare()` repeats the unguarded map read on the hot path**, before every RPC.
   TSan does not flag it because no insertions remain by then; the window is concurrent startup.
   Fix: resolve the per-thread pointer once into a `thread_local` — also removes two lookups per
   RPC.
3. **`GetFrontend` returns an object that is not the one registered.** It `new`s a `Frontend`,
   `Init()` creates *another* and registers that one, and the later `insert` is a no-op. It does
   not break because the methods that matter ignore `this` and re-resolve through the map. One
   leaked `Frontend` per thread, and a trap for whoever next writes a method that uses `this`.
4. **Teardown races**: one TSan run reported 38, all in destructors, plus a SIGSEGV inside
   log4cplus (which is reconfigured *per thread*). Not reproduced in the four runs after. Not
   proven absent.
5. **An exception escaping a UCX callback** remains the most plausible route to killing the
   backend (`acquire_rx_slot` throws on a failed pinned allocation, on the AM handler path).
   Unverified — the pool never reached that state.
6. **44 orphan samplers on dpu-01.**

**The rule that still holds: before citing any number from this campaign, ask for its raw file.**
