# 01 — Testbed Setup & Phase 0 Bring-up

Status: **Phase 0 complete** (backend runs, end-to-end smoke test passes).
Last updated: 2026-07-17.

---

## 1. Hardware & roles

Two directly-wired nodes at AAU, each with an NVIDIA **L40S** GPU:

| Node | Role | GPU | Driver |
|------|------|-----|--------|
| `es-dpu-01` | **Backend** (GPU server) | L40S (46 GB) | 580.95.05 |
| `es-dpu-02` | **Frontend** (client) | L40S (46 GB) | 580.95.05 |

Both nodes actually have a GPU — required because **GPUDirect on the frontend**
needs the client NIC to DMA into client GPU memory.

## 2. Network

RDMA NIC: Mellanox `mlx5_1` → netdev `ens1f1np1`, RoCEv2, **GID index 3**.

| Node | `ens1f1np1` (RoCE, 25 Gb/s) | `ens1f0np0` |
|------|------------------------------|-------------|
| `es-dpu-01` (backend) | `25.25.25.2/24` | `24.24.24.2/24` |
| `es-dpu-02` (frontend) | `25.25.25.1/24` | `24.24.24.1/24` |

> **⚠️ IMPORTANT — IPs are swapped vs. older docs.** Earlier docs
> (`docs/GPUDIRECT.md`, `docs/UCX_GUIDE.md`) assume the backend `es-dpu-01` is
> `25.25.25.1`. On the *current* physical setup that is reversed: `25.25.25.1`
> is the **frontend** (es-dpu-02) and `25.25.25.2` is the **backend** (es-dpu-01).
> Always set `server_address` to the **backend node's real IP = `25.25.25.2`**.

RoCE link verified: `ping 25.25.25.1` from es-dpu-01 → 0% loss, ~0.12 ms RTT.
`show_gids mlx5_1` on es-dpu-01 confirms index 3 = RoCEv2 (`v2`) bound to `25.25.25.2`.

## 3. SSH access (from the Windows dev box)

Autonomous non-interactive SSH works via an SSH config with a ProxyJump through
the AAU gateway:

```
Host aaugw
    HostName sshgw.aau.dk
    User kz08ey@student.aau.dk
Host es-dpu-01 es-dpu-02
    HostName %h.srv.aau.dk
    User kz08ey@student.aau.dk
    ProxyJump aaugw
```

- Key: `~/.ssh/id_ed25519` (no passphrase), authorized on gateway + both nodes.
- Because each shell call is a fresh process (no SSH multiplexing on Windows),
  every remote step is a discrete `ssh <node> "<cmd>"`; long-running services
  (the backend) run **detached** so they persist across calls.

## 4. Repository state

- Both nodes: `~/GVirtuS`, branch **`marcel/ucx-comm/testing`** (commit `88fac1f`).
- The branch was local-only on the dev box; it was propagated by pushing to
  `origin/marcel/ucx-comm/testing` (via a git bundle imported on es-dpu-01, which
  has SSH push rights), then fetched + checked out on both nodes.

## 5. Build / run harness (this branch)

Docker-based. Required images already present on both nodes (no rebuild needed):
`aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04`,
`aauce25/simple_matrix_gvirtus:cuda12.6`.

**Config files** (`etc/`):
| File | Suite | Purpose |
|------|-------|---------|
| `properties_ucx.json` | `ucx` | UCX communicator (main). Port `32223`. |
| `properties_plain_rdma.json` | `roce-rdma` | Legacy raw RDMA connector. Port `3333`. |
| `properties.json` | `tcp/ip` | Legacy TCP connector. Port `32222`. |
| `properties_hybrid.json` | hybrid | TCP control + RDMA data. |

`etc/ucx.env` holds the UCX transport preset (loaded by the Makefile). Current
default = **mixed TCP+RDMA**: `UCX_TLS=rc_mlx5,ud_mlx5,tcp,self`,
`UCX_NET_DEVICES=mlx5_1:1,ens1f1np1`, `UCX_IB_GID_INDEX=3`,
`GVIRTUS_UCX_DATAPATH=am`, `GVIRTUS_GPUDIRECT=1`.

**Config change applied (both nodes):** `server_address` `25.25.25.1` → `25.25.25.2`
in `properties_ucx.json` and `properties_plain_rdma.json`.

### Backend launch (es-dpu-01)

The stock target `make run-gvirtus-backend-dev` runs interactively (`-it`), which
does not survive across separate SSH calls. For autonomous driving we launch the
**same container detached** (`-d`), same name `gvirtus-kz08ey` (so the existing
`stop-gvirtus` / `attach-gvirtus-bash` / `run-simple-matrix-test` targets still
interoperate). Launcher lives at `/tmp/gvirtus-backend-run.sh` on es-dpu-01 (kept
out of the repo). The container entrypoint runs `cmake && make -j && make install`
then `gvirtus-backend etc/properties_ucx.json`.

Healthy backend log signature:
```
GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK ...)
rx_pool: initialized 2 slots x ... bytes (host) + 2/2 GPU shadows ...
init_ucx completed host=25.25.25.2 port=32223 mode=am
listener created
```

Stop backend: `ssh es-dpu-01 "docker rm -f gvirtus-kz08ey"` (or `make stop-gvirtus`).

### Frontend run (es-dpu-02)

`make run-simple-matrix-test` — runs the `simple_matrix_gvirtus` container, mounts
`properties_ucx.json`, compiles `examples/simple_matrix/simple_matrix.cu`, and runs
it. One-shot foreground; exits when done.

## 6. Phase 0 result — smoke test

`simple_matrix` (256×256 SGEMM, 10 iters) from es-dpu-02 against the backend:

```
size=256 iters=10 avg_sgemm_ms=0.115277 avg_host_ms=0.506339 check=pass max_abs_err=0
CSV,256,10,0.115277,0.506339
```

Full CUDA lifecycle observed remotely over UCX Active-Message path:
`__cudaRegisterFatBinary` → `cudaMalloc` → `cublasSgemm_v2` → result D2H →
`cudaFree`/`cublasDestroy_v2`/`cudaUnregisterFatBinary` → clean UCX close.

**Conclusion:** the end-to-end remoting pipeline (frontend → UCX/RoCEv2 → backend →
L40S) is functional and numerically correct. Ready for Phase 1 (BabelStream + miniBUDE).

## 7. Reproduce quickly

```bash
# Backend (es-dpu-01)
ssh es-dpu-01 "bash /tmp/gvirtus-backend-run.sh"
ssh es-dpu-01 "docker logs gvirtus-kz08ey 2>&1 | tail -20"   # expect 'listener created'

# Frontend smoke test (es-dpu-02)
ssh es-dpu-02 "cd ~/GVirtuS && make run-simple-matrix-test 2>&1 | grep -E 'check=|^CSV,'"
```
