# GVirtuS UCX Communicator Guide

This guide covers how to configure, deploy, and tune the UCX-based communicator
for GVirtuS. The UCX communicator replaces the original TCP and verb-level RDMA
communicators with a single transport that auto-negotiates the best available
path (TCP, RoCE, InfiniBand) per connection at runtime.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Transport Scenarios](#transport-scenarios)
- [GPUDirect RDMA](#gpudirect-rdma)
- [Environment Variables Reference](#environment-variables-reference)
- [Troubleshooting](#troubleshooting)
- [Architecture Overview](#architecture-overview)
- [Further Reading](#further-reading)

---

## Prerequisites

**Backend node** (GPU host):
- NVIDIA GPU with CUDA 12.x drivers
- UCX >= 1.17 built with CUDA support (`--with-cuda`)
- For RDMA: Mellanox ConnectX-5+ NIC with OFED drivers
- For GPUDirect: `nvidia-peermem` kernel module loaded
- Docker with `--runtime=nvidia` support

**Frontend node** (client):
- UCX >= 1.17 (CUDA support not required)
- Network connectivity to backend (TCP or RDMA fabric)
- No GPU required

---

## Quick Start

### 1. Configure the endpoint

Edit `etc/properties_ucx.json` on both frontend and backend:

```json
{
    "communicator": [
        {
            "endpoint": {
                "suite": "ucx",
                "protocol": "ucx",
                "server_address": "25.25.25.1",
                "port": "32223"
            },
            "plugins": ["cuda", "cudart", "cublas", "curand", "cudnn",
                        "cufft", "cusolver", "cusparse", "nvrtc", "nvml"]
        }
    ],
    "secure_application": false
}
```

Set `server_address` to the backend's IP on the network interface you want
UCX to use (dedicated RDMA fabric recommended for best performance).

### 2. Start the backend

```bash
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
GVIRTUS_GPUDIRECT=1 \
make run-gvirtus-backend-dev
```

### 3. Run a frontend application

```bash
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
make run-simple-matrix-test
```

---

## Configuration

### Endpoint file: `etc/properties_ucx.json`

| Field | Description |
|-------|-------------|
| `suite` | Must be `"ucx"` to select the UCX communicator |
| `protocol` | Must be `"ucx"` |
| `server_address` | Backend IP on the target network interface |
| `port` | Listening port (default: `32223`) |

The same file is used by both backend and frontend. The backend binds to this
address; the frontend connects to it.

### Choosing the network interface

Finding the right values for `UCX_NET_DEVICES`, `UCX_IB_GID_INDEX`, and
`server_address` requires identifying which NIC carries your inter-node fabric
and what RDMA device name UCX expects for it.

**Step 1: List network interfaces and their IPs**

```bash
ip -o -4 addr show
```

Look for the dedicated inter-node link (not the management network). Example:
```
3: ens1f1np1  inet 25.25.25.2/24 ...
```

The IP on the backend side (`25.25.25.1`) goes into `server_address`.

**Step 2: Find the RDMA device name for that interface**

```bash
# List all RDMA devices
ibv_devinfo

# Or map a specific interface to its RDMA device
ibdev2netdev
```

Example output of `ibdev2netdev`:
```
mlx5_0 port 1 ==> ens1f0np0 (Up)
mlx5_1 port 1 ==> ens1f1np1 (Up)    ← this is our 25.25.25.x interface
```

So `UCX_NET_DEVICES=mlx5_1:1` (device `mlx5_1`, port `1`).

**Step 3: Find the correct GID index for RoCE v2**

```bash
show_gids | grep mlx5_1
```

Example output:
```
DEV     PORT  INDEX  GID                                 IPv4            VER   DEV
mlx5_1  1     0      fe80::...                           ---             v1    ens1f1np1
mlx5_1  1     1      fe80::...                           ---             v2    ens1f1np1
mlx5_1  1     2      0000::ffff:25.25.25.2               25.25.25.2      v1    ens1f1np1
mlx5_1  1     3      0000::ffff:25.25.25.2               25.25.25.2      v2    ens1f1np1  ← RoCE v2
```

Pick the index with your IP and `v2` (RoCE v2): `UCX_IB_GID_INDEX=3`.

**Step 4: Verify UCX sees the device**

```bash
ucx_info -d | grep -A2 mlx5_1
```

**Step 5: Check link speed**

```bash
ibstat mlx5_1 | grep -i "rate\|state"
```

Example: `Rate: 200` (200 Gb/s = 4×HDR).

**Summary of values for a typical setup:**

| What | Command | Example value |
|------|---------|---------------|
| Backend IP | `ip -o -4 addr show` on backend | `25.25.25.1` |
| RDMA device | `ibdev2netdev` | `mlx5_1:1` |
| TCP device | `ip link show` | `ens1f1np1` |
| GID index | `show_gids \| grep <IP>` + pick `v2` row | `3` |
| Link speed | `ibstat mlx5_1` | 200 Gb/s |

---

## Transport Scenarios

### TCP only (validation baseline)

Start here to confirm end-to-end connectivity before enabling RDMA.

```bash
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp
```

### RDMA only (strict)

Maximum performance. Requires working RDMA CM in your environment.

```bash
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3
```

### Mixed TCP + RDMA (recommended)

Uses TCP for connection management, RDMA for data. Most robust in containers.

```bash
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_IB_GID_INDEX=3
```

### Verifying which transport was negotiated

Set `UCX_LOG_LEVEL=info` and look for lane tags in the output:
- `tag(rc_mlx5/...)` → RDMA lane active
- `tag(tcp/ens1f1np1)` → TCP lane only

Or check GVirtuS debug output (`GVIRTUS_LOGLEVEL=10000`):
```
[UCX DEBUG] current_connection_supports_cuda: endpoint=0x... -> RDMA (CUDA-capable)
```

---

## GPUDirect RDMA

When enabled, the NIC writes large payloads directly into GPU memory via
`nvidia-peermem`, eliminating the host-memory bounce copy entirely.

### Requirements

1. `nvidia-peermem` module loaded on the backend:
   ```bash
   sudo modprobe nvidia-peermem
   lsmod | grep nvidia_peermem
   ```

2. UCX built with CUDA support (verify: `ucx_info -d | grep cuda`)

3. RDMA-class transport negotiated (not TCP)

### Enabling GPUDirect

Set on the **backend only**:

```bash
GVIRTUS_GPUDIRECT=1
```

The backend performs a runtime probe at startup:
- Allocates 4 KB of GPU memory (`cudaMalloc`)
- Registers it with UCX (`ucp_mem_map` with `UCS_MEMORY_TYPE_CUDA`)
- If both succeed → GPUDirect enabled, GPU shadow slots allocated

Look for this in backend startup output:
```
[GVS] GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK, ...)
[GVS] rx_pool: initialized 2 slots x 1025 MB (host) + 2/2 GPU shadows x 1025 MB
```

### Per-connection transport gate

GPUDirect is only activated on connections that negotiated an RDMA lane. A
single backend can serve both RDMA and TCP frontends simultaneously — TCP
clients transparently fall back to the host-memory path.

### Performance impact

| Payload | Without GPUDirect | With GPUDirect | Speedup |
|---------|-------------------|----------------|---------|
| 64 MB (N=4096) | 27 ms | 17 ms | 1.6× |
| 256 MB (N=8192) | 119 ms | 79 ms | 1.5× |
| 1 GB (N=16384) | 548 ms | 389 ms | 1.4× |

---

## Environment Variables Reference

### GVirtuS-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `GVIRTUS_GPUDIRECT` | `0` | Set to `1` on backend to enable GPUDirect probe |
| `GVIRTUS_RMA_ZEROCOPY` | `0` | Set to `1` to use zero-copy RMA (requires working UCX rcache) |
| `GVIRTUS_LOGLEVEL` | `20000` | Log verbosity: 0=TRACE, 10000=DEBUG, 20000=INFO, 40000=ERROR |
| `GVIRTUS_DUMP_STATS` | off | Set to `1` to dump per-thread transfer stats on exit |

### UCX transport

| Variable | Example | Description |
|----------|---------|-------------|
| `UCX_TLS` | `rc_mlx5,ud_mlx5,tcp,self` | Allowed transports |
| `UCX_NET_DEVICES` | `mlx5_1:1,ens1f1np1` | Pin to specific NIC(s) |
| `UCX_SOCKADDR_TLS_PRIORITY` | `tcp` | Connection manager (tcp or rdmacm) |
| `UCX_IB_GID_INDEX` | `3` | RoCE v2 GID index |
| `UCX_LOG_LEVEL` | `info` | UCX internal logging |
| `UCX_RCACHE_ENABLE` | `n` | Auto-disabled when GVIRTUS_GPUDIRECT=1 |
| `UCX_MEMTYPE_CACHE` | `n` | Auto-disabled when GVIRTUS_GPUDIRECT=1 |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `The TLs list is empty` | Empty `UCX_TLS` | Set valid transports |
| `device ... is not enabled` | `UCX_NET_DEVICES` mismatch | Check `ibv_devinfo` / `ip addr` |
| `rdma_create_event_channel failed` | RDMA CM unavailable in container | Use `UCX_SOCKADDR_TLS_PRIORITY=tcp` |
| Endpoint timeout | Interface/route mismatch | Pin devices explicitly |
| `GPUDirect=disabled (probe FAILED)` | Missing nvidia-peermem or no GPU | Load module, check `nvidia-smi` |
| `cannot find remote protocol for put from cuda memory` | TCP endpoint + GPUDirect | Per-connection gate handles this automatically |
| Segfault in Frontend during `ucp_init` | libuct_cuda fires CUDA calls before connect | Fixed by reentrancy guard (mpInitialized) |

---

## Architecture Overview

```
Frontend (GPU-less)                   Backend (GPU host)
┌─────────────────┐                  ┌─────────────────────────┐
│ CUDA App        │                  │ GVirtuS Backend         │
│   ↓ intercept   │                  │   Process.cpp           │
│ Frontend.cpp    │                  │   ↓ dispatch            │
│   Execute()     │                  │   CudaRtHandler         │
│   ↓ WriteIov    │                  │   ↑ TryAcquireFrame     │
│ UcxCommunicator │                  │ UcxCommunicator         │
│   ↓             │                  │   ↑                     │
│  [AM: header]   │── RPC control ──▶│  [AM receive handler]   │
│  [RMA: payload] │── RDMA put ─────▶│  [Pinned RX slot]       │
│                 │                  │  [GPU shadow slot]       │
└─────────────────┘                  └─────────────────────────┘
          │                                     │
          └───── UCX (rc_mlx5 / tcp) ───────────┘
```

**Data path for a large transfer (e.g., cudaMemcpy H2D 64 MB):**

1. Frontend assembles iov: `[EnvelopeHeader][routine_name][payload_64MB]`
2. `WriteIovRma` stages header+routine into TX scratch, issues `ucp_put_nbx`
   for the payload directly into the backend's pre-registered RX slot
3. GPUDirect path: payload goes to the GPU shadow slot via NIC peer-DMA
4. Tiny `RmaPosted` AM notifies the backend that the slot is ready
5. Backend's `TryAcquireFrame` returns a pointer into the slot (zero-copy)
6. Handler reads GPU data via `cudaMemcpyDeviceToDevice` (no host bounce)

---

## Further Reading

- [UCX_OPTIMIZATIONS.md](UCX_OPTIMIZATIONS.md) — Detailed optimization stack walkthrough
- [GPUDIRECT.md](GPUDIRECT.md) — GPUDirect implementation phases (B1-B4)
- [UCX_PROPERTIES_GUIDE.md](../UCX_PROPERTIES_GUIDE.md) — Makefile variable reference
- [system-requirements-guide.md](system-requirements-guide.md) — Full system setup
