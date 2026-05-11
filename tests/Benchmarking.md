# GVirtuS UCX Benchmarking Guide

This guide explains how to correctly pair backend and frontend configurations for reproducible benchmarking of GVirtuS transport modes. It covers which backend to run for each benchmark mode, what is actually being measured, and how to interpret results fairly.

***

## Overview of Transport Stacks

GVirtuS supports two distinct communicator stacks. They are **not interchangeable** — a frontend and backend must use the same stack to connect.

| Stack | Backend datapath | Wire protocol | Connection manager |
|---|---|---|---|
| **Plain RDMA** | `GVIRTUS_UCX_DATAPATH=rdma` | Raw ibverbs (rdmacm, RC QPs) | rdmacm |
| **UCX AM** | `GVIRTUS_UCX_DATAPATH=am` | UCX Active Messages | TCP CM or rdmacm |

UCX AM can carry data over TCP-only, RDMA-only, or mixed transports depending on `UCX_TLS` and `UCX_NET_DEVICES`. The underlying wire changes, but the GVirtuS protocol layer is always UCX AM.

***

## Benchmark Modes

| Mode | Frontend stack | UCX_TLS | Config file | What it measures |
|---|---|---|---|---|
| `plain_tcp` | Plain TCP (no UCX) | none | `properties.json` | Raw TCP socket communicator baseline |
| `ucx_tcp` | UCX AM | `tcp,self` | `properties_ucx.json` | UCX AM overhead over TCP |
| `plain_rdma` | Plain RDMA (ibverbs) | `rc_mlx5,ud_mlx5,self` | `properties_plain_rdma.json` | Raw RDMA communicator baseline |
| `ucx_rdma` | UCX AM | `rc_mlx5,ud_mlx5,self` | `properties_ucx.json` | UCX AM over strict RDMA (rdmacm CM) |
| `mixed_rdma` | UCX AM | `rc_mlx5,ud_mlx5,tcp,self` | `properties_ucx.json` | UCX AM, RDMA-preferred with TCP fallback |

> **Note:** `plain_tcp` and `plain_rdma` bypass UCX entirely. `ucx_tcp`, `ucx_rdma`, and `mixed_rdma` all go through UCX AM, so their baselines include UCX framing and connection setup overhead.

***

## Backend Requirements Per Mode

Each benchmark mode requires a specific backend. Running the wrong backend will cause `exit=2` on the frontend.

### Mode: `plain_tcp`

Requires a **plain TCP backend** (legacy, non-UCX). This is not the UCX AM backend.

```bash
cd ~/GVirtuS
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties.json \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev
```

> If your deployment no longer has a tag-framed TCP backend, use `ucx_tcp` as the TCP baseline instead.

***

### Mode: `ucx_tcp`

Requires a **UCX AM backend restricted to TCP**.

```bash
make stop-gvirtus || true
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev
```

Frontend command:

```bash
MATRIX_N=2 ./benchmark.sh ucx_tcp 10
```

***

### Mode: `plain_rdma`

Requires a **plain RDMA backend** (raw ibverbs, not UCX).

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=rdma \
GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
make run-gvirtus-backend-dev
```

Prerequisites:
- `etc/properties_plain_rdma.json` must exist with `suite: "roce-rdma"` and the correct backend IP/port.
- `entrypoint.sh` must be updated to use `${GVIRTUS_CONFIG_FILE:-properties_ucx.json}` instead of a hardcoded path.
- The frontend container must have `/dev/infiniband/rdma_cm` and `/sys/class/infiniband/mlx5_1` accessible.

Frontend command:

```bash
MATRIX_N=2 ./benchmark.sh plain_rdma 10
```

***

### Mode: `ucx_rdma`

Requires a **UCX AM backend with strict RDMA and rdmacm CM**.

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties_ucx.json \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev
```

> Only use this when rdmacm is confirmed working in the container environment. If connection fails with `rdma_create_event_channel failed`, use `mixed_rdma` instead.

Frontend command:

```bash
MATRIX_N=2 ./benchmark.sh ucx_rdma 10
```

***

### Mode: `mixed_rdma`

Requires a **UCX AM backend with mixed TCP+RDMA** (the "universal" backend).

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties_ucx.json \
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev
```

This backend also serves `ucx_tcp` frontends simultaneously. UCX selects the best available transport per connection.

Frontend command:

```bash
MATRIX_N=2 ./benchmark.sh mixed_rdma 10
```

***

## Backend–Frontend Compatibility Matrix

| Frontend mode | plain TCP backend | UCX AM / TCP backend | UCX AM / RDMA backend | UCX AM / mixed backend | plain RDMA backend |
|---|:---:|:---:|:---:|:---:|:---:|
| `plain_tcp` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `ucx_tcp` | ❌ | ✅ | ❌ | ✅ | ❌ |
| `plain_rdma` | ❌ | ❌ | ❌ | ❌ | ✅ |
| `ucx_rdma` | ❌ | ❌ | ✅ | ❌ | ❌ |
| `mixed_rdma` | ❌ | ❌ | ❌ | ✅ | ❌ |

***

## Fair Comparison Guidelines

### TCP vs RDMA baseline

To compare raw TCP vs raw RDMA performance:

- Use `plain_tcp` → plain TCP backend vs `plain_rdma` → plain RDMA backend.
- Both bypass UCX, so results reflect only the communicator and network path.

### UCX overhead measurement

To measure UCX AM overhead over TCP:

- Compare `plain_tcp` (no UCX) vs `ucx_tcp` (UCX AM over TCP) against their respective backends.
- The difference is the UCX framing and connection setup cost.

### RDMA transport comparison

To compare RDMA via UCX vs RDMA via raw ibverbs:

- `plain_rdma` → plain RDMA backend (ibverbs, rdmacm, no UCX).
- `ucx_rdma` or `mixed_rdma` → UCX AM backend (UCX over RC lanes).
- The difference reflects UCX AM overhead on top of the RDMA fabric.

> Do **not** run a TCP frontend against a UCX AM backend and call the result a "TCP baseline" — UCX AM adds framing overhead that does not exist in a plain TCP communicator.

***

## Verifying Which Transport Was Actually Used

Set `UCX_LOG_LEVEL=info` on both backend and frontend, then inspect `ep_cfg` lines in the output.

- RDMA lane in use: `tag(rc_mlx5/mlx5_1:1/...)`
- TCP lane in use: `tag(tcp/ens1f1np1/...)`

If only `tag(tcp/...)` appears when RDMA was expected, the RDMA device or rdmacm was not available in the container and UCX fell back to TCP.

***

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `exit=2`, all modes | Wrong backend for the mode, or backend not reachable | Check backend–frontend compatibility matrix above |
| `exit=2`, RDMA modes only | rdmacm unavailable in container, or wrong properties file | Verify `/dev/infiniband/rdma_cm` and `properties_plain_rdma.json` exists |
| UCX falls back to TCP silently | `frontend.sh` TCP fallback guard triggered | Check `/sys/class/infiniband/mlx5_1` exists in container |
| Backend shows `Called Accept()` but frontend fails | Protocol mismatch (e.g. plain RDMA frontend vs UCX AM backend) | Use correct backend per mode table |
| `The TLs list is empty` | `UCX_TLS` set to empty string | Set a valid value or unset it entirely |
| `device ... is not enabled` | `UCX_NET_DEVICES` doesn't match device visible to UCX | Run `ucx_info -d` inside the container to see available devices |

***

## Recommended Benchmarking Workflow

1. **Confirm connectivity** — run `ucx_tcp` mode first (simplest UCX path, most robust in containers).
2. **Establish UCX TCP baseline** — run `ucx_tcp` for 10+ iterations, discard first 1–2 warm-up runs.
3. **Test mixed RDMA** — switch to `mixed_rdma` against the universal backend; verify `rc_mlx5` lanes appear in UCX logs.
4. **Test strict RDMA via UCX** — if rdmacm is confirmed working, run `ucx_rdma` for a direct UCX+RDMA measurement.
5. **Test plain RDMA** — run `plain_rdma` against the plain RDMA backend for an ibverbs-only baseline.
6. **Compare** — use the CSV outputs from `benchmark.sh` to compare `elapsed_ms` across modes at the same `MATRIX_N`.