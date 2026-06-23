# GVirtuS UCX Properties Guide

This guide explains the UCX-related Makefile properties in this repository and how to run each common UCX scenario:

1. TCP only
2. RDMA only (strict)
3. TCP + RDMA (mixed)

It also explains how to choose the correct lane/card and how to verify which lane was actually used.

## Where these properties live

The properties below are Make variables in `Makefile` (not `etc/properties_ucx.json`).

`etc/properties_ucx.json` provides endpoint address/port. The Make variables provide UCX transport behavior.

## UCX properties and responsibility

- `GVIRTUS_UCX_DATAPATH`
  - Controls UCX datapath mode for GVirtuS request handling.
  - Typical value here: `am`.

- `UCX_TLS`
  - Allowed UCX transports.
  - Examples:
    - `tcp,self`
    - `rc_mlx5,ud_mlx5,self`
    - `rc_mlx5,ud_mlx5,tcp,self`

- `UCX_NET_DEVICES`
  - Restricts UCX to specific device(s).
  - TCP device format: `ens1f1np1`
  - RDMA device format: `mlx5_1:1`
  - Mixed format: `mlx5_1:1,ens1f1np1`

- `UCX_SOCKADDR_TLS_PRIORITY`
  - Connection-manager preference for endpoint setup.
  - Typical values:
    - `tcp` (usually robust in containers)
    - `rdmacm` (strict RDMA CM)

- `UCX_LOG_LEVEL`
  - UCX internal logging level (for lane diagnostics).

- `GVIRTUS_LOG_LEVEL`
  - GVirtuS application logging level.

- `SIMPLE_MATRIX_GPU_FLAGS`
  - Optional GPU request for frontend test container.
  - Keep empty on GPU-less client nodes.

## Baseline endpoint config

Set `etc/properties_ucx.json` to backend node IP/port for the target network.

Example dedicated inter-node link:

```json
"server_address": "25.25.25.1",
"port": "32222"
```

## Scenario 1: TCP only

Use this first to validate protocol and connectivity.

Backend node:

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend node:

```bash
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-simple-matrix-test
```

Expected: matrix output succeeds and no endpoint timeout.

## Scenario 2: RDMA only (strict)

Use this only when RDMA/rdmacm container setup is known-good.

Backend node:

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend node:

```bash
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=info \
make run-simple-matrix-test
```

If this fails with rdmacm/device errors, your environment is not currently strict-RDMA-ready in container mode.

## Scenario 3: TCP + RDMA (mixed)

Useful when strict rdmacm is unstable but you still want UCX to consider RDMA lanes.

Backend node:

```bash
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend node:

```bash
GVIRTUS_UCX_DATAPATH=am \
UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=info \
make run-simple-matrix-test
```

## How to choose the correct lane/card

### 1) Choose network path/IP pair

Find the dedicated inter-node link addresses:

```bash
ip -o -4 addr show
```

Choose backend `server_address` from that link.

### 2) Map link IP to RDMA device name

On each node:

```bash
show_gids | grep 25.25.25
```

Example mapping:

- backend 25.25.25.1 -> `mlx5_1:1`, gid index 3 (RoCE v2)
- frontend 25.25.25.3 -> `mlx5_1:1`, gid index 3 (RoCE v2)

### 3) Set `UCX_NET_DEVICES`

- TCP only: use NIC name, e.g. `ens1f1np1`
- RDMA only: use RDMA name, e.g. `mlx5_1:1`
- Mixed: provide both, e.g. `mlx5_1:1,ens1f1np1`

## How to verify which lane was actually used

Use UCX info logs (`UCX_LOG_LEVEL=info`) and inspect `ep_cfg` lines.

- TCP lane evidence:
  - `tag(tcp/ens1f1np1)`

- RDMA lane evidence:
  - `tag(rc_mlx5/...)` or another mlx5 RDMA tag lane

If successful run only shows `tag(tcp/...)`, then that run was UCX-over-TCP, not strict RDMA.

## Fast troubleshooting map

- `The TLs list is empty`
  - `UCX_TLS` was set to empty string. Set a valid value or unset it.

- `device ... is not enabled`
  - `UCX_NET_DEVICES` does not match what UCX sees on that node/container.

- `rdma_create_event_channel failed` / rdmacm connect failures
  - Strict RDMA CM path is not available in current environment.
  - Try mixed profile (`UCX_SOCKADDR_TLS_PRIORITY=tcp`) first.

- Endpoint timeout on first response
  - Usually lane/interface mismatch; pin devices and CM explicitly.

- Frontend segfault after transport failure
  - Secondary symptom after UCX connection failure; fix transport setup first.

## Recommended workflow

1. Start with TCP-only and confirm end-to-end stability.
2. Move to mixed profile to validate broader UCX setup.
3. Attempt strict RDMA only after environment prerequisites are confirmed.
