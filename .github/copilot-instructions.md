# GitHub Copilot Instructions

## ⚠️ STANDING RULE — ALWAYS CLEAN UP STALLED PROCESSES / ZOMBIES / CONTEXTS / CONTAINERS WHILE TESTING/BENCHMARKING

Leftover state silently corrupts results and makes runs look "slow"/"hung" when the code is fine.
**After every test/benchmark — especially after any `timeout`, crash, Ctrl-C, or interrupted run —
clean up before the next run:**
- Kill stalled/zombie frontend processes (they hold connections + GPU memory and clog the backend):
  `docker exec <fe-container> bash -lc 'for p in $(pgrep -f "llama|simple_matrix|<bin>"); do kill -9 $p; done'`.
  A `timeout`-killed client usually leaves the process and its backend-side connection alive.
- Restart the backend fresh between transport/GPUDirect phases and after any client crash/kill
  (`docker rm -f <backend>` then relaunch) — a crashed CUDA context poisons the persistent backend.
- Verify GPU memory is actually freed on BOTH nodes (`nvidia-smi --query-gpu=memory.used --format=csv,noheader`)
  before trusting a run; orphaned containers can leak many GB and starve GPUDirect (symptoms:
  `cudaMalloc(4K) failed`, stalls, exit-134). Non-root `kill` can't reap root-owned container procs — use `docker rm -f`.
- NEVER diagnose a "regression"/"slowness" before ruling out leftover state. A multi-minute "hang"
  was repeatedly just accumulated zombies + a poisoned backend; cleanup returned the run to seconds.
  Clean first, measure second. Confirm a clean slate (`docker ps` / `pgrep`) before each measurement.

## Project Overview

**GVirtuS** (Generic Virtualization Service) is a framework that transparently virtualizes NVIDIA CUDA GPU resources over a network. Applications on GPU-less machines link against a GVirtuS frontend stub that intercepts CUDA API calls, serializes them, and forwards them to a backend host with a physical GPU.

### Architecture (simplified)

```
Application → Frontend (stub) → Communicator (TCP/RDMA) → Backend → Plugin (.so) → real CUDA GPU
```

- **Frontend** (`src/frontend/`): Per-thread singleton. Serializes CUDA call name + args into a `Buffer`, sends via `Communicator::Write()`, blocks for result.
- **Backend** (`src/backend/`): Reads `properties.json`, forks one `Process` per endpoint. Each `Process` accepts connections, dispatches to the matching Plugin via `dlopen`.
- **Communicator** (`src/communicators/`): Pluggable transport layer (TCP, RDMA, Hybrid, etc.). Selected at runtime by `CommunicatorFactory` based on the endpoint `suite` in `properties.json`.
- **Plugins** (`plugins/`): One shared library per CUDA sub-API (cudart, cublas, cudnn, cufft, curand, cusolver, cusparse, nvrtc, nvml). Each executes the real CUDA call and serializes the return value.

> **Important**: Do NOT modify files under `plugins/` unless the task explicitly requires it.

---

## Hardware & Network Setup

| Machine | Interface | IP | Purpose |
|---------|-----------|-----|---------|
| `es-dpu-01` (GPU server) | `ens7f0np0` | `24.24.24.1/24` | GVirtuS backend listening address |
| `es-dpu-01` | `ens7f1np1` | `25.25.25.1/24` | Dedicated RDMA link |
| `es-dpu-01` | `bond0` | `172.19.8.95/24` | Campus/management |
| `es-dpu-02` (client) | — | `25.25.25.3/24` | Dedicated RDMA link |

- **GPU**: NVIDIA L40s (on `es-dpu-01`)
- **NIC (RDMA)**: Mellanox BlueField-3 (`mlx5_2`, PCI 82:00), RoCEv2, GID index 3, IB port 1
- **NIC (secondary)**: Mellanox ConnectX-6 Lx (PCI 02:00)

---

## Configuration

### `etc/properties.json`

```json
{
  "communicator": [{
    "endpoint": {
      "suite": "tcp/ip",
      "protocol": "tcp",
      "server_address": "24.24.24.1",
      "port": "2222"
    },
    "plugins": ["cuda","cudart","cublas","curand","cudnn","cufft","cusolver","cusparse","nvrtc","nvml"]
  }],
  "secure_application": false
}
```

- `server_address` — IP the backend binds to (must match a real interface on the server node).
- `port` — TCP port for the communicator.
- `plugins` — list of CUDA sub-APIs this endpoint exposes.

---

## Building & Running

### Dev container (Docker)

```bash
# Build and start the backend (builds inside container, then runs)
make run-gvirtus-backend-dev

# Attach a bash shell to the running container
make attach-gvirtus-bash

# Run unit tests inside the container
make run-gvirtus-tests

# Stop the container
make stop-gvirtus
```

- Docker image: `aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04`
- Source directories are bind-mounted, so code changes on the host are reflected in the container.
- The entrypoint (`docker/dev/entrypoint.sh`) does: `cmake .. && make -j$(nproc) && make install`, then runs the backend.

### Manual build (inside container or native)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
make install
$GVIRTUS_HOME/bin/gvirtus-backend $GVIRTUS_HOME/etc/properties.json
```

---

## Logging

GVirtuS uses **log4cplus** (not log4cpp). The backend configures the root logger in `src/backend/main.cpp`.

### Log levels

| Level | Value | Use |
|-------|-------|-----|
| TRACE | 0 | Internal details (per-byte reads, getstring) |
| DEBUG | 10000 | Routine-level operation flow |
| INFO  | 20000 | Startup, client connect/disconnect (default) |
| WARN  | 30000 | Recoverable issues |
| ERROR | 40000 | Failures |
| FATAL | 50000 | Unrecoverable |
| OFF   | 60000 | Silent |

### Setting the log level

```bash
# Via Makefile (Makefile line 5: GVIRTUS_LOG_LEVEL)
GVIRTUS_LOG_LEVEL=0 make run-gvirtus-backend-dev    # TRACE

# Or directly via Docker
docker run -e GVIRTUS_LOGLEVEL=10000 ...             # DEBUG
```

The env var is `GVIRTUS_LOGLEVEL` (read by both backend and frontend). Default is `20000` (INFO).

### Log format

```
%D{%Y-%m-%d %H:%M:%S.%q} [%-5p] [%c] (%F:%L) - %m%n
```

Example output:
```
2025-06-10 14:30:01.123 [INFO ] [Backend] (Backend.cpp:85) - Setting up endpoint 1/1
2025-06-10 14:30:01.456 [INFO ] [TcpCommunicator] (TcpCommunicator.cpp:42) - Client connected from 25.25.25.3:54321
```

### Named loggers

| Logger name | Where | Typical use |
|---|---|---|
| `GVirtuS` | `main.cpp` | Root/startup messages |
| `Backend` | `Backend.cpp` | Endpoint setup |
| `Process` | `Process.cpp` | Client dispatch, connect/disconnect |
| `Process.getstring` | `Process.cpp` (free function) | Buffer decoding (TRACE) |
| `TcpCommunicator` | `TcpCommunicator.cpp` | TCP transport events |

---

## Source Layout

```
src/
├── backend/         # Backend entry point (main.cpp), Backend class, Process class
├── common/          # Shared utilities (Buffer, MessageHandler, LD_Lib)
├── communicators/   # Transport implementations
│   ├── tcp/         # TcpCommunicator (primary)
│   └── ...          # RDMA, Hybrid, etc.
└── frontend/        # Frontend stub linked by client apps
plugins/             # One directory per CUDA sub-API (DO NOT MODIFY without explicit task)
etc/                 # properties.json configuration
tests/               # Unit tests (.cu files, run via ctest)
docker/dev/          # Dev Dockerfile + entrypoint.sh
```

---

## Key Coding Conventions

- **C++17** standard.
- **log4cplus** for all logging — never use `printf`/`cout` for operational output.
- Log levels: INFO for user-visible events (connections, startup), DEBUG for routine flow, TRACE for internals.
- Plugin handler functions follow the naming pattern: `cuda<LibName>_<functionName>` (e.g., `cudart_cudaMalloc`).
