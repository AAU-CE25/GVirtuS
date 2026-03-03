# GitHub Copilot Instructions

## Project overview

This is **GVirtuS** (Generic Virtualization Service) — a framework that virtualizes NVIDIA CUDA GPU resources over a network so remote machines without a GPU can still run CUDA workloads.

The `ce2-group-1/` folder is a course exercise that explores **RDMA (Remote Direct Memory Access)** communication using Mellanox BlueField-3 DPUs (Data Processing Units) and the `libibverbs` API.

## Hardware context

- **NIC**: Mellanox BlueField-3 (PCI 82:00) — the RDMA-capable device, `mlx5_2`
- **NIC**: Mellanox ConnectX-6 Lx (PCI 02:00) — secondary device
- **RDMA mode**: RoCEv2 (GID index 3)
- **IB port**: 1

## Network topology

Two HPC nodes connected by multiple dedicated links:

| Machine | Interface | IP | Subnet | Purpose |
|---------|-----------|-----|--------|---------|
| `es-dpu-01` | `ens7f0np0` | `24.24.24.1` | `/24` | dedicated link A |
| `es-dpu-01` | `ens7f1np1` | `25.25.25.1` | `/24` | dedicated link B (MPI/RDMA) |
| `es-dpu-01` | `bond0`     | `172.19.8.95` | `/24` | campus/management |
| `es-dpu-02` | —           | `25.25.25.3` | `/24` | dedicated link B (MPI/RDMA) |

- **Preferred link for experiments**: `25.25.25.x` (`ens7f1np1`) — point-to-point, lowest latency, no campus traffic
- **`/etc/hosts` on es-dpu-01** already maps `es-dpu-02 → 25.25.25.3` and `bf3b-mpi → 25.25.25.3`, so hostnames resolve without DNS
- The OS routing table automatically sends `25.25.25.x` traffic out of `ens7f1np1` — no manual interface binding needed in userspace TCP code
- `tmfifo_net0` (`192.168.100.1`) is the BlueField-3 DPU management interface — do not use for data-plane experiments
- `ens4f0–3` are **DOWN** — not connected, do not use

## Language & style

- **C++17** — use `constexpr`, `[[nodiscard]]`, structured bindings, range-for, `std::optional` where appropriate
- Prefer `std::string_view` over raw `const char*` for read-only strings
- Use `ibv_*` C API functions directly — do not wrap them in extra abstraction unless asked
- Error handling: call `die(msg)` for fatal RDMA errors (defined in `rdma_common.h`)
- All shared RDMA helpers live in `rdma_common.h` (header-only, inline functions)
- Keep server and client files minimal — logic belongs in `rdma_common.h`

## Build system

- `ce2-group-1/Makefile` — builds `ce-server` and `ce-client` with `g++ -std=c++17`
- Root `CMakeLists.txt` — builds the full GVirtuS framework
- Link flag for RDMA code: `-libverbs`

## Common patterns

```cpp
// Open RDMA device
ibv_context *ctx = open_device();          // opens DEVICE_NAME ("mlx5_2")

// Allocate resources (PD, MR, CQ, QP)
RdmaRes r;
r.ctx = ctx;
alloc_resources(r);

// Exchange ConnInfo over TCP, then:
connect_qp(r, remote_info);               // INIT -> RTR -> RTS

// Send / receive
post_recv(r);   // always post recv before send
post_send(r);
poll_cq(r);     // busy-poll until completion
```

## Key files

| Path | Purpose |
|------|---------|
| `ce2-group-1/` | Folder for learing the CPP basics (use for example exercises) |
| `src/` | GVirtuS core source (frontend, backend, communicators) |
| `plugins/` | CUDA plugin implementations (cudart, cublas, cudnn, …) |

## GVirtuS core source structure (`src/`)

The `src/` directory is split into four layers:

```
CUDA Application
      │  intercepted API call
      ▼
src/frontend/          — per-thread shim; serialises call + args → Buffer, sends, reads result
      │  transport (TCP / RDMA / SHM / …)
      ▼
src/communicators/     — pluggable transport layer (Communicator interface + Buffer)
      │
      ▼
src/backend/           — GPU-side daemon; forks one Process per endpoint, dispatches to plugins
      │  dlopen plugin .so
      ▼
plugins/               — CUDA API implementations (cudart, cublas, cudnn, …)
```

### `src/frontend/`
- `Frontend` is a **per-thread singleton** (`std::map<pthread_t, Frontend*>`, mutex-protected).
- On first use reads config from `GVIRTUS_CONFIG`, `$GVIRTUS_HOME/etc/properties.json`, or `./properties.json`.
- Creates an `Endpoint` + `Communicator`, calls `Connect()`, allocates three `Buffer`s (`mpInputBuffer`, `mpOutputBuffer`, `mpLaunchBuffer`).
- `Execute()`: write input buffer → read output buffer → decode return value.
- Opt-in stats: `GVIRTUS_DUMP_STATS=on`; log level: `GVIRTUS_LOGLEVEL=<log4cplus int>`.

### `src/backend/`
- `Backend` reads `properties.json`, creates one `Process` per endpoint, calls `fork()` for each.
- `Process` runs the accept loop; spawns a `std::thread` per client; reads function name from Buffer, looks it up in the plugin handler map via `LD_Lib`, writes result back.
- `Property` is a typed wrapper around the JSON config (endpoints, plugin paths, secure flag).

### `src/communicators/`
All transports implement `Communicator` (`Read`, `Write`, `Connect`, `Serve`, `Accept`, `Sync`).

| Transport | Class |
|-----------|-------|
| TCP | `tcp/TcpCommunicator` |
| RDMA (RoCE/IB) | `rdma/RdmaCommunicator` — uses `rdma_cm` / `libibverbs` |
| AF_UNIX | `AfUnixCommunicator` |
| Shared memory | `ShmCommunicator` / `VMShmCommunicator` |
| ZeroMQ | `ZmqCommunicator` |
| VMCI / vsock | `VmciCommunicator` / `VMSocketCommunicator` |
| Virtio serial | `VirtioCommunicator` |
| Hybrid (control+data) | `hybrid/HybridCommunicator` |

`CommunicatorFactory` + `EndpointFactory` pick the implementation from `properties.json` at runtime.

`Buffer` is a heap-grown byte buffer with `Add<T>()` (append) and `Get<T>()` (consume) for POD types; it is the sole serialisation container passed between all layers.

### `src/common/`
| File | Purpose |
|------|---------|
| `Encoder` / `Decoder` | Base64-style codec for text-safe binary transfer |
| `JSON<T>` | Template: parse `properties.json` → typed config object |
| `LD_Lib` | RAII `dlopen`/`dlsym` — loads plugin `.so` files |
| `MessageDispatcher` | Observer-based function-name → handler router |
| `Observable` / `Observer` | Classic observer pattern (Process ↔ plugin handlers) |
| `Mutex` | RAII `pthread_mutex_t` wrapper |
| `SignalException` / `SignalState` | Converts POSIX signals to C++ exceptions for clean shutdown |

### `properties.json` (config)
```json
{
  "endpoints": [{ "communicator": "tcp", "server_address": "…", "server_port": 9999 }],
  "plugins": [["libCudaRt.so"]],
  "secure": false
}
```
Both frontend and backend read the same file. Change `"communicator"` to `"rdma"`, `"unix"`, `"shm"`, `"zmq"`, or `"hybrid"` to switch transport.

## What NOT to do

- Do not use `using namespace std;` globally
- Do not hardcode IP addresses — accept them as CLI arguments
- Do not close the RDMA device before destroying all QPs/CQs/MRs/PDs
- Do not poll indefinitely without a timeout in production code
