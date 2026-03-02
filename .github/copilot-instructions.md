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
| `ce2-group-1/rdma_common.h` | Shared RDMA helpers (device open, QP setup, send/recv) |
| `ce2-group-1/ce-server.cpp` | RDMA server — receives message |
| `ce2-group-1/ce-client.cpp` | RDMA client — sends message |
| `ce2-group-1/step1/helloHPC.cpp` | Simple C++ hello-world (intro exercise) |
| `cpp_first_steps/step2-ping-cpp/helloServer.cpp` | TCP ping server — run on es-dpu-02 |
| `cpp_first_steps/step2-ping-cpp/helloClient.cpp` | TCP ping client — run on es-dpu-01, target es-dpu-02 |
| `src/` | GVirtuS core source (frontend, backend, communicators) |
| `plugins/` | CUDA plugin implementations (cudart, cublas, cudnn, …) |

## What NOT to do

- Do not use `using namespace std;` globally
- Do not hardcode IP addresses — accept them as CLI arguments
- Do not close the RDMA device before destroying all QPs/CQs/MRs/PDs
- Do not poll indefinitely without a timeout in production code
