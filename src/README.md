# GVirtuS — Source Directory

This directory contains the core C++ source code for the GVirtuS framework. GVirtuS virtualizes NVIDIA CUDA GPU resources over a network so that remote machines without a physical GPU can transparently execute CUDA workloads.

## Folder Overview

| Folder | Responsibility |
|--------|---------------|
| [`backend/`](backend/) | GPU-side daemon — accepts connections, dispatches CUDA calls via plugin system |
| [`common/`](common/) | Shared utilities — serialization (Encoder/Decoder), JSON config parsing, signal handling, observer pattern |
| [`communicators/`](communicators/) | Transport layer — TCP, RDMA, UNIX socket, shared memory, ZMQ, virtio, hybrid transports; plus the `Buffer` abstraction used on both sides |
| [`frontend/`](frontend/) | Client-side shim — intercepts CUDA API calls from the application and forwards them to the backend over the configured transport |

## Request Flow (high level)

```
CUDA Application
      │  calls CUDA API (e.g. cudaMemcpy)
      ▼
  Frontend (frontend/)
      │  serialises function id + arguments into a Buffer
      │  sends over chosen Communicator (TCP / RDMA / …)
      ▼
  Backend (backend/)
      │  receives Buffer, looks up function in loaded Plugin
      ▼
  Plugin (plugins/cudart, plugins/cublas, …)
      │  executes real CUDA call on local GPU
      │  serialises result back into a Buffer
      ▼
  Backend  ──transport──▶  Frontend  ──return value──▶  Application
```

See [architecture.md](architecture.md) for a detailed description of every component.
