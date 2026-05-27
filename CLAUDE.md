# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GVirtuS (GPU Virtualization Service) transparently virtualizes NVIDIA CUDA GPU resources over a network. Applications on GPU-less machines link against a frontend stub that intercepts CUDA API calls, serializes them, and forwards them to a backend host with a physical GPU.

**Call flow:** Application → Frontend (stub) → Communicator (TCP/RDMA/UCX) → Backend → Plugin (.so) → real CUDA GPU

## Build & Run Commands

All development uses Docker. The dev container builds from source on start (cmake + make), then launches the backend.

```bash
# Start backend (builds inside container, mounts local source)
make run-gvirtus-backend-dev

# Run tests (inside already-running container, uses ctest)
make run-gvirtus-tests

# Attach bash to running container
make attach-gvirtus-bash

# Stop the backend container
make stop-gvirtus

# Discover available UCX transports/devices (host or container)
make ucx-discover
make ucx-discover-docker
```

### Manual build (inside container or native Linux)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
make install
$GVIRTUS_HOME/bin/gvirtus-backend $GVIRTUS_HOME/etc/properties.json
```

### Running a single test

Tests are registered with ctest individually. Inside the container:
```bash
cd /gvirtus/build
ctest -R test_cudart --output-on-failure   # run only cudart tests
ctest -R test_cublas --output-on-failure   # run only cublas tests
```

### Selecting a transport / backend config

The entrypoint defaults to `properties_ucx.json`. Override via:
```bash
BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties.json make run-gvirtus-backend-dev  # TCP
BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties_ucx.json make run-gvirtus-backend-dev  # UCX (default)
```

UCX transport tuning lives in `etc/ucx.env` (loaded by the Makefile). Edit that file to switch presets (TCP-only, RDMA-only, mixed). Command-line overrides work too:
```bash
UCX_TLS=tcp,self make run-gvirtus-backend-dev
```

### Log verbosity

Set `GVIRTUS_LOGLEVEL` env var before launching (0=TRACE, 10000=DEBUG, 20000=INFO default, 40000=ERROR, 60000=OFF):
```bash
GVIRTUS_LOG_LEVEL=10000 make run-gvirtus-backend-dev
```

## Architecture

### Core libraries (src/)

- **gvirtus-frontend** (`src/frontend/`): Per-thread singleton linked by client apps. Serializes CUDA call name + arguments into a `Buffer`, sends via communicator, blocks for result.
- **gvirtus-backend** (`src/backend/`): Reads a properties JSON, forks one `Process` per endpoint. Each `Process` accepts connections and dispatches to the matching plugin via `dlopen`.
- **gvirtus-communicators** (`src/communicators/`): Pluggable transport layer. Built as separate .so libraries per transport: `gvirtus-communicators-tcp`, `gvirtus-communicators-ib` (RDMA), `gvirtus-communicators-hybrid`, `gvirtus-communicators-ucx`. Selected at runtime by `CommunicatorFactory` based on endpoint `suite` in properties JSON.
- **gvirtus-common** (`src/common/`): Shared utilities (Encoder/Decoder, JSON config, dynamic library loading, message dispatch).

### Communicator interface (`include/gvirtus/communicators/Communicator.h`)

Base virtual methods: `Serve()`, `Accept()`, `Connect()`, `Read()`, `Write()`, `Sync()`, `Close()`.

Extended for high-performance transports:
- `WriteIov()` — gather-send (avoids staging large payloads into contiguous buffer)
- `TryAcquireFrame()` / `ReleaseFrame()` — zero-copy frame handoff for message-oriented transports
- `current_frame_gpu()` — exposes GPU-resident payload (GPUDirect)
- `current_connection_supports_cuda()` — per-connection RDMA capability gate

### Plugin system (plugins/)

Each plugin is a shared library wrapping one CUDA sub-API. Plugins have:
- `backend/` — Handler class with a `map<string, RoutineHandler>` dispatching CUDA function names to implementations. Exports `create_t()` factory.
- `frontend/` — Stub library that mimics the real CUDA .so (versioned soname via linker script). Intercepts API calls and forwards via `Frontend::Execute()`.

Plugins: cudart, cudadr, cublas, cudnn, cufft, curand, cusolver, cusparse, nvrtc, nvml.

The Handler interface (`include/gvirtus/backend/Handler.h`) requires:
- `bool CanExecute(std::string routine)` — check if this plugin handles the named routine
- `shared_ptr<Result> Execute(std::string routine, shared_ptr<Buffer> input_buffer)` — execute it

### Adding a new CUDA function to a plugin

1. Add the backend handler in `plugins/<lib>/backend/CudaXxxHandler_<category>.cpp` — register a function pointer in the handler map.
2. Add the frontend stub in `plugins/<lib>/frontend/CudaXxx_<category>.cpp` — serialize args into a Buffer, call `Frontend::Execute("<routine_name>")`, deserialize result.
3. Add a test in `tests/test_<lib>.cu` using GoogleTest (`CUDA_CHECK` macro pattern).
4. Add the new test file to `tests/CMakeLists.txt` `TEST_SOURCES` list if creating a new file.

## Configuration

`etc/properties*.json` files define endpoints (transport suite, address, port) and which plugins to load:
- `properties.json` — TCP transport
- `properties_ucx.json` — UCX transport (default in dev container)
- `properties_hybrid.json` — Hybrid (TCP control + RDMA data)
- `properties_plain_rdma.json` — Raw RDMA/RoCEv2

`etc/ucx.env` — UCX environment variables (transport selection, device binding, GVirtuS options like `GVIRTUS_UCX_DATAPATH`, `GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`).

## Key Conventions

- C++23 standard (set in CMakeLists.txt).
- Logging via **log4cplus** — never printf/cout for operational output.
- Plugin handler functions follow: `cuda<LibName>_<functionName>`.
- Tests use GoogleTest with CUDA (.cu files). Each test file becomes a separate ctest target.
- Do NOT modify plugin files unless the task explicitly requires it.

## Hardware Context

Development targets: NVIDIA L40s GPU, Mellanox BlueField-3 NIC (RoCEv2), Ubuntu 22.04, CUDA 12.6.3, cuDNN 9.5.1.
