# GitHub Copilot Instructions

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

See [src/architecture.md](src/architecture.md) for the full component breakdown.

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

---

## Spark RAPIDS Integration with GVirtuS

### Overview

GVirtuS can virtualise the GPU for **Apache Spark with RAPIDS Accelerator**. The Spark executor/driver loads RAPIDS cuDF native libraries (`libcudf.so`, `libcudfjni.so`) which in turn `dlopen()` CUDA libraries. By hiding the real CUDA libraries and making the GVirtuS frontend stubs the only ones visible to the linker, all CUDA calls are transparently forwarded to the remote backend.

### Example: `examples/spark_simple_matrix/`

| Component | Version |
|-----------|---------|
| Base image | `nvidia/cuda:12.6.3-cudnn-{devel,runtime}-ubuntu22.04` |
| Spark | 3.5.2 |
| Scala | 2.12.18 |
| RAPIDS Accelerator | 26.02.1 (`rapids-4-spark_2.12-26.02.1.jar`) |
| cuDF | 26.02.0 |
| Java | 17 |
| Python | 3.10 |

**Makefile targets:**

```bash
make docker-build-spark          # Build the Spark+RAPIDS+GVirtuS Docker image
make run-spark-gvirtus           # Run benchmark via GVirtuS (no --runtime=nvidia)
make test-spark-gvirtus          # Run the 16-call GVirtuS connectivity test
make run-spark-docker-rapids     # Run with real GPU (--runtime=nvidia)
```

### CUDA Library Hiding Technique

The NVIDIA base Docker image ships real CUDA libraries in `/usr/local/cuda/lib64/` (and `/usr/local/cuda/targets/x86_64-linux/lib/`). RAPIDS cuDF JNI's native libraries (`libcudf.so`, `libcudfjni.so`) have **no** static `NEEDED` entries for `libcudart.so` — they `dlopen()` it at runtime. The dynamic linker resolves `libcudart.so.12` to the real CUDA libs unless we intervene.

**Solution (implemented in `entrypoint.sh`):**

1. **Move** real CUDA libs (`libcudart.so*`, `libcuda.so*`, `libcublas.so*`, etc.) to `real-backup/` subdirectories.
2. **Create symlinks** in the GVirtuS frontend directory (`/opt/GVirtuS/lib/frontend/`) — unversioned (`libcudart.so`) pointing to versioned stubs (`libcudart.so.12`).
3. **Rebuild ldconfig** cache with GVirtuS frontend dir first (`/etc/ld.so.conf.d/gvirtus.conf`).
4. **Set `LD_LIBRARY_PATH`** to `${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib:${NATIVE_DIR}`.

### Critical: Do NOT use LD_PRELOAD

**`LD_PRELOAD` must NOT be used with GVirtuS in JVM/Spark environments.**

The GVirtuS Frontend is a **per-thread singleton** whose constructor opens a TCP connection to the backend. With `LD_PRELOAD`:
- Every process (JVM, `ls`, `grep`, shell scripts) gets the GVirtuS stub forced into it.
- The constructor fires immediately, creating a TCP connection that sends 0 bytes and disconnects.
- The backend sees spurious `Client connected` → `Read() returned 0 bytes` → `Client disconnected` sequences.
- This interferes with JVM startup and causes Spark to hang during `SparkSession.builder.getOrCreate()`.

**Correct approach:** Hide real CUDA libs + set `LD_LIBRARY_PATH` + rebuild ldconfig. When cuDF's native code does `dlopen("libcudart.so.12")`, it finds the GVirtuS stub through the normal library search path. The Frontend singleton is only created when a CUDA function is actually called, in the correct thread context.

### RAPIDS Native Library Extraction

The RAPIDS JAR contains native `.so` files under `amd64/Linux/`. These must be extracted to a directory on `LD_LIBRARY_PATH`:

```bash
NATIVE_DIR="/tmp/rapids-native"
unzip -q -j "$RAPIDS_JAR" "amd64/Linux/*.so" -d "$NATIVE_DIR"
# Create versioned symlinks for nvcomp
ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.5"
ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.4"
```

### GVirtuS Frontend Stubs Available

| Stub Library | Soname | CUDA Sub-API |
|---|---|---|
| `libcudart.so.12` | CUDA Runtime | Core runtime (malloc, memcpy, launch, etc.) |
| `libcuda.so.1` | CUDA Driver | Driver API (context, module, memory) |
| `libcublas.so.12` | cuBLAS | BLAS operations |
| `libcufft.so.12` | cuFFT | FFT operations |
| `libcurand.so.12` | cuRAND | Random number generation |
| `libcusolver.so.12` | cuSOLVER | Linear algebra solvers |
| `libcusparse.so.12` | cuSPARSE | Sparse matrix operations |
| `libcudnn.so.9` | cuDNN | Deep neural networks |

---

## CUDA Version Compatibility

### Current baseline: CUDA 12.6.3

The container base images and GVirtuS plugins are built against **CUDA 12.6.3**. The host GPU driver (580.95.05, CUDA 13.0) is forward-compatible — CUDA 12.x applications work on a 13.x driver.

### CUDA 13.0 Breaking Changes (known)

If upgrading to CUDA 13.0 base images in the future, these GVirtuS plugin files need fixes:

| File | Issue |
|---|---|
| `plugins/cudadr/backend/CudaDrHandler_context.cpp` | `cuCtxCreate` renamed to `cuCtxCreate_v4` with different parameters |
| `plugins/cudart/backend/CudaRtHandler_stream.cpp` | `cudaStreamGetCaptureInfo` parameter signature changed |
| `plugins/cudart/backend/CudaRtHandler_thread.cpp` | `cudaThreadExit` and `cudaThreadSynchronize` removed |
| `plugins/cudart/backend/CudaRtHandler_execution.cpp` | `cudaSetDoubleForDevice` and `cudaSetDoubleForHost` removed |
| `plugins/cudart/demo/matrixMul.cu` | `cudaDeviceProp::computeMode` removed (already fixed with `#if defined` guard) |

> These are NOT applied yet — only relevant when upgrading from CUDA 12.x to 13.0+.

---

## Debugging Tips

### Diagnosing `cudaErrorInsufficientDriver`

This error means the CUDA runtime found a GPU driver that is too old for the runtime version, **or** no driver at all.
- **With `--runtime=nvidia`**: Check driver/runtime version mismatch (`nvidia-smi` vs container CUDA version).
- **Without `--runtime=nvidia` (GVirtuS)**: Means the real `libcudart.so` is being loaded instead of the GVirtuS stub. Verify with `ldconfig -p | grep cudart` inside the container.

### Verifying GVirtuS stubs are loaded

```bash
# Inside the container, check what ldconfig resolves:
ldconfig -p | grep -E "libcudart|libcuda\.so"
# Should show paths under /opt/GVirtuS/lib/frontend/, NOT /usr/local/cuda/

# Quick functional test (Python):
python3 -c "import ctypes; c=ctypes.CDLL('libcudart.so.12'); n=ctypes.c_int(); c.cudaGetDeviceCount(ctypes.byref(n)); print(f'Devices: {n.value}')"
# Should print "Devices: 1" if GVirtuS backend is running

# Use LD_DEBUG to trace library resolution:
LD_DEBUG=libs python3 -c "import ctypes; ctypes.CDLL('libcudart.so.12')" 2>&1 | grep cudart
```

### Backend log patterns

| Pattern | Meaning |
|---------|---------|
| `Client connected` → `Read() returned 0 bytes` → `Client disconnected` | Spurious connection (LD_PRELOAD side effect or constructor-only load) |
| `Client connected` → actual CUDA calls → `Client disconnected` | Normal operation |
| No connections at all | GVirtuS stubs not being loaded; check LD_LIBRARY_PATH and ldconfig |

---

## TODO: Missing CUDA Driver API Functions in `cuGetProcAddress`

### Current Status (2026-04-13)

The `cuGetProcAddress` bootstrap fix is working: `cuInit` and `cuDriverGetVersion` succeed, and the CUDA runtime resolves ~70 driver API functions through GVirtuS stubs. However, `cudaGetDeviceCount` still fails with **error 36 (`cudaErrorCallRequiresNewerDriver`)** because the runtime detects missing functions for the reported driver version.

**Root cause:** `cuDriverGetVersion` returns the backend's real driver version (e.g., 13000 for CUDA 13.0). The CUDA runtime then expects ALL driver functions up to that version to be available via `cuGetProcAddress`. When it finds ~120 functions as NOT FOUND, it concludes the driver doesn't support the required API level.

**Possible fix strategies:**
1. **Intercept `cuDriverGetVersion` in the frontend** — return a version matching our implementation coverage instead of forwarding to backend. Risk: if we return too low a version, the runtime thinks the driver is too old for CUDA 12.6 → error 35.
2. **Implement stub functions** for the missing APIs — forward to backend where possible, or return sensible defaults for probing-only functions.
3. **Return dummy pointers from `cuGetProcAddress`** for NOT FOUND functions — a generic stub that returns `CUDA_ERROR_NOT_SUPPORTED`. The runtime sees the function as "present" during probing, and only fails if actually called. Risk: may crash if the runtime calls the dummy function with unexpected arguments.

### Priority 1 — Likely Blocking RAPIDS Init (basic driver API)

These are fundamental functions any CUDA 12.x driver must have. The runtime likely checks for these during `cudaGetDeviceCount` initialization:

| Function | Since | Notes |
|----------|-------|-------|
| `cuStreamWaitEvent` | v3020 | **Very basic** — needed for stream synchronization |
| `cuDeviceGetUuid` | v9020 | Device identification — RAPIDS uses this |
| `cuCtxGetFlags` | v7000 | Context flag query |
| `cuDevicePrimaryCtxSetFlags` | v11000 | Primary context management |
| `cuStreamCreateWithPriority` | v5050 | Priority stream creation |
| `cuStreamGetPriority` | v5050 | Stream priority query |
| `cuStreamGetFlags` | v5050 | Stream flag query |
| `cuStreamGetCtx` | v9020 | Get context from stream |
| `cuMemcpy` | v4000 | Generic memcpy (unified addressing) |
| `cuMemcpyAsync` | v4000 | Generic async memcpy |
| `cuMemAllocManaged` | v6000 | Unified/managed memory |
| `cuPointerGetAttributes` | v7000 | Pointer attribute query |
| `cuCtxGetApiVersion` | v3020 | Context API version |
| `cuDeviceGetByPCIBusId` | v4010 | PCI bus lookup |
| `cuDeviceGetPCIBusId` | v4010 | PCI bus ID query |

### Priority 2 — RAPIDS GPU Operations (async memory, pools)

RAPIDS cuDF heavily uses async memory allocation and memory pools:

| Function | Since | Notes |
|----------|-------|-------|
| `cuMemAllocAsync` | v11020 | Async memory allocation |
| `cuMemFreeAsync` | v11020 | Async memory free |
| `cuMemAllocFromPoolAsync` | v11020 | Pool-based async alloc |
| `cuMemPoolCreate` | v11020 | Create memory pool |
| `cuMemPoolDestroy` | v11020 | Destroy memory pool |
| `cuMemPoolSetAttribute` | v11020 | Pool attributes |
| `cuMemPoolGetAttribute` | v11020 | Pool attributes |
| `cuMemPoolSetAccess` | v11020 | Pool access control |
| `cuMemPoolGetAccess` | v11020 | Pool access query |
| `cuDeviceGetDefaultMemPool` | v11020 | Default pool |
| `cuDeviceSetMemPool` | v11020 | Set device pool |
| `cuDeviceGetMemPool` | v11020 | Get device pool |
| `cuMemHostRegister` | v6050 | Pin host memory |
| `cuMemHostUnregister` | v4000 | Unpin host memory |
| `cuStreamAddCallback` | v5000 | Stream callbacks |
| `cuStreamWaitValue32` | v8000 | Stream ordered wait |
| `cuLaunchCooperativeKernel` | v9000 | Cooperative launch |
| `cuLaunchHostFunc` | v10000 | Host function on stream |
| `cuMemcpyPeer` / `Async` | v4000 | Peer-to-peer memcpy |

### Priority 3 — Advanced Features (may not block init)

| Category | Functions | Notes |
|----------|-----------|-------|
| Module loading mode | `cuModuleGetLoadingMode` | v11070, module config |
| Library API (CUDA 12+) | `cuLibrary*`, `cuKernel*` | 15 functions, new in CUDA 12.0 |
| Graph API | `cuGraph*` (~40 functions) | CUDA graphs, v10000+ |
| Stream capture | `cuStreamBeginCapture`, `cuStreamEndCapture`, etc. | v10000+ |
| Texture/Surface objects | `cuTexObject*`, `cuSurfObject*` | v5000+ |
| External memory | `cuImportExternalMemory`, etc. | v10000+ |
| IPC | `cuIpcGet/Open/CloseMemHandle` | v4010+ |
| Graphics interop | `cuGL*`, `cuEGL*`, `cuVDPAU*` | Not needed for compute |
| Profiler | `cuProfilerStart/Stop` | v4000, not critical |
| Event with flags | `cuEventRecordWithFlags` | v11010 |
| Async notifications | `cuDeviceRegister/UnregisterAsyncNotification` | v12040 |

### Functions Already Implemented (FOUND)

These work correctly through GVirtuS (~70 functions):

- **Init:** `cuInit`, `cuDriverGetVersion`, `cuGetProcAddress`
- **Device:** `cuDeviceGet`, `cuDeviceGetCount`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`, `cuDeviceCanAccessPeer`
- **Primary context:** `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxGetState`, `cuDevicePrimaryCtxReset`
- **Context:** `cuCtxCreate`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxDetach`, `cuCtxGetDevice`, `cuCtxGetLimit`, `cuCtxSetLimit`, `cuCtxSynchronize`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`
- **Memory:** `cuMemAlloc`, `cuMemFree`, `cuMemAllocPitch`, `cuMemGetInfo`, `cuMemGetAddressRange`, `cuMemFreeHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuPointerGetAttribute`
- **Memcpy:** `cuMemcpyHtoD/Async`, `cuMemcpyDtoH/Async`, `cuMemcpyDtoD/Async`, `cuMemcpy2D/Async/Unaligned`, `cuMemcpy3D/Async`
- **Memset:** `cuMemsetD8`, `cuMemsetD2D8`
- **Arrays:** `cuArrayCreate`, `cuArray3DCreate`, `cuArrayDestroy`, `cuArrayGetDescriptor`, `cuArray3DGetDescriptor`
- **Modules:** `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadFatBinary`, `cuModuleUnload`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleGetTexRef`
- **Linker:** `cuLinkCreate`, `cuLinkAddData`, `cuLinkAddFile`, `cuLinkComplete`, `cuLinkDestroy`
- **Execution:** `cuFuncGetAttribute`, `cuFuncSetAttribute`, `cuFuncSetCacheConfig`, `cuLaunchKernel`, `cuLaunchKernelEx`
- **Streams:** `cuStreamCreate`, `cuStreamDestroy`, `cuStreamQuery`, `cuStreamSynchronize`, `cuStreamWriteValue32`
- **Events:** `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
- **Occupancy:** `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, `cuOccupancyAvailableDynamicSMemPerBlock`, `cuOccupancyMaxPotentialClusterSize`, `cuOccupancyMaxActiveClusters`
- **Errors:** `cuGetErrorString`, `cuGetErrorName`

---

## GVirtuS with JVM-Based Applications (Spark/RAPIDS)

### Problem: Why LD_PRELOAD Fails with JVM

`LD_PRELOAD` must **NOT** be used with GVirtuS in JVM/Spark environments. Here's why:

1. **GVirtuS Frontend is a per-thread singleton.** Its constructor opens a TCP connection to the backend the moment the library is loaded.
2. **`LD_PRELOAD` forces the library into every process.** This includes the JVM, Python interpreter, shell scripts (`spark-class`, `spark-submit`), and even utilities like `ls` and `grep`.
3. **Each process creates a spurious connection** → connects to backend → sends 0 bytes → disconnects. The backend logs show:
   ```
   Client connected from 24.24.24.3:41412 (fd=54)
   Read() returned 0 bytes
   Client disconnected
   ```
4. **JVM startup hangs** because the GVirtuS constructor fires in JVM threads before Spark/RAPIDS code runs, creating stale connections that interfere with the actual CUDA calls.

### Correct Approach: Library Hiding + LD_LIBRARY_PATH + ldconfig

Instead of `LD_PRELOAD`, we hide real CUDA libraries and let the dynamic linker find GVirtuS stubs naturally:

```
┌─────────────────────────────────────────────────────────┐
│ Container (no --runtime=nvidia)                         │
│                                                         │
│  /usr/local/cuda/lib64/              (EMPTY — moved)    │
│  /usr/local/cuda/lib64/real-backup/  (real libs stored) │
│                                                         │
│  /opt/GVirtuS/lib/frontend/          (GVirtuS stubs)    │
│    libcudart.so.12                                      │
│    libcudart.so → libcudart.so.12                       │
│    libcuda.so.1                                         │
│    libcuda.so → libcuda.so.1                            │
│    libcublas.so.12, libcufft.so.12, ...                 │
│                                                         │
│  /etc/ld.so.conf.d/gvirtus.conf → GVirtuS frontend dir │
│  ldconfig rebuilt → resolves libcudart.so.12 to stubs   │
│                                                         │
│  LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:             │
│                  /opt/GVirtuS/lib:                       │
│                  /tmp/rapids-native                      │
│  LD_PRELOAD=     (unset!)                               │
└─────────────────────────────────────────────────────────┘
```

**Steps (implemented in `entrypoint.sh` for gvirtus mode):**

1. **Move real CUDA libs** from `/usr/local/cuda/lib64/` and `/usr/local/cuda/targets/x86_64-linux/lib/` to `real-backup/` subdirectories
2. **Create unversioned symlinks** in GVirtuS frontend dir (`libcudart.so → libcudart.so.12`, `libcuda.so → libcuda.so.1`)
3. **Rebuild ldconfig cache** with GVirtuS frontend dir first
4. **Set `LD_LIBRARY_PATH`** — GVirtuS frontend + GVirtuS lib + RAPIDS native dir
5. **Unset `LD_PRELOAD`** — critical for JVM stability
6. **Extract RAPIDS native libs** from JAR to `/tmp/rapids-native/` with nvcomp version symlinks

**Spark configuration** (set in `config.py`):
```python
'spark.executor.extraLibraryPath': '/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/tmp/rapids-native',
'spark.driver.extraLibraryPath':   '/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/tmp/rapids-native',
```

### How CUDA Calls Flow Through GVirtuS in Spark

```
1. cuDF JNI (libcudfjni.so) calls cudaGetDeviceCount()
2. Dynamic linker resolves libcudart.so.12 → GVirtuS frontend stub
3. GVirtuS cudart stub serializes "cudaGetDeviceCount" + args → Buffer
4. Buffer sent via TcpCommunicator to backend at 24.24.24.3:2223
5. Backend dispatches to cudart plugin → calls real cudaGetDeviceCount()
6. Result serialized and sent back → GVirtuS stub returns to caller
```

For the **driver API** (`libcuda.so.1`), the GVirtuS stub implements `cuGetProcAddress` which returns local function pointers to GVirtuS stub functions. The CUDA runtime uses this to bootstrap its initialization:

```
1. CUDA runtime calls cuGetProcAddress("cuGetProcAddress") → self-reference
2. Uses returned pointer to resolve cuInit, cuDriverGetVersion, etc.
3. Calls cuInit(0) → forwarded to backend via GVirtuS
4. Calls cuDriverGetVersion → gets backend driver version
5. Probes ~300 driver functions via cuGetProcAddress
6. Any NOT FOUND function → treated as "unsupported by driver"
```

### Verification Commands

```bash
# Inside container — verify stubs are loaded:
ldconfig -p | grep -E "libcudart|libcuda\.so"
# Should show /opt/GVirtuS/lib/frontend/ paths

# Functional test (no LD_PRELOAD needed):
python3 -c "import ctypes; c=ctypes.CDLL('libcudart.so.12'); n=ctypes.c_int(); c.cudaGetDeviceCount(ctypes.byref(n)); print(f'Devices: {n.value}')"

# Check that cuDF JNI loads correctly:
LD_DEBUG=libs python3 -c "import ctypes; ctypes.CDLL('/tmp/rapids-native/libcudfjni.so')" 2>&1 | grep -E "libcuda|calling init"
# Should show libnvcomp.so.5, libcudf.so, libcudfjni.so — NOT libcudart.so (loaded on demand)
```
