# Session Summary — 2026-05-05

## What was done

### 1. Network Setup (es-dpu-02 client machine)
- Brought up BlueField-3 interfaces:
  ```bash
  sudo ip link set ens1f0np0 up
  sudo ip link set ens1f1np1 up
  ```
- Assigned IPs:
  ```bash
  sudo ip addr add 24.24.24.2/24 dev ens1f0np0
  sudo ip addr add 25.25.25.2/24 dev ens1f1np1
  ```
- **Note:** These are non-persistent (lost on reboot). Need netplan/nmcli for persistence.

### 2. GPU Driver Check
- **Driver:** 580.95.05 (supports up to CUDA 13.0)
- **CUDA Toolkit (nvcc):** 12.6.85 — already correct
- **nvidia-smi shows "CUDA 13.0"** — this is just the max supported version, not installed version
- Looked into downgrading to driver 560 but did not execute (user undid)

### 3. Build Warnings (NOT errors)
- GVirtuS builds successfully with deprecated function warnings
- All warnings are `-Wdeprecated-declarations` for old CUDA texture/launch APIs
- Safe to ignore; build completes and produces all `.so` libraries

---

## The Main Problem: Error 36 (`cudaErrorCallRequiresNewerDriver`)

### Symptom
```
ai.rapids.cudf.CudaFatalException: Fatal CUDA error: 36 cudaErrorCallRequiresNewerDriver
```
Spark RAPIDS plugin fails during `cudaGetDeviceCount()` initialization.

### Root Cause Chain

```
cuDF JNI → libcuda.so.1 (GVirtuS cudaDR frontend)
  1. cuGetProcAddress("cuGetProcAddress") → self-reference ✓
  2. cuDriverGetVersion() → backend returns 12060 ✓  
  3. cuInit(0) → backend returns 0 ✓
  4. Runtime probes ~300 functions via cuGetProcAddress → stubs returned ✓
  5. Runtime CALLS some probed functions → stub returns CUDA_SUCCESS with NO DATA
  6. Runtime detects inconsistency → ERROR 36
  7. cudaGetDeviceCount (cudaRT) never reached — fails at driver bootstrap
```

### Why master gives error 35 (different!)
- Master has NO `cuGetProcAddress` implementation
- Runtime does `dlsym(libcuda.so, "cuGetProcAddress_v2")` → NULL
- Concludes driver is ancient (pre-CUDA 9.2) → error 35

### Why current branch gives error 36
- Branch HAS `cuGetProcAddress` + generic stub (`gvirtus_not_implemented_stub`)
- Passes version check (12060 >= minimum)
- But probed functions that get **called** (not just probed) return CUDA_SUCCESS with no output data
- Runtime sees the function exists but returns garbage → error 36

---

## Architecture Understanding

```
Application (cuDF/Spark)
├── libcudart.so.12 (GVirtuS cudaRT frontend)
│     cudaGetDeviceCount() → serialize → TCP → backend → real GPU
│     (NEVER REACHED — failure happens before this)
│
└── libcuda.so.1 (GVirtuS cudaDR frontend)  ← WHERE THE FIX NEEDS TO BE
      cuGetProcAddress() → returns LOCAL stub function pointers
      cuInit() → serialize → TCP → backend
      cuDriverGetVersion() → serialize → TCP → backend → 12060
```

**Key file:** `plugins/cudadr/frontend/CudaDr_driver_entry_point.cpp`

---

## What Needs to Be Done Next

### Immediate: Identify which stubbed functions are CALLED (not just probed)

The `gvirtus_not_implemented_stub` already prints a warning. Run the Spark frontend and look at stderr for:
```
[GVirtuS] WARNING: Called an unimplemented CUDA driver function
```

If that message does NOT appear, the problem is different — possibly the runtime checks return values from functions that ARE in the map but don't return proper data.

### Strategy Options

1. **Add unique identifiers to stubs** — modify `cuGetProcAddress` to create per-function stubs (using a trampoline or logging which function was resolved when it's called)
2. **Implement the missing critical functions** — based on the copilot-instructions.md TODO list:
   - Priority 1 (likely blocking): `cuStreamWaitEvent`, `cuDeviceGetUuid`, `cuCtxGetFlags`, `cuDevicePrimaryCtxSetFlags`, `cuStreamCreateWithPriority`, `cuMemcpy`, `cuMemcpyAsync`, `cuMemAllocManaged`, `cuPointerGetAttributes`
   - Many of these are already implemented on the current branch! Check if the issue is that `cuGetProcAddress` returns the generic stub for the `_v2`/`_v3` versioned names

### Possible Quick Fix: Check versioned name coverage

The CUDA runtime might request `cuDeviceGetUuid_v2` but the function map might only have `cuDeviceGetUuid`. Verify ALL `_v2`/`_v3` variants are in the map for implemented functions.

---

## Files Changed vs Master (plugins only)

| File | Changes |
|------|---------|
| `plugins/cudadr/CMakeLists.txt` | Stubs library search path fix |
| `plugins/cudadr/backend/CudaDrHandler.cpp` | +24 new handler registrations |
| `plugins/cudadr/backend/CudaDrHandler.h` | +24 handler declarations |
| `plugins/cudadr/backend/CudaDrHandler_context.cpp` | +61 lines (CtxGetFlags, ApiVersion, CacheConfig, etc.) |
| `plugins/cudadr/backend/CudaDrHandler_device.cpp` | +66 lines (UUID, PCIBusId) |
| `plugins/cudadr/backend/CudaDrHandler_memory.cpp` | +44 lines (AllocManaged, HostRegister, DtoD) |
| `plugins/cudadr/backend/CudaDrHandler_stream.cpp` | +50 lines (WaitEvent, Priority, GetFlags, GetCtx) |
| `plugins/cudadr/frontend/CudaDr_context.cpp` | +60 lines (matching frontend stubs) |
| `plugins/cudadr/frontend/CudaDr_device.cpp` | +61 lines (UUID, PCIBusId) |
| `plugins/cudadr/frontend/CudaDr_initialization.cpp` | +11 lines |
| `plugins/cudadr/frontend/CudaDr_memory.cpp` | +35 lines |
| `plugins/cudadr/frontend/CudaDr_stream.cpp` | +47 lines |
| `plugins/cudadr/frontend/CudaDr_driver_entry_point.cpp` | **Main cuGetProcAddress implementation** |
| `plugins/cudart/frontend/CudaRt_device.cpp` | +8 lines (debug prints) |

**Total: 14 files changed, +1070 lines**

---

## Configuration Notes

- Backend running on `es-dpu-01` at `24.24.24.1:2222` (TCP)
- Backend reports driver version 12060 correctly
- `cuInit` succeeds through GVirtuS
- The Spark frontend container does NOT have `--runtime=nvidia` (by design — uses GVirtuS)
- `nvidia-smi` not available inside Spark container (expected)
