# TODO: Spark RAPIDS + GVirtuS Integration Issues

**Date**: 2026-04-02

## Problem Summary

Spark RAPIDS connects to GVirtuS backend but immediately disconnects without executing any CUDA routines. The backend logs show:
```
Client connected from 24.24.24.3:xxxxx
Read() returned 0 bytes
Client disconnected (no routine sent - probe connection)
```

## Root Causes Identified

### 1. Missing `libnvJitLink.so` Stub

cuDF dynamically loads `libnvJitLink.so` via `dlopen()`, but GVirtuS doesn't provide this stub.

**Evidence:**
```bash
grep -ao "libnvJitLink" libcudf.so
# Found: libnvJitLink.so.Dynamically
```

**Current GVirtuS frontend libs:**
- `libcuda.so.1` ✅
- `libcudart.so.12` ✅  
- `libnvrtc.so.12` ✅
- `libnvJitLink.so` ❌ **MISSING**

### 2. cuDF Uses `dlopen()` with `RTLD_LOCAL`

cuDF doesn't link CUDA statically. It calls:
```c
dlopen("libcuda.so.1", RTLD_LOCAL)
```

`RTLD_LOCAL` creates an isolated symbol namespace. When Java/JNI probes and closes handles, the connection is destroyed before any routine is called.

### 3. Static Constructor Eager Connection

In `plugins/cudart/frontend/CudaRtFrontend.cpp:37`:
```cpp
CudaRtFrontend msInstance __attribute_used__;
```

The constructor at line 55 calls `GetFrontend()` which immediately opens a TCP connection:
```cpp
CudaRtFrontend::CudaRtFrontend() {
    // ...
    gvirtus::frontend::Frontend::GetFrontend();  // Opens connection!
}
```

This means every `dlopen("libcudart.so")` opens a connection, even if no CUDA call is made.

### 4. Missing `libnvcomp.so.5`

ldd shows:
```
libnvcomp.so.5 => not found
```

cuDF needs nvcomp for compression but it's not in the extracted native libs path.

## Potential Fixes

### Option 1: Add Missing Stubs (Quick Test)
```bash
# In Dockerfile or container:
ln -s libnvrtc.so.12 /opt/GVirtuS/lib/frontend/libnvJitLink.so
ln -s libnvrtc.so.12 /opt/GVirtuS/lib/frontend/libnvJitLink.so.0
```

### Option 2: Lazy Connection (Code Change)

Modify `Frontend::GetFrontend()` to NOT call `Connect()` until `Execute()` is actually called. This prevents wasted probe connections.

**Files to modify:**
- `src/frontend/Frontend.cpp` - delay `Connect()` call
- Check `mpInitialized` flag usage

### Option 3: Process-Level Singleton (Architectural)

Make the communicator connection process-wide rather than per-thread, so multiple `dlopen`/`dlclose` cycles reuse the same connection.

### Option 4: Test Without RAPIDS GPU

Temporarily disable RAPIDS GPU acceleration to verify Spark itself works:
```python
'spark.rapids.sql.enabled': 'false'
```

## Verification Commands

```bash
# Check what GVirtuS frontend provides:
ls -la /opt/GVirtuS/lib/frontend/

# Check cuDF CUDA dependencies:
docker run --rm --env LD_PRELOAD= --entrypoint /bin/bash IMAGE -c \
  'unzip -p JAR "amd64/Linux/libcudf.so" | grep -ao "libcuda\|libnvrtc\|libnvJitLink"'

# Run with TRACE logging:
GVIRTUS_LOGLEVEL=0 make run-gvirtus-backend-dev
```

## Files Changed During Debugging

- `src/backend/Process.cpp` - Added better disconnect logging (line 232-246)

## Next Steps

1. [ ] Create `libnvJitLink.so` frontend stub (if nvJitLink APIs are needed)
2. [ ] Consider implementing lazy connection in Frontend
3. [ ] Extract `libnvcomp.so` from RAPIDS JAR or install separately
4. [ ] Test with `spark.rapids.sql.enabled=false` to isolate issue
