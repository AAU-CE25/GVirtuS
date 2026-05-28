# GVirtuS + Spark RAPIDS Integration

## Overview

Integrating Apache Spark RAPIDS with GVirtuS GPU virtualization to enable GPU-accelerated Spark workloads on GPU-less client machines.

## Architecture

```
Spark Driver/Executor (GPU-less client)
    └── RAPIDS Plugin (cuDF, cuML)
        └── Native libs (libcudf.so, libcudfjni.so)
            └── LD_PRELOAD intercept
                └── GVirtuS Frontend stubs
                    └── TCP Connection (port 2222)
                        └── GVirtuS Backend (GPU server)
                            └── Real CUDA libs → L40s GPU
```

## Key Files

- `examples/spark_simple_matrix/` - Spark RAPIDS test workload
- `examples/spark_simple_matrix/src/config.py` - Spark configuration with GVirtuS paths
- `docker/dev/Dockerfile` - Dev container with GVirtuS + CUDA stubs

## Spark Configuration for GVirtuS

```python
{
    'spark.jars': 'rapids-4-spark_2.12-26.02.1.jar',
    'spark.plugins': 'com.nvidia.spark.SQLPlugin',
    'spark.rapids.sql.enabled': 'true',
    'spark.executor.extraLibraryPath': '/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib',
    'spark.driver.extraLibraryPath': '/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib',
    'spark.executorEnv.LD_PRELOAD': '/opt/GVirtuS/lib/frontend/libcudart.so:/opt/GVirtuS/lib/frontend/libcuda.so',
    'spark.executorEnv.GVIRTUS_HOME': '/opt/GVirtuS',
}
```

## Known Issues

### 1. Probe Connections (Solved)

**Problem:** cuDF native libs trigger `dlopen("libcuda.so")` during initialization. GVirtuS's static constructor opens TCP connections immediately, even if no CUDA call is made. When handles are closed, connections appear as "probe connections" with no routine sent.

**Solution:** Lazy connection - delay `Connect()` until `Execute()` is called.

**Files changed:**
- `include/gvirtus/frontend/Frontend.h` - Added `mpConnected` flag
- `src/frontend/Frontend.cpp` - Moved `Connect()` to first `Execute()` call

### 2. Missing libnvJitLink.so (TODO)

**Problem:** cuDF dynamically loads `libnvJitLink.so` for JIT kernel compilation. GVirtuS doesn't provide this stub.

**Detection:**
```bash
grep -ao "libnvJitLink" libcudf.so
# Returns: libnvJitLink.so.Dynamically
```

**Solution needed:** Create `plugins/nvjitlink/` with handler stubs for:
- `nvJitLinkCreate()`
- `nvJitLinkDestroy()`
- `nvJitLinkAddData()`
- `nvJitLinkComplete()`
- `nvJitLinkGetLinkedCubin()`

### 3. Missing libnvcomp.so.5

**Problem:** cuDF needs nvcomp compression library.

**Detection:**
```bash
ldd libcudf.so | grep nvcomp
# Returns: libnvcomp.so.5 => not found
```

**Solution:** Extract from RAPIDS JAR or install separately. May not need virtualization if compression runs CPU-side.

### 4. RTLD_LOCAL dlopen Issue

**Problem:** cuDF uses `dlopen("libcuda.so.1", RTLD_LOCAL)` which creates isolated symbol namespaces. Each load creates a separate GVirtuS connection.

**Impact:** Multiple connections opened/closed during init, but manageable with lazy connect.

## Debugging Commands

```bash
# Check GVirtuS frontend libs
ls -la /opt/GVirtuS/lib/frontend/

# Available: libcuda.so.1, libcudart.so.12, libnvrtc.so.12, libcublas.so.12, etc.
# Missing: libnvJitLink.so

# Run with TRACE logging
GVIRTUS_LOGLEVEL=0 make run-gvirtus-backend-dev

# Test basic CUDA through GVirtuS
make test-spark-gvirtus

# Check cuDF CUDA dependencies
docker run --rm --env LD_PRELOAD= --entrypoint bash IMAGE -c \
  'unzip -p JAR "amd64/Linux/libcudf.so" | grep -ao "libcuda\|libnvrtc\|libnvJitLink"'
```

## Test Workflow

```bash
# Terminal 1: Start backend
make run-gvirtus-backend-dev

# Terminal 2: Run Spark test
make run-spark-gvirtus
```

## Estimated Work for Full Support

| Task | Effort | Notes |
|------|--------|-------|
| Lazy connection | Done ✓ | Prevents probe connection noise |
| libnvJitLink stub | 1-2 days | Required for JIT compilation |
| libnvcomp handling | 1 day | May run locally without GPU |
| Missing cudadr APIs | 2-5 days | Depends on what cuDF uses |
| Full RAPIDS testing | 1 day | May expose more issues |

## Related Files

- `TODO.md` - Full issue tracking document
- `docs/Matrix_appliation_flow_with_GVirtuS.md` - Data flow documentation
- `src/backend/Process.cpp` - Backend connection handling (enhanced logging)
