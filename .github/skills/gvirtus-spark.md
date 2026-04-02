# GVirtuS + Spark RAPIDS Integration

## Overview

Integrating Apache Spark RAPIDS with GVirtuS GPU virtualization to enable GPU-accelerated Spark workloads on GPU-less client machines.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| GVirtuS Frontend build | ✅ Done | Builds from nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 |
| Basic CUDA tests | ✅ Pass | 16/16 tests pass (cudaGetDeviceCount, cuInit, malloc, etc.) |
| LD_PRELOAD setup | ✅ Done | Set in Dockerfile ENV |
| RAPIDS native lib extraction | ✅ Done | Entrypoint extracts from JAR |
| Logging to stderr | ✅ Fixed | Was breaking Spark shell scripts |
| Spark RAPIDS execution | ⏳ Testing | Multiple connections established, pending full run |

## Architecture

```
Spark Driver/Executor (GPU-less client: es-dpu-02)
    └── RAPIDS Plugin (cuDF JNI)
        └── Native libs extracted from JAR (/tmp/rapids-native/)
            ├── libcudf.so (1.2GB - statically linked CUDA kernels)
            ├── libcudfjni.so (15KB - JNI wrapper)
            └── libnvcomp.so (41MB - compression)
                └── dlopen("libcudart.so", "libcuda.so")
                    └── LD_PRELOAD intercept
                        └── GVirtuS Frontend stubs (/opt/GVirtuS/lib/frontend/)
                            └── TCP Connection (port 2222)
                                └── GVirtuS Backend (es-dpu-01: 24.24.24.1)
                                    └── Real CUDA libs → NVIDIA L40S GPU
```

## Key Files

| File | Purpose |
|------|---------|
| `examples/spark_simple_matrix/Dockerfile.gvirtus` | Multi-stage build: GVirtuS frontend + Spark |
| `examples/spark_simple_matrix/entrypoint-gvirtus.sh` | Extracts RAPIDS native libs, sets up env |
| `examples/spark_simple_matrix/src/config.py` | Spark config with GVirtuS paths |
| `examples/spark_simple_matrix/src/gvirtus_test.py` | Comprehensive CUDA connectivity tests |
| `etc/properties.json` | GVirtuS endpoint and plugins config |
| `src/frontend/Frontend.cpp` | Frontend with stderr logging fix |

## Plugin Naming (Important!)

The plugin naming is confusing:

| Plugin Name | Directory | Provides | Library Name |
|------------|-----------|----------|--------------|
| `cuda` | `plugins/cudadr/` | CUDA Driver API (cuInit, cuDeviceGet, etc.) | `libgvirtus-plugin-cuda.so` |
| `cudart` | `plugins/cudart/` | CUDA Runtime API (cudaMalloc, etc.) | `libgvirtus-plugin-cudart.so` |

**Note:** `"cuda"` in properties.json is the **Driver API**, not a duplicate of cudart!

## Spark Configuration for GVirtuS

```python
# config.py - SPARK_RAPIDS_GVIRTUS_CONFIG
GVIRTUS_HOME_PATH = "/opt/GVirtuS"
GVIRTUS_LIB_PATH = f"{GVIRTUS_HOME_PATH}/lib/frontend:{GVIRTUS_HOME_PATH}/lib"
RAPIDS_NATIVE_PATH = "/tmp/rapids-native"  # Extracted from JAR by entrypoint

{
    'spark.jars': 'rapids-4-spark_2.12-26.02.1.jar',
    'spark.plugins': 'com.nvidia.spark.SQLPlugin',
    'spark.rapids.sql.enabled': 'true',
    'spark.executor.extraLibraryPath': f'{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}',
    'spark.driver.extraLibraryPath': f'{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}',
    'spark.executorEnv.LD_PRELOAD': f'{GVIRTUS_HOME_PATH}/lib/frontend/libcudart.so:{GVIRTUS_HOME_PATH}/lib/frontend/libcuda.so',
    'spark.executorEnv.GVIRTUS_HOME': GVIRTUS_HOME_PATH,
    'spark.executorEnv.LD_LIBRARY_PATH': f'{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}',
}
```

## Dockerfile Environment

```dockerfile
# Set in Dockerfile so ALL processes use GVirtuS stubs
ENV GVIRTUS_HOME=/opt/GVirtuS \
    GVIRTUS_LOGLEVEL=40000 \
    LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib \
    LD_PRELOAD=/opt/GVirtuS/lib/frontend/libcudart.so:/opt/GVirtuS/lib/frontend/libcuda.so
```

## Issues Fixed

### 1. Logging to stdout broke Spark shell scripts ✅ FIXED

**Problem:** GVirtuS used `BasicConfigurator` which logged to stdout. Spark uses shell scripts (spark-submit, find-spark-home) that parse stdout, causing errors like:
```
/usr/local/lib/python3.10/dist-packages/pyspark/./bin/spark-submit: line 21: INFO: command not found
```

**Solution:** Changed to `ConsoleAppender(true, true)` to log to stderr:
```cpp
// src/frontend/Frontend.cpp
SharedAppenderPtr appender(new ConsoleAppender(true, true));  // logToStdErr=true
```

### 2. DEBUG print statement interference ✅ FIXED

**Problem:** Hard-coded `std::cout << "DEBUG: protocol string is..."` in CommunicatorFactory.h

**Solution:** Commented out the debug line.

### 3. RAPIDS native libs not in LD_LIBRARY_PATH ✅ FIXED

**Problem:** Java/JNI extracts libs from JAR to temp dir, but libcudf.so couldn't find libnvcomp.so

**Solution:** Entrypoint extracts libs to `/tmp/rapids-native` and adds to LD_LIBRARY_PATH:
```bash
unzip -q -j "$RAPIDS_JAR" "amd64/Linux/*.so" -d "$NATIVE_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NATIVE_DIR}"
```

### 4. LD_PRELOAD not reaching Java process ✅ FIXED

**Problem:** Setting LD_PRELOAD in entrypoint didn't help Java JNI loading

**Solution:** Set LD_PRELOAD in Dockerfile ENV so it applies to all processes including JVM

## Known Remaining Issues

### 1. Missing libnvJitLink.so (May be needed)

**Problem:** cuDF may dynamically load `libnvJitLink.so` for JIT kernel compilation.

**Detection:**
```bash
strings libcudf.so | grep nvJitLink
# Returns: libnvJitLink.so
```

**Impact:** Unknown - may only be needed for certain operations. If required, needs a new GVirtuS plugin.

### 2. RTLD_LOCAL dlopen creates multiple connections

**Problem:** cuDF uses `dlopen("libcuda.so.1", RTLD_LOCAL)` creating isolated symbol namespaces. Each load triggers a new GVirtuS connection.

**Impact:** Multiple TCP connections during init (seen in logs). Not a bug, just verbose.

### 3. libcudf.so has hardcoded RUNPATH

**Detection:**
```bash
readelf -d libcudf.so | grep RUNPATH
# Returns: $ORIGIN:/usr/local/cuda/lib64:/home/jenkins/...
```

**Impact:** None - our LD_PRELOAD overrides this for CUDA libs.

## Debugging Commands

```bash
# Check env vars in container
docker run --rm --entrypoint bash IMAGE -c 'env | grep -E "LD_|GVIRTUS"'

# Check GVirtuS frontend libs
ls -la /opt/GVirtuS/lib/frontend/
# Should include: libcuda.so.1, libcudart.so.12, libcublas.so.12, etc.

# Run connectivity test
make test-spark-gvirtus

# Run with TRACE logging (frontend)
docker run -e GVIRTUS_LOGLEVEL=0 ...

# Run with DEBUG logging (backend)
GVIRTUS_LOGLEVEL=10000 $GVIRTUS_HOME/bin/gvirtus-backend ...

# Check RAPIDS JAR native libs
unzip -l rapids-4-spark_2.12-26.02.1.jar | grep amd64
# Contents: libcudf.so (1.2GB), libcudfjni.so, libnvcomp.so, etc.

# Check libcudf.so dependencies
cd jars && unzip -q rapids-4-spark*.jar "amd64/Linux/*"
readelf -d amd64/Linux/libcudf.so | grep -E "NEEDED|RUNPATH"
ldd amd64/Linux/libcudf.so | grep "not found"
```

## Test Results (gvirtus_test.py)

```
Testing: cuInit(0)...                   OK
Testing: cuDeviceGetCount...            OK (1 device)
Testing: cuDeviceGet...                 OK
Testing: cuCtxCreate/Destroy...         OK
Testing: Device Count...                OK (1 device)
Testing: Set Device 0...                OK
Testing: Device Properties...           OK (NVIDIA L40S)
Testing: Driver Version...              OK (13000 = CUDA 13.0)
Testing: Runtime Version...             OK (12060 = CUDA 12.6)
Testing: Malloc/Free...                 OK (1MB allocation)
Testing: Memcpy H2D/D2H...              OK (data verified)
Testing: Memory Info...                 OK (43.97 GB free / 44.39 GB total)
Testing: Stream Create/Destroy...       OK
Testing: Device Synchronize...          OK
Testing: Device Reset...                OK

Results: 16 passed, 0 failed
```

## Test Workflow

```bash
# Terminal 1: Start backend (on es-dpu-01)
make run-gvirtus-backend-dev
# Or: GVIRTUS_LOGLEVEL=10000 $GVIRTUS_HOME/bin/gvirtus-backend $GVIRTUS_HOME/etc/properties.json

# Terminal 2: Run connectivity test (on es-dpu-02)
make test-spark-gvirtus

# Terminal 2: Run Spark RAPIDS
make run-spark-gvirtus
```

## Network Setup

| Machine | Interface | IP | Role |
|---------|-----------|-----|------|
| es-dpu-01 | ens7f0np0 | 24.24.24.1 | GVirtuS backend + L40S GPU |
| es-dpu-02 | — | 24.24.24.3 | Spark client (GPU-less) |

## Remaining Work

| Task | Priority | Status |
|------|----------|--------|
| Full Spark RAPIDS test | High | In progress - connections established |
| libnvJitLink stub | Medium | May be needed for JIT ops |
| Performance tuning | Low | After functionality works |
| Documentation | Low | This file |
