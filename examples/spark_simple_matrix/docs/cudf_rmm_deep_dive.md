# RAPIDS cuDF & RMM Deep Dive

This document analyzes the cuDF (CUDA DataFrame) library and RMM (RAPIDS Memory Manager) components from the Spark RAPIDS logs. These are the critical layers that perform actual GPU operations and are the **key intercept points for GVirtuS integration**.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Spark SQL / Catalyst Query Plan                        │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  RAPIDS SQL Plugin (Java/Scala)                            │
│   • Replaces Spark operators with GpuXxx versions                          │
│   • GpuRowToColumnar: Converts Spark rows → columnar batches               │
│   • GpuHashAggregate, GpuProject, GpuColumnarExchange, etc.                │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      JNI Bridge (spark-rapids-jni)                         │
│   ai.rapids.cudf.* Java classes → native function calls                    │
│   • HostMemoryBuffer (pinned CPU memory)                                   │
│   • DeviceMemoryBuffer (GPU memory)                                        │
│   • Table, ColumnVector, Scalar                                            │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │ JNI native calls
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         libcudf.so (C++ library)                           │
│   • Column-oriented GPU operations (filter, join, aggregate, sort)         │
│   • Built for GPU archs: 70, 75, 80, 86, 90, 100, 120                      │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      RMM (RAPIDS Memory Manager)                           │
│   • Pool allocator over cudaMalloc/cudaFree                                │
│   • Manages GPU memory lifecycle                                           │
│   • Spill framework for OOM scenarios                                      │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      CUDA Runtime (libcudart.so)                           │
│   • cudaMalloc, cudaFree, cudaMemcpy                                       │
│   • Kernel launches                                                        │
│   ══════════════════════════════════════════════════════════════════════   │
│   │        THIS IS WHERE GVirtuS INTERCEPTS THE CALLS                  │   │
│   ══════════════════════════════════════════════════════════════════════   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Library Initialization (Version Info)

When the RAPIDS plugin loads, it logs version information for all native components:

### Log: RAPIDS Accelerator Build Info
```log
26/04/12 18:23:08 INFO RapidsPluginUtils: RAPIDS Accelerator build: Map(
    url -> https://github.com/NVIDIA/spark-rapids.git, 
    branch -> HEAD, 
    revision -> 5e6b7f7ff87fbd1a64f200771e39a3590ee1102f, 
    version -> 26.02.1, 
    date -> 2026-03-10T03:31:31Z, 
    cudf_version -> 26.02.0, 
    user -> root
)
```

### Log: JNI Bridge Build Info
```log
26/04/12 18:23:08 INFO RapidsPluginUtils: RAPIDS Accelerator JNI build: Map(
    url -> https://github.com/NVIDIA/spark-rapids-jni.git, 
    branch -> HEAD, 
    gpu_architectures -> 100;120;70;75;80;86;90,   ← Supported GPU architectures
    revision -> 14a182dca4a349519bfe4e5ddf8eebd9a8af95e5, 
    version -> 26.02.0, 
    date -> 2026-02-05T05:04:44Z, 
    user -> root
)
```

### Log: cuDF Native Library Build Info
```log
26/04/12 18:23:08 INFO RapidsPluginUtils: cudf build: Map(
    url -> https://github.com/rapidsai/cudf.git, 
    branch -> HEAD, 
    gpu_architectures -> 100;120;70;75;80;86;90,   ← Same arch list as JNI
    revision -> 9782a269e689140d2b00b5172a93056bdf19e8c2, 
    version -> 26.02.0, 
    date -> 2026-02-05T05:04:41Z, 
    user -> root
)
```

**Key Insight for GVirtuS**: The `gpu_architectures` field shows which GPU compute capabilities the native code supports. Your L40s (SM 8.9) maps to architecture ~90. GVirtuS must emulate/forward calls for these architectures.

---

## 2. GPU Device Manager Initialization

The `GpuDeviceManager` queries the GPU and initializes the RMM memory pool.

### Log: GPU Detection
```log
26/04/12 18:23:17 DEBUG GpuDeviceManager: Initializing GPU device ID to 0
```

**CUDA API Called**: `cudaSetDevice(0)`, `cudaGetDeviceProperties()`

### Log: GPU Memory Query
```log
26/04/12 18:23:17 INFO GpuDeviceManager: availableGpuTotal: 45457.5625 MiB
26/04/12 18:23:17 INFO GpuDeviceManager: availableGpuFree: 44576.0 MiB
```

**CUDA API Called**: `cudaMemGetInfo(&free, &total)`

**GVirtuS Intercept**: These calls must return valid GPU memory stats. If GVirtuS returns incorrect values here, RMM pool sizing will fail.

---

## 3. RMM (RAPIDS Memory Manager) Pool Initialization

RMM pre-allocates a large chunk of GPU memory to avoid repeated `cudaMalloc` overhead.

### Log: Pool Allocation
```log
26/04/12 18:23:17 INFO GpuDeviceManager: Initializing RMM pool size = 43936.0 MB on gpuId 0
26/04/12 18:23:17 INFO GpuDeviceManager: Using per-thread default stream
```

**CUDA API Called**:
```cpp
cudaMalloc(&pool_ptr, 43936 * 1024 * 1024);  // ~43 GB allocation
cudaStreamCreate(&stream);                    // Per-thread stream
```

**GVirtuS Critical Path**: This is a SINGLE 43GB allocation. If GVirtuS cannot handle large allocations, this will fail. Options:
1. Ensure backend GPU has sufficient memory
2. Configure `spark.rapids.memory.gpu.pool` to `NONE` to disable pooling (slower)
3. Set `spark.rapids.memory.gpu.pooling.enabled` to `false`

### Log: Spill Framework
```log
26/04/12 18:23:17 DEBUG GpuDeviceManager: Initialized pool resource for spill operations of None Bytes
26/04/12 18:23:17 INFO SpillFramework: Initialized SpillFramework. Host spill store max size is: 1073741824 B.
```

When GPU memory is exhausted, RAPIDS can "spill" data to host memory (1 GB limit configured).

---

## 4. Row-to-Columnar Conversion (CPU → GPU Data Transfer)

This is where Spark rows are converted to columnar format and transferred to GPU.

### Generated Code: InternalRowToCudfRowIterator

The RAPIDS plugin generates optimized bytecode for each schema:

```log
26/04/12 18:23:19 DEBUG GeneratedInternalRowToCudfRowIterator: code for row#0,col#1,val#2:
```

```java
/* 001 */ public java.lang.Object generate(Object[] references) {
/* 002 */   return new SpecificInternalRowToColumnarBatchIterator(references);
/* 003 */ }
/* 004 */
/* 005 */ final class SpecificInternalRowToColumnarBatchIterator 
              extends com.nvidia.spark.rapids.InternalRowToColumnarBatchIterator {
/* 006 */   private final UnsafeProjection unsafeProj;
...
/* 028 */   public int[] fillBatch(ai.rapids.cudf.HostMemoryBuffer dataBuffer,
/* 029 */       ai.rapids.cudf.HostMemoryBuffer offsetsBuffer,
/* 030 */       long dataLength, int numRows) {
```

### Schema-Specific Row Processing

For your matrix data schema `(row: Int, col: Int, val: Double)`:

```java
/* 054 */           int value_0 = internalRow_0.getInt(0);      // row
/* 055 */           mutableStateArray_0[0].write(0, value_0);
/* 056 */
/* 057 */           int value_1 = internalRow_0.getInt(1);      // col
/* 058 */           mutableStateArray_0[0].write(1, value_1);
/* 059 */
/* 060 */           double value_2 = internalRow_0.getDouble(2); // val
/* 061 */           mutableStateArray_0[0].write(2, value_2);
```

### Data Flow

```
┌─────────────────────┐     copy      ┌─────────────────────┐
│  Spark InternalRow  │ ─────────────→│ ai.rapids.cudf.     │
│  (JVM heap)         │               │ HostMemoryBuffer    │
└─────────────────────┘               │ (pinned CPU memory) │
                                      └──────────┬──────────┘
                                                 │
                                                 │ cudaMemcpyAsync
                                                 │ (H2D transfer)
                                                 ▼
                                      ┌─────────────────────┐
                                      │ ai.rapids.cudf.     │
                                      │ DeviceMemoryBuffer  │
                                      │ (GPU memory)        │
                                      └─────────────────────┘
```

**CUDA APIs Called**:
- `cudaMallocHost()` - Allocate pinned host memory for `HostMemoryBuffer`
- `cudaMalloc()` - Allocate device memory for `DeviceMemoryBuffer`
- `cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream)` - Async transfer

**GVirtuS Note**: These H2D transfers are frequent and performance-critical. GVirtuS must efficiently serialize memory contents across the network.

---

## 5. GPU Execution Plan

After data is on GPU, operations execute as columnar batches:

### Physical Plan with GPU Operators
```log
GpuColumnarToRow false, [loreId=104]
+- GpuHashAggregate (keys=[], functions=[gpucount(1, false)], output=[count#42L])
   +- GpuShuffleCoalesce 1073741824
      +- ResultQueryStage 2
         +- GpuColumnarExchange gpusinglepartitioning$(), ENSURE_REQUIREMENTS
            +- GpuHashAggregate (keys=[], functions=[partial_gpucount(1, false)])
               +- GpuHashAggregate (keys=[i#24, j#25], functions=[], output=[])
                  +- GpuShuffleCoalesce 1073741824
                     +- ShuffleQueryStage 4
                        +- GpuColumnarExchange gpuhashpartitioning(i#24, j#25, 4)
                           +- GpuHashAggregate (keys=[i#24, j#25], output=[i#24, j#25])
                              +- GpuProject [row#0 AS i#24, col#7 AS j#25]
                                 +- GpuSortMergeJoin [col#1], [row#6], Inner
```

### Key GPU Operators

| Operator | cuDF Function | CUDA Kernels |
|----------|---------------|--------------|
| `GpuHashAggregate` | `cudf::groupby::hash::groupby()` | Hash table, reduce |
| `GpuProject` | `cudf::table::select()` | Memory copy |
| `GpuColumnarExchange` | Shuffle via GPU memory | `cudaMemcpy` |
| `GpuSortMergeJoin` | `cudf::merge()` | Sort, merge kernels |
| `GpuRowToColumnar` | `cudf::table::from_rows()` | Data conversion |
| `GpuColumnarToRow` | `cudf::table::to_rows()` | Data conversion |

---

## 6. Critical CUDA API Calls for GVirtuS

Summary of CUDA APIs that **must work correctly** via GVirtuS:

### Device Management
```cpp
cudaGetDeviceCount(&count);           // Returns 1+ for RAPIDS to proceed
cudaSetDevice(0);                     // Select GPU 0
cudaGetDeviceProperties(&props, 0);   // Must return valid SM version, memory, etc.
```

### Memory Management (via RMM)
```cpp
cudaMalloc(&ptr, size);               // Large allocations (GB scale)
cudaFree(ptr);
cudaMallocHost(&ptr, size);           // Pinned memory for async transfers
cudaFreeHost(ptr);
cudaMemGetInfo(&free, &total);        // Memory stats for pool sizing
```

### Data Transfer
```cpp
cudaMemcpy(dst, src, size, kind);                    // Sync transfer
cudaMemcpyAsync(dst, src, size, kind, stream);       // Async transfer (preferred)
```

### Stream Management
```cpp
cudaStreamCreate(&stream);
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

### Kernel Launches
All cuDF operations ultimately launch CUDA kernels. GVirtuS must forward:
```cpp
kernel<<<grid, block, sharedMem, stream>>>(args...);
```

---

## 7. Troubleshooting GVirtuS Integration

### Symptom: "No CUDA devices found"
- `cudaGetDeviceCount()` returning 0
- GVirtuS backend not running or unreachable

### Symptom: RMM pool allocation fails
- `cudaMalloc()` failing for large (43GB) allocation
- Reduce pool: `spark.rapids.memory.gpu.pool=NONE`

### Symptom: Data corruption in results
- `cudaMemcpy` not transferring data correctly
- Check GVirtuS buffer serialization

### Symptom: Timeout/hang during execution
- `cudaStreamSynchronize()` never returning
- Kernel execution stuck on backend

---

## 8. Recommended GVirtuS Test Cases

1. **Device Query Test**:
   ```python
   # Verify cudaGetDeviceCount, cudaGetDeviceProperties
   spark.conf.set("spark.rapids.sql.enabled", "true")
   # Should log: "Initializing GPU device ID to 0"
   ```

2. **Memory Pool Test**:
   ```python
   # Force small pool to test allocation
   spark.conf.set("spark.rapids.memory.gpu.allocFraction", "0.1")
   ```

3. **Small Data Transfer Test**:
   ```python
   # Minimal data to test row-to-columnar conversion
   df = spark.createDataFrame([(1, 2, 3.0)], ["row", "col", "val"])
   df.count()  # Triggers GPU transfer
   ```

4. **Full Pipeline Test**:
   ```python
   # Matrix multiply - tests join, aggregate, shuffle
   result = df_a.join(df_b, df_a.col == df_b.row).groupBy().sum()
   ```
