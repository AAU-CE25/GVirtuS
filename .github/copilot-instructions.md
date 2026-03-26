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

# Smart GPUDirect GVirtuS: Research Project Summary

## 1. Project Overview

### Vision
Implement **Smart GPUDirect** into **GVirtuS** (GPU Virtualization Service) to enable
intelligent, adaptive GPU disaggregation over **SmartNIC-equipped HPC clusters**.
The system dynamically detects which CUDA API calls benefit from GPUDirect RDMA
and routes them through the optimal transport path.

### Hardware Setup
- **Two nodes** connected via **NVIDIA BlueField-3 (BF3)** DPUs
- **NVIDIA L4 GPUs** on the GPU server node
- RDMA fabric between nodes via BF3

### Core Research Question
> At what point does GPU disaggregation via Smart GPUDirect GVirtuS become
> a viable (and cost-efficient) alternative to local GPU access?

---

## 2. Architecture

### High-Level System Design

```
┌──────────────────────┐          RDMA Fabric (BF3)         ┌──────────────────────┐
│     CLIENT NODE      │ ◄──────────────────────────────────►│     SERVER NODE      │
│                      │                                      │                      │
│  App → GVirtuS FE    │    ┌─────────────────────────┐      │  GVirtuS BE → CUDA   │
│       │               │    │  Control Path (TCP/RDMA) │      │       │               │
│       ▼               │    │  Data Path (GPUDirect)   │      │       ▼               │
│  RDMA Transport      │    └─────────────────────────┘      │  RDMA Transport      │
│       │               │                                      │       │               │
│  BF3 NIC ◄───────────┼──────────────────────────────────────┼──► BF3 NIC           │
└──────────────────────┘                                      │       │               │
                                                              │       ▼               │
                                                              │   L4 GPU Memory      │
                                                              └──────────────────────┘
```

### GPUDirect RDMA Data Path

Normal flow (without GPUDirect):
```
GPU Memory → CPU/System RAM → Network Card → Network → Network Card → CPU/System RAM → GPU Memory
```

With GPUDirect RDMA:
```
GPU Memory → Network Card → Network → Network Card → GPU Memory
```

Eliminates multiple memory copies and CPU involvement, reducing latency and increasing throughput.

### Smart Detection Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│           CUDA API Call Classifier                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Is it a data transfer call?                             │
│     (cudaMemcpy, cudaMemcpyAsync, etc.)                    │
│     ├── YES → Check transfer size                           │
│     │   ├── > THRESHOLD → GPUDirect RDMA                    │
│     │   └── ≤ THRESHOLD → RDMA send/recv                    │
│     └── NO → Continue                                       │
│                                                              │
│  2. Is it a control/query call?                             │
│     (cudaGetDeviceProperties, etc.)                         │
│     ├── YES → Check cache                                   │
│     │   ├── Cached → Return locally                         │
│     │   └── Not cached → RDMA send/recv                     │
│     └── NO → Continue                                       │
│                                                              │
│  3. Is it a kernel launch?                                  │
│     ├── YES → RDMA send/recv (small)                        │
│     └── NO → Default RDMA send/recv                         │
│                                                              │
│  Future: ML-based prediction of next calls                  │
│  to pre-register memory / prefetch                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Technology Stack

| Component | Purpose |
|---|---|
| **GVirtuS** | GPU virtualisation framework (our project) |
| **libibverbs** | Low-level RDMA verbs API |
| **librdmacm** | RDMA connection management |
| **CUDA Driver API** | GPU memory management |
| **nvidia-peermem** | Kernel module enabling RDMA registration of GPU memory |
| **GDRCopy** | Low-latency GPU memory copies (optional, for small transfers) |
| **DOCA SDK** | BF3 DPU programming (optional, for DPU offloading) |
| **NVIDIA RAPIDS** | GPU-accelerated Spark (primary benchmark workload) |

### nvidia-peermem Setup
```bash
$ modprobe nvidia-peermem
```
Hooks into InfiniBand subsystem so `ibv_reg_mr()` works with GPU memory pointers,
enabling PCIe peer-to-peer DMA between BF3 NIC and L4 GPU.

### Server-Side RDMA + GPU Memory Registration
```cpp
#include <infiniband/verbs.h>
#include <cuda.h>

CUdeviceptr gpuBuf;
cuMemAlloc(&gpuBuf, bufferSize);

struct ibv_mr *mr = ibv_reg_mr(
    pd, (void*)gpuBuf, bufferSize,
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
);
// Share mr->rkey and gpuBuf address with client via control channel
// Client can now RDMA WRITE directly into GPU memory
```

### Client-Side RDMA Write to Remote GPU
```cpp
void* hostBuf = malloc(bufferSize);
struct ibv_mr *clientMr = ibv_reg_mr(pd, hostBuf, bufferSize, IBV_ACCESS_LOCAL_WRITE);

struct ibv_send_wr wr = {};
wr.opcode = IBV_WR_RDMA_WRITE;
wr.wr.rdma.remote_addr = remoteGpuAddr;
wr.wr.rdma.rkey = remoteRkey;
wr.sg_list = &sge;

ibv_post_send(qp, &wr, &bad_wr);
```

### GPU Memory Registration Manager
```cpp
class RdmaGpuMemoryManager {
    std::unordered_map<CUdeviceptr, ibv_mr*> registrationCache;

    ibv_mr* getOrRegister(CUdeviceptr gpuPtr, size_t size) {
        auto it = registrationCache.find(gpuPtr);
        if (it != registrationCache.end())
            return it->second;

        ibv_mr* mr = ibv_reg_mr(pd, (void*)gpuPtr, size,
                                 IBV_ACCESS_LOCAL_WRITE |
                                 IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ);
        registrationCache[gpuPtr] = mr;
        return mr;
    }
};
```

---

## 4. Implementation Design Decisions

### Split Control and Data Paths
- **Control path**: API call marshalling, small arguments, return codes → RDMA send/recv or TCP
- **Data path**: Large buffer transfers (cudaMemcpy, kernel args) → GPUDirect RDMA

### Transfer Size Threshold
- Configurable threshold (e.g., 4KB-64KB)
- Below threshold: regular RDMA send/recv
- Above threshold: GPUDirect RDMA
- Threshold is a tunable parameter determined by benchmarks, potentially adaptive at runtime

### Memory Registration Strategy
- **Eager**: Pre-register GPU memory pools at `cudaMalloc` time
- **Lazy**: Register on-demand with cache to avoid re-registration
- Hybrid approach recommended

### GVirtuS Integration Points
1. **Transport layer** (communicator classes) → add RDMA communicator
2. **`cudaMemcpy` handler** → intercept and route large transfers through GPUDirect
3. **`cudaMalloc`/`cudaFree` handlers** → manage RDMA memory registration lifecycle
4. **Frontend proxy** → track client-side buffer to remote GPU allocation mappings

### Optimisation Layer Stack
```
Layer 1: GPUDirect RDMA transport      → eliminates copies for large transfers
Layer 2: RDMA send/recv for control    → fast API call delivery
Layer 3: API batching                  → reduces number of round trips
Layer 4: Client-side caching           → eliminates unnecessary round trips
Layer 5: Stream-aware pipelining       → overlaps everything
```

---

## 5. Primary Use Case: Apache Spark with RAPIDS

### Why Spark is Ideal

Spark workloads are primarily CPU-heavy with occasional GPU bursts — the exact profile
where GPU disaggregation wins over local GPU:

```
CPU ████████████████████████████░░░░ GPU
     ◄── Heavy (90%+) ──►  ◄─ Burst ─►
```

### Deployment Architecture

```
Traditional (expensive):
  3 Spark nodes × 1 GPU each = 3 GPUs (mostly idle)

Smart GPUDirect GVirtuS (cost-efficient):
  3 Spark CPU nodes + 1 shared GPU node = 1 GPU (fully utilised)
```

### RAPIDS Operations → Smart Transport Mapping

| Operation | Transfer Size | Compute | Smart Decision |
|---|---|---|---|
| SQL Join | Large (MB-GB) | Heavy | GPUDirect RDMA |
| Aggregation (SUM, AVG) | Medium | Medium | GPUDirect if > threshold |
| Sort | Large | Heavy | GPUDirect RDMA |
| cuML Training | Large upload, small download | Very heavy | GPUDirect upload, regular RDMA download |
| cuML Inference | Small-medium | Medium | Regular RDMA |
| Device Query | Tiny | None | Cached locally |

### Relevant GPU-Accelerated Spark Projects

| Project | What It Offloads |
|---|---|
| NVIDIA RAPIDS Accelerator for Spark | SQL operations, joins, aggregations, shuffles |
| RAPIDS cuDF | DataFrame operations |
| RAPIDS cuML | ML algorithms (PCA, k-means, random forest) |
| Spark + TensorRT | Inference within Spark pipelines |

---

## 6. Benchmarking Framework

### Test Configurations

```
Config A: Local GPU      → App → CUDA → L4 GPU (same node)
Config B: GVirtuS TCP    → App → GVirtuS FE → TCP → GVirtuS BE → L4 GPU
Config C: GVirtuS RDMA   → App → GVirtuS FE → RDMA → GVirtuS BE → L4 GPU
Config D: GVirtuS Smart  → App → GVirtuS FE → Smart GPUDirect → L4 GPU
```

### Metrics

| Metric | Tool |
|---|---|
| End-to-end latency per API call | Custom GVirtuS instrumentation |
| Data transfer throughput (GB/s) | ib_write_bw, custom CUDA benchmarks |
| GPU utilisation | nvidia-smi, DCGM |
| CPU utilisation (both nodes) | perf, mpstat |
| PCIe bandwidth utilisation | nvidia-smi, lspci counters |
| SmartNIC utilisation | DOCA telemetry |
| Application-level throughput | Workload-specific |

### Micro-Benchmarks

| ID | Benchmark | Purpose |
|---|---|---|
| MB1 | cudaMemcpy latency vs transfer size (1B–1GB sweep) | Find GPUDirect crossover point |
| MB2 | cudaMalloc/cudaFree overhead | Pure control path cost |
| MB3 | Kernel launch latency (empty kernel) | Baseline GVirtuS overhead |
| MB4 | Concurrent streams | Pipeline effectiveness |
| MB5 | Bidirectional transfer | RDMA bidirectional bandwidth |
| MB6 | Smart detection overhead | Is classification a bottleneck? |

### Application-Level Benchmarks

| Workload | Compute/Transfer Ratio | Purpose |
|---|---|---|
| ResNet-50 inference | High compute, small transfers | Best case for disaggregation |
| BERT/GPT inference | Medium/medium | Realistic AI serving |
| Image preprocess + inference | CPU-heavy + GPU bursts | Pipelining use case |
| GEMM sweep | Variable | Classic HPC, controllable ratio |
| FFT (cuFFT) | Low compute, large transfers | Worst case stress test |
| Video transcoding | Streaming, continuous GPU | Realistic media workload |

### Spark-Specific Benchmarks

| ID | Benchmark | Description |
|---|---|---|
| SB1 | Large Join | Two large tables, hash join on GPU |
| SB2 | Aggregation Pipeline | GROUP BY + multiple aggregations |
| SB3 | ETL Pipeline | CSV → Transform → Filter → Parquet |
| SB4 | ML Training Pipeline | Preprocessing → Feature eng → cuML train |
| SB5 | ML Inference Pipeline | Batch scoring with trained model |
| SB6 | Iterative Algorithm | PageRank/K-Means, repeated GPU calls |
| SB7 | Mixed Workload | Concurrent queries, multi-tenant GPU sharing |

### Industry Standard Benchmarks

| Benchmark | Operations | Scale |
|---|---|---|
| TPC-H | Analytical queries (joins, aggregations, sorts) | 1GB–1TB |
| TPC-DS | Complex decision support queries | 1GB–1TB |
| TPC-xBB (BigBench) | ML + SQL mixed workload | Configurable |

### Crossover Point Experiment

```
                Performance
                (throughput)
                     │
      Local GPU      │         _______________
                     │        /
                     │       /   Smart GPUDirect GVirtuS
                     │      /
                     │     /
       ─────────────┼────/──────────────────────
                     │   /
                     │  /   GVirtuS TCP
                     │
                     └──────────────────────────── Compute/Transfer
                          ◄── Transfer    Compute ──►
                          heavy           heavy
                              ▲
                         CROSSOVER POINT
```

### Cost-Performance Analysis (Key Result)

```
Config                  | Nodes | GPUs | Cost  | TPC-H Time | Cost-Efficiency
────────────────────────────────────────────────────────────────────────────
3 nodes, each with GPU  |   3   |  3   | $$$   |   100s     |  Low (idle GPUs)
3 nodes, no GPU (CPU)   |   3   |  0   |  $    |   500s     |  Medium
3 nodes + 1 GPU node    |   4   |  1   |  $$   |   ~120s    |  HIGH ← target
(Smart GPUDirect)       |       |      |       |            |
```

### Benchmark Code Structure

```
gvirtus-benchmark/
├── config/
│   ├── local_gpu.yaml
│   ├── gvirtus_tcp.yaml
│   ├── gvirtus_rdma.yaml
│   └── gvirtus_smart_gpudirect.yaml
├── micro/
│   ├── memcpy_sweep.cu
│   ├── malloc_free.cu
│   ├── kernel_launch.cu
│   ├── concurrent_streams.cu
│   ├── bidirectional.cu
│   └── detection_overhead.cu
├── applications/
│   ├── resnet50_inference/
│   ├── bert_inference/
│   ├── preprocessing_pipeline/
│   ├── gemm_sweep/
│   ├── cufft_sweep/
│   └── video_transcode/
├── spark/
│   ├── tpch/
│   ├── tpcds/
│   ├── ml_pipeline/
│   └── mixed_workload/
├── harness/
│   ├── runner.py
│   ├── metrics_collector.py
│   ├── result_parser.py
│   └── plot_generator.py
├── scripts/
│   ├── setup_rdma.sh
│   ├── setup_gvirtus.sh
│   └── setup_nvidia_peermem.sh
└── results/
```

---

## 7. Phased Implementation Plan

| Phase | Goal | Output |
|---|---|---|
| Phase 1 | Basic RDMA transport in GVirtuS (no GPUDirect) | Working RDMA communicator |
| Phase 2 | Add GPUDirect RDMA for cudaMemcpy | nvidia-peermem integration |
| Phase 3 | Implement smart detection/classification | Adaptive transport selection |
| Phase 4 | Micro-benchmarks (MB1–MB6) | Crossover point data |
| Phase 5 | Spark RAPIDS integration + benchmarks | Application-level results |
| Phase 6 | Cost-performance analysis | Publication-ready results |

---

## 8. Research Narrative

### Primary Narrative
> **"Smart GPUDirect: Cost-Efficient GPU Disaggregation for Data Analytics on SmartNIC-Equipped Clusters"**
>
> GPU-accelerated data analytics (Spark RAPIDS) requires expensive GPUs on every node,
> yet GPUs sit idle during CPU-heavy phases. We present Smart GPUDirect GVirtuS, which
> enables transparent GPU disaggregation with intelligent transport selection over
> SmartNIC (BF3) interconnects. Using TPC-H/DS benchmarks on Spark RAPIDS, we show
> that a shared GPU pool achieves X% of local GPU performance at Y% of the cost.

### Core Technical Contribution
> **"Intelligent API-Aware Transport Selection for GPU Virtualisation"**
>
> Not all CUDA API calls benefit equally from GPUDirect RDMA. We present a smart
> classification mechanism that dynamically selects the optimal transport per call,
> achieving near-local GPU performance while minimising RDMA resource consumption.

### Follow-Up Paper
> **"DPU-Accelerated GPU Virtualisation for Multi-Tenant HPC"**
>
> SmartNICs can offload GPU virtualisation overhead from host CPUs. We demonstrate
> GVirtuS with BF3-offloaded Smart GPUDirect, enabling multi-tenant GPU sharing
> with minimal host CPU impact.

### Target Venues
- **HPC**: SC, HPDC, ICS
- **Systems**: USENIX ATC, EuroSys, Middleware, VEE
- **Databases/Analytics**: VLDB, SIGMOD
- **Architecture**: ASPLOS, NSDI, SoCC

---

## 9. Justification: Why Two Nodes > One GPU Node

| Scenario | Why Disaggregation Wins |
|---|---|
| CPU-hungry + occasional GPU bursts | More CPU/RAM available, GPU shared efficiently |
| Cluster cost optimisation | Fewer GPUs needed, higher utilisation |
| Workload pipelining | Node A preprocesses → Node B GPU computes |
| Fault isolation | GPU failure doesn't kill compute job |
| Security/multi-tenancy | Untrusted code never touches GPU node directly |
| Independent hardware refresh | Upgrade GPUs without replacing all nodes |

### Honest Limitation
For latency-sensitive, tightly-coupled GPU workloads (iterative training with small batches),
local GPU will almost always win. The research contribution is **quantifying the crossover point**.

---

## 10. Related Work & Novelty

### Closest Existing Work

| Project | Relation | Gap |
|---|---|---|
| rCUDA | Remote CUDA with RDMA | No smart detection, no SmartNIC, no Spark focus |
| Bitfusion (VMware) | Commercial GPU disaggregation | Proprietary, discontinued/absorbed |
| NVIDIA RAPIDS | GPU-accelerated Spark | Assumes local GPU only |
| MVAPICH2-GDR | GPUDirect RDMA + MPI | Not for virtualisation |
| GVirtuS (current) | GPU virtualisation | No RDMA, no GPUDirect, no smart transport |

### Novel Combination (Not Previously Done)
```
Smart GPUDirect + GVirtuS + SmartNIC (BF3) + Spark RAPIDS + Adaptive transport selection
```

### Recommended Literature Search Queries
```
"GPU disaggregation RDMA Spark"
"remote GPU virtualization RDMA analytics"
"rCUDA GPUDirect"
"GVirtuS RDMA"
"SmartNIC GPU offloading"
"DPU GPU virtualization"
"GPU as a service HPC cluster"
"network attached GPU data analytics"
```

### Key Databases
- Google Scholar, ACM Digital Library, IEEE Xplore
- USENIX proceedings, arXiv (cs.DC, cs.PF, cs.DB)
```
