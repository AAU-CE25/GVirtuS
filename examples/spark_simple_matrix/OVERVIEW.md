# Spark Simple Matrix - Overview

This example demonstrates **Apache Spark** performing matrix multiplication using three different execution environments:

1. **Local (native)** - Run directly on the host machine
2. **Docker (local GPU)** - Run inside Docker with local GPU access
3. **Docker (GVirtuS)** - Run inside Docker with remote GPU via GVirtuS

## What This Example Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MATRIX MULTIPLICATION                               │
│                                                                             │
│   Matrix A (NxN)         Matrix B (NxN)         Matrix C (NxN)             │
│   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│   │ a00 a01 ... │   x    │ b00 b01 ... │   =    │ c00 c01 ... │            │
│   │ a10 a11 ... │        │ b10 b11 ... │        │ c10 c11 ... │            │
│   │ ... ... ... │        │ ... ... ... │        │ ... ... ... │            │
│   └─────────────┘        └─────────────┘        └─────────────┘            │
│                                                                             │
│   Where: C[i][j] = Σ(k) A[i][k] * B[k][j]                                  │
│   Matrix size: N = 100 × SCALE_FACTOR                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

The benchmark uses **Spark DataFrames** with `join` + `groupBy` + `sum` operations, which the **RAPIDS SQL Plugin** can accelerate on GPU.

## Execution Modes

### CPU Mode (`--mode cpu`)
- Pure CPU execution using Spark's Catalyst optimizer
- No GPU required
- Baseline for comparison

### RAPIDS Mode (`--mode rapids`)
- GPU-accelerated execution using NVIDIA RAPIDS
- Requires GPU access (local or via GVirtuS)
- Spark SQL operations are offloaded to GPU

### Overwrite Mode (`--overwrite yes|no`)
- `yes` (default): Replace the results file entirely
- `no`: Merge new results into the existing file (top-level keys are updated/added)

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION ENVIRONMENTS                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  1. LOCAL (Native)                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │  │
│  │  │ Python App  │ -> │   PySpark   │ -> │ Local GPU   │                   │  │
│  │  │ simple_     │    │ + RAPIDS    │    │ (L40s)      │                   │  │
│  │  │ matrix.py   │    │   Plugin    │    │             │                   │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                   │  │
│  │                                                                          │  │
│  │  Command: make run-spark-local                                           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  2. DOCKER (Local GPU)                                                   │  │
│  │  ┌─────────────────────────────────────────────┐                         │  │
│  │  │ Docker Container                            │                         │  │
│  │  │ ┌─────────────┐    ┌─────────────┐         │    ┌─────────────┐      │  │
│  │  │ │ Python App  │ -> │   PySpark   │ ────────┼────│ Local GPU   │      │  │
│  │  │ │             │    │ + RAPIDS    │ --gpus  │    │ (L40s)      │      │  │
│  │  │ └─────────────┘    └─────────────┘         │    └─────────────┘      │  │
│  │  └─────────────────────────────────────────────┘                         │  │
│  │                                                                          │  │
│  │  Command: make run-spark-docker-local                                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  3. DOCKER (GVirtuS)                                                     │  │
│  │  ┌─────────────────────────────────────────────┐                         │  │
│  │  │ Docker Container (GPU-less)                 │                         │  │
│  │  │ ┌─────────────┐    ┌─────────────┐         │        Network          │  │
│  │  │ │ Python App  │ -> │   PySpark   │         │          │              │  │
│  │  │ │             │    │ + RAPIDS    │         │          │              │  │
│  │  │ └─────────────┘    └─────┬───────┘         │          │              │  │
│  │  │                          │                 │          │              │  │
│  │  │                    ┌─────▼───────┐         │          │              │  │
│  │  │                    │  GVirtuS    │ ────────┼──────────┘              │  │
│  │  │                    │  Frontend   │         │                         │  │
│  │  │                    │  (stubs)    │         │                         │  │
│  │  │                    └─────────────┘         │                         │  │
│  │  └─────────────────────────────────────────────┘                         │  │
│  │                              │                                           │  │
│  │                              │ GVIRTUS                                  │  │
│  │                              ▼                                           │  │
│  │                    ┌─────────────────────┐    ┌─────────────┐            │  │
│  │                    │  GVirtuS Backend    │ -> │ Remote GPU  │            │  │
│  │                    │  (es-dpu-01)        │    │ (L40s)      │            │  │
│  │                    └─────────────────────┘    └─────────────┘            │  │
│  │                                                                          │  │
│  │  Command: make run-spark-docker-gvirtus                                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
spark_simple_matrix/
├── OVERVIEW.md              # This file
├── README.md                # Quick start guide
├── requirements.txt         # Python dependencies
│
├── Dockerfile.local         # Docker image for local GPU mode
├── Dockerfile.gvirtus       # Docker image for GVirtuS mode
├── entrypoint-local-gpu.sh  # Entrypoint for Docker local GPU
├── entrypoint-gvirtus.sh    # Entrypoint for Docker GVirtuS
│
├── src/
│   ├── config.py            # Configuration (scale factor, Spark settings)
│   └── simple_matrix.py     # Main benchmark script
│
├── jars/                    # RAPIDS JAR (not in git)
│   └── rapids-4-spark_2.12-26.02.1.jar
│
└── results/                 # Benchmark results
    └── sf{N}/               # Results per scale factor
        └── simple_matrix_{env}_{mode}_results.json
```

## Quick Start

### Prerequisites

- Python 3.10+ and Java 17+ (for native mode)
- Docker with NVIDIA Container Toolkit (for Docker modes)
- NVIDIA GPU with drivers (for GPU modes)
- GVirtuS backend running (for GVirtuS mode)

### Download RAPIDS JAR

```bash
mkdir -p jars
wget -P jars/ https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.02.1/rapids-4-spark_2.12-26.02.1.jar
```

### Run Commands

From the GVirtuS root directory:

```bash
# 1. Local native execution
make run-spark-local-cpu        # CPU only
make run-spark-local-rapids     # GPU with RAPIDS

# 2. Docker with local GPU
make run-spark-docker-local     # Runs both CPU and RAPIDS modes

# 3. Docker with GVirtuS (requires backend running)
make run-spark-docker-gvirtus   # RAPIDS via GVirtuS
```

## Configuration

Edit `src/config.py` to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCALE_FACTOR` | 1 | Matrix size = 100 × SCALE_FACTOR |
| `LOG_LEVEL` | "DEBUG" | Python log level |
| `SPARK_MASTER` | "local[4]" | Spark master URL |
| `SPARK_CONFIG` | ... | Spark configuration for CPU mode |
| `SPARK_RAPIDS_CONFIG` | ... | Spark configuration with RAPIDS enabled |

## Results

Results are saved to `results/sf{SCALE_FACTOR}/simple_matrix_{env}_{mode}_results.json`:

```json
{
  "env": "docker",
  "mode": "rapids",
  "scale_factor": 1,
  "matrix_size": 100,
  "result_elements": 10000,
  "elapsed_seconds": 12.3456,
  "matrix_multiplication_time": 8.1234,
  "spark_config": { ... }
}
```

## How RAPIDS Acceleration Works

1. **Spark SQL Plugin**: The RAPIDS plugin intercepts Spark's Catalyst query plans
2. **GPU Operators**: SQL operations (join, groupBy, sum) are replaced with GPU equivalents
3. **Data Transfer**: Data is transferred to/from GPU memory automatically
4. **With GVirtuS**: CUDA calls are intercepted and forwarded over the network to a remote GPU

```
┌────────────────────────────────────────────────────────────────┐
│                  RAPIDS SQL Pipeline                           │
│                                                                │
│  DataFrame API  -->  Catalyst  -->  GPU Operators  -->  Result │
│      (join)          (plan)         (cuDF)                     │
│      (groupBy)                                                 │
│      (sum)                                                     │
└────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "No RAPIDS JAR found"
Download the JAR as shown in Quick Start, or update `RAPIDS_JAR_PATH` in `config.py`.

### "GPU not available" in Docker
Ensure you have NVIDIA Container Toolkit installed and use `--gpus all` flag.

### GVirtuS connection failed
1. Check that GVirtuS backend is running on the configured host
2. Verify `properties.json` has correct `server_address` and `port`
3. Check network connectivity between client and backend
