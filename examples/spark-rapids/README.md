# E-Commerce Customer Analytics Pipeline — Spark RAPIDS Benchmark

## Use Case

This is a **realistic retail/e-commerce analytics workload** that mirrors what
companies like Amazon, Shopify, and Walmart run daily. It covers:
- Large joins across multiple tables
- Heavy aggregations and window functions
- ML model training (customer segmentation)
- Scalable data generation

The benchmark is designed to compare performance across:
- **Spark CPU** (baseline)
- **Spark RAPIDS with local GPU**
- **Spark RAPIDS with GVirtuS virtual GPU** (remote GPU over SmartNIC)

---

## Prerequisites

### Version Requirements

| Component | Required Version | Notes |
|---|---|---|
| **Python** | 3.10+ | 3.10.12 tested and works |
| **Java (OpenJDK)** | 17+ | PySpark 3.5.x requires Java 17 |
| **PySpark** | 3.5.4 | RAPIDS 24.10 supports up to Spark 3.5.x |
| **RAPIDS Accelerator JAR** | 24.10.0 | Must match Spark version |
| **CUDA** | 12.x | Required on GPU node only |
| **NVIDIA Driver** | 535+ | Required on GPU node only |

### Hardware Setup

| Node | Role | Components |
|---|---|---|
| **Worker Node** (es-dpu-01) | Spark executor, GVirtuS frontend | CPU, RAM, BF3 SmartNIC |
| **GPU Node** (es-dpu-02) | GVirtuS backend, GPU server | CPU, RAM, NVIDIA L4 GPU, BF3 SmartNIC |

---

## Setup Instructions

### Step 1: Install Java 17

PySpark 3.5+ requires Java 17. If your system has an older version, install locally:

```bash
# Check current Java version
java -version
# If it shows Java 11 or older, install Java 17:

# Download OpenJDK 17
cd /tmp
wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz

# Extract and install to home directory
tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
mkdir -p ~/.local
mv jdk-17.0.2 ~/.local/jdk-17

# Clean up
rm /tmp/openjdk-17.0.2_linux-x64_bin.tar.gz

# Add to bashrc
echo 'export JAVA_HOME="$HOME/.local/jdk-17"' >> ~/.bashrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
java -version
# Should show: openjdk version "17.0.2"
```

### Step 2: Create Python Virtual Environment

```bash
# Navigate to project directory
cd ~/GVirtuS/examples/spark-rapids

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# IMPORTANT: Add Java 17 to venv activation script
# so it's set automatically every time you activate
echo '' >> .venv/bin/activate
echo '# Java 17 for PySpark' >> .venv/bin/activate
echo 'export JAVA_HOME="$HOME/.local/jdk-17"' >> .venv/bin/activate
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> .venv/bin/activate

# Deactivate and reactivate to test
deactivate
source .venv/bin/activate

# Verify Java is correct inside venv
java -version
# Should show: openjdk version "17.0.2"
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install pyspark==3.5.4 numpy pandas pyarrow matplotlib seaborn

# Verify installations
python -c "
import pyspark
import numpy
import pandas
import pyarrow
print(f'PySpark:  {pyspark.__version__}')
print(f'NumPy:    {numpy.__version__}')
print(f'Pandas:   {pandas.__version__}')
print(f'PyArrow:  {pyarrow.__version__}')
print('All good!')
"
```

### Step 4: Download RAPIDS Accelerator JAR

The RAPIDS plugin is a Java JAR file that Spark loads at runtime.
It must match your Spark version (3.5.x → RAPIDS 24.10.0).

```bash
# Create jars directory
mkdir -p ~/GVirtuS/examples/spark-rapids/jars

# Download RAPIDS Accelerator for Spark 3.5.x
cd ~/GVirtuS/examples/spark-rapids/jars
wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.10.0/rapids-4-spark_2.12-24.10.0.jar

# Verify download
ls -lh
# Should show: rapids-4-spark_2.12-24.10.0.jar (~500MB)

# Go back to project root
cd ~/GVirtuS/examples/spark-rapids
```

### Step 5: Verify Full Setup

```bash
# Make sure venv is active
source .venv/bin/activate

# Check everything
echo "=== Java ==="
java -version

echo "=== Python ==="
python --version

echo "=== PySpark ==="
python -c "import pyspark; print(f'PySpark: {pyspark.__version__}')"

echo "=== RAPIDS JAR ==="
ls -lh jars/rapids-4-spark_2.12-24.10.0.jar

echo "=== Quick Spark Test ==="
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('test').master('local[*]').getOrCreate()
print(f'Spark version: {spark.version}')
df = spark.range(10)
print(f'Test count: {df.count()}')
spark.stop()
print('Spark is working!')
"
```

Expected output:
```
=== Java ===
openjdk version "17.0.2" 2022-01-18
=== Python ===
Python 3.10.12
=== PySpark ===
PySpark: 3.5.4
=== RAPIDS JAR ===
-rw-r--r-- 1 user user 500M ... rapids-4-spark_2.12-24.10.0.jar
=== Quick Spark Test ===
Spark version: 3.5.4
Test count: 10
Spark is working!
```

---

## Project Structure

```
spark-rapids-benchmark/
├── .venv/                  # Python virtual environment
├── jars/
│   └── rapids-4-spark_2.12-24.10.0.jar  # RAPIDS plugin
├── data/
│   └── sf1/                # Generated data (after running datagen.py)
│       ├── customers.parquet
│       ├── products.parquet
│       ├── orders.parquet
│       ├── order_items.parquet
│       └── clickstream.parquet
├── results/                # Benchmark results (after running pipelines)
├── config.py               # Shared configuration
├── datagen.py              # Scalable synthetic data generator
├── pipeline_spark.py       # Native Spark (CPU) version
├── pipeline_rapids.py      # RAPIDS-accelerated (GPU) version
├── run_benchmark.py        # Benchmark harness
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Configuration

Edit `config.py` to adjust scale and resources:

```python
# config.py

"""
Scalable benchmark configuration.
Adjust SCALE_FACTOR to control data size:
  SCALE_FACTOR = 1    → ~1 GB   (development/testing)
  SCALE_FACTOR = 10   → ~10 GB  (standard benchmark)
  SCALE_FACTOR = 100  → ~100 GB (stress test)
"""

SCALE_FACTOR = 1

# Derived sizes (linear scaling)
NUM_CUSTOMERS = 1_000_000 * SCALE_FACTOR
NUM_PRODUCTS = 100_000
NUM_ORDERS = 10_000_000 * SCALE_FACTOR
NUM_ORDER_ITEMS = 30_000_000 * SCALE_FACTOR
NUM_CLICKSTREAM = 50_000_000 * SCALE_FACTOR

# Paths — use int() to avoid sf0.1 path issues
DATA_DIR = f"data/sf{int(SCALE_FACTOR)}"
RESULTS_DIR = f"results/sf{int(SCALE_FACTOR)}"

# Spark config
SPARK_MASTER = "local[*]"  # Change to cluster URL for distributed
SPARK_EXECUTOR_MEMORY = "8g"
SPARK_DRIVER_MEMORY = "4g"
```

---

## Data Generation

The data generator creates Spark-compatible Parquet files using PyArrow
with microsecond timestamp precision (required by PySpark 4.x / 3.5.x).

```bash
# Generate data at scale factor 1 (~1GB)
python datagen.py
```

### Generated Tables

| Table | Rows (SF=1) | Description |
|---|---|---|
| `customers.parquet` | 1,000,000 | Customer demographics, tier, income |
| `products.parquet` | 100,000 | Product catalog with categories |
| `orders.parquet` | 10,000,000 | Transaction log with dates, status |
| `order_items.parquet` | 30,000,000 | Line items per order |
| `clickstream.parquet` | 50,000,000 | Web browsing events (largest table) |

### Important: Timestamp Compatibility

The data generator uses PyArrow to write timestamps in **microsecond precision**
(`timestamp[us]`). This is required because PySpark rejects nanosecond timestamps
with the error:
```
PARQUET_TYPE_ILLEGAL: Illegal Parquet type: INT64 (TIMESTAMP(NANOS,false))
```

The `save_parquet()` function in `datagen.py` handles this automatically by:
1. Flooring all datetime columns to microsecond precision
2. Casting PyArrow timestamp fields to `pa.timestamp("us")`

---

## Running the Benchmarks

### 1. CPU-Only Baseline (No GPU Required)

```bash
python pipeline_spark.py
```

This runs the full analytics pipeline using only CPU. Works on any node.

### 2. RAPIDS GPU-Accelerated (Requires GPU)

```bash
python pipeline_rapids.py
```

This requires:
- A node with an NVIDIA GPU (or GVirtuS vGPU)
- The RAPIDS JAR in `jars/` directory
- CUDA drivers installed

### 3. RAPIDS with GVirtuS Virtual GPU (Remote GPU)

```bash
python pipeline_rapids.py --gvirtus
```

This runs on a node **without a physical GPU**, using GVirtuS to access
a remote GPU on another node. Requires:
- GVirtuS frontend library installed
- GVirtuS backend running on the GPU server
- Update the GVirtuS paths in `pipeline_rapids.py`

### 4. Full Benchmark Comparison

```bash
# Run all configurations with 3 iterations
python run_benchmark.py --configs cpu gpu gvirtus_smart --scale-factors 1 10 --iterations 3

# Generate data and run
python run_benchmark.py --configs all --scale-factors 1 --iterations 3 --generate-data
```

---

## Pipeline Stages

The analytics pipeline has 7 stages, each representing a real business task:

| Stage | Description | Real-World Parallel | GPU Benefit |
|---|---|---|---|
| 1. Data Loading | Read parquet files | Data lake ingestion | Medium (GPU I/O) |
| 2. Revenue Analytics | 4-table join + aggregation | BI dashboard query | High (join + agg) |
| 3. Customer 360 | Multi-source customer profile | CDP build | High (large joins) |
| 4. RFM Segmentation | Window functions + scoring | Marketing segmentation | Medium (window ops) |
| 5. Cohort Analysis | Time-based retention | Product analytics | Medium (groupBy) |
| 6. Funnel Analysis | Clickstream aggregation | Web analytics | High (largest table) |
| 7. Customer Clustering | K-Means ML training | Data science | Very High (iterative ML) |

---

## RAPIDS Plugin Configuration

The RAPIDS plugin in `pipeline_rapids.py` requires these key configurations:

```python
# RAPIDS JAR — must be on classpath
.config("spark.jars", "jars/rapids-4-spark_2.12-24.10.0.jar")
.config("spark.driver.extraClassPath", "jars/rapids-4-spark_2.12-24.10.0.jar")
.config("spark.executor.extraClassPath", "jars/rapids-4-spark_2.12-24.10.0.jar")

# Enable RAPIDS plugin
.config("spark.plugins", "com.nvidia.spark.SQLPlugin")
.config("spark.rapids.sql.enabled", "true")

# Kryo serialization — RAPIDS REQUIRES this registrator
.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
.config("spark.kryo.registrator", "com.nvidia.spark.rapids.GpuKryoRegistrator")

# GPU resources
.config("spark.executor.resource.gpu.amount", "1")
.config("spark.task.resource.gpu.amount", "0.5")
.config("spark.rapids.sql.concurrentGpuTasks", "2")
```

### Common RAPIDS Errors and Fixes

| Error | Fix |
|---|---|
| `ClassNotFoundException: com.nvidia.spark.SQLPlugin` | Download RAPIDS JAR and add to `spark.jars` config |
| `spark.kryo.registrator` error | Add `.config("spark.kryo.registrator", "com.nvidia.spark.rapids.GpuKryoRegistrator")` |
| `PARQUET_TYPE_ILLEGAL: TIMESTAMP(NANOS)` | Regenerate data using `datagen.py` with PyArrow microsecond timestamps |
| `UnsupportedClassVersionError: class file version 61.0` | Install Java 17 (see Step 1) |
| `PATH_NOT_FOUND: data/sf0.1/` | Set `SCALE_FACTOR = 1` (integer) in `config.py` |

---

## Expected Output

### CPU Pipeline
```
============================================================
  E-Commerce Analytics Pipeline — Native Spark (CPU)
============================================================

  Stage Timings:
    1_data_loading                    12.34s  ( 5.2%) ██
    2_revenue_analytics               45.67s  (19.3%) █████████
    3_customer_360                    78.90s  (33.3%) ████████████████
    4_rfm_segmentation                23.45s  ( 9.9%) ████
    5_cohort_analysis                 15.67s  ( 6.6%) ███
    6_funnel_analysis                 34.56s  (14.6%) ███████
    7_customer_clustering             26.78s  (11.3%) █████

  Pipeline Complete — Total: 237.37s
```

### Benchmark Comparison
```
══════════════════════════════════════════════════════════════
  BENCHMARK COMPARISON REPORT
══════════════════════════════════════════════════════════════

  Scale Factor: 10 (~10GB)
  ────────────────────────────────────────────────────────────
  Configuration                              Avg Time     vs CPU  vs Local GPU
  ────────────────────────────────────────────────────────────
  Spark CPU (No GPU)                          450.0s      1.00x        6.43x
  Spark RAPIDS (Local GPU)                     70.0s      6.43x        1.00x
  Spark RAPIDS (GVirtuS TCP)                  120.0s      3.75x        1.71x
  Spark RAPIDS (GVirtuS RDMA)                  95.0s      4.74x        1.36x
  Spark RAPIDS (GVirtuS Smart GPUDirect)       82.0s      5.49x        1.17x ★
```

---

## Quick Start (Copy-Paste)

```bash
# === ONE-TIME SETUP ===

# Install Java 17 locally
cd /tmp
wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
mkdir -p ~/.local && mv jdk-17.0.2 ~/.local/jdk-17
rm openjdk-17.0.2_linux-x64_bin.tar.gz
echo 'export JAVA_HOME="$HOME/.local/jdk-17"' >> ~/.bashrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create project and venv
cd ~/GVirtuS/examples/spark-rapids
python3 -m venv .venv
source .venv/bin/activate

# Add Java to venv
echo 'export JAVA_HOME="$HOME/.local/jdk-17"' >> .venv/bin/activate
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> .venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install pyspark==3.5.4 numpy pandas pyarrow matplotlib seaborn

# Download RAPIDS JAR
mkdir -p jars
cd jars
wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.10.0/rapids-4-spark_2.12-24.10.0.jar
cd ..

# === EVERY TIME ===

# Activate environment
cd ~/GVirtuS/examples/spark-rapids
source .venv/bin/activate

# Generate data
python datagen.py

# Run CPU baseline
python pipeline_spark.py

# Run RAPIDS (needs GPU node)
python pipeline_rapids.py

# Run with GVirtuS (needs GVirtuS setup)
python pipeline_rapids.py --gvirtus
```

---

## Troubleshooting

### Java version mismatch inside venv
If `java -version` shows Java 11 after activating the venv:
```bash
export JAVA_HOME="$HOME/.local/jdk-17"
export PATH="$JAVA_HOME/bin:$PATH"
```

### Disk space issues with apt
If `sudo apt install` fails due to disk space, install Java locally
(see Step 1 — no sudo required).

### Parquet timestamp errors
Delete old data and regenerate:
```bash
rm -rf data/
python datagen.py
```

### Verify parquet timestamp format
```bash
python -c "
import pyarrow.parquet as pq
schema = pq.read_schema('data/sf1/orders.parquet')
print(schema)
# Should show timestamp[us] NOT timestamp[ns]
"
