# Spark RAPIDS E-Commerce Benchmark (Current Workspace Version)

This folder contains a practical Spark analytics benchmark with:
- CPU baseline (`pipeline_spark.py`)
- RAPIDS/GPU run (`pipeline_rapids.py`)
- Synthetic data generation (`datagen.py`)
- Multi-config benchmark harness (`run-benchmark.py`)

It is used to compare CPU vs GPU performance and (optionally) GVirtuS remote GPU transport modes.

## What is in this folder

- `config.py` — benchmark scale + Spark memory settings (currently `SCALE_FACTOR = 10`)
- `datagen.py` — generates parquet input datasets into `data/sfX/`
- `pipeline_spark.py` — full 7-stage analytics pipeline on CPU
- `pipeline_rapids.py` — same logic, intended to run with RAPIDS-enabled Spark submit
- `run-benchmark.py` — automation harness for configs/scale-factors/iterations
- `test_gpu.py` — quick Spark GPU sanity tests
- `jars/rapids-4-spark_2.12-25.02.1.jar` — RAPIDS plugin jar currently present
- `results/` — saved JSON outputs and markdown comparisons

## Current results already generated

- `results/sf1/spark_cpu_results.json`
- `results/sf1/rapids_gpu_results.json`
- `results/sf1/comparison.md`
- `results/sf10/spark_cpu_results.json`
- `results/sf10/rapids_gpu_results.json`
- `results/sf10/comparison_sf10.md`
- `results/comparison_all.md`

## Performance summary (from this workspace)

### sf1
- CPU total: **36.11s**
- GPU total: **40.43s**
- Overall speedup (CPU/GPU): **0.89x** (CPU slightly faster at small scale)

### sf10
- CPU total: **431.69s**
- GPU total: **92.63s**
- Overall speedup (CPU/GPU): **4.66x** (GPU clearly faster at larger scale)

### Takeaway
At small data size (`sf1`), GPU/plugin overhead can dominate. At larger size (`sf10`), GPU acceleration provides major gains, especially in heavy stages like clustering and segmentation.

## Prerequisites

- Python 3.10+
- Java 17+
- NVIDIA GPU + driver (for RAPIDS runs)
- PySpark and Python deps from `requirements.txt`

Install dependencies:

```bash
cd ~/GVirtuS/examples/spark-rapids
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Edit `config.py`:

- `SCALE_FACTOR` controls data/output paths (`data/sfX`, `results/sfX`)
- memory settings:
  - `SPARK_EXECUTOR_MEMORY`
  - `SPARK_DRIVER_MEMORY`

Current default in this workspace is `SCALE_FACTOR = 10`.

## Run commands

### 1) Generate data

```bash
cd ~/GVirtuS/examples/spark-rapids
source .venv/bin/activate
python datagen.py
```

### 2) CPU pipeline

```bash
python pipeline_spark.py
```

Writes:
- `results/sfX/spark_cpu_results.json`

### 3) RAPIDS GPU pipeline

`pipeline_rapids.py` creates a Spark session but does not inject RAPIDS jar/plugin settings itself. Run it with `spark-submit` and explicit RAPIDS configs:

```bash
spark-submit \
  --master local[4] \
  --driver-memory 4g \
  --jars jars/rapids-4-spark_2.12-25.02.1.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.explain=ALL \
  --conf spark.rapids.memory.gpu.pool=NONE \
  --conf spark.sql.shuffle.partitions=4 \
  --conf spark.sql.session.timeZone=UTC \
  pipeline_rapids.py
```

Writes:
- `results/sfX/rapids_gpu_results.json`

### 4) Automated benchmark harness

Use the actual filename in this folder (`run-benchmark.py`):

```bash
python run-benchmark.py --configs cpu gpu --scale-factors 1 10 --iterations 3
```

Or include all modes:

```bash
python run-benchmark.py --configs all --scale-factors 1 10 --iterations 3 --generate-data
```

> Note: the harness internally refers to config keys like `gvirtus_tcp`, `gvirtus_rdma`, and `gvirtus_smart`.

## Output format

Both CPU and GPU runs save:

```json
{
  "timings": {
    "1_data_loading": 0.0,
    "2_revenue_analytics": 0.0,
    "3_customer_360": 0.0,
    "4_rfm_segmentation": 0.0,
    "5_cohort_analysis": 0.0,
    "6_funnel_analysis": 0.0,
    "7_customer_clustering": 0.0
  },
  "results": {
    "row_counts": {"customers": 0, "products": 0, "orders": 0, "order_items": 0, "clickstream": 0}
  }
}
```

## Quick checks

- GPU visibility:

```bash
nvidia-smi
```

- Spark GPU smoke test in this folder:

```bash
python test_gpu.py
```

## Known folder-specific notes

- The benchmark runner file is named `run-benchmark.py` (hyphen).
- Current RAPIDS jar present is `25.02.1` under `jars/`.
- Combined comparison markdown is already available at `results/comparison_all.md`.
