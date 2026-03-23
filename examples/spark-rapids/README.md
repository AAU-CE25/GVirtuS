# Spark RAPIDS E-Commerce Benchmark

Practical Spark analytics benchmark for comparing:
- CPU pipeline
- RAPIDS GPU pipeline
- Hybrid pipeline (some stages GPU, some CPU)

## Project Structure

```text
examples/spark-rapids/
├── .venv/                         # Local virtual environment
├── data/                          # Generated parquet data (data/sfX)
├── jars/
│   └── rapids-4-spark_2.12-25.02.1.jar
├── results/
│   ├── comparison_all.md
│   ├── sf1/
│   │   ├── spark_cpu_results.json
│   │   ├── rapids_gpu_results.json
│   │   └── comparison.md
│   └── sf10/
│       ├── spark_cpu_results.json
│       ├── rapids_gpu_results.json
│       └── comparison_sf10.md
├── src/
│   ├── __init__.py
│   ├── config.py                  # Scale + Spark config + RAPIDS jar path
│   ├── datagen.py                 # Synthetic parquet data generator
│   ├── ecomerce_pipeline.py       # Shared 7-stage base pipeline logic
│   ├── pipeline_spark.py          # CPU pipeline runner
│   ├── pipeline_rapids.py         # RAPIDS GPU pipeline runner
│   ├── pipeline_hybrid.py         # Hybrid CPU+RAPIDS per-stage runner
│   └── run-benchmark.py           # Multi-config benchmark harness
└── README.md
```

## Prerequisites

- Python 3.10+
- Java 17+
- NVIDIA GPU + driver (for RAPIDS/hybrid GPU stages)
- Dependencies from `requirements.txt`

```bash
cd ~/GVirtuS/examples/spark-rapids
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Edit `src/config.py`:
- `SCALE_FACTOR` (controls `data/sfX` and `results/sfX`)
- `SPARK_MASTER`, `SPARK_EXECUTOR_MEMORY`, `SPARK_DRIVER_MEMORY`
- `RAPIDS_JAR_PATH`

## Run

### 1) Generate data

```bash
python src/datagen.py
```

### 2) CPU pipeline

```bash
python src/pipeline_spark.py
```

Output:
- `results/sfX/spark_cpu_results.json`

### 3) RAPIDS GPU pipeline

```bash
python src/pipeline_rapids.py
```

Output:
- `results/sfX/rapids_gpu_results.json`

### 4) Hybrid pipeline (GPU for selected stages)

Default GPU stages are `2,4,5,6,7`:

```bash
python src/pipeline_hybrid.py
```

Custom stage split example:

```bash
python src/pipeline_hybrid.py --gpu-stages 2,7
```

Output:
- `results/sfX/hybrid_results.json`

### 5) Benchmark harness

```bash
python src/run-benchmark.py --configs cpu gpu --scale-factors 1 10 --iterations 3
```

All configs:

```bash
python src/run-benchmark.py --configs all --scale-factors 1 10 --iterations 3 --generate-data
```

## Current Workspace Results

### sf1
- CPU total: **36.11s**
- GPU total: **40.43s**
- Speedup (CPU/GPU): **0.89x**

### sf10
- CPU total: **431.69s**
- GPU total: **92.63s**
- Speedup (CPU/GPU): **4.66x**

See combined comparison: `results/comparison_all.md`

## Quick Checks

```bash
nvidia-smi
python src/pipeline_rapids.py --help
python src/pipeline_hybrid.py --help
```
