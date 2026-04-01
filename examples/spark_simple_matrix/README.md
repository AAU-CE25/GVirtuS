## Prerequisites

- Python 3.10+
- Java 17+
- NVIDIA GPU + driver (for RAPIDS/hybrid GPU stages)
- Dependencies from `requirements.txt`

```bash
cd ~/GVirtuS/examples/spark_simple_matrix
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

### RAPIDS JAR

The RAPIDS Accelerator JAR must be placed in a `jars/` folder at the project root:

```bash
mkdir -p jars
# Download the JAR (pick the version matching your Spark):
wget -P jars/ https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.02.1/rapids-4-spark_2.12-26.02.1.jar
```

The JAR filename is configured in `src/config.py` via `RAPIDS_JAR_PATH`:

```python
RAPIDS_JAR_PATH = "../jars/rapids-4-spark_2.12-26.02.1.jar"
```

If you use a different version, update this path to match.

### General settings

Edit `src/config.py`:
- `SCALE_FACTOR` — matrix size = 100 × SCALE_FACTOR (also controls `results/sfX` path)
- `SPARK_MASTER` — Spark master URL (default: `local[4]`)
- `RAPIDS_JAR_PATH` — relative path to the RAPIDS JAR (see above)

## Running Simple Matrix

| Strategy | API | GPU-accelerated by RAPIDS? |
|----------|-----|---------------------------|
| **DataFrame** | `join` + `groupBy` + `sum` | Yes (Catalyst/SQL plan) |

### Basic usage

```bash
# Go to the src dir
cd src

# CPU-only (default) — overwrites any previous results file
python simple_matrix.py --mode cpu

# RAPIDS GPU — the JAR is loaded automatically via PYSPARK_SUBMIT_ARGS in config.py
python simple_matrix.py --mode rapids

# Run both modes sequentially
python simple_matrix.py --mode both
```

### Result-file behaviour

By default the results JSON is **overwritten** on each run.
Use `--no-overwrite` to **merge** new keys into an existing file instead
(handy for accumulating results from multiple runs):

```bash
# First run — creates the file
python simple_matrix.py --mode cpu

# Second run — merges new keys into the same file
python simple_matrix.py --mode cpu --no-overwrite
```

Results are saved to `results/simple_matrix_<mode>_results.json`.