# Spark Simple Matrix (RAPIDS + GVirtuS)

Matrix multiplication benchmark using Apache Spark with optional RAPIDS GPU acceleration via GVirtuS.

## Prerequisites

- Python 3.10+, Java 17+
- RAPIDS JAR in a `jars/` directory (sibling to the GVirtuS repo):

```bash
mkdir -p ../jars
wget -P ../jars/ https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.02.1/rapids-4-spark_2.12-26.02.1.jar
```

For local (non-Docker) runs, create a venv:

```bash
cd ~/GVirtuS/examples/spark_simple_matrix
python3 -m venv spark-venv && source spark-venv/bin/activate
pip install -r requirements.txt
```

## Run (from GVirtuS root)

| Mode | Command | Requirements |
|------|---------|--------------|
| Local CPU | `make run-spark-local-cpu` | Host Python + Java + venv |
| Local RAPIDS | `make run-spark-local-rapids` | + NVIDIA GPU + driver |
| GVirtuS Frontend | `make run-spark-frontend` | Docker + GVirtuS backend running |

### Build images (one-time)

```bash
make docker-build-frontend         # Base GVirtuS frontend image
make docker-build-spark-frontend   # Adds Java + PySpark on top
```

## Configuration

Edit `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCALE_FACTOR` | 1 | Matrix size = 100 × SCALE_FACTOR |
| `SPARK_MASTER` | `local[1]` | Spark master URL |
| `LOG_LEVEL` | `DEBUG` | Python logging level |

## CLI Options

```
python3 simple_matrix.py <env> [--mode cpu|rapids] [--overwrite yes|no] [--minimal]
```

| `env` | `local`, `docker`, `gvirtus` | — | Execution environment |
| `--mode` | `cpu`, `rapids` | `cpu` | Compute mode |
| `--overwrite` | `yes`, `no` | `yes` | Overwrite or merge results |
| `--minimal` | flag | off | Run small GPU test before full benchmark |

## Output

Results: `results/sf{N}/simple_matrix_{env}_{mode}_results.json`
Logs: `logs/{env}/`



# ── Extract RAPIDS native libs from JAR (needed for both modes) ──
if [[ -f "$RAPIDS_JAR" ]] && [[ ! -d "$NATIVE_DIR" ]]; then
    mkdir -p "$NATIVE_DIR"
    unzip -q -j "$RAPIDS_JAR" "amd64/Linux/*.so" -d "$NATIVE_DIR" 2>/dev/null || true
    
    if [[ -f "$NATIVE_DIR/libnvcomp.so" ]]; then
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.5"
        ln -sf libnvcomp.so "$NATIVE_DIR/libnvcomp.so.4"
    fi
    
    [[ -f "$NATIVE_DIR/libcudf.so" ]] && echo "Extracted RAPIDS native libs to $NATIVE_DIR" >&2
fi