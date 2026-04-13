##  1. Configuration

### RAPIDS JAR

The RAPIDS Accelerator JAR must be placed in a `jars/` folder at the project root:

```bash
mkdir -p jars
wget -P jars/ https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.02.1/rapids-4-spark_2.12-26.02.1.jar
```

### 2. Run (from GVirtuS root directory)

| Mode | Command | Description |
|------|---------|-------------|
| **Local Native** | `make run-spark-local` | Run directly on host (CPU + RAPIDS) |
| | `make run-spark-local-cpu` | CPU only |
| | `make run-spark-local-rapids` | RAPIDS GPU only |
| **Docker Local** | `make run-spark-docker-local` | Docker with local GPU (CPU + RAPIDS) |
| | `make run-spark-docker-local-cpu` | Docker, CPU only |
| | `make run-spark-docker-local-rapids` | Docker, RAPIDS GPU only |
| **Docker GVirtuS** | `make run-spark-docker-gvirtus` | Docker with remote GPU via GVirtuS |

### 3. Check Results

Results are saved to `results/sf{N}/simple_matrix_{env}_{mode}_results.json`.

## Configuration

Edit `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCALE_FACTOR` | 1 | Matrix size = 100 x SCALE_FACTOR |
| `SPARK_MASTER` | `local[4]` | Spark master URL |
| `LOG_LEVEL` | `DEBUG` | Python logging level |

## CLI Options

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `env` | `local`, `docker`, `gvirtus` | `local` | Execution environment |
| `--mode` | `cpu`, `rapids` | `cpu` | Execution mode |
| `--overwrite` | `yes`, `no` | `yes` | Overwrite results file, or merge into existing |

## Prerequisites

- **Local Native**: Python 3.10+, Java 17+, NVIDIA GPU + driver
- **Docker Local**: Docker, NVIDIA Container Toolkit
- **Docker GVirtuS**: Docker, GVirtuS backend running on remote host


```bash
cd ~/GVirtuS/examples/spark_simple_matrix
python3 -m venv spark-venv
source spark-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
