# GVirtuS Matrix Benchmark

`benchmark.sh` measures end-to-end latency of a SGEMM (single-precision matrix multiply)
offloaded through GVirtuS across four transport modes: plain TCP, UCX over TCP,
plain RDMA, and UCX over RDMA. Matrix sizes ramp geometrically from N=8 to N=16384,
and each stage (malloc, cudaMalloc, H2D, cublasCreate, GEMM, D2H, cleanup) is timed
separately. Results are written to a timestamped CSV in `benchmark_results/`.

---

## Prerequisites

| Component | Where | What must be running |
|---|---|---|
| GVirtuS backend | `es-dpu-01` | `make run-gvirtus-backend-dev` |
| GVirtuS client | `es-dpu-02` | `benchmark.sh` runs here |
| Docker image | `es-dpu-02` | `gvirtus-dev/simple_matrix_gvirtus:cuda12.6` |

Build the client image if not already done:
```bash
make -C ~/GVirtuS local-docker-build-simple-matrix
```

---

## Usage

```bash
./examples/simple_matrix/benchmark.sh [MODE_GROUP] [runs_per_size]
```

| Argument | Default | Description |
|---|---|---|
| `MODE_GROUP` | `all` | Which transport modes to run (see table below) |
| `runs_per_size` | `5` | Number of timed runs per matrix size |

### Mode Groups

| `MODE_GROUP` | Modes run | Backend config needed |
|---|---|---|
| `all` | `plain_tcp` + `ucx_tcp` + `plain_rdma` + `ucx_rdma` | Restart backend between TCP and RDMA groups |
| `tcp` | `plain_tcp` + `ucx_tcp` | `properties.json` then `properties_ucx.json` |
| `rdma` | `plain_rdma` + `ucx_rdma` | `properties_plain_rdma.json` then `properties_ucx.json` |
| `plain_tcp` | TCP only (no UCX) | `properties.json` |
| `ucx_tcp` | UCX over TCP | `properties_ucx.json` |
| `plain_rdma` | RDMA only (no UCX) | `properties_plain_rdma.json` |
| `ucx_rdma` | UCX over RDMA (RoCEv2) | `properties_ucx.json` |

> **Note:** `all` and `tcp` run `plain_tcp` (uses `properties.json`) and `ucx_tcp`
> (uses `properties_ucx.json`) back-to-back. Because each mode requires a different
> backend config, you must restart the backend between them — or run each mode
> individually as shown below.

---

## Step-by-Step: Running Each Mode

### plain_tcp

```bash
# es-dpu-01 — start backend
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties.json \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev

# es-dpu-02 — run benchmark
./examples/simple_matrix/benchmark.sh plain_tcp 5
```

### ucx_tcp

```bash
# es-dpu-01 — start backend
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties_ucx.json \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev

# es-dpu-02 — run benchmark
./examples/simple_matrix/benchmark.sh ucx_tcp 5
```

### plain_rdma ### This does not work currently

```bash
# es-dpu-01 — start backend
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=rdma \
GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev

# es-dpu-02 — run benchmark
./examples/simple_matrix/benchmark.sh plain_rdma 5
```

### ucx_rdma

```bash
# es-dpu-01 — start backend
make stop-gvirtus || true
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties_ucx.json \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev

# es-dpu-02 — run benchmark
./examples/simple_matrix/benchmark.sh ucx_rdma 5
```

---

## Output

Results are saved to: benchmark_results/benchmark_results_<MODE_GROUP>_<YYYYMMDD_HHMMSS>.csv
### CSV Columns

| Column | Description |
|---|---|
| `mode` | Transport mode (`plain_tcp`, `ucx_tcp`, etc.) |
| `matrix_n` | Matrix dimension N (NxN matrix) |
| `run` | Run index within this size |
| `warmup` | `true` if discarded warmup run, `false` if measured |
| `elapsed_ms` | Total end-to-end time (ms) |
| `exit_code` | Process exit code (0 = success) |
| `ucx_tls` | UCX transport layer string used |
| `ucx_net_devices` | Network device used |
| `malloc_ms` | Host memory allocation time |
| `cudamalloc_ms` | Device memory allocation time |
| `h2d_ms` | Host-to-device transfer time |
| `cublas_create_ms` | cuBLAS handle creation time |
| `gemm_ms` | SGEMM execution time |
| `d2h_ms` | Device-to-host transfer time |
| `cleanup_ms` | cuBLAS destroy + cudaFree time |
| `result_check` | `PASS`/`FAIL` correctness check |
| `timestamp` | ISO-8601 timestamp of the run |

### Console Output

During a run you will see live per-size output:
=========================================
MODE: ucx_tcp (TLS=tcp,self, CFG=properties_ucx.json)
=========================================
[WARMUP] 3x N=8 — results discarded
warmup 1/3: 142ms (discarded)
...

── N=512 ──
run 1/5 ... 198ms (exit=0) | malloc=0ms cudamalloc=1ms h2d=12ms create=44ms gemm=18ms d2h=8ms
...
→ median=201ms min=196ms max=214ms (n=5)

text

---

## Warmup Behaviour

The first 3 runs at N=8 are always discarded regardless of `runs_per_size`.
This follows the [Criterion.rs](https://bheisler.github.io/criterion.rs/book/criterion_rs.html)
methodology: warmup eliminates RDMA QP connection setup, CPU/GPU cache cold-start,
and cuBLAS JIT-compilation penalties from the measured data.
Warmup runs are still written to the CSV with `warmup=true` so they can be
inspected or filtered during analysis.

---

## Filtering Results in Python

```python
import pandas as pd

df = pd.read_csv("benchmark_results/benchmark_results_all_20260512_145835.csv")

# Drop warmup rows and failed runs
measured = df[(df["warmup"] == False) & (df["exit_code"] == 0)]

# Median elapsed time per mode and matrix size
summary = measured.groupby(["mode", "matrix_n"])["elapsed_ms"].median().unstack("mode")
print(summary)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `exit=2`, all stage times empty | Backend not running or wrong config | Check backend is up with the correct `properties_*.json` |
| `WARNING: NVIDIA Driver not detected` | `--gpus all` missing from Docker run | Rebuild image or check Makefile `run-simple-matrix-test` target |
| `N/A` for `elapsed_ms` | Binary not printing `BENCHMARK_RESULT_MS=` | Rebuild `local-docker-build-simple-matrix` |
| `RESULT_CHECK=FAIL` | GEMM result incorrect | Check backend GPU has enough memory; verify `simple_matrix.cu` |
| `ucp_listener_create failed: Device is busy` | RDMA CM device held by another process | Wait 15s, retry; or run `ucx_tcp` mode instead |
| `Permission denied` running script | Script not executable | Run `chmod +x ./examples/simple_matrix/benchmark.sh` |