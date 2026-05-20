# GVirtuS Benchmarking Guide

This guide explains how to run `examples/Benchmarking/benchmark.sh` in a reproducible way.

The benchmark has two sides:

- **Backend**: the GVirtuS backend container, started from the repository root.
- **Frontend**: the benchmark script, started from `examples/Benchmarking`.

The benchmark supports these modes:

| Mode | Transport | Config file |
|---|---|---|
| `tcp` | Plain TCP | `properties.json` |
| `rdma` | Plain RDMA / RoCE | `properties_plain_rdma.json` |
| `ucx_tcp` | UCX over TCP | `properties_ucx.json` |
| `ucx_rdma` | UCX over RDMA / RoCE | `properties_ucx.json` |

For reporting, use `aggregate_summary.csv`.

- `timing_ms` is the selected benchmark timing.
- For `simple_matrix`, `timing_ms` comes from `benchmark_result_ms`.
- For OpenCV examples, `timing_ms` uses `inference_ms` when parsed.
- `elapsed_ms` is the outer harness wall-clock time and includes wrapper overhead.

## 1. Set common variables

Use variables instead of hard-coded user-specific paths.

From the repository root:

```bash
cd /path/to/GVirtuS

export GVIRTUS_REPO_ROOT="$PWD"
export BENCHMARK_DIR="$GVIRTUS_REPO_ROOT/examples/Benchmarking"

# Backend container name used by the Makefile / benchmark setup.
# Adjust this if your local Makefile uses another name.
export GVIRTUS_BACKEND_CONTAINER="${GVIRTUS_BACKEND_CONTAINER:-gvirtus-${USER}}"

# OpenCV installation used by the OpenCV examples.
# Adjust if OpenCV is installed elsewhere.
export OPENCV_PREFIX="${OPENCV_PREFIX:-$HOME/opencv-local}"
```

If your project uses another backend container name, check it with:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

## 2. Repository layout

Expected paths:

```text
GVirtuS/
├── etc/
│   ├── properties.json
│   ├── properties_plain_rdma.json
│   └── properties_ucx.json
├── examples/
│   ├── Benchmarking/
│   │   └── benchmark.sh
│   ├── face-recognition/
│   │   └── run.sh
│   ├── opencv-dnn/
│   │   └── run.sh
│   └── opencv-yolo/
│       └── run.sh
└── lib/
    ├── libgvirtus-frontend.so
    ├── libgvirtus-communicators*.so
    ├── frontend/
    │   ├── libcudart.so.12
    │   └── libcublas.so.12
    └── ucx/
        └── libuct_*.so
```

The `lib/` directory contains copied/generated runtime libraries. These files are needed locally for host-side examples, but they should not be committed to Git.

## 3. One-time runtime preparation

The host-side examples need repo-local GVirtuS frontend libraries and, for UCX RDMA, repo-local UCX transport modules.

First start or build the backend container at least once. Then, in another terminal:

```bash
cd "$GVIRTUS_REPO_ROOT"

mkdir -p ./lib ./lib/ucx

# GVirtuS frontend/communicator/runtime libraries
docker cp "$GVIRTUS_BACKEND_CONTAINER:/usr/local/gvirtus/lib/." ./lib/

# UCX transport modules required for UCX RDMA
docker cp "$GVIRTUS_BACKEND_CONTAINER:/usr/lib/ucx/." ./lib/ucx/
```

Verify that important files exist:

```bash
cd "$GVIRTUS_REPO_ROOT"

find lib -maxdepth 3 -type f \
  \( -name 'libgvirtus-frontend.so*' \
  -o -name 'libgvirtus-communicators*.so*' \
  -o -name 'libcudart.so*' \
  -o -name 'libcublas.so*' \
  -o -name 'libucp.so*' \
  -o -name 'libuct*.so*' \) \
  | sort
```

For UCX RDMA, verify that `libuct_ib.so.0` resolves from repo-local `lib/ucx`, not from `/usr/lib/ucx`:

```bash
cd "$GVIRTUS_REPO_ROOT"

LD_LIBRARY_PATH="$PWD/lib:$PWD/lib/ucx:$PWD/lib/frontend:${LD_LIBRARY_PATH:-}" \
ldd ./lib/ucx/libuct_ib_mlx5.so | grep -E "ucp|uct|ucs|ucm|ibverbs|mlx5|rdmacm|not found"
```

Expected important line:

```text
libuct_ib.so.0 => <repo-root>/lib/ucx/libuct_ib.so.0
```

## 4. Backend commands

Start the backend from the repository root:

```bash
cd "$GVIRTUS_REPO_ROOT"
```

Stop old backend/frontend containers before switching transport modes:

```bash
docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true
```

### 4.1 Plain TCP backend

```bash
GVIRTUS_CONFIG_FILE=properties.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

### 4.2 Plain RDMA backend

```bash
GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

The plain RDMA config should point to the active RoCE address, for example:

```json
{
  "endpoint": {
    "suite": "roce-rdma",
    "protocol": "ib",
    "server_address": "<roce-ip-address>",
    "port": "3333"
  }
}
```

### 4.3 UCX TCP backend

Use the TCP network interface that owns the IP configured in `etc/properties_ucx.json`.

Example:

```bash
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=<tcp-netdev> \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Example values from one tested setup:

```bash
UCX_NET_DEVICES=ens1f1np1
```

### 4.4 UCX RDMA backend

Use the RDMA device associated with the RoCE path configured in `etc/properties_ucx.json`.

Example:

```bash
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=<rdma-device:port> \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Example values from one tested setup:

```bash
UCX_NET_DEVICES=mlx5_1:1
```

## 5. Frontend benchmark commands

Run the benchmark script from:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"
```

General form:

```bash
./benchmark.sh \
  --mode <mode> \
  --examples <examples> \
  --matrix-n <size|all> \
  --warmups <n> \
  --runs <n>
```

Default output is compact. For full commands and debug paths:

```bash
BENCHMARK_VERBOSE=1 ./benchmark.sh ...
```

## 6. Supported examples

Available examples:

| Example name | Description |
|---|---|
| `simple_matrix` | Matrix multiplication benchmark |
| `face_recon` | Face recognition host-side example |
| `opencv_dnn` | OpenCV DNN benchmark |
| `opencv_yolo` | OpenCV YOLO benchmark |

Run multiple examples with a comma-separated list:

```bash
--examples face_recon,opencv_dnn,opencv_yolo
```

## 7. Quick smoke tests

Run these after starting the matching backend.

### 7.1 Plain TCP smoke

Backend:

```bash
cd "$GVIRTUS_REPO_ROOT"

docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode tcp \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 7.2 Plain RDMA smoke

Backend:

```bash
cd "$GVIRTUS_REPO_ROOT"

docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode rdma \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 7.3 UCX TCP smoke

Backend:

```bash
cd "$GVIRTUS_REPO_ROOT"

docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=<tcp-netdev> \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode ucx_tcp \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 7.4 UCX RDMA smoke

Backend:

```bash
cd "$GVIRTUS_REPO_ROOT"

docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=<rdma-device:port> \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode ucx_rdma \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

## 8. Full smoke suites

A full smoke run uses one measured run for every configured example.

### Plain TCP full smoke

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode tcp \
  --matrix-n all \
  --runs 1
```

### Plain RDMA full smoke

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode rdma \
  --matrix-n all \
  --runs 1
```

### UCX TCP full smoke

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode ucx_tcp \
  --matrix-n all \
  --runs 1
```

### UCX RDMA full smoke

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode ucx_rdma \
  --matrix-n all \
  --runs 1
```

## 9. Full benchmark runs

For real measurements, omit `--runs 1` and use the script defaults:

```bash
cd "$GVIRTUS_REPO_ROOT/examples/Benchmarking"

./benchmark.sh \
  --mode tcp \
  --matrix-n all

./benchmark.sh \
  --mode rdma \
  --matrix-n all

./benchmark.sh \
  --mode ucx_tcp \
  --matrix-n all

./benchmark.sh \
  --mode ucx_rdma \
  --matrix-n all
```

Benchmark only a specific example:

```bash
./benchmark.sh \
  --mode ucx_rdma \
  --examples face_recon \
  --warmups 0 \
  --runs 1
```

Benchmark only OpenCV examples:

```bash
./benchmark.sh \
  --mode ucx_rdma \
  --examples opencv_dnn,opencv_yolo \
  --warmups 0 \
  --runs 1
```

## 10. Matrix sizes

For `simple_matrix`, `--matrix-n all` expands to the configured list in `benchmark.sh`.

Common examples:

```bash
# Small sanity check
./benchmark.sh --mode ucx_rdma --examples simple_matrix --matrix-n 256 --warmups 0 --runs 1

# Largest matrix only
./benchmark.sh --mode ucx_rdma --examples simple_matrix --matrix-n 16384 --warmups 0 --runs 1

# All configured matrix sizes
./benchmark.sh --mode ucx_rdma --examples simple_matrix --matrix-n all --runs 1
```

For non-matrix examples, `matrix-n` is not relevant and the output displays:

```text
Matrix sizes: -
```

## 11. Output and result files

Each run creates a timestamped directory:

```text
examples/Benchmarking/benchmark_results/<timestamp>_<mode>/
```

Important files:

```text
results.csv      measured benchmark results
static.csv       static metadata
metadata.json    run metadata and environment
logs/            per-run stdout/stderr logs
```

Default output is compact:

```text
[face_recon][measured 1]                         OK   3592ms
```

Verbose mode prints full commands and debug paths:

```bash
BENCHMARK_VERBOSE=1 ./benchmark.sh \
  --mode ucx_rdma \
  --examples face_recon \
  --warmups 0 \
  --runs 1
```

## 12. Common troubleshooting

### Backend container is missing

Check running containers:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

Start the backend for the selected mode before running `benchmark.sh`.

### Wrong backend mode

Restart the backend when switching between `rdma`, `ucx_tcp`, and `ucx_rdma`.

```bash
docker rm -f "$GVIRTUS_BACKEND_CONTAINER" simple_matrix_test_container-${USER} 2>/dev/null || true
```

Then start the backend again using the matching backend command.

### Find the correct TCP and RDMA devices

Useful commands:

```bash
ip -br addr

rdma link show 2>/dev/null || true
ibv_devices 2>/dev/null || true
ibdev2netdev 2>/dev/null || true
```

Use the TCP network device for UCX TCP, and the RDMA device/port for UCX RDMA.

### UCX TCP says destination is unreachable

Make sure `UCX_NET_DEVICES` is set to the network interface for the IP in `properties_ucx.json`.

Example:

```bash
UCX_TLS=tcp,self
UCX_NET_DEVICES=<tcp-netdev>
UCX_SOCKADDR_TLS_PRIORITY=tcp
```

### UCX RDMA aborts with `tl_rkey_size <= UINT8_MAX`

This usually means UCX core libraries and UCX transport modules are mismatched.

Make sure:

```bash
UCX_MODULE_DIR="$GVIRTUS_REPO_ROOT/lib/ucx"
LD_LIBRARY_PATH="$GVIRTUS_REPO_ROOT/lib:$GVIRTUS_REPO_ROOT/lib/ucx:$GVIRTUS_REPO_ROOT/lib/frontend:${LD_LIBRARY_PATH:-}"
```

Verify:

```bash
cd "$GVIRTUS_REPO_ROOT"

LD_LIBRARY_PATH="$PWD/lib:$PWD/lib/ucx:$PWD/lib/frontend:${LD_LIBRARY_PATH:-}" \
ldd ./lib/ucx/libuct_ib_mlx5.so | grep libuct_ib
```

Expected:

```text
libuct_ib.so.0 => <repo-root>/lib/ucx/libuct_ib.so.0
```

### OpenCV examples cannot find `libopencv_dnn.so.410`

Make sure `OPENCV_PREFIX/lib` is in `LD_LIBRARY_PATH`.

The benchmark script and OpenCV `run.sh` files should add:

```bash
$OPENCV_PREFIX/lib
```

Set `OPENCV_PREFIX` if needed:

```bash
export OPENCV_PREFIX=/path/to/opencv-local
```

### Host runtime libs missing

If host examples fail because `libgvirtus-frontend.so` is missing, copy runtime libs again from the backend container:

```bash
cd "$GVIRTUS_REPO_ROOT"

mkdir -p ./lib ./lib/ucx
docker cp "$GVIRTUS_BACKEND_CONTAINER:/usr/local/gvirtus/lib/." ./lib/
docker cp "$GVIRTUS_BACKEND_CONTAINER:/usr/lib/ucx/." ./lib/ucx/
```

## 13. Recommended workflow

1. Start the backend for the desired mode.
2. Run a small smoke test, for example `simple_matrix --matrix-n 256`.
3. Run host-side examples individually.
4. Run the full smoke suite with `--matrix-n all --runs 1`.
5. Run full benchmarks without `--runs 1`.
6. Commit only source/config/scripts/docs, not copied runtime binaries.
