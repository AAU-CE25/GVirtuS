# GVirtuS Benchmarking Guide

This guide explains how to run the benchmark harness in `examples/Benchmarking/benchmark.sh`.

The benchmark has two sides:

- **Backend**: the GVirtuS backend container, started from the repository root.
- **Frontend**: the benchmark script, started from `examples/Benchmarking`.

The benchmark supports these modes:

| Mode | Transport | Config file |
|---|---|---|
| `tcp` | plain TCP | `properties.json` |
| `rdma` | plain RDMA / RoCE | `properties_plain_rdma.json` |
| `ucx_tcp` | UCX over TCP | `properties_ucx.json` |
| `ucx_rdma` | UCX over RDMA / RoCE | `properties_ucx.json` |

## 1. Repository layout

Expected repository paths:

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

The `lib/` directory contains copied/generated runtime libraries. These files are needed locally for host-side examples, but they should **not** be committed to Git.

## 2. One-time runtime preparation

After the backend image has been built at least once, copy the runtime libraries from the backend container into the repo-local `lib/` folder.

Start or rebuild the backend container first, then in another terminal run:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

mkdir -p ./lib ./lib/ucx

# GVirtuS frontend/communicator/runtime libs
docker cp gvirtus-ul11nh:/usr/local/gvirtus/lib/. ./lib/

# UCX transport modules required for UCX RDMA
docker cp gvirtus-ul11nh:/usr/lib/ucx/. ./lib/ucx/
```

Verify that the important files exist:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

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
cd /home/student.aau.dk/ul11nh/GVirtuS

LD_LIBRARY_PATH="$PWD/lib:$PWD/lib/ucx:$PWD/lib/frontend:${LD_LIBRARY_PATH:-}" \
ldd ./lib/ucx/libuct_ib_mlx5.so | grep -E "ucp|uct|ucs|ucm|ibverbs|mlx5|rdmacm|not found"
```

Expected important line:

```text
libuct_ib.so.0 => /home/student.aau.dk/ul11nh/GVirtuS/lib/ucx/libuct_ib.so.0
```

## 3. Backend commands

Run the backend from the repository root:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS
```

Stop old backend/frontend containers before switching transport modes:

```bash
docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true
```

### 3.1 Plain TCP backend

```bash
GVIRTUS_CONFIG_FILE=properties.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

### 3.2 Plain RDMA backend

```bash
GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

The plain RDMA config should point to the active RoCE address, for example:

```json
"server_address": "25.25.25.2",
"port": "3333",
"protocol": "ib",
"suite": "roce-rdma"
```

### 3.3 UCX TCP backend

Use `ens1f1np1` for the `25.25.25.2` network:

```bash
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

### 3.4 UCX RDMA backend

Use the RDMA device associated with the `25.25.25.2` RoCE path:

```bash
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

The backend should print that it loaded `properties_ucx.json` and is listening on `25.25.25.2:2222`.

## 4. Frontend benchmark commands

Run the benchmark script from:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking
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

The script prints a compact summary by default. For full commands and debug paths, use:

```bash
BENCHMARK_VERBOSE=1 ./benchmark.sh ...
```

## 5. Supported examples

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

## 6. Quick smoke tests

### 6.1 Plain TCP smoke

Backend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode tcp \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 6.2 Plain RDMA smoke

Backend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
GVIRTUS_LOG_LEVEL=10000 \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode rdma \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 6.3 UCX TCP smoke

Backend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=tcp,self \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode ucx_tcp \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

### 6.4 UCX RDMA smoke

Backend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true

GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=10000 \
GVIRTUS_LOGLEVEL=10000 \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_LOG_LEVEL=info \
make run-gvirtus-backend-dev
```

Frontend:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode ucx_rdma \
  --examples simple_matrix \
  --matrix-n 256 \
  --warmups 0 \
  --runs 1
```

## 7. Full smoke suites

A full smoke run uses one measured run for every configured example.

### Plain RDMA full smoke

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode rdma \
  --matrix-n all \
  --runs 1
```

### UCX TCP full smoke

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode ucx_tcp \
  --matrix-n all \
  --runs 1
```

### UCX RDMA full smoke

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

./benchmark.sh \
  --mode ucx_rdma \
  --matrix-n all \
  --runs 1
```

## 8. Full benchmark runs

For real measurements, omit `--runs 1` and let the script use its default run count:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS/examples/Benchmarking

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

To benchmark only a specific example:

```bash
./benchmark.sh \
  --mode ucx_rdma \
  --examples face_recon \
  --warmups 0 \
  --runs 1
```

To benchmark only OpenCV examples:

```bash
./benchmark.sh \
  --mode ucx_rdma \
  --examples opencv_dnn,opencv_yolo \
  --warmups 0 \
  --runs 1
```

## 9. Matrix sizes

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

For non-matrix examples, `matrix-n` is not relevant and the output displays `Matrix sizes: -`.

## 10. Output and result files

Each benchmark run creates a timestamped directory:

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

## 11. Common troubleshooting

### Backend container is missing

If the benchmark prints that the backend container is not detected, start the backend first.

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

### Wrong backend mode

Restart the backend when switching between `rdma`, `ucx_tcp`, and `ucx_rdma`.

```bash
docker rm -f gvirtus-ul11nh simple_matrix_test_container-ul11nh 2>/dev/null || true
```

Then start the backend again using the matching backend command.

### UCX TCP says destination is unreachable

Use the TCP netdev for `25.25.25.2`:

```bash
UCX_TLS=tcp,self
UCX_NET_DEVICES=ens1f1np1
UCX_SOCKADDR_TLS_PRIORITY=tcp
```

### UCX RDMA aborts with `tl_rkey_size <= UINT8_MAX`

This usually means UCX core libraries and UCX transport modules are mismatched.

Make sure:

```bash
UCX_MODULE_DIR=/home/student.aau.dk/ul11nh/GVirtuS/lib/ucx
LD_LIBRARY_PATH=/home/student.aau.dk/ul11nh/GVirtuS/lib:/home/student.aau.dk/ul11nh/GVirtuS/lib/ucx:/home/student.aau.dk/ul11nh/GVirtuS/lib/frontend:...
```

Verify:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

LD_LIBRARY_PATH="$PWD/lib:$PWD/lib/ucx:$PWD/lib/frontend:${LD_LIBRARY_PATH:-}" \
ldd ./lib/ucx/libuct_ib_mlx5.so | grep libuct_ib
```

Expected:

```text
libuct_ib.so.0 => /home/student.aau.dk/ul11nh/GVirtuS/lib/ucx/libuct_ib.so.0
```

### OpenCV examples cannot find `libopencv_dnn.so.410`

Make sure `OPENCV_PREFIX/lib` is in `LD_LIBRARY_PATH`.

The benchmark script and OpenCV `run.sh` files should add:

```bash
$OPENCV_PREFIX/lib
```

For this setup:

```bash
OPENCV_PREFIX=/home/student.aau.dk/ul11nh/opencv-local
```

### Host runtime libs missing

If host examples fail because `libgvirtus-frontend.so` is missing, copy runtime libs again from the backend container:

```bash
cd /home/student.aau.dk/ul11nh/GVirtuS

mkdir -p ./lib ./lib/ucx
docker cp gvirtus-ul11nh:/usr/local/gvirtus/lib/. ./lib/
docker cp gvirtus-ul11nh:/usr/lib/ucx/. ./lib/ucx/
```

## 12. Git hygiene

Do not commit copied runtime binaries:

```text
lib/*.so*
lib/frontend/*.so*
lib/ucx/*.so*
lib/ucx/*.a
lib/ucx/*.la
```

These files are generated/copied runtime artifacts and should be ignored by Git.

Recommended `.gitignore` entries:

```gitignore
/lib/*.so*
/lib/frontend/*.so*
/lib/ucx/*.so*
/lib/ucx/*.a
/lib/ucx/*.la
```
