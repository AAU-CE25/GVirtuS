# OpenCV-DNN GVirtuS Full-Metrics Benchmark

This document describes how to build, run, validate, and summarize the OpenCV-DNN benchmark through GVirtuS using the final full-metrics workflow.

The benchmark compares four transport configurations:

- `tcp`
- `rdma`
- `ucx-tcp`
- `ucx-rdma`

The application workload uses OpenCV-DNN with MobileNetV2 ONNX. The transport benchmark is considered valid when the frontend log contains a completed inference result, including:

```text
Running inference
Image:
True Class ID:
Predicted Class ID:
Time taken:
Saved total timings
Final Results:
Total images:
Accuracy:
```

For the final measurements, each transport should have:

```text
50 measured runs
5 warmup runs
status = OK
exit_code = 0
valid_output = true
```

---

## General path setup

This document uses environment variables instead of user-specific absolute paths.

Set these once in each terminal before running commands:

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

export TCP_CONFIG="${TCP_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
export RDMA_CONFIG="${RDMA_CONFIG:-$GVIRTUS_HOME/etc/properties_plain_rdma.json}"
export UCX_CONFIG="${UCX_CONFIG:-$GVIRTUS_HOME/etc/properties_ucx.json}"

# Optional: change this if your backend container has a different name.
export GVIRTUS_CONTAINER="${GVIRTUS_CONTAINER:-gvirtus-}"
```

If your installation is not under `$HOME`, set the variables explicitly:

```bash
export GVIRTUS_HOME="/path/to/GVirtuS"
export LZ4_HOME="/path/to/lz4-install"
export OPENCV_HOME="/path/to/opencv-local"
export CUDNN_ROOT="/path/to/cudnn-9.5.1"
```

---

## Required files

The folder should contain the benchmark source and runner:

```text
examples/opencv-dnn/
├── benchmark.sh
├── benchmark.md
├── main.cu
├── run.sh
├── mobilenetv2-10.onnx
├── imagenet_classes.txt
├── imagenet_test_1000/
│   └── 0_dummy.png
└── sample
```

Some of these files are required to **run** the benchmark, but should normally not be committed.

Commit these benchmark files:

```text
examples/opencv-dnn/benchmark.sh
examples/opencv-dnn/benchmark.md
examples/opencv-dnn/main.cu
examples/opencv-dnn/run.sh
```

Do not commit generated or downloaded runtime artifacts unless explicitly required:

```text
examples/opencv-dnn/sample
examples/opencv-dnn/benchmark_results/
examples/opencv-dnn/total_times_*.txt
examples/opencv-dnn/*.onnx
examples/opencv-dnn/imagenet_test_1000/
examples/opencv-dnn/*.bak*
```

---

## Runtime patches needed to reproduce the benchmark

If this benchmark is moved to another branch and the branch must be able to **rerun** the final measurements, include the relevant GVirtuS runtime fixes as well.

Important files:

```text
plugins/cudnn/backend/CudnnHandler.cpp
plugins/cudart/frontend/CudaRt_internal.cpp
plugins/cudart/backend/CudaRtHandler.cpp
```

These contain the benchmark-critical fixes used during the final runs:

```text
cuDNN 9 version compatibility reporting for OpenCV-DNN
one-shot cudaUnregisterFatBinary teardown handling
disabled unlinked CUDA/OpenGL interop handler registrations
```

Without these fixes, OpenCV/cuDNN compatibility, RDMA teardown, or backend plugin loading may fail again.

---

## Download/generate required assets

Run these commands from:

```bash
cd "$GVIRTUS_HOME/examples/opencv-dnn"
```

### Download MobileNetV2 ONNX model

The benchmark uses `mobilenetv2-10.onnx` from the ONNX Model Zoo.

```bash
wget -O mobilenetv2-10.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx

ls -lh mobilenetv2-10.onnx
```

Expected size is approximately 14 MB.

### Generate ImageNet class labels

For transport benchmarking, exact labels are not important. The example only needs a valid 1000-line class list.

```bash
python3 - <<'PY'
from pathlib import Path

out = Path("imagenet_classes.txt")
with out.open("w") as f:
    for i in range(1000):
        f.write(f"class_{i}\n")

print(f"Wrote {out}")
PY
```

If real class names are required later, replace `imagenet_classes.txt` with a proper ImageNet 1000-class label file.

### Generate a valid dummy PNG image

The dummy image is enough for transport benchmarking. Accuracy is expected to be meaningless for this generated input.

```bash
mkdir -p imagenet_test_1000

python3 - <<'PY'
from pathlib import Path
import struct
import zlib

width, height = 224, 224
rgb = bytes([128, 128, 128]) * width
raw = b"".join(b"\x00" + rgb for _ in range(height))

def chunk(kind, data):
    return (
        struct.pack(">I", len(data)) +
        kind +
        data +
        struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    )

png = (
    b"\x89PNG\r\n\x1a\n" +
    chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
    chunk(b"IDAT", zlib.compress(raw, 9)) +
    chunk(b"IEND", b"")
)

out = Path("imagenet_test_1000/0_dummy.png")
out.write_bytes(png)
print(f"Wrote {out}")
PY

file imagenet_test_1000/0_dummy.png
```

---

## Build the frontend sample

Use the locally built OpenCV installation.

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"

cd "$GVIRTUS_HOME/examples/opencv-dnn"

export PKG_CONFIG_PATH="$OPENCV_HOME/lib/pkgconfig:$OPENCV_HOME/lib64/pkgconfig:$OPENCV_HOME/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$OPENCV_HOME/lib:$OPENCV_HOME/lib64:${LD_LIBRARY_PATH:-}"

g++ -x c++ main.cu -o sample $(pkg-config --cflags --libs opencv4)

ls -lh sample
```

The `-x c++` flag is required because the source file is named `main.cu`, but this OpenCV-DNN example is compiled as normal C++.

---

## Common frontend environment

Use this setup in the frontend terminal before running manual tests or `benchmark.sh`.

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

cd "$GVIRTUS_HOME/examples/opencv-dnn"

export CUDNN_LIB="$CUDNN_ROOT/lib"
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"

export LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"
```

If `libcuda.so` interception is needed in a specific environment, add it to `LD_PRELOAD`, but the final OpenCV-DNN runs used CUDA runtime, cuBLAS, and cuDNN frontend libraries.

---

## Backend commands

Run the backend in a separate terminal on the backend node.

The examples below use `$GVIRTUS_CONTAINER` for the backend container name. Check the active name with:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep gvirtus || true
```

### TCP backend

```bash
cd "$GVIRTUS_HOME"

docker rm -f "$GVIRTUS_CONTAINER" 2>/dev/null || true

GVIRTUS_CONFIG_FILE="${TCP_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### Plain RDMA backend

```bash
cd "$GVIRTUS_HOME"

docker rm -f "$GVIRTUS_CONTAINER" 2>/dev/null || true

GVIRTUS_CONFIG_FILE="${RDMA_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### UCX-TCP backend

Use this for the UCX-over-TCP transport.

```bash
cd "$GVIRTUS_HOME"

docker rm -f "$GVIRTUS_CONTAINER" 2>/dev/null || true

UCX_TLS=tcp \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=n \
UCX_RNDV_THRESH=inf \
UCX_ZCOPY_THRESH=inf \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE="${UCX_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### UCX-RDMA backend

Use this for the UCX-over-RDMA transport.

```bash
cd "$GVIRTUS_HOME"

docker rm -f "$GVIRTUS_CONTAINER" 2>/dev/null || true

UCX_TLS=rc_x,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=y \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE="${UCX_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```


> `GVIRTUS_CONFIG_FILE` is the config filename expected by the existing backend Makefile. The commands above assume the corresponding files are available under `$GVIRTUS_HOME/etc/`.

Important notes:

```text
GVIRTUS_UCX_DATAPATH=am is used for the GVirtuS UCX communicator.
Plain RDMA and UCX-RDMA are different communicator paths.
UCX-TCP and UCX-RDMA should be treated as separate benchmark transports.
```

---

## Manual frontend tests

Run these from:

```bash
cd "$GVIRTUS_HOME/examples/opencv-dnn"
```

### TCP frontend

```bash
GVIRTUS_CONFIG="$TCP_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 60 ./sample 2>&1 | tee /tmp/opencv_dnn_tcp_run.log
```

### Plain RDMA frontend

```bash
GVIRTUS_CONFIG="$RDMA_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 120 ./sample 2>&1 | tee /tmp/opencv_dnn_rdma_run.log
```

### UCX-TCP frontend

```bash
UCX_TLS=tcp \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=n \
UCX_RNDV_THRESH=inf \
UCX_ZCOPY_THRESH=inf \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG="$UCX_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 120 ./sample 2>&1 | tee /tmp/opencv_dnn_ucx_tcp_run.log
```

### UCX-RDMA frontend

```bash
UCX_TLS=rc_x,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=y \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG="$UCX_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 120 ./sample 2>&1 | tee /tmp/opencv_dnn_ucx_rdma_run.log
```

Validate a manual run with:

```bash
grep -E "Running inference|Image:|Saved total timings|Final Results|Total images|Accuracy|Failed status|Execution exception|UCX endpoint error|terminate|Aborted" \
  /tmp/opencv_dnn_tcp_run.log
```

Change the log filename for RDMA, UCX-TCP, or UCX-RDMA.

---

## Run benchmark.sh

Make the benchmark script executable:

```bash
cd "$GVIRTUS_HOME/examples/opencv-dnn"

chmod +x benchmark.sh
```

Run one mode at a time. Start the matching backend first.

### TCP

```bash
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
TCP_CONFIG="$TCP_CONFIG" \
FRONTEND_CMD="./sample" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=120 \
GVIRTUS_LOGLEVEL=50000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh tcp
```

### Plain RDMA

```bash
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
RDMA_CONFIG="$RDMA_CONFIG" \
FRONTEND_CMD="./sample" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh rdma
```

### UCX-TCP

```bash
UCX_TLS=tcp \
UCX_NET_DEVICES=ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=n \
UCX_RNDV_THRESH=inf \
UCX_ZCOPY_THRESH=inf \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
UCX_CONFIG="$UCX_CONFIG" \
FRONTEND_CMD="./sample" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the output folder to:

```text
benchmark_results/OpenCVDNN Final Metrics UCX-TCP
```

### UCX-RDMA

```bash
UCX_TLS=rc_x,tcp,self \
UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp \
UCX_PROTO_ENABLE=y \
UCX_MAX_EAGER_RAILS=1 \
UCX_MAX_RNDV_RAILS=1 \
UCX_LOG_LEVEL=warn \
UCX_WARN_UNUSED_ENV_VARS=n \
GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
UCX_CONFIG="$UCX_CONFIG" \
FRONTEND_CMD="./sample" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the output folder to:

```text
benchmark_results/OpenCVDNN Final Metrics UCX-RDMA
```

---

## Output folders

The final result folders are manually renamed to:

```text
benchmark_results/Opencv-dnn Final Metrics TCP/
benchmark_results/Opencv-dnn Final Metrics RDMA/
benchmark_results/OpenCVDNN Final Metrics UCX-TCP/
benchmark_results/OpenCVDNN Final Metrics UCX-RDMA/
```

Each final folder should contain:

```text
results.csv
routine_calls.csv
routine_summary.csv
nic_counters.csv
logs/
counters/
system/
```

---

## Metrics

The final `results.csv` contains:

```text
timestamp
mode
phase
run
status
exit_code
wall_s
inference_ms
total_images
accuracy_pct
correct_predictions
predicted_class
confidence_pct
valid_output
calls
gvirtus_in_B
gvirtus_out_B
nic_rx_B
nic_tx_B
config
log_file
error
routine_counts
```

The most important metrics are:

```text
wall_s              Full frontend runtime.
inference_ms        Application-reported OpenCV-DNN inference time.
calls               Number of GVirtuS routine calls.
gvirtus_in_B        Bytes sent from frontend to backend at the GVirtuS routine layer.
gvirtus_out_B       Bytes returned from backend to frontend at the GVirtuS routine layer.
nic_rx_B            NIC receive byte delta during the run.
nic_tx_B            NIC transmit byte delta during the run.
accuracy_pct        Accuracy reported by the application. For dummy input this is not scientifically meaningful.
```

`routine_summary.csv` aggregates per-routine call counts and transferred bytes.  
`routine_calls.csv` contains per-call records.  
`nic_counters.csv` contains raw NIC counter deltas per interface.

---

## Success criteria

A run is valid when:

```text
status = OK
exit_code = 0
valid_output = true
```

and the frontend log contains:

```text
Saved total timings
Final Results:
Total images:
```

Older debugging runs could use `SOFT-OK` if inference completed but RDMA cleanup/shutdown later failed. The final benchmark summaries should use clean `OK` rows.

---

## Final summary generation

After the four final folders exist, generate a compact summary with:

```bash
cd "$GVIRTUS_HOME/examples/opencv-dnn"

python3 - <<'PY'
import csv, statistics as st
from pathlib import Path

base = Path("benchmark_results")
folders = {
    "TCP": base / "Opencv-dnn Final Metrics TCP",
    "RDMA": base / "Opencv-dnn Final Metrics RDMA",
    "UCX-TCP": base / "OpenCVDNN Final Metrics UCX-TCP",
    "UCX-RDMA": base / "OpenCVDNN Final Metrics UCX-RDMA",
}

fields = [
    "transport", "n", "ok",
    "wall_mean_s", "wall_std_s",
    "inference_mean_ms", "inference_std_ms",
    "calls_mean",
    "gvirtus_in_mean_B", "gvirtus_out_mean_B",
    "nic_rx_mean_B", "nic_tx_mean_B",
]

def vals(rows, field):
    out = []
    for r in rows:
        v = str(r.get(field, "")).strip()
        if not v or v.lower() in {"nan", "none", "null"}:
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out

def mean(rows, field):
    x = vals(rows, field)
    return st.mean(x) if x else float("nan")

def std(rows, field):
    x = vals(rows, field)
    if len(x) > 1:
        return st.stdev(x)
    if len(x) == 1:
        return 0.0
    return float("nan")

rows_out = []
for name, folder in folders.items():
    with open(folder / "results.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("phase") == "measure"]

    ok = [
        r for r in rows
        if r.get("status") == "OK"
        and r.get("exit_code") == "0"
        and r.get("valid_output") == "true"
    ]

    rows_out.append({
        "transport": name,
        "n": len(rows),
        "ok": len(ok),
        "wall_mean_s": f"{mean(ok, 'wall_s'):.6f}",
        "wall_std_s": f"{std(ok, 'wall_s'):.6f}",
        "inference_mean_ms": f"{mean(ok, 'inference_ms'):.3f}",
        "inference_std_ms": f"{std(ok, 'inference_ms'):.3f}",
        "calls_mean": f"{mean(ok, 'calls'):.1f}",
        "gvirtus_in_mean_B": f"{mean(ok, 'gvirtus_in_B'):.1f}",
        "gvirtus_out_mean_B": f"{mean(ok, 'gvirtus_out_B'):.1f}",
        "nic_rx_mean_B": f"{mean(ok, 'nic_rx_B'):.1f}",
        "nic_tx_mean_B": f"{mean(ok, 'nic_tx_B'):.1f}",
    })

out = base / "OpenCVDNN_Final_Summary.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(",".join(fields))
for r in rows_out:
    print(",".join(str(r[f]) for f in fields))

print(f"\nWrote: {out}")
PY
```