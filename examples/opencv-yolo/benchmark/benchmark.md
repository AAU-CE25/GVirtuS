# OpenCV-YOLO GVirtuS Full-Metrics Benchmark

This document describes how to build, run, validate, and summarize the OpenCV-YOLO benchmark through GVirtuS using the final full-metrics workflow.

The benchmark compares four transport configurations:

- `tcp`
- `rdma`
- `ucx-tcp`
- `ucx-rdma`

The workload uses OpenCV-DNN with a YOLOv5 ONNX model. The benchmark is considered valid when the frontend log contains:

```text
Attempting to use CUDA acceleration
Detection finished. Results saved to output.jpg
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

The folder should contain:

```text
examples/opencv-yolo/
├── benchmark.sh
├── benchmark.py
├── benchmark.md
├── main.cu
├── run.sh
├── setup.sh
├── class.names
├── images/
│   └── zidane.jpg
├── weights/
│   └── yolov5n.onnx
└── sample
```
---

## Runtime patches needed to reproduce the benchmark

If this benchmark is moved to another branch and the branch must be able to **rerun** the final measurements, include the GVirtuS runtime fixes used during benchmarking.

Important files:

```text
plugins/cudnn/backend/CudnnHandler.cpp
plugins/cudart/frontend/CudaRt_internal.cpp
plugins/cudart/backend/CudaRtHandler.cpp
```

These contain the benchmark-critical fixes used during the final runs:

```text
cuDNN 9 version compatibility reporting for OpenCV
one-shot cudaUnregisterFatBinary teardown handling
disabled unlinked CUDA/OpenGL interop handler registrations
```

Without these fixes, OpenCV/cuDNN compatibility, RDMA teardown, or backend plugin loading may fail again.

---

## Source compatibility notes

During setup, the unused OpenCV highgui include was removed because the local OpenCV build did not include the highgui header:

```cpp
#include <opencv2/highgui.hpp>
```

The image codec header was added because `imread` and `imwrite` are used:

```cpp
#include <opencv2/imgcodecs.hpp>
```

The benchmark source should therefore include `opencv2/imgcodecs.hpp` and should not require `opencv2/highgui.hpp` unless the OpenCV installation includes highgui.

---

## Build the frontend sample

Use the locally built OpenCV installation.

Run from:

```bash
cd "$GVIRTUS_HOME/examples/opencv-yolo"
```

Build:

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

cd "$GVIRTUS_HOME/examples/opencv-yolo"

export CUDNN_LIB="$CUDNN_ROOT/lib"
export LZ4_LIB="$LZ4_HOME/lib"

export PKG_CONFIG_PATH="$OPENCV_HOME/lib/pkgconfig:$OPENCV_HOME/lib64/pkgconfig:$OPENCV_HOME/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_LIB:${LD_LIBRARY_PATH:-}"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4)"

nvcc main.cu -o sample $OPENCV_CFLAGS \
  -L"$OPENCV_HOME/lib" \
  -L"$OPENCV_HOME/lib64" \
  -L"$CUDNN_LIB" \
  $OPENCV_LIBS \
  -lcudnn -lcuda -lcublas -lcudart

ls -lh ./sample
```

Check linked libraries:

```bash
LD_LIBRARY_PATH="$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_LIB:${LD_LIBRARY_PATH:-}" \
ldd ./sample | grep -E "opencv|cuda|cublas|cudnn|cudart|gvirtus"
```

---

## Native sanity test

Run without GVirtuS preload first:

```bash
cd "$GVIRTUS_HOME/examples/opencv-yolo"

rm -f output.jpg

env -u LD_PRELOAD \
LD_LIBRARY_PATH="$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}" \
timeout 60 ./sample 2>&1 | tee /tmp/opencv_yolo_native_test.log

tail -120 /tmp/opencv_yolo_native_test.log
ls -lh output.jpg
```

Expected log lines:

```text
Attempting to use CUDA acceleration
Detection finished. Results saved to output.jpg
```

A cuDNN version warning may appear if OpenCV reports cuDNN 9 using a different encoding, but the benchmark is still usable if detection completes and `output.jpg` is written.

---

## Common frontend environment

Use this setup in the frontend terminal before manual tests or `benchmark.sh`.

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

cd "$GVIRTUS_HOME/examples/opencv-yolo"

export CUDNN_LIB="$CUDNN_ROOT/lib"
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"

export LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"
```

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
cd "$GVIRTUS_HOME/examples/opencv-yolo"
```

### TCP frontend

```bash
GVIRTUS_CONFIG="$TCP_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 120 ./sample 2>&1 | tee /tmp/opencv_yolo_tcp_run.log
```

### Plain RDMA frontend

```bash
GVIRTUS_CONFIG="$RDMA_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
timeout 180 ./sample 2>&1 | tee /tmp/opencv_yolo_rdma_run.log
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
timeout 180 ./sample 2>&1 | tee /tmp/opencv_yolo_ucx_tcp_run.log
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
timeout 180 ./sample 2>&1 | tee /tmp/opencv_yolo_ucx_rdma_run.log
```

Validate a manual run with:

```bash
grep -E "Attempting|Detection finished|output.jpg|Failed status|Execution exception|UCX endpoint error|terminate|Aborted" \
  /tmp/opencv_yolo_tcp_run.log

ls -lh output.jpg
```

Change the log filename for RDMA, UCX-TCP, or UCX-RDMA.

---

## Run benchmark.sh

Make the benchmark scripts executable:

```bash
cd "$GVIRTUS_HOME/examples/opencv-yolo"

chmod +x benchmark.sh benchmark.py
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
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh tcp
```

Rename the output folder to:

```text
benchmark_results/OpenCVYOLO_Final_Metrics_TCP
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

Rename the output folder to:

```text
benchmark_results/OpenCVYOLO_Final_Metrics_RDMA
```

Plain RDMA initially showed teardown/shutdown fragility around repeated CUDA runtime cleanup, especially `cudaUnregisterFatBinary`. The final run used the patched one-shot teardown handling and completed cleanly.

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
benchmark_results/OpenCVYOLO_Final_Metrics_UCX-TCP
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
benchmark_results/OpenCVYOLO_Final_Metrics_UCX-RDMA
```

---

## Output folders

The final result folders are manually renamed to:

```text
benchmark_results/OpenCVYOLO_Final_Metrics_TCP/
benchmark_results/OpenCVYOLO_Final_Metrics_RDMA/
benchmark_results/OpenCVYOLO_Final_Metrics_UCX-TCP/
benchmark_results/OpenCVYOLO_Final_Metrics_UCX-RDMA/
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
detection_ms
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

Important metrics:

```text
wall_s              Full frontend runtime.
detection_ms        Application-reported detection time, if available. Some final runs may leave this empty/NA.
valid_output        True when detection completed and output.jpg was written.
calls               Number of GVirtuS routine calls.
gvirtus_in_B        Bytes sent from frontend to backend at the GVirtuS routine layer.
gvirtus_out_B       Bytes returned from backend to frontend at the GVirtuS routine layer.
nic_rx_B            NIC receive byte delta during the run.
nic_tx_B            NIC transmit byte delta during the run.
routine_counts      Semicolon-separated routine call summary.
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
Attempting to use CUDA acceleration
Detection finished. Results saved to output.jpg
```

If `detection_ms` is empty, the run can still be valid as long as the output image was generated and the process exited successfully.

---

## Final summary generation

After the four final folders exist, generate a compact summary with:

```bash
cd "$GVIRTUS_HOME/examples/opencv-yolo"

python3 - <<'PY'
import csv, sys, statistics as st
from pathlib import Path

csv.field_size_limit(sys.maxsize)

base = Path("benchmark_results")
folders = {
    "TCP": base / "OpenCVYOLO_Final_Metrics_TCP",
    "RDMA": base / "OpenCVYOLO_Final_Metrics_RDMA",
    "UCX-TCP": base / "OpenCVYOLO_Final_Metrics_UCX-TCP",
    "UCX-RDMA": base / "OpenCVYOLO_Final_Metrics_UCX-RDMA",
}

fields = [
    "transport", "n", "ok",
    "wall_mean_s", "wall_std_s",
    "detection_mean_ms", "detection_std_ms",
    "calls_mean",
    "gvirtus_in_mean_B", "gvirtus_out_mean_B",
    "nic_rx_mean_B", "nic_tx_mean_B",
]

def vals(rows, field):
    out = []
    for r in rows:
        v = str(r.get(field, "")).strip()
        if not v or v.lower() in {"nan", "none", "null", "na"}:
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
        "detection_mean_ms": f"{mean(ok, 'detection_ms'):.3f}",
        "detection_std_ms": f"{std(ok, 'detection_ms'):.3f}",
        "calls_mean": f"{mean(ok, 'calls'):.1f}",
        "gvirtus_in_mean_B": f"{mean(ok, 'gvirtus_in_B'):.1f}",
        "gvirtus_out_mean_B": f"{mean(ok, 'gvirtus_out_B'):.1f}",
        "nic_rx_mean_B": f"{mean(ok, 'nic_rx_B'):.1f}",
        "nic_tx_mean_B": f"{mean(ok, 'nic_tx_B'):.1f}",
    })

out = base / "OpenCVYOLO_Final_Summary.csv"
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

Expected final summary shape:

```text
transport,n,ok,wall_mean_s,wall_std_s,detection_mean_ms,detection_std_ms,calls_mean,gvirtus_in_mean_B,gvirtus_out_mean_B,nic_rx_mean_B,nic_tx_mean_B
TCP,50,50,...
RDMA,50,50,...
UCX-TCP,50,50,...
UCX-RDMA,50,50,...
```

---