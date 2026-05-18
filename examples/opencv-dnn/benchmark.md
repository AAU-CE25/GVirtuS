# OpenCV-DNN GVirtuS Benchmark

This example benchmarks OpenCV-DNN inference through three GVirtuS frontend/backend transports:

- `tcp`
- `rdma`
- `ucx`

The benchmark is considered valid when the frontend log contains:

```text
Saved total timings
Final Results:
Total images:
```

For plain RDMA, a run may be marked as `SOFT-OK` if inference completes but cleanup/shutdown later reports RDMA transport errors. This is acceptable for inference benchmarking because the useful OpenCV-DNN result has already been produced.

---

## Required files

The folder should contain:

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

Do not commit generated files such as:

```text
sample
benchmark_results/
total_times_*.txt
*.onnx
imagenet_test_1000/
*.bak*
```

---

## Download/generate required assets

Run these commands from:

```bash
cd /GVirtuS/examples/opencv-dnn
```

### Download MobileNetV2 ONNX model

The benchmark uses `mobilenetv2-10.onnx` from the ONNX Model Zoo.

```bash
wget -O mobilenetv2-10.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx

ls -lh mobilenetv2-10.onnx
```

Expected size is about 14 MB.

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

The dummy image is enough for transport benchmarking. Accuracy is expected to be meaningless.

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
cd /GVirtuS/examples/opencv-dnn

export OPENCV_HOME=/opencv-local
export PKG_CONFIG_PATH="$OPENCV_HOME/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$OPENCV_HOME/lib:$LD_LIBRARY_PATH"

g++ -x c++ main.cu -o sample $(pkg-config --cflags --libs opencv4)

ls -lh sample
```

The `-x c++` flag is required because the file is named `main.cu`, but this OpenCV-DNN example is compiled as normal C++.

---

## Common frontend environment

Use this setup in the frontend terminal before running manual tests or `benchmark.sh`.

```bash
cd /GVirtuS/examples/opencv-dnn

export GVIRTUS_HOME=/gvirtus-install
export GVIRTUS_LOGLEVEL=30000

export OPENCV_HOME=/opencv-local
export NPP_DIR=//.local/lib/python3.10/site-packages/nvidia/npp/lib
export CUDNN_ROOT=/cudnn-9.5.1
export CUDNN_LIB=$CUDNN_ROOT/lib

export GV_LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$OPENCV_HOME/lib:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:/lz4-install/lib"
export GV_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcuda.so:$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"

unset LD_PRELOAD
```

---

## Backend commands

Run the backend in a separate terminal.

### TCP backend

```bash
cd /GVirtuS

make stop-gvirtus || true

GVIRTUS_CONFIG_FILE=properties.json \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

If using a dedicated TCP config, replace `properties.json` with the correct file, for example:

```bash
GVIRTUS_CONFIG_FILE=properties_tcp_25_2.json \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### Plain RDMA backend

```bash
cd /GVirtuS

make stop-gvirtus || true

GVIRTUS_CONFIG_FILE=properties_plain_rdma.json \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### UCX backend

```bash
cd /GVirtuS

make stop-gvirtus || true

GVIRTUS_UCX_DATAPATH=am \
GVIRTUS_CONFIG_FILE=properties_ucx.json \
GVIRTUS_LOG_LEVEL=30000 \
UCX_TLS=rc_mlx5,ud_mlx5,self \
UCX_NET_DEVICES=mlx5_1:1 \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_GID_INDEX=3 \
UCX_RNDV_THRESH=inf \
UCX_ZCOPY_THRESH=inf \
UCX_LOG_LEVEL=warn \
make run-gvirtus-backend-dev
```

Use `GVIRTUS_UCX_DATAPATH=am` for UCX. Do not use `rdma` here; plain RDMA and UCX are different communicator paths.

---

## Manual frontend tests

Run these from:

```bash
cd /GVirtuS/examples/opencv-dnn
```

### TCP frontend

```bash
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json

timeout 15s env \
  LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
  LD_PRELOAD="$GV_PRELOAD" \
  ./sample 2>&1 | tee /tmp/opencv_dnn_tcp_run.log
```

If using a dedicated TCP config:

```bash
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_tcp_25_2.json
```

### Plain RDMA frontend

```bash
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_plain_rdma.json

timeout 15s env \
  LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
  LD_PRELOAD="$GV_PRELOAD" \
  ./sample 2>&1 | tee /tmp/opencv_dnn_rdma_run.log
```

### UCX frontend

```bash
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_ucx.json

timeout 15s env \
  GVIRTUS_UCX_DATAPATH=am \
  UCX_TLS=rc_mlx5,ud_mlx5,self \
  UCX_NET_DEVICES=mlx5_1:1 \
  UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
  UCX_IB_GID_INDEX=3 \
  UCX_RNDV_THRESH=inf \
  UCX_ZCOPY_THRESH=inf \
  UCX_LOG_LEVEL=warn \
  LD_LIBRARY_PATH="$GV_LD_LIBRARY_PATH" \
  LD_PRELOAD="$GV_PRELOAD" \
  ./sample 2>&1 | tee /tmp/opencv_dnn_ucx_run.log
```

Validate a manual run with:

```bash
grep -E "Running inference|Image:|Saved total timings|Final Results|Total images|Accuracy|Failed status|Execution exception|UCX endpoint error|terminate|Aborted" \
  /tmp/opencv_dnn_tcp_run.log
```

Change the log filename for RDMA or UCX.

---

## Run benchmark.sh

Make the benchmark script executable:

```bash
cd /GVirtuS/examples/opencv-dnn

chmod +x benchmark.sh
```

Run one mode at a time:

```bash
export GVIRTUS_HOME=/gvirtus-install

RUNS=3 WARMUPS=1 IFACES="ens1f1np1" ./benchmark.sh tcp
RUNS=3 WARMUPS=1 IFACES="ens1f1np1" ./benchmark.sh rdma
RUNS=3 WARMUPS=1 IFACES="ens1f1np1" ./benchmark.sh ucx
```

Run all modes:

```bash
RUNS=3 WARMUPS=1 IFACES="ens1f1np1" ./benchmark.sh all
```

Choose a different frontend command:

```bash
FRONTEND_CMD="./sample" RUNS=5 WARMUPS=1 ./benchmark.sh ucx
```

The script writes:

```text
benchmark_results/frontend_<timestamp>_<mode>/
├── results.csv
├── nic_counters.csv
├── meta.txt
├── logs/
└── counters/
```

---

## Metrics

`results.csv` contains:

```text
mode
run_type
run
status
exit_code
wall_s
inference_ms
total_images
accuracy
correct_predictions
predicted_class
confidence_pct
valid_output
log_file
timestamp
```

`nic_counters.csv` contains:

```text
mode
run_type
run
iface
rx_before
rx_after
rx_delta
tx_before
tx_after
tx_delta
```

Use:

- `wall_s` for full frontend runtime.
- `inference_ms` for OpenCV-DNN image inference time.
- NIC byte deltas for approximate transport traffic.

---

## Success criteria

A run is valid if it contains:

```text
Saved total timings
Final Results:
Total images:
```

Status meanings:

- `OK`: process exited cleanly.
- `SOFT-OK`: valid inference output exists, but cleanup/timeout happened later.
- `FAILED`: no valid inference output.

Plain RDMA may be `SOFT-OK` because inference can complete before RDMA cleanup/shutdown errors.

---

## Recommended `.gitignore`

```gitignore
examples/opencv-dnn/benchmark_results/
examples/opencv-dnn/total_times_*.txt
examples/opencv-dnn/sample
examples/opencv-dnn/*.onnx
examples/opencv-dnn/imagenet_test_1000/
examples/opencv-dnn/*.bak*
```

---

## Files to commit

Recommended files to commit for this benchmark:

```text
examples/opencv-dnn/benchmark.sh
examples/opencv-dnn/benchmark.md
examples/opencv-dnn/main.cu
src/backend/Process.cpp
src/communicators/rdma/RdmaCommunicator.cpp
```

Commit config files only if they are intended to be shared:

```text
etc/properties.json
etc/properties_plain_rdma.json
etc/properties_ucx.json
```

Do not commit generated benchmark outputs, the compiled `sample` binary, downloaded ONNX model files, or generated image folders unless explicitly required.
