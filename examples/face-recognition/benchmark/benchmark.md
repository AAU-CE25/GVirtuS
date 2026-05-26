# FaceRecon GVirtuS Full-Metrics Benchmark

This document describes how to build, run, validate, and summarize the FaceRecon benchmark through GVirtuS using the final full-metrics workflow.

The benchmark compares four transport configurations:

- `tcp`
- `rdma`
- `ucx-tcp`
- `ucx-rdma`

The workload performs face recognition through GVirtuS. The benchmark is considered valid when the frontend process exits successfully and the result CSV reports a valid accuracy value.

For the final measurements, each transport should have:

```text
50 measured runs
5 warmup runs
exit_code = 0
accuracy_pct present
```

The final selected FaceRecon runs all completed:

```text
50/50 measured runs
100% accuracy
```

---

## General path setup

This document uses environment variables instead of user-specific absolute paths.

Set these once in each terminal before running commands:

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

# Optional: only needed if FaceRecon also uses a local OpenCV build.
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"

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
export CUDNN_ROOT="/path/to/cudnn-9.5.1"
```

---

## Required files

The folder should contain the FaceRecon example source, runner, and benchmark scripts.

Typical structure:

```text
examples/face-recognition/
├── benchmark.sh
├── benchmark.md
├── run.sh
├── setup.sh
├── main/source files
├── dataset/input files
└── benchmark_results/
```

Check the exact files in the local checkout with:

```bash
cd "$GVIRTUS_HOME/examples/face-recognition"

find . -maxdepth 2 -type f | sort
find . -maxdepth 1 -type f -executable -print
```

Commit the benchmark/source files needed to reproduce the run, but normally do not commit raw benchmark output folders.

Recommended benchmark files to commit if present:

```text
examples/face-recognition/benchmark.sh
examples/face-recognition/benchmark.md
examples/face-recognition/run.sh
examples/face-recognition/setup.sh
```

Also commit the FaceRecon source files and small metadata/class/input files required by the example.

Do not commit generated artifacts unless explicitly required:

```text
examples/face-recognition/benchmark_results/
examples/face-recognition/*.bak*
examples/face-recognition/*~
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
cuDNN 9 version compatibility reporting
one-shot cudaUnregisterFatBinary teardown handling
disabled unlinked CUDA/OpenGL interop handler registrations
```

Without these fixes, OpenCV/cuDNN compatibility, RDMA teardown, or backend plugin loading may fail again depending on the branch state and workload.

---

## Common frontend environment

Use this setup in the frontend terminal before manual tests or `benchmark.sh`.

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"
export CUDNN_ROOT="${CUDNN_ROOT:-$HOME/cudnn-9.5.1}"
export NPP_DIR="${NPP_DIR:-$HOME/.local/lib/python3.10/site-packages/nvidia/npp/lib}"

cd "$GVIRTUS_HOME/examples/face-recognition"

export CUDNN_LIB="$CUDNN_ROOT/lib"
export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$CUDNN_LIB:$NPP_DIR:/usr/local/cuda-12.6/lib64:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so:$GVIRTUS_HOME/lib/frontend/libcudnn.so.9"
```

If the FaceRecon build also requires OpenCV, add the local OpenCV library paths:

```bash
export OPENCV_HOME="${OPENCV_HOME:-$HOME/opencv-local}"
export LD_LIBRARY_PATH="$OPENCV_HOME/lib:$OPENCV_HOME/lib64:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$OPENCV_HOME/lib/pkgconfig:$OPENCV_HOME/lib64/pkgconfig:$OPENCV_HOME/share/pkgconfig:${PKG_CONFIG_PATH:-}"
```

---

## Backend commands

Run the backend in a separate terminal on the backend node.

The examples below use the `gvirtus-` container name. Some environments may use `gvirtus-<user>`; check with:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep gvirtus || true
```

### TCP backend

```bash
cd "$GVIRTUS_HOME"

docker rm -f gvirtus- gvirtus-<user> 2>/dev/null || true

GVIRTUS_CONFIG_FILE="${TCP_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### Plain RDMA backend

```bash
cd "$GVIRTUS_HOME"

docker rm -f gvirtus- gvirtus-<user> 2>/dev/null || true

GVIRTUS_CONFIG_FILE="${RDMA_CONFIG##*/}" \
GVIRTUS_LOG_LEVEL=30000 \
make run-gvirtus-backend-dev
```

### UCX-TCP backend

Use this for the UCX-over-TCP transport.

```bash
cd "$GVIRTUS_HOME"

docker rm -f gvirtus- gvirtus-<user> 2>/dev/null || true

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

docker rm -f gvirtus- gvirtus-<user> 2>/dev/null || true

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

## Run benchmark.sh

Make the benchmark script executable:

```bash
cd "$GVIRTUS_HOME/examples/face-recognition"

chmod +x benchmark.sh
```

Run one mode at a time. Start the matching backend first.

### TCP

```bash
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
TCP_CONFIG="$TCP_CONFIG" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh tcp
```

Rename the selected final output folder to:

```text
benchmark_results/FaceRecon Final Metrics TCP

```

### Plain RDMA

```bash
GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
RDMA_CONFIG="$RDMA_CONFIG" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh rdma
```

Rename the selected final output folder to:

```text
benchmark_results/FaceRecon Final Metrics RDMA

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
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the selected final output folder to:

```text
benchmark_results/FaceRecon Final Metrics UCX-TCP
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
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=180 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the selected final output folder to:

```text
benchmark_results/FaceRecon Final Metrics UCX-RDMA
```
---

## Output folders

Each final folder should contain:

```text
results.csv
logs/
counters/
```

Depending on the wrapper version, folders may also contain:

```text
routine_calls.csv
routine_summary.csv
nic_counters.csv
system/
```

---

## Metrics

FaceRecon `results.csv` uses these final columns:

```text
timestamp
mode
phase
run
exit_code
wall_s
accuracy_pct
execution_s
per_image_s
gvirtus_calls
gvirtus_in_bytes
gvirtus_out_bytes
nic_rx_bytes
nic_tx_bytes
nic_rx_packets
nic_tx_packets
config
frontend_log
backend_log
routine_summary
```

Some older TCP folders may also include detailed call timing columns:

```text
soft_ok
failed_gvirtus_calls
total_server_exec_ms
total_send_ms
total_recv_ms
total_comm_ms
total_call_ms
nic_rx_errors
nic_tx_errors
nic_rx_dropped
nic_tx_dropped
```

Important metrics:

```text
wall_s                Full frontend runtime.
execution_s           Application execution time.
per_image_s           Application execution time per image.
accuracy_pct          FaceRecon accuracy.
gvirtus_calls         Number of GVirtuS routine calls.
gvirtus_in_bytes      Bytes sent from frontend to backend at the GVirtuS routine layer.
gvirtus_out_bytes     Bytes returned from backend to frontend at the GVirtuS routine layer.
nic_rx_bytes          NIC receive byte delta during the run.
nic_tx_bytes          NIC transmit byte delta during the run.
routine_summary       Semicolon-separated routine call summary.
```

---

## Success criteria

A FaceRecon measured run is valid when:

```text
phase = measure
exit_code = 0
accuracy_pct is present
```

For the final selected folders, each transport completed:

```text
50 measured runs
50 valid runs
100% mean accuracy
```

---

## Final summary generation

After the final folders exist, generate the selected summary with:

```bash
cd "$GVIRTUS_HOME/examples/face-recognition"

python3 - <<'PY'
import csv, sys, statistics as st
from pathlib import Path

csv.field_size_limit(sys.maxsize)

base = Path("benchmark_results")

transport_folders = {
    "TCP": [
        base / "FaceRecon Final Metrics TCP",
        base / "FaceRecon Final Metrics TCP_2",
    ],
    "RDMA": [
        base / "FaceRecon Final Metrics RDMA",
        base / "FaceRecon Final Metrics RDMA_2",
    ],
    "UCX-TCP": [
        base / "FaceRecon Final Metrics UCX-TCP",
        base / "FaceRecon Final Metrics UCX-TCP_2",
    ],
    "UCX-RDMA": [
        base / "FaceRecon Final Metrics UCX-RDMA",
        base / "FaceRecon Final Metrics UCX-RDMA_2",
    ],
}

fields = [
    "transport",
    "selected_folder",
    "n",
    "ok",
    "wall_mean_s",
    "wall_std_s",
    "execution_mean_s",
    "execution_std_s",
    "execution_mean_ms",
    "per_image_mean_s",
    "per_image_mean_ms",
    "accuracy_mean_pct",
    "calls_mean",
    "gvirtus_in_mean_B",
    "gvirtus_out_mean_B",
    "nic_rx_mean_B",
    "nic_tx_mean_B",
]

def read_rows(folder):
    results = folder / "results.csv"
    if not results.exists():
        return [], []

    rows = [
        r for r in csv.DictReader(open(results, newline=""))
        if r.get("phase") == "measure"
    ]

    ok = []
    for r in rows:
        exit_code = str(r.get("exit_code", "")).strip()
        acc = str(r.get("accuracy_pct", "")).strip()
        if exit_code == "0" and acc:
            ok.append(r)

    return rows, ok

def nums(rows, field):
    vals = []
    for r in rows:
        v = str(r.get(field, "")).strip()
        if not v or v.lower() in {"nan", "none", "null"}:
            continue
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return vals

def mean(rows, field):
    vals = nums(rows, field)
    return st.mean(vals) if vals else float("nan")

def std(rows, field):
    vals = nums(rows, field)
    if len(vals) > 1:
        return st.stdev(vals)
    if len(vals) == 1:
        return 0.0
    return float("nan")

def summarize(transport, folder):
    rows, ok = read_rows(folder)

    execution_s = mean(ok, "execution_s")
    per_image_s = mean(ok, "per_image_s")

    return {
        "transport": transport,
        "selected_folder": folder.name,
        "n": len(rows),
        "ok": len(ok),
        "wall_mean_s": f"{mean(ok, 'wall_s'):.6f}",
        "wall_std_s": f"{std(ok, 'wall_s'):.6f}",
        "execution_mean_s": f"{execution_s:.6f}",
        "execution_std_s": f"{std(ok, 'execution_s'):.6f}",
        "execution_mean_ms": f"{execution_s * 1000:.3f}",
        "per_image_mean_s": f"{per_image_s:.6f}",
        "per_image_mean_ms": f"{per_image_s * 1000:.3f}",
        "accuracy_mean_pct": f"{mean(ok, 'accuracy_pct'):.3f}",
        "calls_mean": f"{mean(ok, 'gvirtus_calls'):.1f}",
        "gvirtus_in_mean_B": f"{mean(ok, 'gvirtus_in_bytes'):.1f}",
        "gvirtus_out_mean_B": f"{mean(ok, 'gvirtus_out_bytes'):.1f}",
        "nic_rx_mean_B": f"{mean(ok, 'nic_rx_bytes'):.1f}",
        "nic_tx_mean_B": f"{mean(ok, 'nic_tx_bytes'):.1f}",
    }

selected_rows = []

for transport, folders in transport_folders.items():
    candidates = []

    for folder in folders:
        if not folder.exists():
            continue

        rows, ok = read_rows(folder)
        candidates.append((folder, rows, ok))

    if not candidates:
        continue

    # Prefer folder with most OK rows; if tied, prefer _2.
    candidates.sort(key=lambda x: (len(x[2]), x[0].name.endswith("_2")), reverse=True)
    selected_rows.append(summarize(transport, candidates[0][0]))

out = base / "FaceRecon_Final_Summary_Selected.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(selected_rows)

print(",".join(fields))
for r in selected_rows:
    print(",".join(str(r[f]) for f in fields))

print(f"\nWrote: {out}")
PY
```
