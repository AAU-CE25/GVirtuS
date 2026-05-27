# SimpleMatrix GVirtuS Full-Metrics Benchmark

This document describes how to build, run, validate, and summarize the SimpleMatrix benchmark through GVirtuS using the final full-metrics workflow.

The benchmark compares four transport configurations:

- `tcp`
- `rdma`
- `ucx-tcp-am`
- `ucx-rdma`

The workload runs a CUDA/cuBLAS matrix operation through GVirtuS. Matrix size is controlled with the `MATRIX_N` environment variable.

The benchmark is considered valid when the frontend log reports:

```text
BENCHMARK_MATRIX_N=<N>
RESULT_CHECK=PASS
BENCHMARK_RESULT_MS=<time>
```

For the final measurements, each transport and matrix size should have:

```text
50 measured runs
5 warmup runs
status = OK
exit_code = 0
valid_output = true
RESULT_CHECK = PASS
```

---

## General path setup

This document uses environment variables instead of user-specific absolute paths.

Set these once in each terminal before running commands:

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"

export TCP_CONFIG="${TCP_CONFIG:-$GVIRTUS_HOME/etc/properties.json}"
export RDMA_CONFIG="${RDMA_CONFIG:-$GVIRTUS_HOME/etc/properties_plain_rdma.json}"
export UCX_CONFIG="${UCX_CONFIG:-$GVIRTUS_HOME/etc/properties_ucx.json}"

# Optional: only needed if your Docker container name differs from the default.
export GVIRTUS_CONTAINER="${GVIRTUS_CONTAINER:-gvirtus-}"
```

If your installation is not under `$HOME`, set the variables explicitly:

```bash
export GVIRTUS_HOME="/path/to/GVirtuS"
export LZ4_HOME="/path/to/lz4-install"
```

---

## Matrix sizes

The final benchmark uses:

```text
N = 256, 512, 1024, 2048, 4096, 8192, 16384
```

Payload labels are based on one `float32` matrix:

```text
256    -> 256KB
512    -> 1MB
1024   -> 4MB
2048   -> 16MB
4096   -> 64MB
8192   -> 256MB
16384  -> 1GB
```

The full operation transfers approximately:

```text
frontend to backend: A + B matrices
backend to frontend: C matrix
```

so the GVirtuS byte counters are approximately 2x payload in and 1x payload out.

---

## Required files

The folder should contain:

```text
examples/simple_matrix/
├── benchmark.py
├── benchmark.sh
├── benchmark.md
├── simple_matrix.cu
├── simple_matrix
├── backend.sh
├── frontend.sh
└── Dockerfile
```

Commit the benchmark/source files needed to reproduce the final workflow:

```text
examples/simple_matrix/benchmark.py
examples/simple_matrix/benchmark.sh
examples/simple_matrix/benchmark.md
examples/simple_matrix/simple_matrix.cu
```

Optional helper files:

```text
examples/simple_matrix/backend.sh
examples/simple_matrix/frontend.sh
examples/simple_matrix/Dockerfile
```

Do not commit generated artifacts unless explicitly required:

```text
examples/simple_matrix/simple_matrix
examples/simple_matrix/benchmark_results/
examples/simple_matrix/*.bak*
examples/simple_matrix/*~
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

For SimpleMatrix specifically, the `cudaUnregisterFatBinary` teardown patch is important for stable repeated RDMA runs.

---

## Build SimpleMatrix

Run from:

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"
```

If the binary already exists and is current, rebuild is not needed. Otherwise build according to the local project setup, for example through the existing Makefile/script if available.

Check the binary:

```bash
ls -lh ./simple_matrix
file ./simple_matrix
```

SimpleMatrix uses `MATRIX_N` to select the matrix size. It does **not** use a positional CLI argument for the final benchmark workflow.

Correct:

```bash
MATRIX_N=1024 ./simple_matrix
```

Incorrect for final benchmarking:

```bash
./simple_matrix 1024
```

---

## Common frontend environment

Use this setup in the frontend terminal before manual tests or `benchmark.sh`.

```bash
export GVIRTUS_HOME="${GVIRTUS_HOME:-$HOME/GVirtuS}"
export LZ4_HOME="${LZ4_HOME:-$HOME/lz4-install}"

cd "$GVIRTUS_HOME/examples/simple_matrix"

export LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LZ4_HOME/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$GVIRTUS_HOME/lib/frontend/libcudart.so:$GVIRTUS_HOME/lib/frontend/libcublas.so"
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

### UCX-TCP-AM backend

Use this for the UCX-over-TCP active-message transport.

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
UCX-TCP-AM and UCX-RDMA should be treated as separate benchmark transports.
```

---

## Manual frontend tests

Run these from:

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"
```

### TCP frontend

```bash
GVIRTUS_CONFIG="$TCP_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
MATRIX_N=1024 \
timeout 60 ./simple_matrix 2>&1 | tee /tmp/simplematrix_tcp_run.log
```

### Plain RDMA frontend

```bash
GVIRTUS_CONFIG="$RDMA_CONFIG" \
GVIRTUS_HOME="$GVIRTUS_HOME" \
GVIRTUS_LOGLEVEL=10000 \
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
LD_PRELOAD="$LD_PRELOAD" \
MATRIX_N=1024 \
timeout 120 ./simple_matrix 2>&1 | tee /tmp/simplematrix_rdma_run.log
```

### UCX-TCP-AM frontend

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
MATRIX_N=1024 \
timeout 120 ./simple_matrix 2>&1 | tee /tmp/simplematrix_ucx_tcp_run.log
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
MATRIX_N=1024 \
timeout 120 ./simple_matrix 2>&1 | tee /tmp/simplematrix_ucx_rdma_run.log
```

Validate a manual run with:

```bash
grep -E "BENCHMARK_MATRIX_N|RESULT_CHECK|BENCHMARK_RESULT_MS|STAGE_|FAILED|ERROR|terminate|Aborted" \
  /tmp/simplematrix_tcp_run.log
```

Change the log filename for RDMA, UCX-TCP-AM, or UCX-RDMA.

---

## Full-metrics benchmark wrapper

Make the wrapper executable:

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

chmod +x benchmark.py benchmark.sh
```

The final wrapper writes:

```text
benchmark_results/simplematrix_full_metrics_<timestamp>_<mode>/
├── results.csv
├── routine_calls.csv
├── routine_summary.csv
├── nic_counters.csv
├── logs/
├── counters/
└── system/
```

The wrapper runs the frontend using:

```text
FRONTEND_CMD_TEMPLATE="MATRIX_N={size} ./simple_matrix"
```

This is important because the final benchmark uses `MATRIX_N`, not CLI arguments.

---

## Run benchmark.sh

Run one mode at a time. Start the matching backend first.

### TCP

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
TCP_CONFIG="$TCP_CONFIG" \
FRONTEND_CMD_TEMPLATE="MATRIX_N={size} ./simple_matrix" \
SIZES="256 512 1024 2048 4096 8192 16384" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=300 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh tcp
```

Rename the final folder to:

```text
benchmark_results/SimpleMatrix_Final_Metrics_TCP
```

### Plain RDMA

Plain RDMA was more stable when run in per-size chunks with backend restarts between chunks. A combined long run stalled after many successful launches, so the final RDMA data was assembled from valid 50/50 OK size-level runs.

For small sizes, a combined run completed the first sizes cleanly:

```text
256
512
1024
```

The remaining sizes were rerun per-size:

```text
2048
4096
8192
16384
```

Run one size at a time after restarting the RDMA backend:

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

GVIRTUS_HOME="$GVIRTUS_HOME" \
LZ4_HOME="$LZ4_HOME" \
RDMA_CONFIG="$RDMA_CONFIG" \
FRONTEND_CMD_TEMPLATE="MATRIX_N={size} ./simple_matrix" \
SIZES="2048" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=300 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh rdma
```

Rename per-size folders as needed:

```text
benchmark_results/SimpleMatrix_Final_Metrics_RDMA_n2048
benchmark_results/SimpleMatrix_Final_Metrics_RDMA_n4096
benchmark_results/SimpleMatrix_Final_Metrics_RDMA_n8192
benchmark_results/SimpleMatrix_Final_Metrics_RDMA_n16384
```

The partial combined folder used for the first three sizes was preserved as:

```text
benchmark_results/SimpleMatrix_RDMA_partial_hang_n2048_run28
```

Only the fully completed 50/50 OK sizes from that partial folder are used in the final summary.

### UCX-TCP-AM

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

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
FRONTEND_CMD_TEMPLATE="MATRIX_N={size} ./simple_matrix" \
SIZES="256 512 1024 2048 4096 8192 16384" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=300 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the final folder to:

```text
benchmark_results/SimpleMatrix_Final_Metrics_UCX-TCP_AM_forced
```

Important UCX-TCP note:

```text
UCX-TCP is reported as UCX-TCP-AM because the functional configuration uses the GVirtuS UCX active-message datapath with rendezvous/zcopy disabled. Enabling UCX protocol/rendezvous for TCP was not compatible with the current implementation and failed during remote-key handling.
```

### UCX-RDMA

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

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
FRONTEND_CMD_TEMPLATE="MATRIX_N={size} ./simple_matrix" \
SIZES="256 512 1024 2048 4096 8192 16384" \
RUNS=50 \
WARMUPS=5 \
RUN_TIMEOUT=300 \
GVIRTUS_LOGLEVEL=10000 \
IFACES="ens1f1np1 ens1f0np0 bond0" \
./benchmark.sh ucx
```

Rename the final folder to:

```text
benchmark_results/SimpleMatrix_Final_Metrics_UCX-RDMA
```

---


## Metrics

The final `results.csv` contains:

```text
timestamp
mode
phase
matrix_n
payload_label
run
status
exit_code
wall_s
benchmark_result_ms
malloc_ms
cudamalloc_ms
h2d_ms
cublas_create_ms
gemm_ms
d2h_ms
cleanup_ms
result_check
valid_output
calls
gvirtus_in_B
gvirtus_out_B
nic_rx_B
nic_tx_B
config
frontend_cmd
log_file
error
routine_counts
```

Important metrics:

```text
wall_s                  Full frontend runtime.
benchmark_result_ms     Application-reported total benchmark time.
malloc_ms               Host allocation / initialization stage.
cudamalloc_ms           CUDA allocation stage.
h2d_ms                  Host-to-device transfer stage.
gemm_ms                 cuBLAS GEMM stage.
d2h_ms                  Device-to-host transfer stage.
cleanup_ms              Cleanup/free stage.
result_check            Correctness check. Must be PASS.
calls                   Number of GVirtuS routine calls.
gvirtus_in_B            Bytes sent from frontend to backend at the GVirtuS routine layer.
gvirtus_out_B           Bytes returned from backend to frontend at the GVirtuS routine layer.
nic_rx_B                NIC receive byte delta during the run.
nic_tx_B                NIC transmit byte delta during the run.
routine_counts          Semicolon-separated routine call summary.
```

`routine_summary.csv` aggregates per-routine call counts and transferred bytes.  
`routine_calls.csv` contains per-call records.  
`nic_counters.csv` contains raw NIC counter deltas per interface.

---

## Success criteria

A SimpleMatrix measured run is valid when:

```text
phase = measure
status = OK
exit_code = 0
valid_output = true
result_check = PASS
```

Each final transport/size combination should have:

```text
50 measured runs
50 valid runs
```

---

## Final summary generation

After the final folders exist, generate the combined summary with:

```bash
cd "$GVIRTUS_HOME/examples/simple_matrix"

python3 - <<'PY'
import csv, sys, statistics as st
from pathlib import Path

csv.field_size_limit(sys.maxsize)

base = Path("benchmark_results")

sources = {
    "TCP": [base / "SimpleMatrix_Final_Metrics_TCP"],
    "RDMA": [
        base / "SimpleMatrix_RDMA_partial_hang_n2048_run28",
        base / "SimpleMatrix_Final_Metrics_RDMA_n2048",
        base / "SimpleMatrix_Final_Metrics_RDMA_n4096",
        base / "SimpleMatrix_Final_Metrics_RDMA_n8192",
        base / "SimpleMatrix_Final_Metrics_RDMA_n16384",
    ],
    "UCX-TCP-AM": [base / "SimpleMatrix_Final_Metrics_UCX-TCP_AM_forced"],
    "UCX-RDMA": [base / "SimpleMatrix_Final_Metrics_UCX-RDMA"],
}

wanted = [256, 512, 1024, 2048, 4096, 8192, 16384]

def collect_rows(transport):
    rows_by_n = {n: [] for n in wanted}
    for folder in sources[transport]:
        results = folder / "results.csv"
        if not results.exists():
            print(f"WARNING missing: {results}", file=sys.stderr)
            continue

        with open(results, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("phase") != "measure":
                    continue
                if r.get("status") != "OK" or r.get("exit_code") != "0" or r.get("valid_output") != "true":
                    continue

                n = int(r["matrix_n"])

                # For the partial RDMA folder, only take sizes that fully completed.
                if transport == "RDMA" and folder.name == "SimpleMatrix_RDMA_partial_hang_n2048_run28" and n not in {256, 512, 1024}:
                    continue

                if n in rows_by_n:
                    rows_by_n[n].append(r)

    return rows_by_n

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

out = base / "SimpleMatrix_Final_Summary_All.csv"

header = [
    "transport", "matrix_n", "payload_label", "n", "ok",
    "wall_mean_s", "wall_std_s",
    "benchmark_result_mean_ms", "benchmark_result_std_ms",
    "calls_mean",
    "gvirtus_in_mean_B", "gvirtus_out_mean_B",
    "nic_rx_mean_B", "nic_tx_mean_B",
]

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    print(",".join(header))

    for transport in ["TCP", "RDMA", "UCX-TCP-AM", "UCX-RDMA"]:
        rows_by_n = collect_rows(transport)

        for n in wanted:
            rows = rows_by_n[n]
            if not rows:
                line = [transport, n, "MISSING", 0, 0] + ["nan"] * 9
            else:
                line = [
                    transport,
                    n,
                    rows[0]["payload_label"],
                    len(rows),
                    len(rows),
                    f"{mean(rows, 'wall_s'):.6f}",
                    f"{std(rows, 'wall_s'):.6f}",
                    f"{mean(rows, 'benchmark_result_ms'):.3f}",
                    f"{std(rows, 'benchmark_result_ms'):.3f}",
                    f"{mean(rows, 'calls'):.1f}",
                    f"{mean(rows, 'gvirtus_in_B'):.1f}",
                    f"{mean(rows, 'gvirtus_out_B'):.1f}",
                    f"{mean(rows, 'nic_rx_B'):.1f}",
                    f"{mean(rows, 'nic_tx_B'):.1f}",
                ]

            w.writerow(line)
            print(",".join(map(str, line)))

print(f"\nWrote: {out}")
PY