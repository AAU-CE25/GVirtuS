#!/bin/bash
# Frontend launcher for opencv-yolo over GVirtuS+UCX.
# Benchmark-friendly: supports BENCH_INTERNAL_RUNS and low logs.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

IMAGE="ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6"
CONTAINER_NAME="opencv-yolo-frontend-ucx-${USER%@*}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS:-1}"
FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT:-120}"

echo "=== BENCH_INTERNAL_RUNS=${BENCH_INTERNAL_RUNS} ==="
echo "=== FRONTEND OPENCV-YOLO UCX ==="
echo "=== FRONTEND_TIMEOUT=${FRONTEND_TIMEOUT}s ==="

docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --device /dev/infiniband \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -e GVIRTUS_HOME=/opt/GVirtuS \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_UCX_DATAPATH=am \
  -e GVIRTUS_GPUDIRECT=1 \
  -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 \
  -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
  -e UCX_IB_GID_INDEX=3 \
  -e UCX_LOG_LEVEL=warn \
  -e UCX_WARN_UNUSED_ENV_VARS=n \
  -e GVIRTUS_LOGLEVEL=30000 \
  -e BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS}" \
  -e FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT}" \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/usr/local/cuda/lib64 \
  -v "$(pwd):/app:rw" \
  -v "$HOME/GVirtuS/etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json:ro" \
  "${IMAGE}" \
  -c '
    set -euo pipefail
    cd /app

    echo "=== [1/5] sanity: OpenCV DNN-CUDA backend enabled? ==="
    python3 -c "
import cv2
info = cv2.getBuildInformation()
print(\"OpenCV:\", cv2.__version__)
for k in (\"NVIDIA CUDA:\", \"cuDNN:\", \"DNN:\", \"OpenCV DNN CUDA:\"):
    for line in info.splitlines():
        if k.lower() in line.lower():
            print(line.strip())
            break
" || echo "(python3 probe failed, proceeding)"

    echo ""
    echo "=== [2/5] config/env ==="
    cat "${GVIRTUS_CONFIG}"
    env | grep -E "GVIRTUS|UCX|RMA|GPUDIRECT|BENCH_INTERNAL_RUNS|FRONTEND_TIMEOUT" | sort || true

    echo ""
    echo "=== [3/5] compiling main.cu against GVirtuS frontend stubs ==="
    if [ ! -f yolo_test ]; then
      nvcc main.cu -o yolo_test -g \
        $(pkg-config --cflags --libs opencv4) \
        -L${GVIRTUS_HOME}/lib/frontend \
        -L${GVIRTUS_HOME}/lib \
        -lcudart -lcublas -lcudnn
    else
      echo "=== using existing yolo_test ==="
    fi

    echo ""
    echo "=== [4/5] verifying ldd picks the GVirtuS stubs ==="
    ldd yolo_test | grep -E "cudnn|cudart|cublas|gvirtus" || true

    echo ""
    echo "=== dataset/model check ==="
    ls -lh main.cu yolo_test weights/yolov5n.onnx images/zidane.jpg class.names

    echo ""
    echo "=== [5/5] running YOLO benchmark ==="
    set +e
    timeout --preserve-status "${FRONTEND_TIMEOUT}s" ./yolo_test
    rc=$?
    echo "=== yolo_test exit code: ${rc} ==="

    echo ""
    echo "=== done. output.jpg should have been overwritten ==="
    ls -la output.jpg 2>/dev/null || echo "WARN: output.jpg not produced"

    exit "$rc"
  '
