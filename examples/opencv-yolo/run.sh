#!/bin/bash
# Frontend launcher for opencv-yolo over GVirtuS+UCX.
# Pre-req: backend running on es-dpu-01:25.25.25.1:32223.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

IMAGE="ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6"
CONTAINER_NAME="opencv-yolo-frontend-${USER%@*}"

# Make sure no stale container from a previous run.
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

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
  -e UCX_LOG_LEVEL=info \
  -e GVIRTUS_LOGLEVEL=10000 \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/usr/local/cuda/lib64 \
  -v "$(pwd):/app:rw" \
  -v "$HOME/GVirtuS/etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json:ro" \
  "${IMAGE}" \
  -c '
    set -euo pipefail
    cd /app

    echo "=== [1/4] sanity: OpenCV DNN-CUDA backend enabled? ==="

    python3 -c "
import cv2
info = cv2.getBuildInformation()

print(\"OpenCV:\", cv2.__version__)

keys = (\"NVIDIA CUDA:\", \"cuDNN:\", \"DNN:\", \"OpenCV DNN CUDA:\")
for k in keys:
    for line in info.splitlines():
        if k.lower() in line.lower():
            print(line.strip())
            break
" || echo "(python3 probe failed, proceeding)"

    echo ""
    echo "=== [2/4] compiling main.cu against GVirtuS frontend stubs ==="

    nvcc main.cu -o yolo_test -g \
      $(pkg-config --cflags --libs opencv4) \
      -L${GVIRTUS_HOME}/lib/frontend \
      -L${GVIRTUS_HOME}/lib \
      -lcudart -lcublas -lcudnn

    echo ""
    echo "=== [3/4] verifying ldd picks the GVirtuS stubs, not real CUDA ==="

    ldd yolo_test | grep -E "cudnn|cudart|cublas|gvirtus" || true

    echo ""
    echo "Expected: libcudnn / libcudart / libcublas should resolve from:"
    echo "  /opt/GVirtuS/lib/frontend/"

    echo ""
    echo "=== [4/4] running yolo on images/zidane.jpg ==="

    ./yolo_test

    echo ""
    echo "=== done. output.jpg should have been overwritten ==="

    ls -la output.jpg 2>/dev/null || echo "WARN: output.jpg not produced"
  '
