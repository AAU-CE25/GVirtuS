#!/bin/bash
# Frontend launcher for opencv-dnn over GVirtuS+UCX.
# Reuses the opencv-yolo image (same OpenCV+CUDA-DNN + GVirtuS frontend stack).
# Pre-req: (1) backend running on es-dpu-01:25.25.25.1:32223,
#          (2) models + imagenet_test_1000 placed in this directory (see setup.sh).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

IMAGE=ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6
CONTAINER_NAME="opencv-dnn-frontend-${USER%@*}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Pre-flight: warn if dataset/models missing so we do not waste a 40min build.
for m in mobilenetv2-10.onnx squeezenet1.1-7.onnx resnet18-v1-7.onnx vgg16-7.onnx; do
  [ -f "$m" ] || echo "WARN: $m not found in $(pwd) - run setup.sh first"
done
[ -d imagenet_test_1000 ] || echo "WARN: imagenet_test_1000/ not found - run setup.sh first"

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
    echo "=== compiling main.cu ==="
    nvcc main.cu -o dnn_test -g \
      $(pkg-config --cflags --libs opencv4) \
      -L${GVIRTUS_HOME}/lib/frontend -L${GVIRTUS_HOME}/lib \
      -lcudart -lcublas -lcudnn

    echo "=== ldd check ==="
    ldd dnn_test | grep -E "cudnn|cudart|cublas" || true

    echo "=== running ==="
    ./dnn_test
  '
