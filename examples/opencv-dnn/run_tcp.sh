#!/bin/bash
# Frontend launcher for opencv-dnn over GVirtuS plain TCP.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

IMAGE=ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6
CONTAINER_NAME="opencv-dnn-frontend-${USER%@*}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS:-1}"

echo "=== BENCH_INTERNAL_RUNS=${BENCH_INTERNAL_RUNS} ==="
echo "=== MODE: plain TCP ==="

for m in mobilenetv2-10.onnx squeezenet1.1-7.onnx vgg16-7.onnx; do
  [ -f "$m" ] || echo "WARN: $m not found in $(pwd) - run setup.sh first"
done

[ -d imagenet_test_1000 ] || echo "WARN: imagenet_test_1000/ not found - run setup.sh first"

docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  -e GVIRTUS_HOME=/opt/GVirtuS \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties.json \
  -e GVIRTUS_LOGLEVEL=30000 \
  -e BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS}" \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/usr/local/cuda/lib64 \
  -v "$(pwd):/app:rw" \
  -v "$HOME/GVirtuS/etc/properties.json:/opt/GVirtuS/etc/properties.json:ro" \
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
    ldd dnn_test | grep -E "cudart|cublas|cudnn|cuda|gvirtus" || true

    echo "=== pwd/files ==="
    pwd
    ls -lh main.cu dnn_test mobilenetv2-10.onnx squeezenet1.1-7.onnx vgg16-7.onnx
    find imagenet_test_1000 -type f | wc -l

    echo "=== running BENCH_INTERNAL_RUNS=${BENCH_INTERNAL_RUNS} ==="
    ./dnn_test
  '