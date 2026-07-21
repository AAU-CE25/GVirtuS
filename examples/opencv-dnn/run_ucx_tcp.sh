#!/bin/bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

IMAGE=ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6
CONTAINER_NAME="opencv-dnn-frontend-ucx-tcp-${USER%@*}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS:-1}"
FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT:-120}"

echo "=== BENCH_INTERNAL_RUNS=${BENCH_INTERNAL_RUNS} ==="
echo "=== FRONTEND UCX TCP ==="
echo "=== FRONTEND_TIMEOUT=${FRONTEND_TIMEOUT}s ==="

docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  -e GVIRTUS_HOME=/opt/GVirtuS \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_LOGLEVEL=30000 \
  -e GVIRTUS_UCX_DATAPATH=am \
  -e GVIRTUS_RMA_ZEROCOPY=0 \
  -e GVIRTUS_GPUDIRECT=0 \
  -e UCX_TLS=tcp,self \
  -e UCX_NET_DEVICES=ens1f1np1 \
  -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
  -e UCX_LOG_LEVEL=warn \
  -e UCX_WARN_UNUSED_ENV_VARS=n \
  -e BENCH_INTERNAL_RUNS="${BENCH_INTERNAL_RUNS}" \
  -e FRONTEND_TIMEOUT="${FRONTEND_TIMEOUT}" \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/usr/local/cuda/lib64 \
  -v "$(pwd):/app:rw" \
  -v "$HOME/GVirtuS/etc/properties_ucx.json:/opt/GVirtuS/etc/properties_ucx.json:ro" \
  "${IMAGE}" \
  -c '
    set -euo pipefail
    cd /app

    echo "=== config ==="
    cat "${GVIRTUS_CONFIG}"

    echo "=== env check ==="
    env | grep -E "GVIRTUS|UCX|RMA|GPUDIRECT|FRONTEND_TIMEOUT" | sort || true

    if [ ! -f dnn_test ]; then
      echo "=== compiling main.cu ==="
      nvcc main.cu -o dnn_test -g \
        $(pkg-config --cflags --libs opencv4) \
        -L${GVIRTUS_HOME}/lib/frontend -L${GVIRTUS_HOME}/lib \
        -lcudart -lcublas -lcudnn
    else
      echo "=== using existing dnn_test ==="
    fi

    echo "=== ldd check ==="
    ldd dnn_test | grep -E "cudart|cublas|cudnn|cuda|gvirtus" || true

    echo "=== dataset/model check ==="
    pwd
    ls -lh main.cu dnn_test mobilenetv2-10.onnx squeezenet1.1-7.onnx vgg16-7.onnx
    find imagenet_test_1000 -type f | wc -l

    echo "=== running BENCH_INTERNAL_RUNS=${BENCH_INTERNAL_RUNS} with timeout ${FRONTEND_TIMEOUT}s ==="
    set +e
    timeout --preserve-status "${FRONTEND_TIMEOUT}s" ./dnn_test
    rc=$?
    echo "=== dnn_test exit code: ${rc} ==="
    exit "$rc"
  '
