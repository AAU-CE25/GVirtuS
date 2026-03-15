#!/bin/bash
set -euo pipefail

export OPENPOSE_ROOT=/opt/openpose
export GVIRTUS_HOME=/opt/GVirtuS
export LOCAL_GVIRTUS_ROOT=/workspace/GVirtuS
export LD_LIBRARY_PATH="$OPENPOSE_ROOT/build/src/openpose:$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:${LD_LIBRARY_PATH:-}"

echo "Building local GVirtuS from mounted workspace..."
cmake -S "$LOCAL_GVIRTUS_ROOT" -B /tmp/gvirtus-build -DCMAKE_INSTALL_PREFIX="$GVIRTUS_HOME"
cmake --build /tmp/gvirtus-build -j"$(nproc)"
cmake --install /tmp/gvirtus-build

echo "Compiling OpenPose test..."
cd /opt/openpose/examples/gvirtus
OPENCV_PKG=$(pkg-config --exists opencv4 && echo opencv4 || echo opencv)

nvcc 00_test.cpp -o 00_test -g -O0 -std=c++17 \
  -I"$OPENPOSE_ROOT/include" \
  -I"$OPENPOSE_ROOT/3rdparty/caffe/include" \
  -L"$OPENPOSE_ROOT/build/src/openpose" \
  -L"$OPENPOSE_ROOT/build/caffe/lib" \
  -lopenpose -lcaffe -lgflags \
  $(pkg-config --cflags --libs "$OPENCV_PKG")

echo "Running OpenPose test..."
cd "$OPENPOSE_ROOT"
./examples/gvirtus/00_test
