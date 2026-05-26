#!/bin/bash
set -e

export OPENPOSE_ROOT=/opt/openpose
export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=$OPENPOSE_ROOT/build/src/openpose:$GVIRTUS_HOME/lib:$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH

echo "🛠️ Compiling OpenPose multi-image benchmark..."
cd /opt/openpose/examples/gvirtus

nvcc test_multiple.cpp -o test_multiple -g \
  -I$OPENPOSE_ROOT/include \
  -I$OPENPOSE_ROOT/3rdparty/caffe/include \
  -L$OPENPOSE_ROOT/build/src/openpose \
  -L$OPENPOSE_ROOT/build/caffe/lib \
  -lopenpose -lcaffe -lgflags \
  $(pkg-config --cflags --libs opencv4)

NET_RES="${NET_RES:--1x368}"
echo "🚀 Running OpenPose multi-image benchmark (test_multiple)..."
echo "Config: $GVIRTUS_CONFIG"
echo "Net resolution: $NET_RES"
cd $OPENPOSE_ROOT
./examples/gvirtus/test_multiple --custom_net_resolution=$NET_RES
