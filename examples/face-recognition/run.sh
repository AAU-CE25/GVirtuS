#!/bin/bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend
nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu  -lcudart -lcublas
ldd libextension.so
env LD_PRELOAD=${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so \
python3 cnn.py