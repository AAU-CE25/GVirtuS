#!/bin/bash
export GVIRTUS_HOME=/opt/GVirtuS
export LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/build/plugins/cudadr:/opt/GVirtuS/build/plugins/cudart:/opt/GVirtuS/build/plugins/cublas:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/GVirtuS/lib/frontend/libcuda.so.1:/opt/GVirtuS/lib/frontend/libcudart.so.12:/opt/GVirtuS/lib/frontend/libcublas.so.12:/opt/GVirtuS/lib/frontend/libcublasLt.so.12
echo "=== python3 rapids_matrix.py (GPUDirect ON) ==="
python3 /opt/GVirtuS/examples/rapids_matrix/rapids_matrix.py
echo "=== PYTHON_EXIT=$? ==="
