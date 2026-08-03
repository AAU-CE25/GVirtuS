#!/bin/bash
# Regresion de llama tras los cambios de A1/I13, I10 e I12. tg16, 3 repeticiones.
# La comparacion es contra 531,3 t/s (2026-08-03 tarde) y 528,6 +- 64,8 (referencia previa).
set -u
cd /home/student.aau.dk/ll33pq/GVirtuS
LP=/opt/GVirtuS/examples/llama
IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6
docker rm -f gv_lb >/dev/null 2>&1
timeout 1200 docker run --rm --name gv_lb --network host --cap-add IPC_LOCK --ulimit memlock=-1 \
  --entrypoint bash --device /dev/infiniband \
  -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json -e GVIRTUS_UCX_DATAPATH=am \
  -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 -e GVIRTUS_RMA_SLOTS=8 \
  -e GVIRTUS_RMA_SLOT_CAP_MB=128 -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_LOG_LEVEL=error \
  -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 -e UCX_RCACHE_ENABLE=n \
  -e UCX_MEMTYPE_CACHE=n -e LP="$LP" \
  -v "$PWD":/opt/GVirtuS:ro "$IMG" \
  -c 'export GVIRTUS_HOME=/opt/GVirtuS GVIRTUS_LOGLEVEL=40000
      export LD_LIBRARY_PATH=/opt/GVirtuS/lib:/opt/GVirtuS/lib/frontend:$LP/llama.cpp/build_cuda/bin:$LD_LIBRARY_PATH
      $LP/llama.cpp/build_cuda/bin/llama-bench -m $LP/models/tinyllama-1.1b-q4.gguf -p 0 -n 16 -r 3 --mmap 0'
