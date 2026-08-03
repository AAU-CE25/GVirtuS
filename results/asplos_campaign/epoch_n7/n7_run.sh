#!/bin/bash
# n7_run.sh <etiqueta> <ablate> <nopark 0|1> [iters]
# Corre epochharm contra el backend con el fault de reanuncio.  El log del cliente queda en
# /tmp/gusto_n7/<etiqueta>.log
set -u
ET="$1"; ABL="${2:-full}"; NOPARK="${3:-0}"; IT="${4:-40}"
OUT=/tmp/gusto_n7; mkdir -p "$OUT"
cd /home/student.aau.dk/ll33pq/GVirtuS || exit 1
IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6
RT=/opt/GVirtuS/examples/rmatest
docker rm -f gv_n7 >/dev/null 2>&1
docker run --rm --name gv_n7 --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash -e LD_PRELOAD= \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e UCX_MEMTYPE_CACHE=n -e GVIRTUS_UCX_DATAPATH=am \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
  -e UCX_LOG_LEVEL=error -e UCX_RCACHE_ENABLE=n -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e GVIRTUS_HOME=/opt/GVirtuS -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_LOGLEVEL=30000 \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/lib:/usr/local/cuda/lib64 \
  -e GVIRTUS_RMA_MIN_BYTES=8192 -e GVIRTUS_RMA_SCALAR_FLOOR=8192 -e GUSTO_RMA_METRICS_MS=0 \
  -e GVIRTUS_ASYNC_DISPATCH=1 \
  -e GVS_ABLATE="$ABL" -e GVS_FAULT_NOPARK="$NOPARK" \
  -v "$PWD":/opt/GVirtuS:ro "$IMG" \
  -c "ulimit -c 0; LD_PRELOAD=libcuda.so.1:libcudart.so.12 timeout 240 $RT/epochharm $IT" \
  > "$OUT/$ET.log" 2>&1
echo "exit=$? -> $OUT/$ET.log"
