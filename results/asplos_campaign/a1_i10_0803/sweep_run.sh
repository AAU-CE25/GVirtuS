#!/bin/bash
# sweep_run.sh -- como i10_run.sh pero con un volumen ESCRIBIBLE para el CSV.
# El de i10 monta el arbol :ro y sweep_bench aborta con "no puedo escribir ..." antes de
# mover un solo byte: sale con admit_rma=0 y parece una corrida, no lo es.
set -u
ET="$1"; BIN="$2"; shift 2
EXTRA=""; for kv in "$@"; do EXTRA="$EXTRA -e $kv"; done
mkdir -p /home/student.aau.dk/ll33pq/cross_out
cd /home/student.aau.dk/ll33pq/GVirtuS
docker rm -f gv_sw >/dev/null 2>&1
docker run --rm --name gv_sw --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash -e LD_PRELOAD= \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e UCX_MEMTYPE_CACHE=n -e GVIRTUS_UCX_DATAPATH=am \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
  -e UCX_LOG_LEVEL=error -e UCX_RCACHE_ENABLE=n -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
  -e GVIRTUS_HOME=/opt/GVirtuS -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
  -e GVIRTUS_LOGLEVEL=30000 $EXTRA \
  -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/lib:/usr/local/cuda/lib64 \
  -v "$PWD":/opt/GVirtuS:ro \
  -v /home/student.aau.dk/ll33pq/cross_out:/out \
  ll33pq/cudf_gvirtus_dyncudf:cuda12.6 \
  -c "ulimit -c 0; LD_PRELOAD=libcuda.so.1:libcudart.so.12 timeout 900 $BIN" \
  > /tmp/sw_$ET.log 2>&1
echo "exit=$?"
