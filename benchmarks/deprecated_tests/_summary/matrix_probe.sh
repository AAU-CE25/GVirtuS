#!/bin/bash
# matrix regression probe. $1 = async (0|1). One size N=16000, warmup + 3 reps.
# Emits: async,rep,sgemm_ms,host_ms  and the backend path taken (from stderr grep).
set -u
ASYNC=$1
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties_ucx.json
export GVIRTUS_LOGLEVEL=40000
export GVIRTUS_ASYNC_DISPATCH=$ASYNC
export GVIRTUS_UCX_DATAPATH=am
export UCX_TLS=rc_mlx5,ud_mlx5,tcp,self
export UCX_NET_DEVICES=mlx5_1:1,ens1f1np1
export UCX_IB_GID_INDEX=3
export UCX_SOCKADDR_TLS_PRIORITY=tcp
export UCX_LOG_LEVEL=error
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib:/usr/local/gvirtus/lib/frontend
cd /gvirtus/examples/simple_matrix
echo "async,rep,sgemm_ms,host_ms"
timeout 300 ./simple_matrix 16000 5 1 >/dev/null 2>&1   # warmup
for r in 1 2 3; do
  out=$(timeout 300 ./simple_matrix 16000 5 1 2>/dev/null)
  line=$(echo "$out" | grep -a "^CSV,")
  s=$(echo "$line" | cut -d, -f4); h=$(echo "$line" | cut -d, -f5)
  echo "$ASYNC,$r,$s,$h"
done
