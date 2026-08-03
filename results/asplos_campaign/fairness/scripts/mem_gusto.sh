#!/bin/bash
# mem_gusto.sh N — huella de memoria por tenant del brazo remoto.
#
# El brazo de Gusto de la tabla de memoria estaba HEREDADO de la campana previa mientras que
# nativo y nativo+MPS se midieron frescos, asi que la comparacion mezclaba sesiones. Esto lo
# mide en la misma sesion y con el mismo metodo.
#
# La diferencia con el arnes local: la memoria que importa esta en dpu-01 (el backend), no en
# dpu-02. El muestreo lo hace quien invoca; aqui solo se levantan los pods y se ejercitan.
set -u
N="${1:?pods}"
LP=/opt/GVirtuS/examples/llama
MODEL=$LP/models/mistral-7b-q4.gguf
IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6
cd /home/student.aau.dk/ll33pq/GVirtuS || exit 99
limpia() { for i in $(seq 1 "$N"); do docker rm -f "gpod$i" >/dev/null 2>&1; done; }
limpia

for i in $(seq 1 "$N"); do
  P=$((8600+i))
  docker run -d --name "gpod$i" --network host --cap-add IPC_LOCK --ulimit memlock=-1 \
    --entrypoint bash -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
    -e GVIRTUS_UCX_DATAPATH=am -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
    -e GVIRTUS_RMA_SLOTS=8 -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
    -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_LOG_LEVEL=error \
    -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
    -e UCX_RCACHE_ENABLE=n -e UCX_MEMTYPE_CACHE=n --device /dev/infiniband \
    -v "$PWD":/opt/GVirtuS:ro "$IMG" \
    -c "export GVIRTUS_HOME=/opt/GVirtuS GVIRTUS_LOGLEVEL=40000; export LD_LIBRARY_PATH=/opt/GVirtuS/lib:/opt/GVirtuS/lib/frontend:$LP/llama.cpp/build_cuda/bin:\$LD_LIBRARY_PATH; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null
done

up=0
for t in $(seq 1 200); do
  sleep 4; ok=0
  for i in $(seq 1 "$N"); do
    curl -s -m2 http://127.0.0.1:$((8600+i))/health 2>/dev/null | grep -q ok && ok=$((ok+1))
  done
  [ "$ok" = "$N" ] && { up=1; break; }
done
echo "gusto_pods_up=$ok/$N"
[ "$up" = 1 ] || { echo ABORT; limpia; exit 1; }

# Ejercitar cada pod: un llama-server sano puede no haber materializado todo hasta la primera
# peticion, igual que en el arnes local.
for i in $(seq 1 "$N"); do
  curl -s -m 40 -X POST http://127.0.0.1:$((8600+i))/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"hola","n_predict":8,"stream":false}' >/dev/null 2>&1 &
done
wait
echo "LISTO_PARA_MUESTREAR"
sleep 30          # ventana en la que quien invoca muestrea la GPU de dpu-01
limpia
echo "DERRIBADO"
