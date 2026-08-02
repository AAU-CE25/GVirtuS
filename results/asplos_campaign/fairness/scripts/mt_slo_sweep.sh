#!/bin/bash
# mt_slo_sweep.sh SYS N REP "l1 l2 ..." — barrido de carga con SLO, multi-tenant llama 7B.
#
# POR QUE EXISTE. El barrido anterior tenia dos puntos y nada en medio:
#   lambda=1.0 total  -> ESTABLE, TTFT p50 186 ms, 100% cumple SLO de 5 s, pero la carga
#                        ofrecida es plana y el goodput queda clavado en 128.0 t/s para todo
#                        N: la comparacion no discrimina.
#   lambda=N          -> el goodput separa (1.18x/1.29x/1.37x) pero los 6 puntos estan
#                        marcados UNSTABLE, TTFT p50 son 14-23 SEGUNDOS, y el 77% de las
#                        peticiones muere en el deadline de 25 s. `pct_slo_1s` = 0%.
# No hay ningun punto medido donde GVirtuS gane al nativo Y sirva. Este barrido cubre el
# hueco y permite el titular correcto: GOODPUT QUE CUMPLE SLO frente a carga ofrecida.
#
# La tasa sostenible se estima en goodput/NPRED ~ 300/128 ~ 2.4 req/s, asi que la
# resolucion se concentra entre 1.5 y 3.0.
#
# LOS PODS SE MANTIENEN VIVOS entre puntos de lambda dentro de una repeticion: arrancar 8
# llama-server de 7B por punto multiplicaria el reloj por cinco. El precio es que no se
# reinicia el backend entre puntos; se compensa (a) aleatorizando el orden de lambda dentro
# de la repeticion y (b) haciendo cada REPETICION con derribo completo. Queda declarado.
set -u
SYS="${1:?sys: ucx|bm|bmmps}"; N="${2:?pods}"; REP="${3:?rep}"; LAMS="${4:?lista de lambda}"
WINDOW="${WINDOW:-30}"
LP=/opt/GVirtuS/examples/llama
MODEL=$LP/models/mistral-7b-q4.gguf
OUT=/home/student.aau.dk/ll33pq/GVirtuS/results/asplos_campaign/llama_slo_sweep
mkdir -p "$OUT"
cd /home/student.aau.dk/ll33pq/GVirtuS || exit 99

case "$SYS" in
  ucx)   IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6; BASE=8200; PFX=pod;    TAG=ucx ;;
  bm)    IMG=aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04; BASE=8300; PFX=bmpod;  TAG=bm ;;
  bmmps) IMG=aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04; BASE=8400; PFX=mpspod; TAG=bmmps ;;
  *) echo "SYS desconocido"; exit 2 ;;
esac

limpia() { for i in $(seq 1 "$N"); do docker rm -f "${PFX}$i" >/dev/null 2>&1; done; }
trap limpia EXIT
limpia

if [ "$SYS" = bmmps ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  mkdir -p /tmp/mps_pipe /tmp/mps_log
  pgrep -f "nvidia-cuda-mps-contro[l]" >/dev/null || nvidia-cuda-mps-control -d
  sleep 2
fi
U=$(id -u); G=$(id -g)

for i in $(seq 1 "$N"); do
  P=$((BASE+i)); NAME="${PFX}$i"
  case "$SYS" in
    ucx)
      docker run -d --name "$NAME" --network host --cap-add IPC_LOCK --ulimit memlock=-1 \
        --entrypoint bash -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
        -e GVIRTUS_UCX_DATAPATH=am -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
        -e GVIRTUS_RMA_SLOTS=8 -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
        -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_LOG_LEVEL=error \
        -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
        -e UCX_RCACHE_ENABLE=n -e UCX_MEMTYPE_CACHE=n --device /dev/infiniband \
        -e GVIRTUS_ASYNC_DISPATCH=1 -v "$PWD":/opt/GVirtuS:ro "$IMG" \
        -c "export GVIRTUS_HOME=/opt/GVirtuS GVIRTUS_LOGLEVEL=40000; export LD_LIBRARY_PATH=/opt/GVirtuS/lib:/opt/GVirtuS/lib/frontend:$LP/llama.cpp/build_cuda/bin:\$LD_LIBRARY_PATH; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null ;;
    bm)
      docker run -d --name "$NAME" --network host --gpus all --entrypoint bash \
        -v "$PWD":/opt/GVirtuS:ro "$IMG" \
        -c "export LD_LIBRARY_PATH=$LP/llama.cpp/build_cuda/bin:/usr/local/cuda/lib64; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null ;;
    bmmps)
      docker run -d --name "$NAME" --network host --gpus all --ipc=host --user "$U:$G" \
        -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe \
        -v /tmp/mps_pipe:/tmp/mps_pipe -v /tmp/mps_log:/tmp/mps_log \
        --entrypoint bash -v "$PWD":/opt/GVirtuS:ro "$IMG" \
        -c "export LD_LIBRARY_PATH=$LP/llama.cpp/build_cuda/bin:/usr/local/cuda/lib64; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null ;;
  esac
done

SRVS=""; for i in $(seq 1 "$N"); do SRVS="${SRVS:+$SRVS,}http://127.0.0.1:$((BASE+i))"; done
up=0
for t in $(seq 1 180); do
  sleep 4; ok=0
  for i in $(seq 1 "$N"); do
    curl -s -m2 http://127.0.0.1:$((BASE+i))/health 2>/dev/null | grep -q ok && ok=$((ok+1))
  done
  [ "$ok" = "$N" ] && { up=1; break; }
done
echo "[$TAG rep$REP] pods_up=$up/$N"
[ "$up" = 1 ] || { echo "ABORT: no arrancaron todos los pods"; exit 1; }

if [ "$SYS" = bmmps ]; then
  echo "[$TAG] servidores MPS nuevos: $(grep -c 'NEW SERVER' /tmp/mps_log/control.log 2>/dev/null || echo 0)"
fi

# Orden ALEATORIO de lambda dentro de la repeticion, con semilla = REP para reproducirlo.
ORD=$(echo $LAMS | tr ' ' '\n' | shuf --random-source=<(yes "$REP") | tr '\n' ' ')
echo "[$TAG rep$REP] orden de lambda: $ORD"

for L in $ORD; do
  LBL="slo_${TAG}_n${N}_l${L}_r${REP}"
  OUTDIR=$OUT SERVERS=$SRVS MODE=open RATE=$L UNIQUE=1 SEED=$REP LABEL=$LBL \
    CAMPAIGN_ID=asplos_slo_sweep WINDOW=$WINDOW WARM=10 NPRED=128 REQ_TIMEOUT=25 \
    GVIRTUS_GPUDIRECT=$([ "$SYS" = ucx ] && echo 1 || echo "") \
    python3 $HOME/bench.py 2>&1 | tail -1
  sleep 8   # dejar drenar antes del siguiente punto
done
echo "SWEEP_${TAG}_N${N}_REP${REP}_DONE"
