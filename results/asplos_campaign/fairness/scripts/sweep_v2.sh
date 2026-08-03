#!/bin/bash
# sweep_v2.sh SYS N REP — barrido de capacidad con VENTANA ESCALADA.
#
# POR QUE V2. La v1 usaba ventana fija de 30 s a toda carga. A lambda=0,25 eso son 7,5
# peticiones ofrecidas, y el goodput resultante variaba de 29,9 a 59,7 t/s entre repeticiones
# -- casi x2. No era el sistema: era ruido de conteo de Poisson. Con n=1 eso produjo un
# +17,4 % que NO sobrevivio a n=3.
#
# Aqui la ventana se escala para que cada punto vea >=40 peticiones ofrecidas:
#     WINDOW = max(30, 40/lambda)
# El goodput sigue siendo una tasa, asi que alargar la ventana solo reduce su varianza; no
# cambia lo que se mide.
#
# Se barre solo la REGION DECISIVA (0,5 a 1,5): por debajo todo cumple el SLO y por encima
# nada lo cumple, asi que los extremos no aportan.
set -u
SYS="${1:?bm|bmmps|ucx}"; N="${2:?pods}"; REP="${3:?rep}"
LAMS="${LAMS:-0.5 0.75 1.0 1.5}"
LP=/opt/GVirtuS/examples/llama
MODEL=$LP/models/mistral-7b-q4.gguf
OUT=/home/student.aau.dk/ll33pq/GVirtuS/results/asplos_campaign/llama_slo_sweep_v2
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

MPSENV=(); MPSMNT=()
if [ "$SYS" = bmmps ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  mkdir -p /tmp/mps_pipe /tmp/mps_log
  pgrep -f "nvidia-cuda-mps-contro[l]" >/dev/null || nvidia-cuda-mps-control -d
  sleep 3
  MPSENV=(--ipc=host --user "$(id -u):$(id -g)" -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe)
  MPSMNT=(-v /tmp/mps_pipe:/tmp/mps_pipe -v /tmp/mps_log:/tmp/mps_log)
fi

for i in $(seq 1 "$N"); do
  P=$((BASE+i)); NAME="${PFX}$i"
  if [ "$SYS" = ucx ]; then
    docker run -d --name "$NAME" --network host --cap-add IPC_LOCK --ulimit memlock=-1 \
      --entrypoint bash -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
      -e GVIRTUS_UCX_DATAPATH=am -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
      -e GVIRTUS_RMA_SLOTS=8 -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self \
      -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_LOG_LEVEL=error \
      -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
      -e UCX_RCACHE_ENABLE=n -e UCX_MEMTYPE_CACHE=n --device /dev/infiniband \
      -e GVIRTUS_ASYNC_DISPATCH=1 -v "$PWD":/opt/GVirtuS:ro "$IMG" \
      -c "export GVIRTUS_HOME=/opt/GVirtuS GVIRTUS_LOGLEVEL=40000; export LD_LIBRARY_PATH=/opt/GVirtuS/lib:/opt/GVirtuS/lib/frontend:$LP/llama.cpp/build_cuda/bin:\$LD_LIBRARY_PATH; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null
  else
    docker run -d --name "$NAME" --network host --gpus all "${MPSENV[@]}" "${MPSMNT[@]}" \
      --entrypoint bash -v "$PWD":/opt/GVirtuS:ro "$IMG" \
      -c "export LD_LIBRARY_PATH=$LP/llama.cpp/build_cuda/bin:/usr/local/cuda/lib64; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null
  fi
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
echo "[$TAG N=$N rep$REP] pods_up=$ok/$N"
[ "$up" = 1 ] || { echo "ABORT"; exit 1; }

ORD=$(echo $LAMS | tr ' ' '\n' | shuf --random-source=<(yes "$REP") | tr '\n' ' ')
for L in $ORD; do
  W=$(python3 -c "print(int(max(30, 40/$L)))")
  LBL="v2_${TAG}_n${N}_l${L}_r${REP}"
  OUTDIR=$OUT SERVERS=$SRVS MODE=open RATE=$L UNIQUE=1 SEED=$REP LABEL=$LBL \
    CAMPAIGN_ID=slo_v2 WINDOW=$W WARM=10 NPRED=128 REQ_TIMEOUT=25 \
    python3 $HOME/bench.py 2>&1 | tail -1
done
echo "V2_${TAG}_N${N}_REP${REP}_DONE"
