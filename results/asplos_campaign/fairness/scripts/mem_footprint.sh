#!/bin/bash
# mem_footprint.sh SYS N — huella de memoria de GPU por tenant.
#
#   SYS = bm | bmmps      (ambos en la L40S local de dpu-02, misma GPU, misma sesion)
#
# Por que hace falta: el paquete tiene la huella de contextos nativos independientes y la de
# Gusto (~463 MiB/tenant ahorrados a N=8), pero NO la de Native+MPS. Y es la columna que
# decide el argumento: MPS consolida contextos igual que hace el backend por construccion,
# asi que sin ella el ahorro se atribuye al remoting cuando podria deberse solo a la
# consolidacion de contexto.
#
# Definicion, la misma que usa mt_pods.sh: por tenant = (pico - linea base) / N. Es un valor
# DERIVADO, no una medida directa: nvidia-smi no puede atribuir memoria por tenant cuando los
# tenants comparten proceso (Gusto) o contexto (MPS). Se declara como tal.
set -u
SYS="${1:?bm|bmmps}"; N="${2:?pods}"
LP=/opt/GVirtuS/examples/llama
MODEL=$LP/models/mistral-7b-q4.gguf
IMG=aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04
OUT=/home/student.aau.dk/ll33pq/GVirtuS/results/asplos_campaign/memoria
mkdir -p "$OUT"
cd /home/student.aau.dk/ll33pq/GVirtuS || exit 99
CSV="$OUT/mem_footprint.csv"
[ -f "$CSV" ] || echo "system,N,baseline_mib,peak_mib,after_mib,per_tenant_mib,pods_up,samples,mps_servers" > "$CSV"

case "$SYS" in
  bm)    BASE=8300; PFX=bmpod;  MPS=off ;;
  bmmps) BASE=8400; PFX=mpspod; MPS=on  ;;
  *) echo "SYS desconocido"; exit 2 ;;
esac
limpia() { for i in $(seq 1 "$N"); do docker rm -f "${PFX}$i" >/dev/null 2>&1; done; }
trap limpia EXIT
limpia; sleep 3

MPSENV=(); MPSMNT=(); NSRV=0
if [ "$MPS" = on ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  mkdir -p /tmp/mps_pipe /tmp/mps_log
  pgrep -f "nvidia-cuda-mps-contro[l]" >/dev/null || nvidia-cuda-mps-control -d
  sleep 3
  MPSENV=(-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe --ipc=host --user "$(id -u):$(id -g)")
  MPSMNT=(-v /tmp/mps_pipe:/tmp/mps_pipe -v /tmp/mps_log:/tmp/mps_log)
else
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
  pkill -f "nvidia-cuda-mps-contro[l]" >/dev/null 2>&1
  sleep 2
fi

BL=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[$SYS N=$N] linea base = $BL MiB"

for i in $(seq 1 "$N"); do
  P=$((BASE+i))
  docker run -d --name "${PFX}$i" --network host --gpus all "${MPSENV[@]}" "${MPSMNT[@]}" \
    --entrypoint bash -v "$PWD":/opt/GVirtuS:ro "$IMG" \
    -c "export LD_LIBRARY_PATH=$LP/llama.cpp/build_cuda/bin:/usr/local/cuda/lib64; $LP/llama.cpp/build_cuda/bin/llama-server -m $MODEL --host 127.0.0.1 --port $P -ngl 99 --no-mmap -c 2048 --parallel 1 --metrics" >/dev/null
done

up=0
for t in $(seq 1 180); do
  sleep 4; ok=0
  for i in $(seq 1 "$N"); do
    curl -s -m2 http://127.0.0.1:$((BASE+i))/health 2>/dev/null | grep -q ok && ok=$((ok+1))
  done
  [ "$ok" = "$N" ] && { up=1; break; }
done
echo "[$SYS N=$N] pods_up=$up/$N"

# El pico se toma DESPUES de ejercitar cada pod: un llama-server sano puede no haber
# materializado todo su contexto hasta la primera peticion.
for i in $(seq 1 "$N"); do
  curl -s -m 30 -X POST http://127.0.0.1:$((BASE+i))/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"hola","n_predict":8,"stream":false}' >/dev/null 2>&1 &
done
wait

PK=0; NS=0
for _ in $(seq 1 25); do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$M" -gt "$PK" ] && PK=$M
  NS=$((NS+1)); sleep 0.6
done
[ "$MPS" = on ] && NSRV=$(grep -c "NEW SERVER" /tmp/mps_log/control.log 2>/dev/null || echo 0)

limpia; sleep 6
AF=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
PT=$(awk -v p="$PK" -v b="$BL" -v n="$N" 'BEGIN{printf "%.1f",(p-b)/n}')
echo "$SYS,$N,$BL,$PK,$AF,$PT,$up/$N,$NS,$NSRV" >> "$CSV"
echo "[$SYS N=$N] pico=$PK  tras derribo=$AF  **por tenant=$PT MiB**  servidores MPS=$NSRV"
