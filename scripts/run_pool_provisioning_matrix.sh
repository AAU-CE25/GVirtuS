#!/bin/bash
# Matriz de provisioning del pool RMA.
#
#   slots     4, 8, 12, 16, 24, 32, 48, 64
#   clientes  1, 2, 4, 8
#   politica  scalar, quadrant
#   reps      3   (>=3 por punto, que es el minimo que el encargo admite en cargas caras)
#
# 8*4*2*3 = 192 corridas. El backend solo se reinicia cuando cambia el numero de slots, que es
# lo unico que vive en su lado: politica y concurrencia son del frontend. Asi son 8 reinicios
# en vez de 192, y la comparacion entre politicas es contra EL MISMO pool.
#
# Cada corrida deja su fila en raw/matrix.csv y su metadato en raw/meta/<etiqueta>.json, con
# commit, hardware, transporte, entorno y marca de tiempo, para que un punto suelto sea
# reproducible sin leer este guion.
set -u
SMA=${SMA:-/usr/bin/secure-machine-access}
ROOT=${ROOT:-/home/ethanadams/.claude/jobs/20f7d7dd/tmp/results/pool_provisioning}
RAW="$ROOT/raw"; META="$RAW/meta"
mkdir -p "$RAW" "$META" "$ROOT/processed" "$ROOT/figures"
CSV="$RAW/matrix.csv"
LOG="$ROOT/matrix.log"
IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6
RT=/opt/GVirtuS/examples/rmatest
VENTANA=${VENTANA:-30}
CALENTAMIENTO=8
REPS=${REPS:-3}
TAM=4194304

say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -f "$CSV" ]; then
  echo "etiqueta,politica,slots,clientes,rep,completadas_total,completadas_por_cliente,h2d_fail,d2h_fail,reservas,admit_rma,admit_am,rma_pct,peak_inflight,avg_inflight,peak_waiters,waited,wait_us_avg,decline_capacity,decline_timeout,ack_applied,ack_on_free,gen_mismatch,jain,min_max_ratio,backend,ts_utc" > "$CSV"
fi

# --- contexto del banco, una vez -----------------------------------------------------------
COMMIT=$($SMA es-dpu-02 'cd ~/GVirtuS && git rev-parse HEAD' | tr -d '\r')
GPU=$($SMA es-dpu-01 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' | tr -d '\r')
UCXV=$($SMA es-dpu-02 'ucx_info -v 2>&1 | head -1' | tr -d '\r')
say "commit=$COMMIT | gpu=$GPU | $UCXV"

reset_be() {
  $SMA es-dpu-01 "GVIRTUS_RMA_MIN_BYTES=8192 GVS_SLOTS=$1 GVS_SLOT_MIN_MB=4 GVS_SLOT_CAP_MB=32 GVS_PREALLOC=1 bash ~/reset_backend_pool.sh" >>"$LOG" 2>&1
  local l; l=$($SMA es-dpu-01 'ss -ltn | grep -c 32222' | tr -d '\r')
  [ "${l:-0}" = "1" ]
}

punto() {   # $1 slots  $2 politica  $3 clientes  $4 rep
  local S="$1" POL="$2" N="$3" R="$4"
  local ET="s${S}_${POL}_n${N}_r${R}"
  grep -q "^$ET," "$CSV" 2>/dev/null && { say "  $ET ya medido, salto"; return; }

  for i in $(seq 1 "$N"); do
    $SMA es-dpu-02 "docker rm -f mcli$i >/dev/null 2>&1; cd ~/GVirtuS && docker run -d --name mcli$i --network host --device /dev/infiniband \
      --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash -e LD_PRELOAD= \
      -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e UCX_MEMTYPE_CACHE=n -e GVIRTUS_UCX_DATAPATH=am \
      -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e UCX_IB_GID_INDEX=3 \
      -e UCX_LOG_LEVEL=error -e UCX_RCACHE_ENABLE=n -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
      -e GVIRTUS_HOME=/opt/GVirtuS -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json -e GVIRTUS_LOGLEVEL=30000 \
      -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/lib:/usr/local/cuda/lib64 \
      -e GVIRTUS_RMA_POLICY=$POL -e GVIRTUS_RMA_MIN_BYTES=8192 -e GVIRTUS_RMA_SCALAR_FLOOR=4194304 -e GUSTO_RMA_METRICS_MS=3000 \
      -v \"\$PWD\":/opt/GVirtuS:ro $IMG \
      -c 'ulimit -c 0; LD_PRELOAD=libcuda.so.1:libcudart.so.12 $RT/rma_checksum $TAM 1000000'" >/dev/null 2>&1
  done
  sleep $CALENTAMIENTO
  sleep $VENTANA

  local tot=0 lista="" hf=0 df=0 rv=0 ar=0 aa=0 pi=0 pw=0 wd=0 dc=0 dt=0 aap=0 aof=0 gm=0
  local ai="0" wus="0" pct="0" mn=999999999 mx=0 suma2=0
  for i in $(seq 1 "$N"); do
    local L="$RAW/${ET}_c${i}.log"
    $SMA es-dpu-02 "docker logs mcli$i 2>&1" > "$L" 2>&1
    local c; c=$(grep -ac '^[0-9]*,' "$L"); c=${c:-0}
    lista="$lista $c"; tot=$((tot+c)); suma2=$((suma2+c*c))
    [ "$c" -lt "$mn" ] && mn=$c; [ "$c" -gt "$mx" ] && mx=$c
    local x; x=$(grep -ac ',FAIL,' "$L"); hf=$((hf+${x:-0}))
    x=$(grep -ac ',pass,FAIL' "$L"); df=$((df+${x:-0}))
    local m; m=$(grep -a GUSTO_METRIC "$L" | tail -1)
    g(){ local v; v=$(echo "${m:-}" | grep -o "$1=[0-9.]*" | cut -d= -f2 | head -1); echo "${v:-0}"; }
    rv=$((rv+$(g reservations))); ar=$((ar+$(g admit_rma))); aa=$((aa+$(g admit_am)))
    wd=$((wd+$(g waited))); dc=$((dc+$(g decline_capacity))); dt=$((dt+$(g decline_timeout)))
    aap=$((aap+$(g ack_applied))); aof=$((aof+$(g ack_on_free))); gm=$((gm+$(g ack_gen_mismatch)))
    local p; p=$(g peak_inflight); [ "${p%%.*}" -gt "$pi" ] 2>/dev/null && pi=${p%%.*}
    p=$(g peak_waiters); [ "${p%%.*}" -gt "$pw" ] 2>/dev/null && pw=${p%%.*}
    ai=$(g avg_inflight); wus=$(g wait_us_avg); pct=$(g rma_pct)
    $SMA es-dpu-02 "docker rm -f mcli$i >/dev/null 2>&1"
  done
  local jain ratio be ts
  jain=$(python3 -c "s=$tot;s2=$suma2;n=$N;print('%.4f'%((s*s)/(n*s2)) if s2 else 0)" 2>/dev/null)
  ratio=$(python3 -c "print('%.3f'%($mn/$mx) if $mx else 0)" 2>/dev/null)
  be=$($SMA es-dpu-01 'ss -ltn | grep -q 25.25.25.2:32222 && echo VIVO || echo MUERTO' | tr -d '\r')
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  say "  $ET -> tot=$tot ($lista) rma=${pct}% pico=$pi esperas=$wd jain=$jain be=$be"
  echo "$ET,$POL,$S,$N,$R,$tot,\"$lista\",$hf,$df,$rv,$ar,$aa,$pct,$pi,$ai,$pw,$wd,$wus,$dc,$dt,$aap,$aof,$gm,$jain,$ratio,$be,$ts" >> "$CSV"

  cat > "$META/$ET.json" <<JSON
{"etiqueta":"$ET","commit":"$COMMIT","hostname_frontend":"es-dpu-02","hostname_backend":"es-dpu-01",
 "gpu":"$GPU","ucx":"$UCXV","nic":"mlx5_1:1/ens1f1np1","transporte":"rc_mlx5,ud_mlx5,tcp,self",
 "politica":"$POL","slots_backend":$S,"slot_cap_mb":32,"slot_min_mb":4,"prealloc":1,
 "clientes":$N,"rep":$R,"tam_transferencia":$TAM,"ventana_s":$VENTANA,"calentamiento_s":$CALENTAMIENTO,
 "carga":"rma_checksum","semilla":"determinista: el payload deriva del indice de transferencia",
 "comando":"LD_PRELOAD=libcuda.so.1:libcudart.so.12 $RT/rma_checksum $TAM 1000000",
 "env":{"GVIRTUS_RMA_POLICY":"$POL","GVIRTUS_RMA_MIN_BYTES":"8192","GVIRTUS_RMA_SCALAR_FLOOR":"4194304",
        "GVIRTUS_GPUDIRECT":"1","GVIRTUS_RMA_ZEROCOPY":"1","UCX_RCACHE_ENABLE":"n","UCX_MEMTYPE_CACHE":"n"},
 "ts_utc":"$ts"}
JSON
}

for S in 4 8 12 16 24 32 48 64; do
  say "=== pool de $S slots ==="
  reset_be "$S" || { say "  *** backend no arranca con $S slots, salto el bloque"; continue; }
  for POL in scalar quadrant; do
    for N in 1 2 4 8; do
      for R in $(seq 1 $REPS); do
        punto "$S" "$POL" "$N" "$R"
      done
    done
  done
done
say "=== MATRIZ COMPLETA ==="
wc -l "$CSV" | tee -a "$LOG"
echo FIN_MATRIZ
