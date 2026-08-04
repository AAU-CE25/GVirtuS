#!/bin/bash
# acumula_abort.sh -- acumula corridas de la receta del abort y, si salta, captura la
# atribucion asincrona del backend, que es para lo que se instrumento.
#
# Cada bloque: N iteraciones, y tras CADA una se vuelca lo que el backend haya emitido con
# [GVS ASYNC] desde el arranque del bloque. La instrumentacion vive en el BACKEND, asi que su
# salida NO esta en el log del servidor de llama: hay que ir a buscarla a dpu-01.
set -u
N="${1:-30}"; VENT="${2:-45}"
D=/home/student.aau.dk/ll33pq/abort_acumula; mkdir -p "$D"
SMA=/usr/bin/secure-machine-access
: > "$D/resumen.txt"

for i in $(seq 1 "$N"); do
  T0=$(date -u +%s)
  bash /home/student.aau.dk/ll33pq/llama_abort_repro.sh 1 "$VENT" > "$D/iter_$i.out" 2>&1
  LIN=$(grep -E "^iter=1" "$D/iter_$i.out" | tail -1)
  # Evidencia de que el brazo NO es inerte, en CADA iteracion.
  EV=$(grep -oE "admit_rma=[0-9]+ admit_am=[0-9]+" /tmp/llama_abort/iter1_srv.log 2>/dev/null | tail -1)
  POL=$(grep -o "four-quadrant placement.*" /tmp/llama_abort/iter1_srv.log 2>/dev/null | head -1)
  cp /tmp/llama_abort/iter1_srv.log "$D/srv_$i.log" 2>/dev/null

  # La atribucion asincrona vive en el backend.
  $SMA es-dpu-01 "docker logs gvirtus-ll33pq 2>&1 | grep -A 40 'GVS ASYNC' | tail -60" \
      > "$D/async_$i.log" 2>/dev/null
  NA=$(grep -c "GVS ASYNC" "$D/async_$i.log" 2>/dev/null || echo 0)

  ABORT=no
  grep -qaE "ggml_abort|CUDA error" "$D/srv_$i.log" 2>/dev/null && ABORT=SI
  printf "iter=%-3s %-42s %-34s abort=%s async_lines=%s dur=%ss\n" \
    "$i" "$LIN" "$EV" "$ABORT" "$NA" "$(( $(date -u +%s) - T0 ))" | tee -a "$D/resumen.txt"

  # PUERTA DE VITALIDAD. Si las dos primeras iteraciones no arrancan, la campana esta
  # midiendo aire y hay que pararla, no dejarla una hora. Aprendido esta noche: un simbolo
  # indefinido tumbo el backend, llama no conectaba, y el arnes seguia leyendo el log ANTERIOR
  # -- daba admit_rma rancio y parecia un brazo vivo.
  if [ "$i" -le 2 ] && echo "$LIN" | grep -q ARRANQUE_FALLIDO; then
    fallos_inicio=$((${fallos_inicio:-0}+1))
    if [ "${fallos_inicio}" -ge 2 ]; then
      echo "*** ABORTADA: 2 arranques fallidos seguidos. El backend no sirve; revisar" \
           "'docker logs gvirtus-ll33pq | grep -iE \"undefined|error:\"'" | tee -a "$D/resumen.txt"
      exit 2
    fi
  fi
  [ "$ABORT" = SI ] && { echo "  >>> ABORT en iter $i; atribucion en $D/async_$i.log"; \
                         grep -aE "ggml_abort|CUDA error|cudaStreamSync" "$D/srv_$i.log" | head -3 | tee -a "$D/resumen.txt"; }
done
echo "=== FIN: $(grep -c "abort=SI" "$D/resumen.txt") abortos de $N ===" | tee -a "$D/resumen.txt"
