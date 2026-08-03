#!/bin/bash
# a1_matrix.sh -- coste de las politicas A1 con REPETICIONES INDEPENDIENTES.
#
# Unidad experimental = una corrida de PROCESO, no una transferencia. La tabla anterior
# tomaba 14 transferencias de UNA corrida y las trataba como 14 replicas; no lo son
# (comparten proceso, conexion, pool y estado de cache de registro). Aqui:
#   * mediana de las transferencias 3-16 DENTRO de cada corrida  -> un valor por corrida
#   * mediana + bootstrap ENTRE corridas                          -> la celda
#   * orden de las celdas ALEATORIZADO, para que una deriva termica o de frecuencia no se
#     alinee con una politica
set -u
REPS="${REPS:-6}"
OUT="${OUT:-/tmp/a1_matrix.csv}"
SEED="${SEED:-20260803}"
POLS="${POLS:-assume fence flush strict}"

echo "cell,rep,transfer,h2d_ms,h2d_GBps,device_ck_ok,host_ck_ok" > "$OUT"

# Plan aleatorizado: todas las (celda,rep) barajadas con semilla fija y reproducible.
PLAN=$(for p in $POLS; do for r in $(seq 1 "$REPS"); do echo "$p $r"; done; done \
       | awk -v s="$SEED" 'BEGIN{srand(s)} {print rand()"\t"$0}' | sort -k1,1 | cut -f2-)

echo "$PLAN" > /tmp/a1_matrix.plan
N=$(echo "$PLAN" | wc -l)
i=0
echo "$PLAN" | while read -r POL REP; do
  i=$((i+1))
  printf "[%2d/%2d] cell=%-7s rep=%d ... " "$i" "$N" "$POL" "$REP"
  bash /home/student.aau.dk/ll33pq/i10_run.sh "a1_${POL}_${REP}" \
      /opt/GVirtuS/examples/rmatest/rma_checksum \
      "GVIRTUS_A1_POLICY=${POL}" >/dev/null 2>&1
  LOG=/tmp/i10_a1_${POL}_${REP}.log
  # La linea de politica es la PRUEBA de que la celda se ejercito. Sin ella la fila no vale.
  VIS=$(grep -o "A1 teardown: policy=[a-z]* assumed=[0-9]* fences=[0-9]* flushes=[0-9]* declined=[0-9]*" "$LOG" | tail -1)
  NP=$(grep -cE "^[0-9]+,[0-9.]+,[0-9.]+," "$LOG")
  echo "filas=$NP  $VIS"
  echo "# $POL rep$REP :: $VIS" >> /tmp/a1_matrix.evidence
  grep -E "^[0-9]+,[0-9.]+,[0-9.]+," "$LOG" | while IFS=, read -r t ms gb exp dck hck bd ch; do
    echo "${POL},${REP},${t},${ms},${gb},${bd},${ch}" >> "$OUT"
  done
done
echo "=== hecho -> $OUT ==="
