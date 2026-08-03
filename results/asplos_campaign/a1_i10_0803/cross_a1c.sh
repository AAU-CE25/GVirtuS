#!/bin/bash
# cross_a1c.sh -- ¿sigue el cruce de colocacion donde estaba, ahora que todo camino admitido
# descarga ordenacion de transporte (I13) y visibilidad de device (I10)?
#
# DOS variables, dos trabajos, y confundirlas ya produjo una vez cuatro curvas identicas:
#   GVIRTUS_RMA_MIN_BYTES     dimensiona el POOL. Lo lee el BACKEND, que es quien lo construye.
#   GVIRTUS_RMA_SCALAR_FLOOR  es la PUERTA de la politica escalar, por transferencia, cliente.
# El backend tiene que estar arrancado con MIN_BYTES=4096 o el pool no existe y todo cae a AM.
set -u
REPS="${REPS:-3}"
D=/home/student.aau.dk/ll33pq/cross_a1; mkdir -p "$D"
OUTD=/home/student.aau.dk/ll33pq/cross_out; mkdir -p "$OUTD"
SIZES="${SIZES:-4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304}"
ALTO=1073741824
: > "$D/evidencia.txt"

for r in $(seq 1 "$REPS"); do
for MEM in pinned pageable; do
  for ARM in $(printf "am\nassume\nfence\nflush\n" | awk -v s="$((20260803+r))" 'BEGIN{srand(s)}{print rand()"\t"$0}' | sort | cut -f2); do
    TAG="${ARM}_${MEM}_r${r}"
    printf "  %-24s ... " "$TAG"
    if [ "$ARM" = am ]; then
      FLOOR="GVIRTUS_RMA_SCALAR_FLOOR=$ALTO"; POL="GVIRTUS_A1_POLICY=flush"
    else
      FLOOR="GVIRTUS_RMA_SCALAR_FLOOR=4096";  POL="GVIRTUS_A1_POLICY=$ARM"
    fi
    bash /home/student.aau.dk/ll33pq/sweep_run.sh "$TAG" \
      "/opt/GVirtuS/examples/rmatest/sweep_bench" \
      "SIZES=$SIZES" "MEM=$MEM" "REG=cached" "DIRS=both" "ITERS=20" "WARMUP=5" \
      "TAG=$TAG" "OUT=/out/out_$TAG.csv" \
      "GVIRTUS_RMA_MIN_BYTES=4096" "$FLOOR" "$POL" >/dev/null 2>&1
    L=/tmp/sw_$TAG.log
    EV=$(grep -o "admit_rma=[0-9]* admit_am=[0-9]*" "$L" | tail -1)
    A1=$(grep -o "policy=[a-z]* assumed=[0-9]* fences=[0-9]* flushes=[0-9]* declined=[0-9]*" "$L" | tail -1)
    NR=0; [ -f "$OUTD/out_$TAG.csv" ] && NR=$(wc -l < "$OUTD/out_$TAG.csv")
    echo "csv=$NR  $EV  $A1"
    echo "$TAG :: csv=$NR :: $EV :: $A1" >> "$D/evidencia.txt"
  done
done
done
echo "=== hecho ==="
