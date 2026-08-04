#!/bin/bash
# micros_final.sh -- reconfirma los CUATRO umbrales desde el commit definitivo, con las tres
# politicas. Criterio del usuario para dar Quadrant por listo.
#   pinned H2D 16 KiB · pageable H2D 1 MiB · pinned D2H 1 MiB · pageable D2H 2 MiB
#
# El brazo AM se fuerza con GVIRTUS_RMA_SCALAR_FLOOR alto (la PUERTA), no con
# GVIRTUS_RMA_MIN_BYTES (que dimensiona el POOL y lo lee el backend). Confundirlas da cuatro
# curvas identicas, que ya paso una vez.
set -u
REPS="${REPS:-3}"
D=/home/student.aau.dk/ll33pq/micros_final; mkdir -p "$D"
OUTD=/home/student.aau.dk/ll33pq/cross_out; mkdir -p "$OUTD"
S="SIZES=4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304"
ALTO=1073741824
: > "$D/evidencia.txt"

for rep in $(seq 1 "$REPS"); do
for MEM in pinned pageable; do
  for ARM in am scalar quadrant oracle; do
    T=mf_${ARM}_${MEM}_r${rep}
    case $ARM in
      am)       POL="GVIRTUS_RMA_POLICY=scalar";   FL="GVIRTUS_RMA_SCALAR_FLOOR=$ALTO" ;;
      scalar)   POL="GVIRTUS_RMA_POLICY=scalar";   FL="GVIRTUS_RMA_SCALAR_FLOOR=4096" ;;
      quadrant) POL="GVIRTUS_RMA_POLICY=quadrant"; FL="GVIRTUS_RMA_SCALAR_FLOOR=4096" ;;
      oracle)   POL="GVIRTUS_RMA_POLICY=oracle";   FL="GVIRTUS_RMA_SCALAR_FLOOR=4096" ;;
    esac
    bash /home/student.aau.dk/ll33pq/sweep_run.sh "$T" \
      /opt/GVirtuS/examples/rmatest/sweep_bench \
      "$S" MEM=$MEM REG=cached DIRS=both ITERS=20 WARMUP=5 \
      "TAG=$T" "OUT=/out/out_$T.csv" \
      GVIRTUS_RMA_MIN_BYTES=4096 "$FL" "$POL" >/dev/null 2>&1
    A=$(grep -oE "admit_rma=[0-9]+" /tmp/sw_$T.log | tail -1 | cut -d= -f2)
    N=0; [ -f "$OUTD/out_$T.csv" ] && N=$(wc -l < "$OUTD/out_$T.csv")
    printf "%-26s admit_rma=%-7s csv=%s\n" "$T" "${A:-?}" "$N" | tee -a "$D/evidencia.txt"
  done
done
done
echo "=== micros hechos ===" | tee -a "$D/evidencia.txt"
