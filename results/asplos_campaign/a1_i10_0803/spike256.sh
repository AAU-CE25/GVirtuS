#!/bin/bash
# ¿Es el -12 % de 256 KiB un ESCALON (un umbral de protocolo) o un bache?
# Barrido fino alrededor, assume vs flush, 3 corridas. Si es escalon, el sitio del salto
# nombra el umbral; si es bache, no es un umbral y hay que buscar otra cosa.
set -u
D=/home/student.aau.dk/ll33pq/cross_out; mkdir -p $D
echo "brazo,rep,bytes,gbps"
for rep in 1 2 3; do
  for arm in assume flush; do
    T=sp_${arm}_r$rep
    bash /home/student.aau.dk/ll33pq/sweep_run.sh "$T" \
      /opt/GVirtuS/examples/rmatest/sweep_bench \
      "SIZES=131072,163840,196608,229376,262144,294912,327680,393216,524288" \
      MEM=pinned REG=cached DIRS=h2d ITERS=30 WARMUP=8 \
      "TAG=$T" "OUT=/out/out_$T.csv" \
      GVIRTUS_RMA_MIN_BYTES=4096 GVIRTUS_RMA_SCALAR_FLOOR=4096 \
      "GVIRTUS_A1_POLICY=$arm" >/dev/null 2>&1
    A=$(grep -o "admit_rma=[0-9]*" /tmp/sw_$T.log | tail -1 | cut -d= -f2)
    [ "${A:-0}" = 0 ] && { echo "$arm,$rep,INERTE,-"; continue; }
    awk -F, -v a="$arm" -v r="$rep" 'NR>1{print a","r","$2","$8}' $D/out_$T.csv
  done
done
