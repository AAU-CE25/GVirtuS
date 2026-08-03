#!/bin/bash
# cuDF ETL bajo la configuracion SEGURA (A1=flush por defecto + descarga de visibilidad).
# Pregunta: ¿sobrevive el resultado extremo a extremo? Referencias publicadas (60 batches,
# 5 reps): GPUDirect 356,1 ms en N=1 y 631,5 en N=8; nativo 319,8 y 637,0.
# Aqui se corre una version reducida (batches/reps menores) -- se compara la FORMA y la
# retencion, no se pretende reemplazar la tabla publicada.
set -u
cd /home/student.aau.dk/ll33pq/harness
B=${B:-20}; REPS=${REPS:-2}
echo "cfg,n,rep,wall_s,tx_mib,rx_mib,records,ready"
for rep in $(seq 1 "$REPS"); do
  for n in 1 8; do
    for cfg in native gpudirect; do
      if [ "$cfg" = native ]; then gvs=0; gd=0; else gvs=1; gd=1; fi
      L=$(bash ./cudf_point.sh "$cfg" "$n" "$rep" "$B" "$gvs" "$gd" 2>&1 | tail -1)
      W=$(echo "$L" | grep -o "WALL=[0-9.]*" | cut -d= -f2)
      TX=$(echo "$L" | grep -o "TX=[0-9.]*" | cut -d= -f2)
      RX=$(echo "$L" | grep -o "RX=[0-9.]*" | cut -d= -f2)
      RE=$(echo "$L" | grep -o "REC=[0-9]*" | cut -d= -f2)
      RD=$(echo "$L" | grep -o "READY=[0-9]*/[0-9]*" | cut -d= -f2)
      echo "$cfg,$n,$rep,$W,$TX,$RX,$RE,$RD"
    done
  done
done
