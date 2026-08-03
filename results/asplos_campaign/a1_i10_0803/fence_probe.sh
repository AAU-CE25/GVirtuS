#!/bin/bash
# ¿Que hace realmente `fence` en esta maquina? UCX_FENCE_MODE=auto y UCX_RC_MLX5_FENCE=auto,
# y la doc de UCX dice que el fence de IB puede ser `none` (no-op). Si auto, weak y none dan
# lo mismo, el fence no esta haciendo nada medible -- que es un dato, no un fallo.
set -u
echo "celda,rep,gbps,fences,flushes"
for rep in 1 2 3; do
  for cel in auto weak none strong; do
    case $cel in
      auto)   E1="UCX_RC_MLX5_FENCE=auto"; E2="UCX_FENCE_MODE=auto" ;;
      weak)   E1="UCX_RC_MLX5_FENCE=weak"; E2="UCX_FENCE_MODE=weak" ;;
      none)   E1="UCX_RC_MLX5_FENCE=none"; E2="UCX_FENCE_MODE=auto" ;;
      strong) E1="UCX_RC_MLX5_FENCE=auto"; E2="UCX_FENCE_MODE=strong" ;;
    esac
    bash /home/student.aau.dk/ll33pq/i10_run.sh "fp_${cel}_$rep" \
      /opt/GVirtuS/examples/rmatest/rma_checksum \
      GVIRTUS_A1_POLICY=fence "$E1" "$E2" >/dev/null 2>&1
    L=/tmp/i10_fp_${cel}_$rep.log
    G=$(grep -E "^[0-9]+," "$L" | awk -F, '$1>2{s+=$3;n++} END{if(n)printf "%.3f",s/n; else print "NA"}')
    F=$(grep -o "fences=[0-9]*" "$L" | tail -1 | cut -d= -f2)
    FL=$(grep -o "flushes=[0-9]*" "$L" | tail -1 | cut -d= -f2)
    echo "$cel,$rep,$G,$F,$FL"
  done
done
