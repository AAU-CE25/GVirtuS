#!/bin/bash
# A 64 MiB todo lo tapa el tiempo de cable. A 4-16 KiB el fence SI se vio mas lento que
# `assume`, asi que aqui se puede distinguir si hace algo. Si fence/auto == fence/none, el
# coste no viene del fence de IB.
set -u
D=/home/student.aau.dk/ll33pq/cross_out; mkdir -p $D
echo "brazo,rep,bytes,dir,gbps"
for rep in 1 2 3; do
  for arm in assume fence_auto fence_none flush; do
    case $arm in
      assume)     P="GVIRTUS_A1_POLICY=assume"; F="UCX_RC_MLX5_FENCE=auto" ;;
      fence_auto) P="GVIRTUS_A1_POLICY=fence";  F="UCX_RC_MLX5_FENCE=auto" ;;
      fence_none) P="GVIRTUS_A1_POLICY=fence";  F="UCX_RC_MLX5_FENCE=none" ;;
      flush)      P="GVIRTUS_A1_POLICY=flush";  F="UCX_RC_MLX5_FENCE=auto" ;;
    esac
    T=fs_${arm}_r$rep
    bash /home/student.aau.dk/ll33pq/sweep_run.sh "$T" \
      /opt/GVirtuS/examples/rmatest/sweep_bench \
      "SIZES=4096,16384,65536,262144" MEM=pinned REG=cached DIRS=h2d ITERS=30 WARMUP=8 \
      "TAG=$T" "OUT=/out/out_$T.csv" \
      GVIRTUS_RMA_MIN_BYTES=4096 GVIRTUS_RMA_SCALAR_FLOOR=4096 "$P" "$F" >/dev/null 2>&1
    A=$(grep -o "admit_rma=[0-9]*" /tmp/sw_$T.log | tail -1 | cut -d= -f2)
    [ "${A:-0}" = 0 ] && { echo "$arm,$rep,INERTE,-,-"; continue; }
    awk -F, -v a="$arm" -v r="$rep" 'NR>1{print a","r","$2","$3","$8}' $D/out_$T.csv
  done
done
