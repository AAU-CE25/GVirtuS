#!/bin/bash
# Driver de los tests unitarios del selector. Un proceso por politica: rma_policy() cachea el
# modo en un static a proposito, asi que cambiarla dentro de una ejecucion es imposible (y esa
# inmutabilidad es en si misma una propiedad deseable: la politica no cambia a mitad de vuelo).
set -u
cd /home/student.aau.dk/ll33pq/GVirtuS
BIN=/tmp/test_rma_policy
g++ -std=c++17 -O1 -Iinclude -o $BIN /tmp/test_rma_policy.cpp || { echo "BUILD FALLIDO"; exit 1; }

fallos=0
corre() {  # $1 = etiqueta, resto = env
  local etiqueta="$1"; shift
  echo "--- $etiqueta ---"
  if env "$@" $BIN "$etiqueta"; then :; else fallos=$((fallos+1)); fi
  echo
}

corre scalar          GVIRTUS_RMA_POLICY=scalar
corre quadrant        GVIRTUS_RMA_POLICY=quadrant
corre oracle          GVIRTUS_RMA_POLICY=oracle
corre override        GVIRTUS_RMA_POLICY=quadrant GVIRTUS_RMA_MIN_H2D_PINNED=16384
corre floor_override  GVIRTUS_RMA_POLICY=scalar   GVIRTUS_RMA_SCALAR_FLOOR=65536
corre desconocida     GVIRTUS_RMA_POLICY=chorizo

echo "======================================"
if [ $fallos -eq 0 ]; then echo "TODOS LOS GRUPOS PASAN"; else echo "$fallos GRUPOS CON FALLOS"; fi
exit $fallos
