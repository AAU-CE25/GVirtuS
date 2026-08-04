#!/bin/bash
# reset_backend_serial.sh -- relanza el backend con la MISMA configuracion que la corrida en
# curso (capturada de docker inspect el 2026-08-04) y las perillas de diagnostico de la
# serializacion. Cualquier variable extra se pasa en GVS_EXTRA (formato "K=V K=V").
#
# Por que un script y no una linea suelta: la campaña ya perdio brazos enteros porque una
# perilla no llegaba al contenedor. Aqui se imprime el entorno EFECTIVO leido de dentro del
# contenedor despues de arrancar, que es la unica prueba de que llego.
set -uo pipefail
cd /home/student.aau.dk/ll33pq/GVirtuS

EXTRA=""
for kv in ${GVS_EXTRA:-}; do EXTRA="$EXTRA -e $kv"; done

docker rm -f gvirtus-ll33pq >/dev/null 2>&1
for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':32222' || break; sleep 2; done

docker run -d --name gvirtus-ll33pq --ulimit memlock=-1 \
  --runtime=nvidia --privileged --network host --shm-size=8G \
  --entrypoint /entrypoint.sh \
  -v "$PWD/etc":/gvirtus/etc -v "$PWD/tests":/gvirtus/tests -v "$PWD/cmake":/gvirtus/cmake \
  -v "$PWD/examples":/gvirtus/examples -v "$PWD/include":/gvirtus/include -v "$PWD/src":/gvirtus/src \
  -v "$PWD/plugins":/gvirtus/plugins -v "$PWD/tools":/gvirtus/tools \
  -v "$PWD/CMakeLists.txt":/gvirtus/CMakeLists.txt -v "$PWD/docker/dev/entrypoint.sh":/entrypoint.sh \
  -e UCX_LOG_LEVEL=error -e UCX_SOCKADDR_TLS_PRIORITY=tcp -e GVIRTUS_UCX_PROGRESS_TIMEOUT_MS=0 \
  -e BACKEND_CONFIG=/usr/local/gvirtus/etc/properties_ucx.json \
  -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e GVS_FAULT=none -e GVS_FAULT_MS=50 \
  -e GVIRTUS_RMA_SLOTS=8 -e GVIRTUS_LOGLEVEL=40000 -e GVIRTUS_UCX_DATAPATH=am \
  -e UCX_IB_GID_INDEX=3 -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_MIN_BYTES=4194304 \
  -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e GVIRTUS_RMA_ZEROCOPY=1 \
  $EXTRA \
  ll33pq/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04 >/dev/null || { echo "FALLO docker run"; exit 1; }

echo "relanzado; esperando compilacion + listen..."
LISTO=no
for t in $(seq 1 120); do
  if ss -ltn 2>/dev/null | grep -q '25.25.25.2:32222'; then LISTO=si; echo "LISTENING t=${t}x3s $(date -u +%H:%M:%S)"; break; fi
  sleep 3
done
[ "$LISTO" = si ] || { echo "NO LEVANTO -- ultimas lineas:"; docker logs gvirtus-ll33pq 2>&1 | tail -25; exit 2; }
# Prueba de que las perillas LLEGARON (no de que se pasaron en la linea de comandos).
echo "=== entorno efectivo dentro del contenedor:"
docker exec gvirtus-ll33pq bash -c 'tr "\0" "\n" < /proc/$(pgrep -f gvirtus-backend | head -1)/environ | grep -E "^(GVS|GVIRTUS_GPU|GVIRTUS_RMA|UCX_TLS)" | sort'
docker logs gvirtus-ll33pq 2>&1 | grep -iE "GPUDirect|Backend config|POLICY" | tail -3
