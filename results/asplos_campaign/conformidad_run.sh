#!/bin/bash
# conformidad_run.sh -- fase 2, suite de conformidad semantica en TRES brazos.
#
# POR QUE TRES. Los resultados que ya existen en results/asplos_campaign/ptds/ tienen tres
# fallos (driver_ptds ptsz, event_crossstream ptsz intermitente, graph_ptds en las DOS
# variantes) y NO se puede decir si son defectos de conformidad o del propio test, porque no
# hay brazo nativo. El "baseline" que hay es ptds_repro corriendo por GVirtuS, no nativo.
#
#   native        el mismo binario contra la L40S local, sin GVirtuS
#   gusto_handle  por GVirtuS, compilacion normal
#   gusto_ptsz    por GVirtuS, compilado con --default-stream per-thread
#
# La regla de lectura, y es toda la fase:
#   falla en native Y en gusto  -> fallo del TEST (o de CUDA), no del remoting
#   pasa en native, falla en gusto -> DEFECTO DE CONFORMIDAD del remoting
#   falla solo en gusto_ptsz       -> defecto de la superficie _ptsz
#
# Cuarto brazo opcional, el control de la inyeccion: GVS_ABLATE=pointer_keyed debe hacer
# fallar las propiedades de MEMORIA y solo esas. Sin ese control, "todo pasa" no distingue
# "el sistema es correcto" de "el test no prueba nada".
#
#   uso: conformidad_run.sh <suite> <hilos> <iters> [semilla] [solo_test]
#        suite = semantic | ptds
set -u
SUITE="${1:-semantic}"; HILOS="${2:-1}"; ITERS="${3:-18}"; SEM="${4:-1}"; SOLO="${5:-}"
OUT=$HOME/GVirtuS/results/asplos_campaign/semantic_conformance
mkdir -p "$OUT"
cd "$HOME/GVirtuS" || exit 1
IMG=ll33pq/cudf_gvirtus_dyncudf:cuda12.6
TD=$HOME/GVirtuS/tests/semantic
# Las dos suites NO tienen la misma firma y confundirlas hace que el test se autolesione:
#   semantic_conformance  <hilos> <iters> <semilla> [solo]
#   ptds_conformance      <hilos> <iters> <BYTES> <semilla> [solo]
# Pasar la semilla en la posicion de bytes lanza kernels de 1 byte y todo devuelve
# cudaErrorInvalidConfiguration -- en NATIVO tambien, que es como se detecto.
BYTES="${BYTES:-1048576}"
case "$SUITE" in
  semantic) BIN=semantic_conformance; ARGS="$HILOS $ITERS $SEM" ;;
  ptds)     BIN=ptds_conformance;     ARGS="$HILOS $ITERS $BYTES $SEM" ;;
  *) echo "suite desconocida: $SUITE"; exit 2 ;;
esac
[ -n "$SOLO" ] && ARGS="$ARGS $SOLO"

# --- brazo nativo: la GPU local, sin nada en medio -------------------------------------
corre_nativo() {
  local et="$1" b="$2"
  ( cd "$TD" && ./"$b" $ARGS ) \
      > "$OUT/${et}.log" 2>&1
  echo "  native exit=$?"
}

# --- brazos por GVirtuS -----------------------------------------------------------------
corre_gusto() {
  local et="$1" b="$2" ablate="${3:-full}"
  docker rm -f gv_conf >/dev/null 2>&1
  docker run --rm --name gv_conf --network host --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1 --entrypoint bash -e LD_PRELOAD= \
    -e UCX_TLS=rc_mlx5,ud_mlx5,tcp,self -e UCX_MEMTYPE_CACHE=n -e GVIRTUS_UCX_DATAPATH=am \
    -e UCX_NET_DEVICES=mlx5_1:1,ens1f1np1 -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
    -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=error -e UCX_RCACHE_ENABLE=n \
    -e GVIRTUS_GPUDIRECT=1 -e GVIRTUS_RMA_ZEROCOPY=1 \
    -e GVIRTUS_HOME=/opt/GVirtuS -e GVIRTUS_CONFIG=/opt/GVirtuS/etc/properties_ucx.json \
    -e GVIRTUS_LOGLEVEL=30000 \
    -e LD_LIBRARY_PATH=/opt/GVirtuS/lib/frontend:/opt/GVirtuS/lib:/opt/GVirtuS/build:/usr/local/lib:/usr/local/cuda/lib64 \
    -e GVIRTUS_RMA_MIN_BYTES=8192 -e GVIRTUS_RMA_SCALAR_FLOOR=8192 \
    -e GVS_ABLATE="$ablate" \
    -v "$PWD":/opt/GVirtuS:ro "$IMG" \
    -c "ulimit -c 0; cd /opt/GVirtuS/tests/semantic && LD_PRELOAD=libcuda.so.1:libcudart.so.12 timeout 900 ./$b $ARGS" \
    > "$OUT/${et}.log" 2>&1
  echo "  $et exit=$?"
}

echo "=== suite=$SUITE hilos=$HILOS iters=$ITERS semilla=$SEM ${SOLO:+solo=$SOLO} ==="
corre_nativo "${SUITE}_native_t${HILOS}"        "${BIN}"
corre_gusto  "${SUITE}_gusto_handle_t${HILOS}"  "${BIN}"
corre_gusto  "${SUITE}_gusto_ptsz_t${HILOS}"    "${BIN}_ptsz"
if [ "${CONF_ABLATE:-0}" = "1" ]; then
  corre_gusto "${SUITE}_ablate_pointerkeyed_t${HILOS}" "${BIN}" pointer_keyed
fi
echo "=== filas obtenidas ==="
grep -hc '^CSVROW' "$OUT"/${SUITE}_*_t${HILOS}.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || true
