#!/bin/bash
# canonica2.sh -- campana canonica: TODOS los titulares desde UNA configuracion.
#
# POR QUE EXISTE. El paper demuestra que un umbral ESCALAR tiene la forma equivocada y luego
# publica numeros medidos CON el escalar. Esta campana vuelve a medir con la politica que se
# defiende, para que la politica que se promueve sea la que produjo las cifras.
#
# CONFIGURACION UNICA: commit limpio · A1 = flush (defecto del codigo) · I10 activo ·
# placement = QUADRANT (16K / 1024K / 1024K / 2048K) · misma imagen · UCX 1.20.0 (la del
# contenedor; la del host es 1.17.0 y NO es la del camino de datos).
#
# TRES SALVAGUARDAS, cada una por un fallo concreto de esta campana:
#
#   1. POLITICA VERIFICADA EN EL LOG DE CADA CORRIDA, no supuesta del entorno. Ningun arnes leia
#      `GVS_EXTRA_ENV` salvo sweep_run.sh (que la toma por argumentos), asi que los bloques de
#      llama y cuDF de la version anterior habrian corrido con la politica POR DEFECTO y se
#      habrian anotado como quadrant. Se exige el banner `[GVS POLICY]` con los cuatro umbrales.
#   2. DETECCION DE LOG RANCIO. Cuando un contenedor no arranca, `docker logs` falla y el fichero
#      conserva el de la corrida ANTERIOR: tres corridas seguidas se anotaron "ok" con
#      admit_rma identico al digito. Se exige que el log sea mas nuevo que el inicio de la corrida.
#   3. EL CRUDO SIEMPRE. Cada bloque escribe una fila por repeticion en raw/. Los percentiles se
#      calculan aparte con estadistica.py. Citar un numero sin poder ver sus replicas ya produjo
#      dos errores: 14 transferencias de UNA corrida contadas como 14 replicas, y una "banda" que
#      eran dos configuraciones exactas.
#
#   uso: bash canonica2.sh [bloques]     p.ej. "1 2" o "todos" (defecto)
set -u
H=/home/student.aau.dk/ll33pq
D=$H/canonica2; mkdir -p "$D/raw" "$D/logs"
L=$D/canonica.log
BLOQUES="${*:-todos}"
Q="GVIRTUS_RMA_POLICY=quadrant GVIRTUS_RMA_SCALAR_FLOOR=4194304"
S="GVIRTUS_RMA_POLICY=scalar GVIRTUS_RMA_SCALAR_FLOOR=4194304"
UMBRALES="H2D pinned 16K / H2D pageable 1024K / D2H pinned 1024K / D2H pageable 2048K"

say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }
quiere(){ [ "$BLOQUES" = todos ] && return 0; case " $BLOQUES " in *" $1 "*) return 0;; esac; return 1; }

# Comprobacion POSITIVA: el banner solo se imprime cuando la politica NO es escalar, y trae los
# cuatro umbrales. Que aparezca demuestra que quadrant llego al proceso.
politica_quadrant(){
  [ -f "$1" ] || return 1
  grep -qa "GVS POLICY.*four-quadrant placement ($UMBRALES)" "$1"
}
fresco(){ [ -f "$1" ] && [ "$(stat -c %Y "$1")" -ge "$2" ]; }

# Cerrojo. Al lanzarla la primera vez arrancaron DOS instancias (la llamada se relanzo al pasar
# a segundo plano) y comparten un unico backend, asi que se habrian contaminado en silencio.

# Cerrojo por FICHERO DE PID, no por flock. Con flock el descriptor lo HEREDAN los hijos: al
# matar una campana su `timeout` se quedaba con el cerrojo, y el siguiente lanzamiento se
# declaraba duplicado y SALIA EN SILENCIO -- el modo de fallo peligroso, porque no da error, no
# mide. Un PID se comprueba y no se hereda.
# (Este bloque estuvo DUPLICADO: el marcador con el que lo inserte casaba tambien con la linea
# de "CAMPANA CANONICA COMPLETA" del final.)
if [ -f "$D/.pid" ] && kill -0 "$(cat "$D/.pid" 2>/dev/null)" 2>/dev/null; then
  echo "ya hay una campana corriendo (pid $(cat "$D/.pid")); salgo"; exit 1
fi
echo $$ > "$D/.pid"
trap 'rm -f "$D/.pid"' EXIT

say "################ CAMPANA CANONICA -- bloques: $BLOQUES ################"
bash $H/procedencia.sh canonica2 "$D/procedencia.json" >/dev/null 2>&1
[ -s "$D/procedencia.json" ] && say "procedencia guardada" || say "AVISO: procedencia VACIA -- la config no queda registrada"

# ---------------------------------------------------------------- 1. sweep H2D/D2H, 4 brazos
if quiere 1; then
  say "=== 1. micros: sweep H2D/D2H -- am / scalar / quadrant / oracle, 3 reps ==="
  REPS=3 timeout 5400 bash $H/micros_final.sh > "$D/logs/micros.log" 2>&1
  say "  micros terminados; crudo en logs/micros.log ($(grep -c . "$D/logs/micros.log") lineas)"
  # Los umbrales que el brazo quadrant declara TIENEN que ser los cuatro esperados.
  # El banner vive en la salida del CONTENEDOR de cada corrida (/tmp/sw_<tag>.log), no en el
  # resumen que escribe micros_final.sh. Buscarlo en el resumen daba un aviso falso.
  if ls /tmp/sw_mf_quadrant_*.log >/dev/null 2>&1 && \
     grep -qah "four-quadrant placement ($UMBRALES)" /tmp/sw_mf_quadrant_*.log; then
    say "  umbrales confirmados en las corridas quadrant del sweep: $UMBRALES"
  else
    say "  *** AVISO: ninguna corrida quadrant del sweep declara los cuatro umbrales -- no citar"
  fi
fi

# ---------------------------------------------------------------- 2. miniBUDE: LA carga clave
# Sus pagos (256 KB - 1,5 MB) caen EXACTAMENTE en la banda donde quadrant difiere de scalar
# (16 KiB - 4 MiB). Es la unica carga evaluada cuyas transferencias caen ahi, y nunca se ha
# medido con ellas yendo por RMA: la etiqueta "transport-invariant" se puso con el suelo a 4 MiB.
if quiere 2; then
    # MEDIDO, y corrige la premisa con la que se eligio esta carga. Bisecando el suelo escalar,
  # miniBUDE mueve 66 transferencias y NINGUNA llega a 1 MiB (suelo 4K->8 admitidas, 16K->7,
  # 64K y 256K->6, 1M->0). Quadrant aplica 1 MiB a H2D paginable y 1-2 MiB a D2H, luego NO
  # admite ninguna: miniBUDE es un CONTROL que no debe cambiar, no el diferenciador. Yo habia
  # supuesto que sus 256 KB eran memoria FIJADA, donde el umbral seria 16 KiB.
  say "=== 2. miniBUDE: CONTROL scalar contra quadrant (no debe cambiar), 15 reps ==="
  echo "politica,rep,admit_rma,gflops,ms,politica_verificada" > "$D/raw/minibude.csv"
  BUDE=/opt/GVirtuS/examples/minibude/miniBUDE/build/cuda-bude-gvirtus
  DECK=/opt/GVirtuS/examples/minibude/miniBUDE/data/bm1
  for POL in scalar quadrant; do
    [ "$POL" = quadrant ] && E="$Q" || E="$S"
    for rep in $(seq 1 15); do
      T=mb_${POL}_r$rep; T0=$(date +%s)
      timeout 900 bash $H/sweep_run.sh "$T" "$BUDE -n 65536 -i 8 --deck $DECK" $E >/dev/null 2>&1
      LG=/tmp/sw_$T.log
      if ! fresco "$LG" "$T0"; then
        say "  miniBUDE $POL rep$rep: LOG RANCIO o ausente -- no cuenta"
        echo "$POL,$rep,,,,rancio" >> "$D/raw/minibude.csv"; continue
      fi
      cp "$LG" "$D/logs/$T.log"
      A=$(grep -aoE "admit_rma=[0-9]+" "$LG" | tail -1 | cut -d= -f2)
      G=$(grep -aoE "[0-9]+\.[0-9]+ GFLOP/s" "$LG" | tail -1 | cut -d" " -f1)
      MS=$(grep -aoE "[0-9]+\.[0-9]+ ms" "$LG" | tail -1 | cut -d" " -f1)
      if [ "$POL" = quadrant ]; then
        politica_quadrant "$LG" && V=si || V=NO
        [ "$V" = NO ] && say "  miniBUDE quadrant rep$rep: *** el banner de politica NO aparece -- brazo INERTE"
      else V=na; fi
      echo "$POL,$rep,${A:-},${G:-},${MS:-},$V" >> "$D/raw/minibude.csv"
    done
    say "  $POL hecho"
  done
  say "  --- miniBUDE, estadistica ---"
  for POL in scalar quadrant; do
    grep -E "^($POL|politica)," "$D/raw/minibude.csv" > /tmp/mb_$POL.csv
    python3 $H/estadistica.py /tmp/mb_$POL.csv gflops "miniBUDE $POL GFLOP/s" 2>&1 | tee -a "$L"
    python3 $H/estadistica.py /tmp/mb_$POL.csv admit_rma "miniBUDE $POL admit_rma" 2>&1 | tee -a "$L"
  done
fi

# ---------------------------------------------------------------- 3. llama
if quiere 3; then
  # LLAMA ES EL DIFERENCIADOR, medido: con el escalar a 4 MiB una corrida da admit_rma=13, y
  # con quadrant 6 731 -- un factor ~500 en admisiones. Es la carga cuyo camino cambia la
  # politica, asi que se corre head-to-head en vez de solo con quadrant.
  say "=== 3. llama tg16 head-to-head: scalar contra quadrant, 2 modelos, 3 reps ==="
  echo "politica,modelo,rep,tps,admit_rma,politica_verificada" > "$D/raw/llama_tg16.csv"
  for POL in scalar quadrant; do
    [ "$POL" = quadrant ] && E="$Q" || E="$S"
    for M in tinyllama-1.1b-q4 mistral-7b-q4; do
      for rep in 1 2 3; do
        OUT="$D/logs/tg16_${POL}_${M}_r$rep.log"
        GVS_MODEL=$M GVS_EXTRA_ENV="$E" timeout 1800 bash $H/llama_tg16.sh > "$OUT" 2>&1
        TPS=$(grep -aoE "[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+" "$OUT" | tail -1 | cut -d" " -f1)
        A=$(grep -aoE "admit_rma=[0-9]+" "$OUT" | tail -1 | cut -d= -f2)
        if [ "$POL" = quadrant ]; then
          politica_quadrant "$OUT" && V=si || V=NO
          [ "$V" = NO ] && say "  *** tg16 quadrant $M rep$rep: banner ausente -- INERTE"
        else
          # El brazo escalar se verifica al reves: el banner NO debe aparecer.
          grep -qa "GVS POLICY" "$OUT" && V=BANNER_INESPERADO || V=si
        fi
        echo "$POL,$M,$rep,${TPS:-},${A:-},$V" >> "$D/raw/llama_tg16.csv"
        say "  tg16 $POL $M rep$rep: ${TPS:-SIN_CIFRA} t/s admit_rma=${A:-?} ($V)"
      done
    done
  done
  for POL in scalar quadrant; do
    for M in tinyllama-1.1b-q4 mistral-7b-q4; do
      (head -1 "$D/raw/llama_tg16.csv"; grep "^$POL,$M," "$D/raw/llama_tg16.csv") > /tmp/tg.csv
      python3 $H/estadistica.py /tmp/tg.csv tps "llama $POL $M t/s" 2>&1 | tee -a "$L"
    done
  done

  say "=== 3b. llama servidor N=8, scalar contra quadrant: TTFT p50/p95/p99 ==="
  echo "politica,rep,veredicto,admit_rma,ttft_ms" > "$D/raw/llama_srv.csv"
  for POL in scalar quadrant; do
  [ "$POL" = quadrant ] && E="$Q" || E="$S"
  for rep in 1 2 3 4 5; do
    T0=$(date +%s)
    GVS_EXTRA_ENV="$E" timeout 400 bash $H/llama_abort_repro.sh 1 45 > "$D/logs/srv_${POL}_r$rep.out" 2>&1
    LG=/tmp/llama_abort/iter1_srv.log
    if ! fresco "$LG" "$T0"; then say "  srv $POL rep$rep: LOG RANCIO -- no cuenta"; continue; fi
    cp "$LG" "$D/logs/srv_${POL}_r$rep.log"
    A=$(grep -aoE "admit_rma=[0-9]+" "$LG" | tail -1 | cut -d= -f2)
    grep -qaE "ggml_abort|CUDA error" "$LG" && V=ABORT || V=ok
    # Una fila por peticion: el p95 de un servidor se calcula sobre PETICIONES, no sobre corridas.
    grep -aoE "prompt eval time = *[0-9]+\.[0-9]+ ms" "$LG" | grep -oE "[0-9]+\.[0-9]+" \
      | while read t; do echo "$POL,$rep,$V,${A:-},$t" >> "$D/raw/llama_srv.csv"; done
    say "  srv $POL rep$rep: $V admit_rma=${A:-?} peticiones=$(grep -ca "prompt eval time" "$LG")"
  done
  done
  for POL in scalar quadrant; do
    (head -1 "$D/raw/llama_srv.csv"; grep "^$POL," "$D/raw/llama_srv.csv") > /tmp/srv.csv
    python3 $H/estadistica.py /tmp/srv.csv ttft_ms "llama srv $POL TTFT ms" 2>&1 | tee -a "$L"
  done
fi

# ---------------------------------------------------------------- 4. cuDF
if quiere 4; then
  say "=== 4. cuDF ETL N=1 y N=8, quadrant, 3 reps ==="
  echo "n,rep,wall_s,tx_mib,rx_mib,records,ready" > "$D/raw/cudf.csv"
  cd $H/harness || exit 1
  for rep in 1 2 3; do
    for n in 1 8; do
      Lo=$(GVS_EXTRA_ENV="$Q" timeout 3000 bash ./cudf_point.sh gpudirect $n $rep 20 1 1 2>&1 | tail -1)
      W=$(echo "$Lo"|grep -o "WALL=[0-9.]*"|cut -d= -f2); TX=$(echo "$Lo"|grep -o "TX=[0-9.]*"|cut -d= -f2)
      RX=$(echo "$Lo"|grep -o "RX=[0-9.]*"|cut -d= -f2); RE=$(echo "$Lo"|grep -o "REC=[0-9]*"|cut -d= -f2)
      RD=$(echo "$Lo"|grep -o "READY=[0-9]*/[0-9]*"|cut -d= -f2)
      # PUERTA DE VALIDEZ. Una corrida con ready=8/8 pero rec=0 movio 15,7 GB en vez de 93,8 y
      # no produjo un solo registro: aborto a mitad. Sin esta puerta entraba en la estadistica y
      # subia la sd de N=8 a 8,65 con min=11,88. Lo esperado es 21 registros por cliente.
      ESP=$((21 * n))
      if [ "${RE:-0}" != "$ESP" ]; then
        say "  *** cuDF N=$n rep$rep: rec=${RE:-0}, esperados $ESP -- corrida DESCARTADA"
        echo "$n,$rep,,,,${RE:-0},${RD:-}" >> "$D/raw/cudf.csv"; continue
      fi
      echo "$n,$rep,${W:-},${TX:-},${RX:-},${RE:-},${RD:-}" >> "$D/raw/cudf.csv"
      V=NO; for f in /tmp/cl_gpudirect_n${n}_r${rep}_*.log; do politica_quadrant "$f" && V=si && break; done
      say "  cuDF N=$n rep$rep: wall=${W:-?}s rec=${RE:-?} ready=${RD:-?} politica=$V"
    done
  done
  cd $H
  for n in 1 8; do
    (head -1 "$D/raw/cudf.csv"; grep "^$n," "$D/raw/cudf.csv") > /tmp/cudf_$n.csv
    python3 $H/estadistica.py /tmp/cudf_$n.csv wall_s "cuDF N=$n wall s" 2>&1 | tee -a "$L"
  done
fi

# ---------------------------------------------------------------- 5. OSU (referencia, NO Gusto)
# OSU es MPI de DOS NODOS sobre RoCE con buffers de dispositivo, y corre NATIVO: no pasa por
# GVirtuS. Mide el TECHO del fabric, que es contra lo que se compara el sweep. Es
# INDEPENDIENTE de la politica de colocacion, asi que no lleva brazo quadrant ni comprobacion de
# banner: pedirle uno habria producido un numero sin significado.
if quiere 5; then
  say "=== 5. OSU baremetal (referencia del fabric; independiente de la politica), 3 reps ==="
  echo "prueba,rep,tam_bytes,valor" > "$D/raw/osu.csv"
  export PATH=/usr/mpi/gcc/openmpi-4.1.7a1/bin:$PATH
  AZ=/home/es.aau.dk/az05mg/repos/dpu-pqc/third_party/ucx-install-x86_64
  CO=$H/osu-micro-benchmarks-7.5/c
  LDP=$AZ/lib:/usr/local/cuda/lib64:/usr/mpi/gcc/openmpi-4.1.7a1/lib
  for P in osu_bw osu_latency; do
    for rep in 1 2 3; do
      O="$D/logs/${P}_r$rep.log"
      timeout 200 mpirun -np 2 --host es-dpu-02,25.25.25.2 \
        --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" --mca pml ucx --mca btl ^openib \
        --mca coll ^hcoll -x UCX_NET_DEVICES=mlx5_1:1 \
        -x UCX_TLS=rc_mlx5,ud_mlx5,tcp,cuda_copy,self -x UCX_IB_GID_INDEX=3 \
        -x LD_LIBRARY_PATH=$LDP "$CO/mpi/pt2pt/standard/$P" -d cuda D D > "$O" 2>&1
      n=$(grep -aE "^[0-9]+ +[0-9.]+" "$O" | awk -v p=$P -v r=$rep '{print p","r","$1","$2}' \
            | tee -a "$D/raw/osu.csv" | wc -l)
      [ "$n" = 0 ] && say "  *** $P rep$rep: 0 puntos -- corrida INERTE, no cuenta"
    done
    say "  $P: $(grep -c "^$P," "$D/raw/osu.csv") puntos en 3 reps"
  done
  # El titular es el pico; se saca del tamano mayor, con sus 3 replicas a la vista.
  awk -F, '$1=="osu_bw"{if($3+0>m)m=$3+0} END{print m}' "$D/raw/osu.csv" > /tmp/osu_max
  MX=$(cat /tmp/osu_max)
  say "  osu_bw a $MX bytes: $(awk -F, -v m=$MX '$1=="osu_bw" && $3+0==m+0 {printf "%s ", $4}' "$D/raw/osu.csv")GB/s (3 reps)"
fi

# ---------------------------------------------------------------- 6. grafos + N7
if quiere 6; then
  say "=== 6. conformidad de grafos y vida de slot / epoch ==="
  for t in graphprobe4 graphprobe6 graphsem graphvis graphvis2 d2hpool d2hreclass; do
    GVS_EXTRA_ENV="$Q" bash $H/gv_run_bin.sh "/opt/GVirtuS/tests/semantic/$t" >/dev/null 2>&1
    R=$(grep -avE "^\[GVS|^UCX|^DEBUG|GUSTO_METRIC|^\[GUSTO" /tmp/gv_bin.out | grep -iE "SUMMARY|CORRECT|VERDICT|PROBE6,all" | tail -1)
    say "  $t: ${R:-sin linea de resumen}"
  done
  for rep in 1 2 3; do
    timeout 1200 bash $H/n7_run2.sh "canon_n7_r$rep" full 0 40 >/dev/null 2>&1
    R=$(grep -aoE "(admit_rma|admit_am|ack_applied|ack_gen_mismatch|ack_on_free|parked|buffers)=[0-9]+" /tmp/gusto_n7/canon_n7_r$rep.log 2>/dev/null | tr "\n" " ")
    say "  N7 rep$rep: ${R:-sin contadores}"
  done
fi

# ---------------------------------------------------------------- 7. CloverLeaf + BabelStream
if quiere 7; then
  say "=== 7. no-regresion: CloverLeaf y BabelStream, 5 reps ==="
  echo "carga,rep,valor" > "$D/raw/noreg.csv"
  for rep in $(seq 1 5); do
    T0=$(date +%s); unset B
    GVS_EXTRA_ENV="$Q" timeout 1800 bash $H/gv_run_bin.sh \
      "/experiments/babelstream/build/cuda-stream -s 33554432 -n 20" >/dev/null 2>&1
    fresco /tmp/gv_bin.out "$T0" || { say "  babelstream rep$rep: SALIDA RANCIA -- no cuenta"; B=""; }
    [ -n "${B+x}" ] || B=$(grep -aE "^Copy" /tmp/gv_bin.out | awk '{print $2}' | tail -1)
    cp /tmp/gv_bin.out "$D/logs/babelstream_r$rep.log" 2>/dev/null
    echo "babelstream,$rep,${B:-}" >> "$D/raw/noreg.csv"
    T1=$(date +%s); unset C
    # Tres cosas que hacian falta y ninguna daba un error legible en el resumen: clover_leaf es
    # MPI (necesita mpirun, y `--mca plm isolated` porque dentro del contenedor no hay ssh),
    # enlaza libgfortran que la imagen no trae, y ESCRIBE en su directorio de trabajo -- que en
    # este arnes se monta :ro. Se ejecuta desde /tmp con el deck copiado.
    GVS_EXTRA_ENV="$Q" timeout 1800 bash $H/gv_run_bin.sh \
      "bash -c \"mkdir -p /tmp/cl && cd /tmp/cl && cp /experiments/cloverleaf/build/clover.in . && /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun --allow-run-as-root --mca plm isolated -np 1 /experiments/cloverleaf/build/clover_leaf\"" >/dev/null 2>&1
    fresco /tmp/gv_bin.out "$T1" || { say "  cloverleaf rep$rep: SALIDA RANCIA -- no cuenta"; C=""; }
    [ -n "${C+x}" ] || C=$(grep -aoE "Wall clock +[0-9.]+" /tmp/gv_bin.out | tail -1 | awk '{print $3}')
    cp /tmp/gv_bin.out "$D/logs/cloverleaf_r$rep.log" 2>/dev/null
    echo "cloverleaf,$rep,${C:-}" >> "$D/raw/noreg.csv"
    say "  rep$rep: babelstream=${B:-?} MB/s  cloverleaf=${C:-?} s"
  done
  for w in babelstream cloverleaf; do
    (head -1 "$D/raw/noreg.csv"; grep "^$w," "$D/raw/noreg.csv") > /tmp/nr_$w.csv
    python3 $H/estadistica.py /tmp/nr_$w.csv valor "$w" 2>&1 | tee -a "$L"
  done
fi

# ---------------------------------------------------------------- 8. fairness (el mas largo)
if quiere 8; then
  say "=== 8. fairness multi-inquilino (bloque mas largo) ==="
  for N in 1 2 4 8; do
    GVS_EXTRA_ENV="$Q" timeout 3600 bash $H/mt_pods.sh ucx "$N" 1.0 30 > "$D/logs/fairness_n$N.log" 2>&1 || true
    R=$(grep -aiE "jain|goodput|TTFT" "$D/logs/fairness_n$N.log" | tail -3 | tr "\n" " | ")
    say "  fairness N=$N: ${R:-SIN RESUMEN -- revisar logs/fairness_n$N.log}"
  done
fi

# Cerrojo. Al lanzarla la primera vez arrancaron DOS instancias (la llamada se relanzo al pasar
# a segundo plano) y comparten un unico backend, asi que se habrian contaminado en silencio.

say "################ CAMPANA CANONICA COMPLETA ################"
say "crudo en $D/raw/ · logs en $D/logs/ · resumen en $L"
