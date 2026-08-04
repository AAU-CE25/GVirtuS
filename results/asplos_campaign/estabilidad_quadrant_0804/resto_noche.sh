#!/bin/bash
# resto_noche.sh -- micros de umbrales + no-regresion, encadenados y desatendidos.
set -u
H=/home/student.aau.dk/ll33pq
L=$H/resto_noche.log
: > "$L"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }

say "=== 1/3 micros de umbrales (scalar / quadrant / oracle / AM) ==="
REPS=3 bash $H/micros_final.sh >> "$L" 2>&1
say "micros hechos"

say "=== 2/3 no-regresion: llama tg16 (1B, config final) ==="
bash $H/llama_tg16.sh 2>&1 | grep -E "tg16|model" | tail -2 >> "$L"

say "=== 3/3 no-regresion: rmatest + grafos ==="
for t in rma_verdict growtest dst_realloc src_realloc; do
  bash $H/gv_run_bin.sh "/opt/GVirtuS/examples/rmatest/$t" >/dev/null 2>&1
  A=$(grep -oE "admit_rma=[0-9]+" /tmp/gv_bin.out | tail -1)
  R=$(grep -vE "^\[GVS|^UCX|^DEBUG|GUSTO_METRIC|^\[GUSTO" /tmp/gv_bin.out | grep -iE "PASS|FAIL|failed_transfers|bad=" | tail -1)
  say "  $t  $A  $R"
done
for t in graphprobe4 graphprobe6 graphsem graphvis graphvis2 d2hpool ptds_mt; do
  bash $H/gv_run_bin.sh "/opt/GVirtuS/tests/semantic/$t" >/dev/null 2>&1
  R=$(grep -vE "^\[GVS|^UCX|^DEBUG|GUSTO_METRIC|^\[GUSTO" /tmp/gv_bin.out | grep -iE "SUMMARY|CORRECT|PROBE6,all" | tail -1)
  say "  $t  ${R:-sin linea de resumen}"
done
say "=== RESTO DE LA NOCHE COMPLETO ==="
