#!/bin/bash
# simple_matrix N-sweep over clean GVirtuS TCP. Per-rep raw rows.
set -u
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties.json
export GVIRTUS_LOGLEVEL=60000
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:/usr/local/gvirtus/lib
cd /benchmarks/gvs-clean
OUT=/benchmarks/gvs-clean/gvs-clean-simple-matrix.csv
echo "bench,transport,N,rep,iters,avg_sgemm_ms,avg_host_ms,check" > "$OUT"
SIZES="256 512 1024 2048 4096 8192 16384"
ITERS=10
REPS=5
for N in $SIZES; do
  # warmup rep (discarded)
  timeout 600 ./simple_matrix "$N" "$ITERS" 2 >/dev/null 2>&1
  for r in $(seq 1 $REPS); do
    line=$(timeout 600 ./simple_matrix "$N" "$ITERS" 2 2>/dev/null | grep -a "^CSV,")
    rc=$?
    if [ -z "$line" ]; then
      echo "[sm] FAIL N=$N rep=$r rc=$rc" >&2
      echo "simple_matrix,tcp,$N,$r,$ITERS,FAIL,FAIL,fail" >> "$OUT"
      continue
    fi
    # line: CSV,N,iters,avg_sgemm_ms,avg_host_ms
    n=$(echo "$line" | cut -d, -f2)
    it=$(echo "$line" | cut -d, -f3)
    sg=$(echo "$line" | cut -d, -f4)
    ho=$(echo "$line" | cut -d, -f5)
    echo "simple_matrix,tcp,$n,$r,$it,$sg,$ho,pass" >> "$OUT"
    echo "[sm] OK N=$N rep=$r sgemm_ms=$sg host_ms=$ho" >&2
  done
done
echo "[sm] DONE"
