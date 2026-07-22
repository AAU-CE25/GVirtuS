#!/bin/bash
set -u
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties.json
export GVIRTUS_LOGLEVEL=60000
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:/usr/local/gvirtus/lib
BIN=/benchmarks/miniBUDE/build/cuda-bude-gvirtus
OUT=/benchmarks/gvs-clean/gvs-clean-minibude.csv
DECK=${DECK:-data/bm1}
ITER=${ITER:-8}
REPS=${REPS:-5}
cd /benchmarks/miniBUDE
echo "bench,transport,rep,deck,iter,gflops,giga_interactions_s,avg_ms,context_ms,valid" > "$OUT"
# warmup discarded
timeout 300 "$BIN" --deck "$DECK" --iter "$ITER" >/dev/null 2>&1
for r in $(seq 1 $REPS); do
  out=$(timeout 300 "$BIN" --deck "$DECK" --iter "$ITER" 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ]; then echo "[mb] FAIL rep=$r rc=$rc" >&2; continue; fi
  gflops=$(echo "$out" | grep -aE "gflop/s:" | head -1 | grep -oE "[0-9.]+")
  gints=$(echo "$out" | grep -aE "giga_interactions/s:" | head -1 | grep -oE "[0-9.]+")
  avg=$(echo "$out" | grep -aE "avg_ms:" | tail -1 | grep -oE "[0-9.]+" | head -1)
  ctx=$(echo "$out" | grep -aE "context_ms:" | head -1 | grep -oE "[0-9.]+")
  valid=$(echo "$out" | grep -aoE "valid: (true|false)" | head -1 | awk "{print \$2}")
  echo "minibude,tcp,$r,$DECK,$ITER,$gflops,$gints,$avg,$ctx,$valid" >> "$OUT"
  echo "[mb] OK rep=$r gflops=$gflops gints=$gints avg_ms=$avg valid=$valid" >&2
done
echo "[mb] DONE"
