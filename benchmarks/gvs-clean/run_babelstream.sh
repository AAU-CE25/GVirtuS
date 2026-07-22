#!/bin/bash
set -u
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties.json
export GVIRTUS_LOGLEVEL=60000
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:/usr/local/gvirtus/lib
BIN=/benchmarks/cuda-stream-fe
OUT=/benchmarks/gvs-clean/gvs-clean-babelstream.csv
ITERS=${ITERS:-50}
REPS=${REPS:-5}
SIZES="1048576 2097152 4194304 8388608 16777216 33554432 67108864"
echo "bench,transport,rep,function,num_times,n_elements,sizeof,max_MB_per_sec,min_runtime,max_runtime,avg_runtime" > "$OUT"
for s in $SIZES; do
  # warmup (discarded)
  timeout 120 "$BIN" -s "$s" -n "$ITERS" --csv >/dev/null 2>&1
  for r in $(seq 1 $REPS); do
    out=$(timeout 120 "$BIN" -s "$s" -n "$ITERS" --csv 2>/dev/null)
    rc=$?
    if [ $rc -ne 0 ]; then echo "[bs] FAIL s=$s rep=$r rc=$rc" >&2; continue; fi
    echo "$out" | awk -v r="$r" -F, '$1!="function" && NF>=8 {print "babelstream,tcp,"r","$0}' >> "$OUT"
    echo "[bs] OK s=$s rep=$r" >&2
  done
done
echo "[bs] DONE"
