#!/bin/bash
# Rigorous rep runner. Warmup (1, discarded) + REPS measured runs.
# Emits per-rep raw values so mean+95%CI are computed offline.
#   $1 bench = llama|minibude|matrix|babel|transfer
#   $2 mode  = native|tcp|rdma|gd     (gd = run rdma TLS against a GD=1 backend)
#   $3 reps  = measured reps (default 5)
set -u
BENCH=$1; MODE=$2; REPS=${3:-5}
OUTDIR=/benchmarks/_rig; mkdir -p $OUTDIR
CUDA=/usr/local/cuda/lib64
GVL=/usr/local/gvirtus/lib:/usr/local/gvirtus/lib/frontend
LLIBS=/benchmarks/llama.cpp/build_cuda/bin

setup_env() {
  unset GVIRTUS_CONFIG GVIRTUS_UCX_DATAPATH UCX_TLS UCX_NET_DEVICES UCX_IB_GID_INDEX \
        UCX_SOCKADDR_TLS_PRIORITY GVIRTUS_ASYNC_DISPATCH GVIRTUS_GPUDIRECT GVIRTUS_RMA_ZEROCOPY
  if [ "$MODE" = "native" ]; then
    export LD_LIBRARY_PATH=$CUDA:$LLIBS
  else
    export GVIRTUS_HOME=/usr/local/gvirtus
    export GVIRTUS_CONFIG=/gvirtus/etc/properties_ucx.json
    export GVIRTUS_LOGLEVEL=40000
    export GVIRTUS_ASYNC_DISPATCH=1
    export GVIRTUS_UCX_DATAPATH=am
    export UCX_NET_DEVICES=mlx5_1:1,ens1f1np1
    export UCX_IB_GID_INDEX=3
    export UCX_SOCKADDR_TLS_PRIORITY=tcp
    export UCX_LOG_LEVEL=error
    export LD_LIBRARY_PATH=$GVL:$LLIBS
    if [ "$MODE" = "tcp" ]; then export UCX_TLS=tcp,self; else export UCX_TLS=rc_mlx5,ud_mlx5,tcp,self; fi
  fi
}
setup_env

case "$BENCH" in
  llama)
    export GGML_CUDA_DISABLE_GRAPHS=1
    B=$LLIBS/llama-bench; M=/benchmarks/models/tinyllama-1.1b-q4.gguf
    O=$OUTDIR/llama_${MODE}.log
    # llama-bench does its own warmup; -r REPS gives mean+-sd over REPS reps.
    timeout 700 $B -m $M -ngl 99 -p 8 -n 16 -r $REPS 2>/dev/null | tee $O
    echo "reps=$REPS mode=$MODE" >> $O ;;
  minibude)
    cd /benchmarks/miniBUDE
    if [ "$MODE" = "native" ]; then B=build/cuda-bude; else B=build/cuda-bude-gvirtus; fi
    O=$OUTDIR/minibude_${MODE}.csv; echo "mode,rep,gflops,context_ms,valid" > $O
    timeout 120 $B --deck data/bm1 --iter 8 >/dev/null 2>&1   # warmup
    for r in $(seq 1 $REPS); do
      out=$(timeout 120 $B --deck data/bm1 --iter 8 2>/dev/null)
      g=$(echo "$out"|grep -aE "gflop/s:"|head -1|grep -oE "[0-9.]+")
      c=$(echo "$out"|grep -aE "context_ms:"|head -1|grep -oE "[0-9.]+")
      v=$(echo "$out"|grep -aoE "valid: (true|false)"|head -1|awk "{print \$2}")
      echo "$MODE,$r,$g,$c,$v" >> $O
    done; cat $O ;;
  matrix)
    cd /gvirtus/examples/simple_matrix
    if [ "$MODE" = "native" ]; then MB=/benchmarks/simple_matrix_native; else MB=./simple_matrix; fi
    O=$OUTDIR/matrix_${MODE}.csv; echo "mode,rep,sgemm_ms,host_ms" > $O
    timeout 300 $MB 16000 5 1 >/dev/null 2>&1   # warmup
    for r in $(seq 1 $REPS); do
      out=$(timeout 300 $MB 16000 5 1 2>/dev/null)
      line=$(echo "$out"|grep -a "^CSV,")
      s=$(echo "$line"|cut -d, -f4); h=$(echo "$line"|cut -d, -f5)
      echo "$MODE,$r,$s,$h" >> $O
    done; cat $O ;;
  babel)
    if [ "$MODE" = "native" ]; then B=/benchmarks/cuda-stream-native; else B=/benchmarks/cuda-stream-fe; fi
    O=$OUTDIR/babel_${MODE}.csv; echo "mode,rep,kernel,n_elements,mbps" > $O
    SIZES="262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864"
    timeout 120 $B -s 262144 -n 50 --csv >/dev/null 2>&1   # warmup
    for r in $(seq 1 $REPS); do
      for s in $SIZES; do
        out=$(timeout 360 $B -s $s -n 100 --csv 2>/dev/null) || continue
        echo "$out"|awk -v m="$MODE" -v r="$r" -F, '$1!="function" && NF>=8 {print m","r","$1","$3","$5}' >> $O
      done
      echo "[babel] mode=$MODE rep=$r done" >&2
    done; echo "rows: $(wc -l < $O)" ;;
  transfer)
    if [ "$MODE" = "native" ]; then B=/benchmarks/transfer_bw2_native; else B=/benchmarks/transfer_bw2; fi
    MAXB=${MAXBYTES:-268435456}
    O=$OUTDIR/transfer_${MODE}.csv; echo "mode,rep,dir,bytes,gbps" > $O
    MINBYTES=4096 timeout 300 $B $MAXB 20 >/dev/null 2>&1   # warmup
    for r in $(seq 1 $REPS); do
      out=$(MINBYTES=4096 timeout 300 $B $MAXB 20 2>/dev/null) || continue
      echo "$out"|awk -v m="$MODE" -v r="$r" -F, '$1!="dir" && NF>=5 {print m","r","$1","$2","$5}' >> $O
      echo "[transfer] mode=$MODE rep=$r done" >&2
    done; echo "rows: $(wc -l < $O)" ;;
esac
