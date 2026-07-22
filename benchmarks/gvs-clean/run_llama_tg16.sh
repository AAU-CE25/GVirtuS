set -u
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties.json
export GVIRTUS_LOGLEVEL=60000
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:/usr/local/gvirtus/lib:/benchmarks/llama.cpp/build_cuda/bin
BENCH=/benchmarks/llama.cpp/build_cuda/bin/llama-bench
MODEL=/benchmarks/models/tinyllama-1.1b-q4.gguf
OUT=/benchmarks/gvs-clean/gvs-clean-llama-tg16.csv
cd /benchmarks/llama.cpp
echo "bench,transport,model,graphs,test,mean_tok_s,note" > "$OUT"
og=$(timeout 600 "$BENCH" -m "$MODEL" -ngl 99 -p 16 -n 16 -r 5 2>/dev/null | grep -avE "UCX DEBUG|PROFILE")
echo "$og"
tgg=$(echo "$og" | awk -F"|" "/tg16/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
ppg=$(echo "$og" | awk -F"|" "/pp16/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "llama,tcp,tinyllama-1.1b-q4,on,pp16,$ppg,cuda_graphs_enabled_clean_master" >> "$OUT"
echo "llama,tcp,tinyllama-1.1b-q4,on,tg16,$tgg,cuda_graphs_enabled_clean_master" >> "$OUT"
od=$(GGML_CUDA_DISABLE_GRAPHS=1 timeout 600 "$BENCH" -m "$MODEL" -ngl 99 -p 16 -n 16 -r 5 2>/dev/null | grep -avE "UCX DEBUG|PROFILE")
echo "$od"
tgd=$(echo "$od" | awk -F"|" "/tg16/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
ppd=$(echo "$od" | awk -F"|" "/pp16/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "llama,tcp,tinyllama-1.1b-q4,off,pp16,$ppd,graphs_disabled_clean_master" >> "$OUT"
echo "llama,tcp,tinyllama-1.1b-q4,off,tg16,$tgd,graphs_disabled_clean_master" >> "$OUT"
echo "[tg16] DONE on=$tgg off=$tgd"

