set -u
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_CONFIG=/gvirtus/etc/properties.json
export GVIRTUS_LOGLEVEL=60000
export LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:/usr/local/gvirtus/lib:/benchmarks/llama.cpp/build_cuda/bin
BENCH=/benchmarks/llama.cpp/build_cuda/bin/llama-bench
MODEL=/benchmarks/models/tinyllama-1.1b-q4.gguf
OUT=/benchmarks/gvs-clean/gvs-clean-llama.csv
cd /benchmarks/llama.cpp
echo "bench,transport,model,graphs,test,mean_tok_s,note" > "$OUT"
# graphs ENABLED (default)
og=$(timeout 600 "$BENCH" -m "$MODEL" -ngl 99 -p 128 -n 128 -r 5 2>/dev/null | grep -avE "UCX DEBUG|PROFILE")
echo "$og"
ppg=$(echo "$og" | awk -F"|" "/pp128/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
tgg=$(echo "$og" | awk -F"|" "/tg128/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "llama,tcp,tinyllama-1.1b-q4,on,pp128,$ppg,cuda_graphs_enabled_clean_master" >> "$OUT"
echo "llama,tcp,tinyllama-1.1b-q4,on,tg128,$tgg,cuda_graphs_enabled_clean_master" >> "$OUT"
# graphs DISABLED (comparison)
od=$(GGML_CUDA_DISABLE_GRAPHS=1 timeout 600 "$BENCH" -m "$MODEL" -ngl 99 -p 128 -n 128 -r 5 2>/dev/null | grep -avE "UCX DEBUG|PROFILE")
ppd=$(echo "$od" | awk -F"|" "/pp128/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
tgd=$(echo "$od" | awk -F"|" "/tg128/ {print \$(NF-1)}" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "llama,tcp,tinyllama-1.1b-q4,off,pp128,$ppd,graphs_disabled_clean_master" >> "$OUT"
echo "llama,tcp,tinyllama-1.1b-q4,off,tg128,$tgd,graphs_disabled_clean_master" >> "$OUT"
echo "[llama] DONE graphs-on tg128=$tgg | graphs-off tg128=$tgd"

