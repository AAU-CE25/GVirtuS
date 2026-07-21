# BabelStream — baseline (clean GVirtuS)

**Placeholder — to be run later.**

`-baseline` = **clean/stock GVirtuS** (upstream, no async dispatcher / RPC optimizations) —
the unoptimized remoting reference. See `../BabelStream-sync/` (optimized, async off) and
`../BabelStream-async/` (async on).

## How to run (later)
Build/adapt BabelStream via `examples/babelstream/setup.sh`, then sweep sizes:
```bash
for s in 262144 1048576 4194304 16777216 67108864; do
  GVIRTUS_LOGLEVEL=40000 cuda-stream -s $s -n 100 --csv
done
```
Save the CSV rows (tag the config) + backend evidence in this folder.
