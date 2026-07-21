# BabelStream — baseline (GVirtuS TCP communicator)

**Placeholder — to be run later.**

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../BabelStream-sync/` (UCX, async off) and `../BabelStream-async/` (UCX, async on).

## How to run (later)
Build/adapt BabelStream via `examples/babelstream/setup.sh`, then sweep sizes over the **TCP** config:
```bash
# backend (es-dpu-01):
gvirtus-backend $GVIRTUS_HOME/etc/properties.json      # suite tcp/ip, port 32222
# frontend (es-dpu-02):
for s in 262144 1048576 4194304 16777216 67108864; do
  GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json GVIRTUS_LOGLEVEL=40000 \
    cuda-stream -s $s -n 100 --csv
done
```
Save the CSV rows (tag the config) + backend evidence here.
