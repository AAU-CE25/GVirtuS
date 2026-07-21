# miniBUDE — baseline (GVirtuS TCP communicator)

**Placeholder — to be run later.**

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../miniBUDE-sync/` (UCX, async off) and `../miniBUDE-async/` (UCX, async on).

> miniBUDE is compute-bound, so the communicator choice mostly shows in the one-time setup transfer,
> not throughput — this baseline quantifies the TCP-communicator setup cost vs UCX.

## How to run (later)
```bash
# backend (es-dpu-01):
gvirtus-backend $GVIRTUS_HOME/etc/properties.json      # suite tcp/ip, port 32222
# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json GVIRTUS_LOGLEVEL=40000 \
  cuda-bude-gvirtus --deck data/bm1 --iter 8
```
Save the raw output (gflop/s, valid:, context_ms) + backend evidence here.
