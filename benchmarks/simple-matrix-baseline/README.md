# simple-matrix — baseline (GVirtuS TCP communicator)

**Placeholder — to be run later.**

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../simple-matrix-sync/` (UCX, async off) and `../simple-matrix-async/` (UCX, async on).

> simple_matrix is bulk-transfer-bound (~1 GB/matrix). The report's TCP figure (N=16384, ~2210 ms)
> is a legacy-TCP-style reference — the current TCP-communicator numbers go here.

## How to run (later)
```bash
# backend (es-dpu-01):
gvirtus-backend $GVIRTUS_HOME/etc/properties.json      # suite tcp/ip, port 32222
# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json GVIRTUS_LOGLEVEL=40000 \
  ./simple_matrix 16000 5 1        # emits CSV,n,iters,sgemm_ms,host_ms
```
Save the CSV line + backend evidence here.
