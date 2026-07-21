# miniBUDE — baseline (GVirtuS TCP communicator)

## Results (measured)

`data/bm1`, iter 8, `GVIRTUS_LOGLEVEL=40000`, TCP communicator (`properties_tcp.json`, port 32222).
Raw: [`tcp_minibude.csv`](tcp_minibude.csv).

| config | GFLOP/s | context_ms | valid |
|--------|---------|-----------|-------|
| **baseline — TCP communicator** | **216.385** | **3.075** | true |

As expected for a compute-bound kernel, throughput is essentially GPU-limited and matches the UCX
`-sync`/`-async` runs — the communicator choice only shows in the one-time setup transfer
(`context_ms`), not in GFLOP/s. This confirms miniBUDE is insensitive to the transport, so async
brings no benefit here (and no regression).

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../miniBUDE-sync/` (UCX, async off) and `../miniBUDE-async/` (UCX, async on).

> miniBUDE is compute-bound, so the communicator choice mostly shows in the one-time setup transfer,
> not throughput — this baseline quantifies the TCP-communicator setup cost vs UCX.

## How to reproduce
```bash
# backend (es-dpu-01):  BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json  # suite tcp/ip, port 32222
# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json GVIRTUS_LOGLEVEL=40000 \
  cuda-bude-gvirtus --deck data/bm1 --iter 8
```
