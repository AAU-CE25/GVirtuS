# simple-matrix — baseline (GVirtuS TCP communicator)

## Results (measured)

N=16000, 5 iters, `GVIRTUS_LOGLEVEL=40000`, TCP communicator (`properties_tcp.json`, port 32222).
Raw: [`tcp_matrix.csv`](tcp_matrix.csv).

| config | sgemm_ms | host_ms (per iter) |
|--------|----------|--------------------|
| **baseline — TCP communicator** | **156.6** | **1352.6** |
| `../simple-matrix-sync/` — UCX, async off | ~157 | ~738 |
| `../simple-matrix-async/` — UCX, async on | ~157 | ~726 |

simple_matrix is bulk-transfer-bound (~1 GB/matrix). The device SGEMM is identical everywhere
(~157 ms), but the end-to-end host time is dominated by the H2D/D2H transfer: the TCP data path is
~1.8× slower than the UCX communicator (1352 vs ~738 ms). Async gives little here because the copies
are large synchronous `cudaMemcpy` (not the fire-and-forget async ops), which is why `-sync` and
`-async` are close — the win comes from the UCX communicator itself.

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../simple-matrix-sync/` (UCX, async off) and `../simple-matrix-async/` (UCX, async on).

> simple_matrix is bulk-transfer-bound (~1 GB/matrix). The report's TCP figure (N=16384, ~2210 ms)
> is a legacy-TCP-style reference — the current TCP-communicator numbers go here.

## How to reproduce
```bash
# backend (es-dpu-01):  BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json  # suite tcp/ip, port 32222
# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json GVIRTUS_LOGLEVEL=40000 \
  ./simple_matrix 16000 5 1        # emits CSV,n,iters,sgemm_ms,host_ms
```
