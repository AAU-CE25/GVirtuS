# BabelStream — baseline (GVirtuS TCP communicator)

## Results (measured)

Size sweep 2¹⁸…2²⁶ elements (double), 100 iters each, `GVIRTUS_LOGLEVEL=40000`, TCP communicator
(`properties_tcp.json`, port 32222). Raw (45 rows, all 5 kernels × 9 sizes):
[`tcp_babel.csv`](tcp_babel.csv).

Peak sustained bandwidth at the largest size (2²⁶ = 512 MB arrays):

| kernel | MB/s @ 2²⁶ |
|--------|-----------|
| Copy | 673 437 |
| Mul | 645 140 |
| Add | 664 136 |
| Triad | 663 519 |
| Dot | 648 869 |

BabelStream kernels are device-resident (no per-iteration host transfer), so at large sizes the TCP
baseline saturates GPU memory bandwidth (~663 GB/s Triad) and matches the UCX `-sync`/`-async` runs —
the transport is off the hot path. The launch-bound small sizes (e.g. 2¹⁸ Triad ≈ 75 GB/s) are where
per-RPC latency shows, and where the async dispatcher helps most (see `../BabelStream-async/`). No
2²² crash observed on this run.

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `properties_tcp.json`,
port 32222) — GVirtuS **without** the UCX communicator. Isolates the UCX communicator's contribution
vs `../BabelStream-sync/` (UCX, async off) and `../BabelStream-async/` (UCX, async on).

## How to reproduce
Build/adapt BabelStream via `examples/babelstream/setup.sh`, then sweep sizes over the **TCP** config:
```bash
# backend (es-dpu-01):  BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json  # suite tcp/ip, port 32222
# frontend (es-dpu-02):
for s in 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864; do
  GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json GVIRTUS_LOGLEVEL=40000 \
    cuda-stream -s $s -n 100 --csv
done
```
