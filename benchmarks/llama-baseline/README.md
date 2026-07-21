# llama — baseline (GVirtuS TCP communicator)

## Results (measured)

TinyLlama-1.1B Q4_K_M, `-ngl 99 -p 8 -n 16 -r 2`, `GVIRTUS_LOGLEVEL=40000`, TCP communicator
(`properties_tcp.json`, suite `tcp/ip`, port 32222). Raw: [`tcp_llama.log`](tcp_llama.log).

| config | pp8 (t/s) | tg16 (t/s) |
|--------|-----------|------------|
| **baseline — TCP communicator** | **163.17 ± 14.63** | **25.13 ± 1.30** |
| `../llama-sync/` — UCX, async off | 250.5 | 40.65 |
| `../llama-async/` — UCX, async on | 528.9 | 87.62 |

The TCP communicator is the slowest of the three: its higher per-RPC latency hurts the launch-bound
decode path most (tg16 25→40 just from switching to UCX; 40→87 from the async dispatcher on top —
**3.5× total** over this baseline). Correct output verified ("…the capital of France is Paris").

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — i.e. GVirtuS **without** the UCX communicator. It is the pre-UCX reference that isolates
the **UCX communicator's** contribution:

- `-baseline` (here) — TCP communicator (`properties.json`).
- `../llama-sync/` — same optimized frontend over the **UCX** communicator, `GVIRTUS_ASYNC_DISPATCH=0`.
- `../llama-async/` — UCX communicator, `GVIRTUS_ASYNC_DISPATCH=1`.

## How to reproduce
Launch the backend with the **TCP** config and run the client over the TCP communicator (no `UCX_*`
env, no UCX datapath):
```bash
# backend (es-dpu-01):  BACKEND_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json  # suite tcp/ip, port 32222
# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties_tcp.json GVIRTUS_LOGLEVEL=40000 \
  llama-bench -m tinyllama-1.1b-q4.gguf -ngl 99 -p 8 -n 16 -r 2
```
