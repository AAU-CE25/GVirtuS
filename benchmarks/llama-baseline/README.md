# llama — baseline (GVirtuS TCP communicator)

**Placeholder — to be run later.**

`-baseline` = GVirtuS over the **legacy TCP communicator** (`tcp/ip` suite, `etc/properties.json`,
port 32222) — i.e. GVirtuS **without** the UCX communicator. It is the pre-UCX reference that isolates
the **UCX communicator's** contribution:

- `-baseline` (here) — TCP communicator (`properties.json`).
- `../llama-sync/` — same optimized frontend over the **UCX** communicator, `GVIRTUS_ASYNC_DISPATCH=0`.
- `../llama-async/` — UCX communicator, `GVIRTUS_ASYNC_DISPATCH=1`.

## How to run (later)
Launch the backend with the **TCP** config and run the client over the TCP communicator (no `UCX_*`
env, no UCX datapath):
```bash
# backend (es-dpu-01):
gvirtus-backend $GVIRTUS_HOME/etc/properties.json      # suite tcp/ip, port 32222

# frontend (es-dpu-02):
GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json GVIRTUS_LOGLEVEL=40000 \
  llama-bench -m tinyllama-1.1b-q4.gguf -ngl 99 -p 8 -n 16 -r 3
# GGML_CUDA_DISABLE_GRAPHS=1 ; capture stdout CSV + backend docker logs here.
```
> For reference, TCP over the **UCX** communicator gave tg16 ~3.6 t/s baseline (RESULTS.md §5);
> the legacy TCP-communicator numbers go here.
