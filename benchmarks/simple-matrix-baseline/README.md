# simple-matrix — baseline (clean GVirtuS)

**Placeholder — to be run later.**

`-baseline` = **clean/stock GVirtuS** (upstream, no async dispatcher / RPC optimizations) —
the unoptimized remoting reference. See `../simple-matrix-sync/` (optimized, async off) and
`../simple-matrix-async/` (async on).

> simple_matrix is bulk-transfer-bound (cuBLAS SGEMM + synchronous `cudaMemcpy`), so async has
> no effect; this baseline confirms no regression and provides a stock-GVirtuS reference for the
> ~1 GB/matrix transfer path.

## How to run (later)
```bash
GVIRTUS_LOGLEVEL=40000 ./simple_matrix 16000 5 1   # n iters warmup ; emits CSV,n,iters,sgemm_ms,host_ms
```
Save the CSV line + backend evidence in this folder.
