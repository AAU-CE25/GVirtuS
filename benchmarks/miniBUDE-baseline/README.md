# miniBUDE — baseline (clean GVirtuS)

**Placeholder — to be run later.**

`-baseline` = **clean/stock GVirtuS** (upstream, no async dispatcher / RPC optimizations) —
the unoptimized remoting reference. See `../miniBUDE-sync/` (optimized, async off) and
`../miniBUDE-async/` (async on).

> miniBUDE is compute-bound, so all three are expected to be ≈ native — this baseline mainly
> confirms the async/opt work introduces no regression on compute-bound workloads.

## How to run (later)
```bash
GVIRTUS_LOGLEVEL=40000 cuda-bude-gvirtus --deck data/bm1 --iter 8
```
Save the raw output (gflop/s, valid:) + backend evidence in this folder.
