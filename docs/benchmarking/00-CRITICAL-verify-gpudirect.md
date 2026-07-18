# ⚠️ CRITICAL RULE — ALWAYS VERIFY GPUDIRECT IS ACTUALLY WORKING ⚠️

**NEVER assume GPUDirect (or any transport mode) is active. NEVER report a GPUDirect result
without positive, logged proof that the GPU peer-DMA path actually carried the data.**

This rule exists because we were burned twice:
1. We first reported "GPUDirect ≈ RDMA, no benefit" — WRONG (a benchmark artifact masked it).
2. We then reported "GPUDirect crashes ≥8 MiB" — WRONG (oversized-reused-buffer bug, not
   GPUDirect; simple_matrix runs GPUDirect fine at 976 MB per cudaMemcpy).
Both came from ASSUMING the mode was engaged/broken instead of VERIFYING. The real, correct
result (GPUDirect ~2.3× D2H speedup) only appeared once we verified properly.

## Mandatory verification checklist before trusting ANY GPUDirect number

1. **Backend probe must say ENABLED** — backend log MUST contain:
   `[GVS] GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK ...)`
   If it says `GPUDirect=disabled` (probe FAILED / env not set) → the run is NOT GPUDirect.
2. **Backend must be launched with `GVIRTUS_GPUDIRECT=1`** (our launcher:
   `GVIRTUS_GPUDIRECT=1 bash /tmp/gvirtus-backend-run.sh`). `=0` (or unset) = plain RDMA.
3. **RDMA transport must actually be selected** — `UCX_TLS=rc_mlx5,...` (NOT `tcp,self`).
   GPUDirect is impossible on TCP.
4. **Confirm the data path in the backend log** — for a real transfer, look for
   `WriteIovRma(zerocopy) ... big=<bytes>` (RMA/`ucp_put`), and for GPU shadows in the
   rx_pool init: `rx_pool: initialized N slots ... + N/N GPU shadows`.
5. **Cross-check against a known-good app** — if a result looks surprising, reproduce with
   `simple_matrix` at large N (e.g. N=16000 = 976 MB/transfer) BEFORE claiming a framework bug.
6. **Isolate the variable** — plain-RDMA config MUST use a `GVIRTUS_GPUDIRECT=0` backend, and
   the GPUDirect config a `GVIRTUS_GPUDIRECT=1` backend, so the two differ ONLY in GPUDirect.
7. **Restart the backend (fresh CUDA context) between GD=0 and GD=1 phases** — a poisoned
   context (after any CUDA 700 / crash) silently invalidates subsequent runs.

## Known caveat that can silently disable GPUDirect
The **frontend** GPUDirect probe currently FAILS (it dlopens the GVirtuS stub `libcudart`, not
the real one) → the client advertises `0 slots with gpu shadow`. For host-source transfers this
is fine (the meaningful GPUDirect is backend-side, since GVirtuS clients are GPU-less by
design), but it means: **the backend `GVIRTUS_GPUDIRECT` flag + backend log is the source of
truth**, not the frontend. Verify on the BACKEND.

## One-line honesty rule
If you cannot point to a backend log line proving GPUDirect engaged for THIS run, do not write
"GPUDirect" in the results. Write "unverified" and go verify.
