# miniBUDE over GVirtuS

[miniBUDE](https://github.com/UoB-HPC/miniBUDE) is a **compute-bound** molecular-docking mini-app (the
Bristol University Docking Engine proxy). It runs a few long, arithmetic-heavy kernels with very
little host↔device traffic, and reports GFLOP/s.

This example runs miniBUDE's CUDA model **through GVirtuS**: the client links the GVirtuS frontend
stubs instead of the real CUDA runtime, so every CUDA call is forwarded to a remote backend that owns
the GPU.

> **Why it's in the suite:** miniBUDE is the **compute-bound extreme** of the roofline — its cost is
> almost entirely on-GPU compute, not RPCs or transfers. Over GVirtuS it runs at **~native** speed
> (<0.2% overhead), so it is the control that shows GVirtuS (and the async dispatcher) add **no
> regression** where the workload isn't launch-/transfer-bound. It brackets the roofline together with
> llama (the launch-bound extreme).

## Files
| File | Purpose |
|------|---------|
| `setup.sh` | Clone miniBUDE + configure it (generates `build/generated`), run once on host. |
| `frontend.sh` | Compile against GVirtuS frontend stubs + run (inside the frontend container). |
| `backend.sh` | Launch the GVirtuS backend with `properties_ucx.json`. |
| `Dockerfile` | Builds a self-contained image (GVirtuS + miniBUDE). |

## Quick start
```bash
# 0) One-time: fetch + configure miniBUDE (on the host, in this directory)
cd examples/minibude && ./setup.sh

# 1) Backend node (owns the GPU): start the GVirtuS backend
#    (e.g. `make run-gvirtus-backend-dev`, which serves properties_ucx.json)

# 2) Frontend node (GPU-less client): build + run miniBUDE over GVirtuS
bash examples/minibude/frontend.sh
#    async on/off is expected to be ~identical (compute-bound):
GVIRTUS_ASYNC_DISPATCH=1 bash examples/minibude/frontend.sh
```

Tunables (env): `DECK` (default `data/bm1`), `ITER` (default 8), `CUDA_ARCH` (default sm_89),
`GVIRTUS_LOGLEVEL` (default 40000 = ERROR). Transport is selected via `etc/properties_ucx.json` +
the `UCX_*` variables (TCP / RDMA / RDMA+GPUDirect).

## Notes
- Build the frontend from the **same GVirtuS source** as the backend.
- Check `valid: true` in the output — miniBUDE self-validates the docking energies.
- Measured result: ~216 GFLOP/s over GVirtuS RDMA, `valid: true`, **identical async on/off** and
  ≈ native — the expected no-regression / transport-insensitive outcome. Data:
  `benchmarks/miniBUDE-{sync,async}/`.
