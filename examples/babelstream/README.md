# BabelStream over GVirtuS

[BabelStream](https://github.com/UoB-HPC/BabelStream) is the standard GPU
**memory-bandwidth** benchmark (the accelerator descendant of STREAM). It runs five
kernels — Copy, Mul, Add, Triad, Dot — over arrays much larger than cache and reports
sustained bandwidth in GB/s.

This example runs BabelStream's CUDA model **through GVirtuS**: the client links the
GVirtuS frontend stubs instead of the real CUDA runtime, so every CUDA call is forwarded
to a remote backend that owns the GPU.

> **Why it's a good GVirtuS benchmark:** the kernels run on the remote GPU while the arrays
> stay in GPU memory, so BabelStream mainly exercises (a) the **control path** — one RPC per
> kernel launch — and (b) the initial/final large H2D/D2H array transfers (the RDMA /
> GPUDirect data path), while establishing the device-bandwidth ceiling.

## Files
| File | Purpose |
|------|---------|
| `setup.sh` | Clone BabelStream + apply the GVirtuS adaptation (run once, on host). |
| `frontend.sh` | Compile against GVirtuS frontend stubs + run (inside the frontend container). |
| `backend.sh` | Launch the GVirtuS backend with `properties_ucx.json`. |
| `nvml_shim.cpp` | No-op NVML shim (cosmetic startup print only — see below). |
| `Dockerfile` | Builds a self-contained image (GVirtuS + adapted BabelStream). |

## Quick start

```bash
# 0) One-time: fetch + adapt BabelStream (on the host, in this directory)
cd examples/babelstream && ./setup.sh

# 1) Backend node (owns the GPU): start the GVirtuS backend
#    (e.g. `make run-gvirtus-backend-dev`, which serves properties_ucx.json)

# 2) Frontend node (GPU-less client): build + run BabelStream over GVirtuS
make run-babelstream-test
#    or, inside a frontend container:  bash examples/babelstream/frontend.sh
```

Tunables (env): `SIZE` (elements, default 33554432), `ITERS` (default 100),
`CUDA_ARCH` (default `sm_89`), `GVIRTUS_LOGLEVEL` (default 40000 = ERROR).
Transport is selected exactly like the other examples via `etc/ucx.env` /
`etc/properties_ucx.json` and the `UCX_*` variables (TCP / RDMA / RDMA+GPUDirect).

## GVirtuS adaptation (important)

BabelStream's dot-reduction partial-sums buffer is normally allocated with
`cudaHostAlloc` (pinned host memory) and written **directly by the kernel** (zero-copy).
GVirtuS implements `cudaHostAlloc` as a **frontend-local `malloc`** — it allocates nothing
on the backend — so that host address is not valid device memory on the backend GPU and the
kernel faults with `CUDA error 700 (illegal memory access)`.

`setup.sh` therefore applies a small, functionally-identical change to
`BabelStream/src/cuda/CUDAStream.cu` (marked `// GVirtuS`): allocate `sums` in **device
memory** (`cudaMalloc`) and copy it back with an explicit `cudaMemcpy` before the host-side
reduction. This does not change what BabelStream measures (the big-array bandwidth); it only
moves a 4.5 KB reduction result off the zero-copy path.

> **Documented GVirtuS limitation:** zero-copy pinned-host memory (`cudaHostAlloc` /
> `cudaMallocHost`) accessed directly inside a kernel is not supported over remoting,
> because host memory is not shared between the frontend and backend address spaces.

## NVML shim

BabelStream calls a few NVML functions only to print a theoretical "PEAK" bandwidth line at
startup. GVirtuS' nvml plugin doesn't implement them, and NVML is not part of the measured
path, so we link `nvml_shim.cpp` (no-ops) instead of `-lnvidia-ml`. The printed PEAK line is
therefore not meaningful under GVirtuS.

## Notes
- Build the frontend from the **same GVirtuS source** as the backend. A frontend built from
  an older image can desync the wire protocol.
- The GVirtuS backend keeps a single persistent CUDA context; if a run faults (700), restart
  the backend before the next run to clear the poisoned context.
