# UCX GPU Support and GVirtuS Notes

## Summary
GVirtuS already uses a remote GPU by executing CUDA calls on the backend. UCX is only the transport for the serialized RPC payloads (host buffers). UCX GPU transports matter only when UCX is asked to move GPU pointers directly. If UCX is not configured with CUDA transports, GPU memory will be treated as host memory and the path will fall back to staging.

Sources:
- UCX FAQ (GPU support and UCX_TLS requirements): https://openucx.readthedocs.io/en/master/faq.html

## GPU Support Requirements
To enable UCX GPU support (GPUDirect RDMA and CUDA-aware transports), UCX must be built with CUDA support and the system must have the required kernel modules.

From the UCX NVIDIA GPU support wiki:
- UCX built with CUDA support (for CUDA memory transports).
- GPUDirect RDMA kernel module `nv_peer_mem` (or equivalent in your stack).
- Optional `gdrcopy` for faster host<->GPU copies.

Source:
- NVIDIA GPU Support (openucx/ucx wiki): https://github-wiki-see.page/m/openucx/ucx/wiki/NVIDIA-GPU-Support

## UCX TLS and GPU Memory Detection
When overriding `UCX_TLS`, UCX requires CUDA transports to detect and handle GPU memory. The FAQ explicitly notes that you must include `cuda` in `UCX_TLS` for GPU memory support when you set `UCX_TLS` yourself.

Source:
- UCX FAQ (Working with GPU / UCX_TLS): https://openucx.readthedocs.io/en/master/faq.html

## Diagnostics
Use `ucx_info` to verify CUDA transports are available:

```
ucx_info -d | grep -E "cuda|gdr"
```

Source:
- UCX FAQ (GPU support troubleshooting): https://openucx.readthedocs.io/en/master/faq.html

## Crucial UCX Flags (with notes)
These are the most relevant flags for UCX + RDMA + GPU scenarios. Most are advanced; change carefully.

- `UCX_TLS`: transport selection. For GPU-aware UCX, include `cuda` (for example, `cuda_copy`).
  - Source: UCX FAQ (GPU support and transport selection) https://openucx.readthedocs.io/en/master/faq.html
- `UCX_NET_DEVICES`: restrict UCX to a specific NIC or HCA (for example `mlx5_1:1`).
  - Source: UCX FAQ (network device selection) https://openucx.readthedocs.io/en/master/faq.html
- `UCX_IB_GID_INDEX`: required for RoCE to select the correct GID index.
  - Source: UCX FAQ (RoCE / GID index) https://openucx.readthedocs.io/en/master/faq.html
- `UCX_MEMTYPE_CACHE`: CUDA memory hook cache (may need to disable for some setups).
  - Sources: UCX-Py config (memory settings) https://ucx-py.readthedocs.io/en/latest/configuration.html
  - UCX NVIDIA GPU support wiki (known issues) https://github-wiki-see.page/m/openucx/ucx/wiki/NVIDIA-GPU-Support
- `UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda`: enable registration cache per CUDA allocation; helps when CUDA memory pools are used.
  - Source: UCX-Py config (memory registration) https://ucx-py.readthedocs.io/en/latest/configuration.html
- `UCX_RNDV_SCHEME`: GPUDirect RDMA tuning often uses `get_zcopy`.
  - Source: UCX NVIDIA GPU support wiki https://github-wiki-see.page/m/openucx/ucx/wiki/NVIDIA-GPU-Support
- `UCX_RNDV_THRESH`: controls when UCX switches to rendezvous protocols; helpful for large transfers.
  - Source: UCX-Py config (rendezvous thresholds) https://ucx-py.readthedocs.io/en/latest/configuration.html
- `UCX_PROTO_ENABLE` and `UCX_PROTO_INFO`: protocol selection and diagnostics (useful for confirming zcopy vs bcopy).
  - Source: UCX-Py config (protocol settings) https://ucx-py.readthedocs.io/en/latest/configuration.html

## Benchmark Config Settings (complete list)
The simple_matrix benchmark script enumerates all environment variables it uses. See [examples/simple_matrix/benchmark.sh](examples/simple_matrix/benchmark.sh).

GVirtuS settings:
- `GVIRTUS_HOME`: root path for GVirtuS in the benchmark container (default `/opt/GVirtuS`).
- `GVIRTUS_CONFIG`: backend config file selected by mode.
- `GVIRTUS_UCX_DATAPATH`: UCX data path for the frontend (set to `am` for UCX modes).
- `GVIRTUS_LOGLEVEL`: log4cplus level for GVirtuS (`30000` by default).
- `LD_LIBRARY_PATH`: prepended with GVirtuS frontend and lib paths for the benchmark.

UCX transport selection:
- `UCX_TLS`: transport list, include `cuda_copy` for CUDA-aware UCX.
- `UCX_NET_DEVICES`: restrict UCX to a specific NIC or HCA.
- `UCX_SOCKADDR_TLS_PRIORITY`: socket transport preference (`tcp` or `rdmacm`).
- `UCX_IB_GID_INDEX`: RoCE GID index (required on RoCE setups).
- `UCX_LOG_LEVEL`: UCX log verbosity (`info` by default).

UCX protocol and memory settings:
- `UCX_PROTO_ENABLE`: enable protocol selection (`y` by default).
- `UCX_PROTO_INFO`: print protocol info (`y` by default).
- `UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES`: register whole CUDA allocations (`cuda` by default).
- `UCX_MEMTYPE_CACHE`: enable CUDA memory hook cache (`y` by default).
- `UCX_RNDV_SCHEME`: rendezvous scheme (`get_zcopy` by default).
- `UCX_RNDV_THRESH`: rendezvous threshold (`inf` by default).

Benchmark controls:
- `MATRIX_SIZES`: space-separated sizes to test (`"1024 2048 4096"` by default).
- `ITERATIONS`: iterations per size (`10` by default).
- `WARMUP`: warmup iterations (`1` by default).
- `MODES`: modes to run (`"tcp ucx-tcp ucx-rdma ucx-mixed"` by default).
- `VARIANTS`: allocation variants (`"transport regcost"` by default).
- `ALLOCATION_MODE`: set per variant (`reuse` or `per_iter`).

Diagnostics and UI:
- `UCX_DIAG`: run `ucx_info` before each mode (`1` by default).
- `LOG_STDOUT`: echo benchmark output to stdout (`1` by default).
- `PROMPT`: prompt to start the backend for each mode (`1` by default).

Testbed constants (edit in script if hardware changes):
- `UCX_TCP_DEV`: TCP interface (default `ens1f1np1`).
- `UCX_RDMA_DEV`: RDMA device (default `mlx5_1:1`).
- `UCX_MIXED_DEVS`: combined RDMA+TCP device list.
- `UCX_GID_INDEX`: RoCE GID index (default `3`).

## Example UCX Modes
These examples assume UCX is CUDA-aware. If not, remove the CUDA transports or rebuild UCX with CUDA support.

### UCX RDMA
```
export UCX_TLS="rc_mlx5,ud_mlx5,self,cuda_copy"
export UCX_NET_DEVICES="mlx5_1:1"
export UCX_SOCKADDR_TLS_PRIORITY="rdmacm"
export UCX_IB_GID_INDEX="3"
```

### UCX TCP
```
export UCX_TLS="tcp,self,cuda_copy"
export UCX_NET_DEVICES="ens1f1np1"
export UCX_SOCKADDR_TLS_PRIORITY="tcp"
```

### UCX Mixed (RDMA + TCP)
```
export UCX_TLS="rc_mlx5,ud_mlx5,tcp,self,cuda_copy"
export UCX_NET_DEVICES="mlx5_1:1,ens1f1np1"
export UCX_SOCKADDR_TLS_PRIORITY="tcp"
export UCX_IB_GID_INDEX="3"
```

## GVirtuS-Specific Notes
GVirtuS serializes CUDA API calls and sends host buffers across the network. Even with UCX GPU support, the transport only moves host-side payloads unless the communicator is extended to send GPU pointers. So enabling CUDA transports helps only if the transport actually sees GPU memory.

To benefit from true GPU-direct transfers, the data path must carry GPU pointers or use an explicit GPU-aware communication layer. Otherwise, the performance limiter is typically synchronization and RPC overhead rather than memory registration.
