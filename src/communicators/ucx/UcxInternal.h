/*
 * UcxCommunicator internal helpers shared across UCX translation units.
 *
 * The UCX communicator's implementation is split across several .cpp files:
 *
 *   UcxCommunicator.cpp  — the Communicator interface (Connect/Serve/Read/...)
 *   UcxRma.cpp           — the RDMA fast path (send_rma_setup, WriteIovRma)
 *   UcxRxPool.cpp        — the pinned RX slot pool and TX scratch buffer
 *   UcxGpu.cpp           — CUDA dlopen helpers + GPUDirect probe + flag
 *
 * This header exposes the small set of free helpers that those files share.
 * Definitions live in the .cpp that owns the underlying state, so each
 * once_flag / atomic has exactly one home.
 */
#pragma once

#include <cstddef>
#include <string>

// Forward-declare the UCX opaque context handle so this header does NOT
// transitively pull in <ucp/api/ucp.h>. Every UCX .cpp already includes
// the real UCX header before including this one.
typedef struct ucp_context *ucp_context_h;

namespace gvirtus::communicators::ucx_internal {

// ---- Debug logging (defined in UcxCommunicator.cpp) ---------------------

// True iff GVIRTUS_LOGLEVEL is set to DEBUG or TRACE (numeric <= 10000).
bool ucx_debug_enabled();

// printf-style debug logger. No-ops when ucx_debug_enabled() is false.
void ucx_debug_log(const char *fmt, ...);

// ---- CUDA / GPUDirect helpers (defined in UcxGpu.cpp) -------------------

// Allocate `n` bytes of pinned host memory. `is_cuda` is set to true if the
// buffer was allocated via cudaHostAlloc (so cudaFreeHost is needed for
// release). Falls back to posix_memalign when libcudart is unavailable.
unsigned char *alloc_pinned_host(std::size_t n, bool &is_cuda);

// Release a buffer allocated via alloc_pinned_host.
void free_pinned_host(unsigned char *p, bool is_cuda);

// Allocate `n` bytes of CUDA device memory. Returns nullptr if libcudart is
// unavailable or cudaMalloc fails.
unsigned char *alloc_gpu_slot(std::size_t n);

// Release a buffer allocated via alloc_gpu_slot. No-op for nullptr.
void free_gpu_slot(unsigned char *p);

// Detect whether `p` is a CUDA device or managed pointer. Returns false on
// host memory, unregistered memory, NULL, or if cudaPointerGetAttributes is
// unavailable. Short-circuits to false when GPUDirect is not active (so the
// frontend never triggers an unwanted cudaPointerGetAttributes RPC).
bool is_gpu_pointer(const void *p);

// Probe: try cudaMalloc(4K) + ucp_mem_map(CUDA) + cleanup. Returns true iff
// peermem + UCX-CUDA cooperate in this process. `reason` is populated with
// the failure description on false.
bool probe_gpudirect(ucp_context_h ctx, std::string &reason);

// Set/get the process-wide GPUDirect flag (true iff GVIRTUS_GPUDIRECT=1 and
// the probe succeeded). Used by RX pool slot allocation to decide whether
// to allocate GPU shadow regions alongside host slots, and by is_gpu_pointer
// as a short-circuit guard.
void set_gpudirect_enabled(bool ok);
bool gpudirect_enabled();

}  // namespace gvirtus::communicators::ucx_internal
