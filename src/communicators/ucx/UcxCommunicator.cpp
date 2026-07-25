#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <malloc.h>
#include <mutex>
#include <netdb.h>
#include <sys/socket.h>
#include <stdexcept>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"
#include "gvirtus/communicators/UcxAmProtocol.h"

using gvirtus::communicators::UcxCommunicator;

namespace {
constexpr unsigned kUcxAmId = 1;

bool ucx_debug_enabled() {
    const char *lvl = std::getenv("GVIRTUS_LOGLEVEL");
    if (lvl == nullptr) return false;

    char *end = nullptr;
    long val = std::strtol(lvl, &end, 10);
    if (end == lvl) return false;
    return val <= 10000;  // DEBUG or TRACE
}

void ucx_debug_log(const char *fmt, ...) {
    if (!ucx_debug_enabled()) return;

    std::fprintf(stderr, "[UCX DEBUG] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

// Pure AM path: tag transport is intentionally removed.

// libcudart resolver — dlopen at runtime so the communicator does not need
// to link against CUDA. Used only to allocate pinned host memory for the RX
// pool; if CUDA isn't available we fall back to plain posix_memalign and
// just lose the auto-DMA-fast-path benefit (cudaMemcpy from non-registered
// memory) — the zero-init avoidance still applies.
using cudaHostAlloc_t = int (*)(void **, size_t, unsigned);
using cudaFreeHost_t  = int (*)(void *);

std::once_flag g_cuda_once;
std::atomic<cudaHostAlloc_t> g_cuda_host_alloc{nullptr};
std::atomic<cudaFreeHost_t>  g_cuda_free_host{nullptr};

void load_cuda_pinned_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto a = reinterpret_cast<cudaHostAlloc_t>(dlsym(h, "cudaHostAlloc"));
        auto f = reinterpret_cast<cudaFreeHost_t>(dlsym(h, "cudaFreeHost"));
        if (a && f) {
            g_cuda_host_alloc.store(a);
            g_cuda_free_host.store(f);
            ucx_debug_log("rx_pool: loaded cudaHostAlloc from %s", candidates[i]);
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("rx_pool: cudaHostAlloc unavailable, falling back to posix_memalign");
}

// Allocate `n` bytes of pinned host memory. is_cuda set to true if the buffer
// was allocated via cudaHostAlloc (so cudaFreeHost is needed for release).
unsigned char *alloc_pinned_host(size_t n, bool &is_cuda) {
    std::call_once(g_cuda_once, load_cuda_pinned_funcs);
    auto fn = g_cuda_host_alloc.load();
    if (fn != nullptr) {
        void *p = nullptr;
        if (fn(&p, n, /*cudaHostAllocDefault*/ 0u) == 0 && p != nullptr) {
            is_cuda = true;
            return static_cast<unsigned char *>(p);
        }
    }
    void *p = nullptr;
    if (posix_memalign(&p, 4096, n) == 0 && p != nullptr) {
        is_cuda = false;
        return static_cast<unsigned char *>(p);
    }
    return nullptr;
}

void free_pinned_host(unsigned char *p, bool is_cuda) {
    if (p == nullptr) return;
    if (is_cuda) {
        auto fn = g_cuda_free_host.load();
        if (fn != nullptr) {
            fn(p);
            return;
        }
    }
    std::free(p);
}

// Allocate `n` bytes of GPU memory (cudaMalloc). Forward-declared usage —
// requires load_cuda_device_funcs() to have been called (probe_gpudirect
// triggers it via std::call_once). Returns nullptr on failure.
//
// Defined LATER in the file so it can see g_cuda_malloc; here we forward-
// declare both to avoid reordering the whole anonymous-namespace section.
unsigned char *alloc_gpu_slot(size_t n);

// The payload size at or above which the RMA path is worth its RmaSetup/RmaPosted
// handshake (measured crossover, see WriteIov). Shared by the sender, which decides
// whether to use RMA, and the receiver, which uses it to decide when a connection has
// proved it needs a slot pool at all.
// Configured full slot capacity. Slots smaller than this exist too (the AM receive
// path appends one per message, sized to that message) but they are not usable for the
// RMA path and must never be advertised as if they were.
size_t ucx_slot_cap_bytes() {
    static const size_t v = []() -> size_t {
        const char *e = std::getenv("GVIRTUS_RMA_SLOT_CAP_MB");
        size_t mb = 1025;
        if (e != nullptr && e[0] != 0) {
            char *end = nullptr;
            unsigned long long parsed = std::strtoull(e, &end, 10);
            if (end != e && parsed > 0) mb = static_cast<size_t>(parsed);
        }
        return mb * 1024u * 1024u + 64u * 1024u;  // + framing slack, see P1b
    }();
    return v;
}

size_t ucx_rma_min_bytes() {
    static const size_t v = []() -> size_t {
        const char *e = std::getenv("GVIRTUS_RMA_MIN_BYTES");
        if (e == nullptr || e[0] == 0) return 4u * 1024u * 1024u;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(e, &end, 10);
        return (end != e) ? static_cast<size_t>(parsed) : 4u * 1024u * 1024u;
    }();
    return v;
}
void           free_gpu_slot(unsigned char *p);

// ---------------------------------------------------------------------------
// GPUDirect probe + flag (Step 1 of GVIRTUS_GPUDIRECT rollout)
// ---------------------------------------------------------------------------
// Runtime resolvers for cudaMalloc/cudaFree/cudaMemcpy/cudaPointerGetAttributes
// (dlopen, no static link to CUDA). cudaPointerGetAttributes lets WriteIovRma
// detect whether an iov fragment lives on the GPU so it can pass
// UCS_MEMORY_TYPE_CUDA to ucp_mem_map (required when rcache is disabled).
// cudaMemcpy is used by am_recv_handler in Step B3 to consolidate a GPU-split
// payload back into the host slot (temporary — B4 removes this consolidation).
using cudaMalloc_t = int (*)(void **, size_t);
using cudaFree_t   = int (*)(void *);
using cudaMemcpy_t = int (*)(void *, const void *, size_t, int /*cudaMemcpyKind*/);
using cudaPointerGetAttributes_t = int (*)(void *, const void *);
using cudaDeviceSynchronize_t = int (*)();
using cudaGetLastError_t = int (*)();

// cudaMemcpyKind values (from cuda_runtime_api.h, stable across CUDA versions).
constexpr int kCudaMemcpyHostToHost     = 0;
constexpr int kCudaMemcpyHostToDevice   = 1;
constexpr int kCudaMemcpyDeviceToHost   = 2;
constexpr int kCudaMemcpyDeviceToDevice = 3;

// Mirror of cudaPointerAttributes (CUDA 11+ layout). `type` 0=Unregistered,
// 1=Host, 2=Device, 3=Managed. Only `type` is read; remaining fields kept for
// ABI alignment.
struct cudaPointerAttributes_layout {
    int   type;
    int   device;
    void *devicePointer;
    void *hostPointer;
};

std::atomic<cudaMalloc_t> g_cuda_malloc{nullptr};
std::atomic<cudaFree_t>   g_cuda_free{nullptr};
std::atomic<cudaMemcpy_t> g_cuda_memcpy{nullptr};
std::atomic<cudaPointerGetAttributes_t> g_cuda_pointer_attrs{nullptr};
std::atomic<cudaDeviceSynchronize_t> g_cuda_device_sync{nullptr};
std::atomic<cudaGetLastError_t> g_cuda_get_last_error{nullptr};
std::once_flag            g_cuda_dev_once;

void load_cuda_device_funcs() {
    const char *candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        auto m  = reinterpret_cast<cudaMalloc_t>(dlsym(h, "cudaMalloc"));
        auto f  = reinterpret_cast<cudaFree_t>(dlsym(h, "cudaFree"));
        auto mc = reinterpret_cast<cudaMemcpy_t>(dlsym(h, "cudaMemcpy"));
        auto a  = reinterpret_cast<cudaPointerGetAttributes_t>(
                     dlsym(h, "cudaPointerGetAttributes"));
        auto ds = reinterpret_cast<cudaDeviceSynchronize_t>(
                     dlsym(h, "cudaDeviceSynchronize"));
        auto gle = reinterpret_cast<cudaGetLastError_t>(
                     dlsym(h, "cudaGetLastError"));
        if (m && f) {
            g_cuda_malloc.store(m);
            g_cuda_free.store(f);
            g_cuda_memcpy.store(mc);          // may be nullptr
            g_cuda_pointer_attrs.store(a);    // may be nullptr; is_gpu_pointer handles that
            g_cuda_device_sync.store(ds);     // may be nullptr; drain_device_if_async_pending checks
            g_cuda_get_last_error.store(gle); // may be nullptr; clears the probe's sticky last error
            ucx_debug_log("gpudirect: loaded cuda runtime symbols from %s "
                          "(memcpy=%s pointer_attrs=%s)",
                          candidates[i], mc ? "yes" : "no", a ? "yes" : "no");
            return;
        }
        dlclose(h);
    }
    ucx_debug_log("gpudirect: cudaMalloc/cudaFree unavailable (libcudart not found)");
}

// Global flag: true iff GVIRTUS_GPUDIRECT=1 and probe succeeded.
// Set once at backend startup by init_ucx(); read by handlers (Step 3) and
// by is_gpu_pointer below as a short-circuit guard.
std::atomic<bool> g_gpudirect_enabled{false};

// Detect whether `p` is a CUDA device or managed pointer. Returns false on
// host memory, unregistered memory, NULL, or if cudaPointerGetAttributes is
// unavailable. Used by WriteIovRma to decide whether to pass
// UCS_MEMORY_TYPE_CUDA on the ucp_mem_map call.
//
// CRITICAL: this function is called from both frontend AND backend (WriteIovRma
// runs on both sides). On the frontend, libcudart.so is the GVirtuS shim that
// REMOTES cudaPointerGetAttributes as an RPC — which is both slow and broken
// for our purposes (the frontend has no local GPU to ask about). So we
// short-circuit to false whenever GPUDirect isn't active: frontend never has
// the env var set → returns false → no RPC storm. Only the backend (where
// GPUDirect probed OK) actually calls into the cuda runtime.
bool is_gpu_pointer(const void *p) {
    if (p == nullptr) return false;
    if (!g_gpudirect_enabled.load()) return false;
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto fn = g_cuda_pointer_attrs.load();
    if (fn == nullptr) return false;
    cudaPointerAttributes_layout attrs{};
    if (fn(&attrs, p) != 0) return false;
    return attrs.type == 2 /*Device*/ || attrs.type == 3 /*Managed*/;
}

// Probe: try cudaMalloc(4K) + ucp_mem_map(CUDA) + cleanup.
// Returns true iff peermem + UCX-CUDA cooperate in this process.
// `reason` is populated with the failure description on false.
bool probe_gpudirect(ucp_context_h ctx, std::string &reason) {
    if (ctx == nullptr) {
        reason = "ucp_context is null";
        return false;
    }
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto cmalloc = g_cuda_malloc.load();
    auto cfree   = g_cuda_free.load();
    if (cmalloc == nullptr || cfree == nullptr) {
        reason = "cudaMalloc/cudaFree symbols unavailable";
        return false;
    }

    void *gpu = nullptr;
    if (cmalloc(&gpu, 4096) != 0 || gpu == nullptr) {
        reason = "cudaMalloc(4K) failed (no GPU? OOM?)";
        return false;
    }

    ucp_mem_map_params_t p{};
    p.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                   UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                   UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    p.address     = gpu;
    p.length      = 4096;
    p.memory_type = UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h memh = nullptr;
    ucs_status_t st = ucp_mem_map(ctx, &p, &memh);
    if (st != UCS_OK) {
        reason  = "ucp_mem_map(CUDA) failed: ";
        reason += ucs_status_string(st);
        cfree(gpu);
        return false;
    }
    ucp_mem_unmap(ctx, memh);
    cfree(gpu);
    reason.clear();
    return true;
}

// Definitions for the forward-declared helpers above. They live AFTER
// probe_gpudirect so the cuda symbol resolver runs at least once (probe
// triggers std::call_once(load_cuda_device_funcs)). If GPUDirect was never
// requested, g_cuda_malloc may still be nullptr — callers must check.
unsigned char *alloc_gpu_slot(size_t n) {
    std::call_once(g_cuda_dev_once, load_cuda_device_funcs);
    auto fn = g_cuda_malloc.load();
    if (fn == nullptr) return nullptr;
    void *p = nullptr;
    if (fn(&p, n) != 0 || p == nullptr) return nullptr;
    return static_cast<unsigned char *>(p);
}

void free_gpu_slot(unsigned char *p) {
    if (p == nullptr) return;
    auto fn = g_cuda_free.load();
    if (fn != nullptr) fn(p);
}
}

UcxCommunicator::UcxCommunicator(const std::string &hostname, std::uint16_t port)
    : hostname_(hostname), port_(port) {}

UcxCommunicator::~UcxCommunicator() { Close(); }

void UcxCommunicator::listener_conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    if (self == nullptr || conn_request == nullptr) return;
    self->enqueue_connection(conn_request);
}

void UcxCommunicator::endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    (void)ep;
    if (self != nullptr) {
        self->endpoint_failed_.store(true);
    }
    std::fprintf(stderr, "UCX endpoint error: %s\n", ucs_status_string(status));
}

// UCX AM receive callback: copy (or rendezvous-receive) the payload into the AM queue.
ucs_status_t UcxCommunicator::am_recv_handler(void *arg, const void *header,
                                              size_t header_length, void *data,
                                              size_t length,
                                              const ucp_am_recv_param_t *param) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    (void)header;
    (void)header_length;

    if (self == nullptr) {
        return UCS_OK;
    }

    ucx_debug_log("am_recv_handler: self=%p length=%zu rndv=%d",
                  (void *)self, length,
                  (param != nullptr && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) ? 1 : 0);

    if (length == 0) {
        self->enqueue_am_message(PooledMsg{});
        return UCS_OK;
    }

    // Quick peek at the envelope header (small messages only, always eager):
    //   * RmaSetup → handshake message, unpack rkeys inline
    //   * RmaPosted → data already RDMA-put into an RX slot, queue a
    //                 PooledMsg pointing at that slot instead of acquiring
    //                 a fresh one + memcpy'ing
    if (length >= sizeof(gvirtus::communicators::ucxam::EnvelopeHeader) &&
        (param == nullptr || !(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV))) {
        gvirtus::communicators::ucxam::EnvelopeHeader peek;
        std::memcpy(&peek, data, sizeof(peek));
        if (peek.magic == gvirtus::communicators::ucxam::kEnvelopeMagic) {
            using gvirtus::communicators::ucxam::MessageType;
            if (peek.message_type == static_cast<std::uint16_t>(MessageType::RmaSetup)) {
                self->handle_rma_setup_am(data, length);
                return UCS_OK;
            }
            if (peek.message_type == static_cast<std::uint16_t>(MessageType::RmaPosted)) {
                const size_t slot_idx   = static_cast<size_t>(peek.reserved0);
                const size_t total      = static_cast<size_t>(peek.payload_size);
                // GPUDirect Step B3: non-zero routine_size means the peer
                // routed `gpu_size` bytes into slot.gpu_addr (via NIC
                // peer-DMA). status_code carries gpu_offset — the position
                // in the logical message where the GPU data folds in (i.e.
                // where to put the consolidation cudaMemcpy destination).
                const size_t gpu_size   = static_cast<size_t>(peek.routine_size);
                const size_t gpu_offset = static_cast<size_t>(peek.status_code);
                std::lock_guard<std::mutex> lk(self->rx_pool_->mu);
                if (slot_idx >= self->rx_pool_->slots.size()) {
                    std::fprintf(stderr,
                                 "RmaPosted: invalid slot_idx=%zu (pool=%zu)\n",
                                 slot_idx, self->rx_pool_->slots.size());
                    return UCS_OK;
                }
                auto &slot = self->rx_pool_->slots[slot_idx];
                // The client should only ever post into a live persistent slot it was
                // advertised. Anything else means it is working from a layout we have
                // replaced -- which the epoch in the tag is there to make survivable,
                // but it should never happen, so say so rather than consume blindly.
                if (!slot.rma_persistent || slot.rma_retired) {
                    std::fprintf(stderr,
                                 "RmaPosted: slot %zu is %s (epoch tag %llu) -- the peer "
                                 "is using a superseded layout\n",
                                 slot_idx,
                                 slot.rma_retired ? "retired" : "not an RMA slot",
                                 (unsigned long long)peek.request_id);
                }
                slot.in_use = true;
                // Remember this slot was filled by a client RMA put so that
                // release_rx_slot sends a SlotConsumed ack (with this
                // generation) back to the client for ABA-safe reuse.
                slot.rma_origin = true;
                slot.rma_generation = peek.request_id;
                // The peer's data is already in this slot. Mark it busy so the server's
                // own view matches reality; release_rx_slot clears it.
                slot.in_use = true;

                PooledMsg msg{slot.addr, total, slot_idx};

                // Step B3 CONSOLIDATION: temporarily cudaMemcpy the GPU
                // portion back into the host slot at offset (total - gpu_size).
                // This preserves the legacy contiguous-host parser path so
                // Buffer/handler dispatch needs no changes. Step B4 removes
                // this copy and teaches Buffer to read GPU directly.
                if (gpu_size > 0) {
                    if (slot.gpu_addr == nullptr ||
                        gpu_offset + gpu_size > total) {
                        std::fprintf(stderr,
                            "RmaPosted B4: gpu_size=%zu offset=%zu but slot %zu has no GPU shadow "
                            "(or offset+size > total=%zu) — protocol mismatch, dropping\n",
                            gpu_size, gpu_offset, slot_idx, total);
                        slot.in_use = false;
                        return UCS_OK;
                    }
                    // Step B4: no consolidation cudaMemcpy. The GPU portion
                    // stays in slot.gpu_addr and we publish it to the
                    // consumer via PooledMsg.gpu_data/gpu_size. Handlers
                    // that recognize the GPU payload (cudaMemcpy H2D) use
                    // cudaMemcpyDeviceToDevice directly from slot.gpu_addr
                    // instead of bouncing through host.
                    msg.gpu_data = slot.gpu_addr;
                    msg.gpu_size = gpu_size;
                    ucx_debug_log("RmaPosted B4: slot=%zu host_bytes=%zu gpu_bytes=%zu offset=%zu (no consolidation)",
                                  slot_idx, total - gpu_size, gpu_size, gpu_offset);
                }

                self->enqueue_am_message(msg);
                return UCS_OK;
            }
            if (peek.message_type ==
                static_cast<std::uint16_t>(MessageType::SlotConsumed)) {
                // Client side: the backend has finished consuming the remote RX
                // slot reserved0 that we filled via ucp_put. Return it to Free
                // (ABA-guarded by request_id = generation) so a WriteIovRma
                // waiter can reuse it. This is the explicit backend-consumption
                // confirmation the slot lifecycle is tied to.
                self->release_remote_slot(static_cast<size_t>(peek.reserved0),
                                          peek.request_id);
                return UCS_OK;
            }
        }
    }

    // Acquire a pinned slot from the RX pool — slot capacity is pre-allocated,
    // no per-message std::vector zero-init.
    // Demand-driven pool: a message at or above the RMA floor proves this connection
    // moves payloads big enough for the RMA path to pay off, so ask for the pool to be
    // built. Deferred rather than done here -- this callback runs under worker_mutex_
    // (ucp_worker_progress holds it) and both the allocation and the advertisement
    // would deadlock or stall progress. Read() picks it up.
    if (length >= ucx_rma_min_bytes()) {
        // Record the largest payload this connection has actually moved. The pool is
        // sized from this rather than from GVIRTUS_RMA_SLOT_CAP_MB, which becomes a
        // ceiling instead of a target.
        size_t prev = self->rma_pool_hint_bytes_.load(std::memory_order_relaxed);
        while (length > prev &&
               !self->rma_pool_hint_bytes_.compare_exchange_weak(
                   prev, length, std::memory_order_relaxed)) {
        }
        // A message at or above the RMA floor arriving EAGERLY means one of two
        // things, and both are answered the same way. Either the pool has not been
        // built yet, or it has been built too small and the sender's WriteIovRma
        // declined the fast path for capacity -- in which case the payload came down
        // this path precisely because no slot could hold it. Ask for (re)build at a
        // capacity derived from the size just observed. No extra control message is
        // needed: the decline is self-reporting.
        if (!self->rma_pool_ready_.load(std::memory_order_acquire) ||
            rma_slot_cap_for(length) >
                self->rma_pool_cap_.load(std::memory_order_acquire)) {
            self->rma_pool_requested_.store(true, std::memory_order_release);
        }
    }

    size_t slot_idx = self->acquire_rx_slot(length);
    PinnedSlot &slot = self->rx_pool_->slots[slot_idx];
    PooledMsg msg{slot.addr, length, slot_idx};

    if ((param != nullptr) && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        ucp_request_param_t recv_param{};
        recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        recv_param.datatype = ucp_dt_make_contig(1);
        if (slot.memh != nullptr) {
            recv_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            recv_param.memh = slot.memh;
        }

        void *request = ucp_am_recv_data_nbx(self->worker_, data, slot.addr, length,
                                             &recv_param);
        if (request == nullptr) {
            self->enqueue_am_message(msg);
            return UCS_OK;
        }
        if (UCS_PTR_IS_ERR(request)) {
            self->release_rx_slot(slot_idx);
            return UCS_PTR_STATUS(request);
        }

        self->enqueue_am_rndv(request, msg);
        return UCS_INPROGRESS;
    }

    std::memcpy(slot.addr, data, length);
    self->enqueue_am_message(msg);

    // For DATA callbacks we copy and return UCS_OK; UCX releases the data.
    return UCS_OK;
}

sockaddr_storage UcxCommunicator::make_sockaddr(const std::string &host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *res = nullptr;
    const std::string port_str = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        throw std::runtime_error("UcxCommunicator: getaddrinfo failed for " + host + ":" +
                                 port_str);
    }

    sockaddr_storage storage{};
    std::memcpy(&storage, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    return storage;
}

void UcxCommunicator::init_ucx() {
    if (initialized_) return;

    // Initialize UCX context/worker and register the AM receive callback.
    am_id_ = kUcxAmId;
    if (!am_state_) {
        am_state_ = std::make_shared<AmState>();
    }

    // Early read of GVIRTUS_GPUDIRECT (BEFORE ucp_init): UCX reads
    // UCX_RCACHE_ENABLE / UCX_MEMTYPE_CACHE at context creation time. The
    // rcache in this container/UCX combo fails on ucp_mem_map(CUDA) with
    // "failed to insert region [0x0..0x0]: Invalid parameter" — same root
    // cause as the production manual memh cache in WriteIovRma. Force-disable
    // rcache + memtype-cache so the CUDA mem_map succeeds. We use overwrite=0
    // so a user-provided value still wins.
    const char *gpudirect_env_early = std::getenv("GVIRTUS_GPUDIRECT");
    const bool gpudirect_env_set = (gpudirect_env_early != nullptr &&
                                    gpudirect_env_early[0] == '1');
    // GPUDirect requires the negotiated UCX transport to support CUDA
    // peer-DMA. UCX-TCP cannot move CUDA memory ("cannot find remote
    // protocol for put from cuda memory to host" error). Even though the
    // backend may have RDMA-class transports listed in UCX_TLS, if a
    // particular client connects over TCP, ucp_put_nbx from GPU mem fails.
    // Guard at process level: if UCX_TLS doesn't include any CUDA-capable
    // transport, do not enable GPUDirect even when GVIRTUS_GPUDIRECT=1.
    // Run a separate backend with UCX_TLS=tcp,self for UCX-TCP benchmarks.
    const bool tls_supports_cuda = []() {
        const char *tls = std::getenv("UCX_TLS");
        if (tls == nullptr) return true;
        std::string s(tls);
        return s.find("rc_mlx5") != std::string::npos ||
               s.find("dc_mlx5") != std::string::npos ||
               s.find("ud_mlx5") != std::string::npos ||
               s.find("ib")      != std::string::npos;
    }();
    const bool gpudirect_requested = gpudirect_env_set && tls_supports_cuda;
    if (gpudirect_requested) {
        setenv("UCX_RCACHE_ENABLE",   "n", /*overwrite=*/0);
        setenv("UCX_MEMTYPE_CACHE",   "n", /*overwrite=*/0);
    }

    ucp_params_t ucp_params{};
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    // AM: small control + legacy data path. RMA: bulk data via ucp_put_nbx
    // into pre-mem_map'd remote slots (avoids per-message rendezvous handshake).
    ucp_params.features = UCP_FEATURE_AM | UCP_FEATURE_RMA;

    ucs_status_t status = ucp_init(&ucp_params, nullptr, &context_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_init failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // GPUDirect probe (Step 1 of GVIRTUS_GPUDIRECT rollout). The early
    // read above already auto-set UCX_RCACHE_ENABLE=n / UCX_MEMTYPE_CACHE=n
    // when requested, so the ucp_mem_map(CUDA) below has a chance to succeed.
    //
    // Side-effect: we also setenv("GVIRTUS_GPUDIRECT_ACTIVE", "1"/"0") so the
    // cudart backend plugin can detect the post-probe state via getenv without
    // needing to link against this UCX library (avoids RTLD_GLOBAL surprises
    // since plugins are dlopen'd separately from libgvirtus-communicators-ucx).
    if (!gpudirect_requested) {
        g_gpudirect_enabled.store(false);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", "0", /*overwrite=*/1);
        if (gpudirect_env_set && !tls_supports_cuda) {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (UCX_TLS=%s has no CUDA-capable transport)\n",
                std::getenv("UCX_TLS") ? std::getenv("UCX_TLS") : "(unset)");
        } else {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (env GVIRTUS_GPUDIRECT not set)\n");
        }
    } else {
        std::string reason;
        const bool ok = probe_gpudirect(context_, reason);
        // The probe's cudaMalloc(4K) runs through the frontend cudart shim
        // during Connect (mpInitialized == false), so it returns the reentrancy-
        // guard init error (cudaErrorInitializationError) and leaves it as the
        // client-side sticky last error. That is an internal probe artifact, not
        // an application error — clear it so a later cudaGetLastError() by the
        // app doesn't spuriously observe it. On the backend this calls the real
        // cudaGetLastError at init time (a harmless no-op before any app work).
        if (auto gle = g_cuda_get_last_error.load()) gle();
        g_gpudirect_enabled.store(ok);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", ok ? "1" : "0", /*overwrite=*/1);
        if (ok) {
            std::fprintf(stderr,
                "[GVS] GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK, "
                "auto-set UCX_RCACHE_ENABLE=n UCX_MEMTYPE_CACHE=n)\n");
        } else {
            std::fprintf(stderr,
                "[GVS] GPUDirect=disabled (GVIRTUS_GPUDIRECT=1 but probe FAILED: %s) "
                "- falling back to host slots, behavior unchanged\n",
                reason.c_str());
        }
    }

    // parameters for ucp_worker
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create(context_, &worker_params, &worker_);
    if (status != UCS_OK) {
        ucp_cleanup(context_);
        context_ = nullptr;
        throw std::runtime_error("UcxCommunicator: ucp_worker_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    ucp_am_handler_param_t am_param{};
    am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                          UCP_AM_HANDLER_PARAM_FIELD_CB |
                          UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am_param.id = am_id_;
    // Copying eager payloads in the callback, so no persistent data is needed.
    am_param.flags = UCP_AM_FLAG_WHOLE_MSG;
    am_param.cb = &UcxCommunicator::am_recv_handler;
    am_param.arg = this;

    status = ucp_worker_set_am_recv_handler(worker_, &am_param);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: failed to set AM handler: " +
                                 std::string(ucs_status_string(status)));
    }

    // Pre-allocate pinned RX pool so the AM handler doesn't have to
    // zero-init a fresh std::vector for every incoming message.
    init_rx_pool();

    initialized_ = true;
    ucx_debug_log("init_ucx completed host=%s port=%u mode=am", hostname_.c_str(), port_);
}

void UcxCommunicator::destroy_ucx() {
    if (!initialized_) return;

    // Tear down UCX resources in reverse order of creation.
    ucx_debug_log("destroy_ucx begin endpoint=%p listener=%p worker=%p context=%p",
                  (void *)endpoint_, (void *)listener_, (void *)worker_, (void *)context_);

    if (endpoint_ != nullptr) {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

        ucp_request_param_t close_params{};
        close_params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        close_params.flags = endpoint_failed_.load() ? UCP_EP_CLOSE_FLAG_FORCE : 0;

        void *close_req = ucp_ep_close_nbx(endpoint_, &close_params);
        if (UCS_PTR_IS_ERR(close_req)) {
            std::fprintf(stderr, "UCX endpoint close failed: %s\n",
                         ucs_status_string(UCS_PTR_STATUS(close_req)));
        } else {
            wait_request_completion(close_req, "ep_close");
        }
        endpoint_ = nullptr;
    }

    // Release pre-registered TX scratch before tearing down the UCP
    // context — ucp_mem_unmap needs context_ still alive.
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        release_tx_scratch_locked();
    }

    if (owns_listener_ && listener_ != nullptr) {
        ucp_listener_destroy(listener_);
        listener_ = nullptr;
    }

    // Destroy RMA state BEFORE the worker/context teardown — ucp_rkey_destroy
    // needs an alive context, and destroy_rx_pool calls ucp_mem_unmap.
    destroy_rma_state();
    destroy_rx_pool();
    current_frame_ = PooledMsg{};

    if (owns_worker_ && worker_ != nullptr) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }

    if (owns_context_ && context_ != nullptr) {
        ucp_cleanup(context_);
        context_ = nullptr;
    }

    initialized_ = false;
    endpoint_failed_.store(false);
    ucx_debug_log("destroy_ucx completed");
}

void UcxCommunicator::enqueue_connection(ucp_conn_request_h conn_request) {
    // Queue incoming connection requests from the listener callback.
    std::lock_guard<std::mutex> lock(conn_mutex_);
    pending_conn_requests_.push(conn_request);
    ucx_debug_log("enqueue_connection request=%p queue_size=%zu", (void *)conn_request,
                  pending_conn_requests_.size());
    conn_cv_.notify_one();
}

ucp_conn_request_h UcxCommunicator::wait_for_connection_request() {
    // Wait for a pending connection request while progressing the worker.
    std::unique_lock<std::mutex> lock(conn_mutex_);
    for (;;) {
        if (!running_) {
            return nullptr;
        }
        if (!pending_conn_requests_.empty()) {
            ucp_conn_request_h req = pending_conn_requests_.front();
            pending_conn_requests_.pop();
            return req;
        }
        conn_cv_.wait_for(lock, std::chrono::milliseconds(5));
        lock.unlock();
        if (worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }
        lock.lock();
    }
}

void UcxCommunicator::wait_request_completion(void *request, const char *op_name) {
    // Progress the worker until the request completes (no sleep for low latency).
    ucx_debug_log("%s: wait_request_completion request=%p", op_name, request);

    if (request == nullptr) {
        ucx_debug_log("%s: immediate completion (null request)", op_name);
        return;
    }

    if (UCS_PTR_IS_ERR(request)) {
        throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                 " request error: " +
                                 ucs_status_string(UCS_PTR_STATUS(request)));
    }

    bool cancel_issued = false;
    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        if (worker_ != nullptr) {
            ucp_worker_progress(worker_);
        }

        progress_am_rndv();

        // If the endpoint has failed (for example, remote peer reset), cancel
        // the in-flight request so callers can unwind instead of hanging.
        if (!cancel_issued && endpoint_failed_.load() && worker_ != nullptr) {
            ucp_request_cancel(worker_, request);
            cancel_issued = true;
        }

    }

    const ucs_status_t final_status = ucp_request_check_status(request);
    ucp_request_free(request);
    if (final_status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ") + op_name +
                                 " completion failed: " +
                                 ucs_status_string(final_status));
    }

    ucx_debug_log("%s: completed status=%s", op_name, ucs_status_string(final_status));
}

void UcxCommunicator::enqueue_am_message(PooledMsg message) {
    // Store a completed AM payload for stream-style Read() / TryAcquireFrame().
    {
        std::lock_guard<std::mutex> lock(am_state_->mutex);
        am_state_->queue.push_back(message);
    }
    am_state_->cv.notify_one();
}

void UcxCommunicator::enqueue_am_rndv(void *request, PooledMsg msg) {
    // Track a rendezvous receive until UCX reports completion.
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    am_state_->rndv.push_back(PendingAmRecv{request, msg});
}

void UcxCommunicator::progress_am_rndv() {
    // Check rendezvous receive requests and move completed payloads into the queue.
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    for (auto it = am_state_->rndv.begin(); it != am_state_->rndv.end();) {
        if (it->request == nullptr) {
            am_state_->queue.push_back(it->msg);
            it = am_state_->rndv.erase(it);
            continue;
        }

        const ucs_status_t status = ucp_request_check_status(it->request);
        if (status == UCS_INPROGRESS) {
            ++it;
            continue;
        }

        ucp_request_free(it->request);
        if (status == UCS_OK) {
            am_state_->queue.push_back(it->msg);
            am_state_->cv.notify_one();
        } else {
            std::fprintf(stderr, "UCX AM rendezvous receive failed: %s\n",
                         ucs_status_string(status));
            // Release the slot since the message was never delivered.
            if (it->msg.slot_idx != static_cast<size_t>(-1)) {
                release_rx_slot(it->msg.slot_idx);
            }
        }
        it = am_state_->rndv.erase(it);
    }
}

void UcxCommunicator::Serve() {
    // Start server listener for UCX client connections.
    init_ucx();

    sockaddr_storage ss = make_sockaddr(hostname_, port_);

    ucp_listener_params_t params{};
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&ss);
    params.sockaddr.addrlen = sizeof(sockaddr_in);
    params.conn_handler.cb = &UcxCommunicator::listener_conn_handler;
    params.conn_handler.arg = this;

    ucs_status_t status = ucp_listener_create(worker_, &params, &listener_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_listener_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    running_ = true;
    std::printf("UCX control-plane ready: Serve (%s:%u)\n", hostname_.c_str(), port_);
    ucx_debug_log("listener created listener=%p", (void *)listener_);
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    // Accept a connection and create a UCX endpoint for the new client.
    auto *self = const_cast<UcxCommunicator *>(this);
    if (!self->running_ || self->listener_ == nullptr) {
        return nullptr;
    }

    ucp_conn_request_h req = self->wait_for_connection_request();
    if (req == nullptr) {
        ucx_debug_log("Accept returned null request (shutdown or no request)");
        return nullptr;
    }

    auto *accepted = new UcxCommunicator(self->hostname_, self->port_);
    accepted->context_ = self->context_;
    accepted->initialized_ = true;
    // Ownership split: shares the listener's UCX context (heavy to create
    // and the rkeys we hand out are scoped to it), but owns a DEDICATED
    // worker so error progress on one accepted connection doesn't poison
    // the worker shared by other connections. Listener stays separate.
    accepted->owns_context_ = false;
    accepted->owns_worker_ = true;
    accepted->owns_listener_ = false;
    accepted->running_ = true;
    accepted->endpoint_failed_.store(false);

    // Per-connection worker. We don't reuse self->worker_; that one only
    // services the listener's conn_handler. Each accepted's data path runs
    // on its own worker -> its own ucp_worker_progress -> isolated request
    // state machine.
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    ucs_status_t status = ucp_worker_create(self->context_, &worker_params,
                                            &accepted->worker_);
    if (status != UCS_OK) {
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server ucp_worker_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // Per-connection AM state, RX pool, and worker mutex — no sharing with
    // the listener or with other accepted connections.
    accepted->worker_mutex_ = std::make_shared<std::mutex>();
    accepted->am_state_ = std::make_shared<AmState>();
    accepted->rx_pool_ = std::make_shared<RxPool>();

    // AM handler bound to THIS accepted's worker, with `arg = accepted` so
    // incoming messages land in its own am_state / rx_pool.
    ucp_am_handler_param_t am_param{};
    am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                          UCP_AM_HANDLER_PARAM_FIELD_CB |
                          UCP_AM_HANDLER_PARAM_FIELD_ARG;
    am_param.id = self->am_id_;
    am_param.flags = UCP_AM_FLAG_WHOLE_MSG;
    am_param.cb = &UcxCommunicator::am_recv_handler;
    am_param.arg = accepted;
    status = ucp_worker_set_am_recv_handler(accepted->worker_, &am_param);
    if (status != UCS_OK) {
        ucp_worker_destroy(accepted->worker_);
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server AM handler register failed: " +
                                 std::string(ucs_status_string(status)));
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.conn_request = req;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = accepted;

    status = ucp_ep_create(accepted->worker_, &ep_params, &accepted->endpoint_);
    if (status != UCS_OK) {
        ucp_worker_destroy(accepted->worker_);
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    std::printf("UCX control-plane accepted connection\n");
    ucx_debug_log("server endpoint created endpoint=%p worker=%p from request=%p",
                  (void *)accepted->endpoint_, (void *)accepted->worker_, (void *)req);

    // Parallel setup: init_rx_pool (~150ms: cudaHostAlloc + ucp_mem_map for
    // each slot) and send_rma_setup (~50ms: pack rkeys + ucp_am_send_nbx)
    // run in a detached thread. The listener can return from Accept()
    // immediately and process the next conn_request while this thread
    // finishes setting up the previous one. The lambda thread spawned by
    // Process.cpp will block at its first incoming AM (via worker progress)
    // until the AM handler can acquire a slot from rx_pool — which is
    // exactly when this setup thread has finished init_rx_pool. Mutex on
    // rx_pool_->mu and am_state_->mutex serialises any actual contention.
    //
    // The client's Connect() waits up to 2 s for server's RmaSetup, so
    // even with setup taking ~250ms in the worst case the client doesn't
    // time out. Net effect for N concurrent connects:
    //   sequential: N × 350 ms serialised in the listener
    //   parallel:   ~max(setup_i) wall time (cudaHostAlloc/ucp_mem_map
    //               serialise at the CUDA/UCX driver level, so ~1.5-2x
    //               speedup rather than perfect N×, but still big).
    // Advertise immediately (with no slots yet) so the client's Connect() does not sit
    // waiting for us to allocate a pool it may never use. The pool is built the first
    // time a message large enough to need it actually arrives.
    std::thread([accepted]() {
        accepted->init_rx_pool();   // no-op unless GVIRTUS_RMA_PREALLOC=1
        accepted->send_rma_setup();
    }).detach();

    return accepted;
}

void UcxCommunicator::Connect() {
    // Connect to the UCX server and create a client endpoint.
    init_ucx();

    sockaddr_storage ss = make_sockaddr(hostname_, port_);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                           UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&ss);
    ep_params.sockaddr.addrlen = sizeof(sockaddr_in);
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = this;

    ucs_status_t status = ucp_ep_create(worker_, &ep_params, &endpoint_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: client ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    running_ = true;
    endpoint_failed_.store(false);
    std::printf("UCX control-plane connected: Connect (%s:%u)\n", hostname_.c_str(), port_);
    ucx_debug_log("client endpoint created endpoint=%p", (void *)endpoint_);

    // Drive worker progress until the server's RmaSetup AM lands. If it
    // doesn't show up within the budget we silently fall back to the AM
    // data path (rma_setup_received_ stays false → WriteIov picks the IOV
    // branch). Useful when talking to an older server build that never
    // sends RmaSetup.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!rma_setup_received_.load() &&
           std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> wl(*worker_mutex_);
            ucp_worker_progress(worker_);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (!rma_setup_received_.load()) {
        ucx_debug_log("Connect: RmaSetup not received within timeout, RMA path disabled");
    } else {
        ucx_debug_log("Connect: RMA path enabled with %zu remote slots (server -> client)",
                      remote_slots_.size());

        // Bidirectional RMA: now that we know the server is RMA-capable
        // (it sent us its rkeys), advertise our own rx_pool's rkeys so it
        // can ucp_put_nbx into our slots for large responses (D2H 64MB
        // etc). Without this the server's WriteIov for the response falls
        // back to the AM-stream path, which is ~1.7s for 64MB without an
        // rcache. With this it becomes a single RDMA write + tiny AM ≈
        // ~10-15ms.
        send_rma_setup();
        ucx_debug_log("Connect: client rkeys advertised (client -> server done)");
    }
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Read called without an active endpoint");
    }
    if (size == 0) {
        return 0;
    }

    // Safe point for the deferred pool build: no worker_mutex_ is held here.
    // Also fires for a REGROW, not just the first build: rma_pool_requested_ is set
    // again whenever a message arrives eagerly that the current slots cannot hold.
    if (rma_pool_requested_.load(std::memory_order_acquire)) {
        materialise_rma_pool();
    }

    // Drain AM queue into the caller buffer, preserving stream semantics.
    // Busy-poll to keep ucp_worker_progress() running continuously.
    size_t copied = 0;
    while (copied < size) {
        if (pending_msg_.data != nullptr &&
            pending_read_offset_ < pending_msg_.size) {
            const size_t available = pending_msg_.size - pending_read_offset_;
            const size_t to_copy = std::min(size - copied, available);
            std::memcpy(buffer + copied,
                        pending_msg_.data + pending_read_offset_,
                        to_copy);
            copied += to_copy;
            pending_read_offset_ += to_copy;
            if (pending_read_offset_ == pending_msg_.size) {
                // Fully consumed — return the pool slot.
                if (pending_msg_.slot_idx != static_cast<size_t>(-1)) {
                    release_rx_slot(pending_msg_.slot_idx);
                }
                pending_msg_ = PooledMsg{};
                pending_read_offset_ = 0;
            }
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(am_state_->mutex);
            if (!am_state_->queue.empty()) {
                pending_msg_ = am_state_->queue.front();
                am_state_->queue.pop_front();
                pending_read_offset_ = 0;
                continue;
            }
        }

        if (worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }

        progress_am_rndv();

        if (endpoint_failed_.load()) {
            return copied == 0 ? 0 : copied;
        }

    }

    return size;
}

bool UcxCommunicator::TryAcquireFrame(const unsigned char *&data, size_t &size) {
    if (endpoint_ == nullptr || worker_ == nullptr) return false;

    // Safe point for the deferred pool build (no worker_mutex_ held yet). The backend
    // takes requests through this path, not Read(), so without this the pool would
    // never materialise on the side that receives the large payloads.
    // Also fires for a REGROW, not just the first build: rma_pool_requested_ is set
    // again whenever a message arrives eagerly that the current slots cannot hold.
    if (rma_pool_requested_.load(std::memory_order_acquire)) {
        materialise_rma_pool();
    }

    // If we already hold a partially-consumed message, give up — the caller
    // mixed stream Read() with frame mode. Conservative: refuse the handoff.
    if (pending_msg_.data != nullptr && pending_read_offset_ > 0) {
        return false;
    }

    // Drain into current_frame_ once a message is available, busy-polling
    // the worker the same way Read() does.
    for (;;) {
        if (current_frame_.data != nullptr) {
            data = current_frame_.data;
            size = current_frame_.size;
            return true;
        }

        // Inherit any message we may have moved into pending_msg_ already
        // (e.g., partial consumption of zero bytes).
        if (pending_msg_.data != nullptr && pending_read_offset_ == 0) {
            current_frame_ = pending_msg_;
            pending_msg_ = PooledMsg{};
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(am_state_->mutex);
            if (!am_state_->queue.empty()) {
                current_frame_ = am_state_->queue.front();
                am_state_->queue.pop_front();
                continue;
            }
        }

        if (worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }
        progress_am_rndv();

        if (endpoint_failed_.load()) return false;
    }
}

void UcxCommunicator::ReleaseFrame() {
    if (current_frame_.slot_idx != static_cast<size_t>(-1)) {
        // Wait for any device work the handler left in flight against this frame
        // (the GPUDirect shadow -> destination copy) BEFORE the slot is declared
        // free. The response has already been sent by now, so this costs the client
        // nothing; skipping it would let the peer's next peer-DMA land on top of a
        // buffer the copy engine is still reading.
        RunFrameDrainHook();
        release_rx_slot(current_frame_.slot_idx);
    }
    current_frame_ = PooledMsg{};
}

// Per-connection GPUDirect gate (Option 2). Returns true iff THIS
// endpoint negotiated an RDMA-class transport (rc_mlx5 / dc_mlx5 /
// ud_mlx5 / ib) capable of carrying CUDA memory operations.
//
// This is a property of the TRANSPORT, not of the local process. In
// particular it does NOT depend on g_gpudirect_enabled (= local probe
// of cudaMalloc + ucp_mem_map(CUDA), which requires nvidia-peermem
// loaded on this host). Reason: frontend Variant B (host → remote GPU
// shadow) puts data FROM host memory, so the local NIC doesn't need
// peer-DMA-from-CUDA capability — only RDMA-class transport plus the
// backend's gpu_rkey suffice.
//
// The "process can locally do CUDA peer-DMA" precondition is enforced
// separately in places where the local side IS the CUDA mem source —
// notably gvirtus_gpudirect_enabled() in CudaRtHandler_memory.cpp,
// which AND-s GVIRTUS_GPUDIRECT_ACTIVE (env, set by init_ucx based
// on the probe) with tls_connection_supports_cuda (this method).
//
// Lazy + cached: ucp_ep_query returns the negotiated lanes only after
// wire-up completes (async, after first AM exchange). The first caller
// (WriteIovRma at the first cudaMemcpy >= 4 MB, or Process.cpp's pre-
// Execute set of tls_connection_supports_cuda) happens well after
// wire-up. ucp_ep_query failures don't cache so a later call retries.
bool UcxCommunicator::current_connection_supports_cuda() const {
    int cached = supports_cuda_cached_.load(std::memory_order_acquire);
    if (cached != -1) return cached == 1;

    if (endpoint_ == nullptr) {
        return false;  // don't cache — endpoint may still be assigned later
    }

    // We use ucp_ep_print_info instead of ucp_ep_query(TRANSPORTS) because
    // in UCX 1.20 (this container) ucp_ep_query returns UCS_OK with
    // num_entries>0 but transport_name/device_name as NULL pointers — a
    // quirk likely tied to lane wire-up state or an ABI mismatch between
    // the pinned header and the loaded .so. ucp_ep_print_info renders the
    // lane info as text to a FILE* and is the API used by ucx_info and
    // verbose UCX logs, so its output is well tested across builds.
    //
    // Captured via open_memstream and grep'd for RDMA-class transport
    // tokens. Expected output lines look like:
    //   #     lane[1]: 2:rc_mlx5/mlx5_1:1.0 md[2] -> md[2]/ib/sysdev[3] ... rma_bw#0 am
    // Tokens rc_mlx5 / dc_mlx5 / ud_mlx5 (mlx5 driver) and rc_verbs /
    // dc_verbs / ud_verbs (generic verbs) indicate an RDMA-class lane.
    char *buf = nullptr;
    size_t buf_size = 0;
    FILE *fp = open_memstream(&buf, &buf_size);
    if (fp == nullptr) {
        return false;  // memstream alloc failed — retry next call
    }
    ucp_ep_print_info(endpoint_, fp);
    std::fclose(fp);

    if (buf == nullptr || buf_size == 0) {
        if (buf) std::free(buf);
        return false;
    }

    const bool supports = (std::strstr(buf, "rc_mlx5") != nullptr) ||
                          (std::strstr(buf, "dc_mlx5") != nullptr) ||
                          (std::strstr(buf, "ud_mlx5") != nullptr) ||
                          (std::strstr(buf, "rc_verbs") != nullptr) ||
                          (std::strstr(buf, "dc_verbs") != nullptr) ||
                          (std::strstr(buf, "ud_verbs") != nullptr);
    std::free(buf);

    supports_cuda_cached_.store(supports ? 1 : 0, std::memory_order_release);
    ucx_debug_log("current_connection_supports_cuda: endpoint=%p -> %s",
                  (void *)endpoint_,
                  supports ? "RDMA (CUDA-capable)" : "non-RDMA (TCP-class)");
    return supports;
}

// Register `slot.addr/capacity` (host) AND `slot.gpu_addr/gpu_capacity` (if
// present) with the UCX context so subsequent ucp_am_recv_data_nbx and
// ucp_put_nbx can use the memh hints and skip on-the-fly IB registration.
// Called with rx_pool_->mu held.
void UcxCommunicator::map_slot_to_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx == nullptr) return;
    if (slot.memh == nullptr && slot.addr != nullptr) {
        ucp_mem_map_params_t map_params{};
        map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        map_params.address = slot.addr;
        map_params.length  = slot.capacity;
        ucs_status_t st = ucp_mem_map(ctx, &map_params, &slot.memh);
        if (st != UCS_OK) {
            slot.memh = nullptr;  // continue without — UCX rcache will register on first use
        }
    }
    // GPUDirect (Step B1): register the GPU shadow if it exists. UCX needs
    // UCS_MEMORY_TYPE_CUDA explicitly here since memtype-cache is disabled.
    if (slot.gpu_memh == nullptr && slot.gpu_addr != nullptr) {
        ucp_mem_map_params_t gpu_params{};
        gpu_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        gpu_params.address     = slot.gpu_addr;
        gpu_params.length      = slot.gpu_capacity;
        gpu_params.memory_type = UCS_MEMORY_TYPE_CUDA;
        ucs_status_t st = ucp_mem_map(ctx, &gpu_params, &slot.gpu_memh);
        if (st != UCS_OK) {
            slot.gpu_memh = nullptr;
            ucx_debug_log("map_slot_to_ucp: gpu_addr map FAILED (%s) — slot will keep host-only path",
                          ucs_status_string(st));
        }
    }
}

void UcxCommunicator::unmap_slot_from_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx != nullptr && slot.memh != nullptr) {
        ucp_mem_unmap(ctx, slot.memh);
        slot.memh = nullptr;
    }
    if (ctx != nullptr && slot.gpu_memh != nullptr) {
        ucp_mem_unmap(ctx, slot.gpu_memh);
        slot.gpu_memh = nullptr;
    }
}

// Pre-allocate N RX slots of an initial size. Slots grow on demand later if
// a message arrives that's bigger than the current capacity.
// Slot capacity to build for the largest payload observed on this connection.
// GVIRTUS_RMA_SLOT_CAP_MB stops being the size we allocate and becomes the most we
// are ever willing to allocate; GVIRTUS_RMA_SLOT_MIN_MB is the least. The payload is
// rounded up to a power of two so a workload whose sizes creep upward regrows a
// bounded number of times rather than once per distinct size, and the framing slack
// is added on top so a transfer of exactly a power of two still fits (the slot has to
// carry the request header, routine name and marshalled argument Buffer alongside the
// payload; without the slack it overshoots by ~90 bytes and silently falls back to
// eager AM at a 3.2x cost).
size_t UcxCommunicator::rma_slot_cap_for(size_t hint_bytes) {
    static constexpr size_t kSlotFramingSlack = 64u * 1024u;
    auto env_mb = [](const char *k, size_t dflt) -> size_t {
        const char *v = std::getenv(k);
        if (v == nullptr || v[0] == '\0') return dflt;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(v, &end, 10);
        return (parsed > 0) ? static_cast<size_t>(parsed) : dflt;
    };
    const size_t ceiling = env_mb("GVIRTUS_RMA_SLOT_CAP_MB", 1025) * 1024u * 1024u +
                           kSlotFramingSlack;
    const size_t floor_bytes = env_mb("GVIRTUS_RMA_SLOT_MIN_MB", 16) * 1024u * 1024u +
                               kSlotFramingSlack;
    if (hint_bytes == 0) return ceiling;  // no evidence yet (eager prealloc)

    // Round the PAYLOAD to a power of two, then add the slack -- not the other way
    // round. hint_bytes is a whole wire message, i.e. payload plus ~80 bytes of
    // framing, so rounding it directly would push a 64 MiB transfer (67108942 B) to
    // the next power of two and allocate 128 MiB for a 64 MiB payload. Taking the
    // framing off first lands exactly on 64 MiB + slack, which is what actually has
    // to fit.
    const size_t payload = (hint_bytes > kSlotFramingSlack)
                               ? (hint_bytes - kSlotFramingSlack)
                               : hint_bytes;
    size_t pow2 = 1;
    while (pow2 < payload && pow2 < ceiling) pow2 <<= 1;
    size_t cap = pow2 + kSlotFramingSlack;
    if (cap < floor_bytes) cap = floor_bytes;
    if (cap > ceiling) cap = ceiling;
    return cap;
}

// Free persistent slots retired at least one full epoch ago. Retirement is NOT
// immediate release: when the pool grows, a client that has not yet processed the new
// advertisement may still have a ucp_put in flight to the old address, and the NIC
// would write into freed, unregistered memory. Holding them for one extra epoch means
// every put has by then been addressed against a layout the client demonstrably has
// (it used it). Caller must hold rx_pool_->mu.
void UcxCommunicator::retire_and_free_locked(std::uint32_t now_epoch) {
    for (auto &sl : rx_pool_->slots) {
        if (!sl.rma_retired || sl.in_use) continue;
        if (now_epoch <= sl.rma_epoch + 1) continue;  // still within the grace epoch
        unmap_slot_from_ucp(context_, sl);
        free_pinned_host(sl.addr, sl.is_cuda_host);
        free_gpu_slot(sl.gpu_addr);
        ucx_debug_log("rx_pool: freed retired slot (%zu bytes, retired at epoch %u)",
                      sl.capacity, sl.rma_epoch);
        sl = PinnedSlot{};  // capacity 0, not persistent, not retired: reusable entry
    }
}

void UcxCommunicator::init_rx_pool() {
    // 2 slots is enough for the current synchronous request/response pattern
    // (the request occupies slot 0 while the response is in flight; once the
    // app receives the response, slot 0 is free for the next request). Was
    // 4 originally but each cudaHostAlloc(64MB) + ucp_mem_map costs ~75ms;
    // halving the count halves per-connection setup time.
    // Slot count and per-slot capacity are configurable for the async
    // dispatcher (Phase 2): more slots let the frontend keep more fire-and-forget
    // large-H2D copies in flight before it must drain. Defaults preserve the
    // original synchronous behaviour (2 x 1025 MB). For async, prefer more,
    // smaller slots to bound memory, e.g. GVIRTUS_RMA_SLOTS=8
    // GVIRTUS_RMA_SLOT_CAP_MB=128. Note total pinned memory is
    // count * cap * (1 + gpudirect_shadow); keep it modest (GPU-leak sensitive).
    auto env_size = [](const char *k, size_t dflt) -> size_t {
        const char *v = std::getenv(k);
        if (v == nullptr || v[0] == '\0') return dflt;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(v, &end, 10);
        return (parsed > 0) ? static_cast<size_t>(parsed) : dflt;
    };
    const size_t kInitialSlotCount = env_size("GVIRTUS_RMA_SLOTS", 2);

    // Lazy by default: a connection that never sends anything at or above the RMA
    // floor can never use these buffers, so it should not pay for them (2.2 s of
    // connect at the stock capacity) nor hold their GPU shadow away from other
    // tenants. materialise_rma_pool() calls this once a large message proves the
    // connection needs it. GVIRTUS_RMA_PREALLOC=1 restores eager provisioning.
    static const bool prealloc = []() {
        const char *e = std::getenv("GVIRTUS_RMA_PREALLOC");
        return e != nullptr && e[0] != '0';
    }();
    if (!prealloc && !rma_pool_requested_.load(std::memory_order_acquire)) return;

    // Size to the evidence, not to the knob. With no evidence (eager prealloc) this
    // returns the ceiling, i.e. the historical behaviour.
    const size_t kInitialSlotCap = rma_slot_cap_for(
        prealloc ? 0 : rma_pool_hint_bytes_.load(std::memory_order_acquire));

    std::lock_guard<std::mutex> lk(rx_pool_->mu);

    const std::uint32_t new_epoch = rma_pool_epoch_.load(std::memory_order_acquire) + 1;
    // Reclaim anything retired two epochs ago before allocating more.
    retire_and_free_locked(new_epoch);

    // NOT "is the pool empty": the AM receive path appends a message-sized slot for
    // every message that arrives, so by the time a large transfer asks for the pool
    // there are already several tiny slots here. Count the LIVE persistent slots that
    // are actually big enough to serve the RMA path at the capacity we now want.
    size_t full_slots = 0;
    for (const auto &sl : rx_pool_->slots)
        if (sl.rma_persistent && !sl.rma_retired && sl.capacity >= kInitialSlotCap)
            ++full_slots;
    if (full_slots >= kInitialSlotCount) {
        rma_pool_cap_.store(kInitialSlotCap, std::memory_order_release);
        return;  // already provisioned at (at least) this capacity
    }

    // Growing: the live persistent slots are too small for what this connection has
    // turned out to move. Retire them -- stop advertising them and stop handing them
    // out -- but do not free them yet (see retire_and_free_locked).
    for (auto &sl : rx_pool_->slots) {
        if (!sl.rma_persistent || sl.rma_retired) continue;
        if (sl.capacity >= kInitialSlotCap) continue;
        sl.rma_retired = true;
        sl.rma_epoch = new_epoch;
        ucx_debug_log("rx_pool: retiring persistent slot (%zu B) for regrow to %zu B",
                      sl.capacity, kInitialSlotCap);
    }

    // GPUDirect (Step B1): when active, each slot ALSO gets a GPU shadow
    // region of the same capacity, mem_map'd as CUDA. The shadow is unused
    // in B1 — purely additive. Step B2 will advertise its rkey to peers;
    // Step B3 will route big H2D payloads here via NIC peer-DMA.
    const bool gpudirect_active = g_gpudirect_enabled.load();
    size_t gpu_allocated_count = 0;

    // Append: resize() would destroy the small on-demand slots, which may be in use.
    const size_t base = rx_pool_->slots.size();
    rx_pool_->slots.resize(base + (kInitialSlotCount - full_slots));
    for (size_t i = base; i < rx_pool_->slots.size(); ++i) {
        bool is_cuda = false;
        unsigned char *p = alloc_pinned_host(kInitialSlotCap, is_cuda);
        if (p == nullptr) {
            throw std::runtime_error("UcxCommunicator: failed to allocate RX pool slot");
        }
        rx_pool_->slots[i] = PinnedSlot{p, kInitialSlotCap, /*in_use*/false, is_cuda, nullptr};
        rx_pool_->slots[i].rma_persistent = true;  // only these may be ucp_put into
        rx_pool_->slots[i].rma_epoch = new_epoch;

        if (gpudirect_active) {
            unsigned char *gp = alloc_gpu_slot(kInitialSlotCap);
            if (gp != nullptr) {
                rx_pool_->slots[i].gpu_addr = gp;
                rx_pool_->slots[i].gpu_capacity = kInitialSlotCap;
                ++gpu_allocated_count;
            } else {
                ucx_debug_log("rx_pool: slot %zu gpu shadow alloc FAILED — host-only", i);
            }
        }

        map_slot_to_ucp(context_, rx_pool_->slots[i]);
    }
    if (gpudirect_active) {
        std::fprintf(stderr,
            "[GVS] rx_pool: initialized %zu slots x %zu bytes (host) + %zu/%zu GPU shadows x %zu bytes\n",
            kInitialSlotCount, kInitialSlotCap,
            gpu_allocated_count, kInitialSlotCount, kInitialSlotCap);
    }
    ucx_debug_log("rx_pool: initialized %zu slots x %zu bytes (gpu_shadows=%zu)",
                  kInitialSlotCount, kInitialSlotCap, gpu_allocated_count);
    rma_pool_cap_.store(kInitialSlotCap, std::memory_order_release);
}

// Build the slot pool and advertise it to the peer. MUST be called from a context that
// does not hold worker_mutex_: send_rma_setup() takes it, and the allocation itself
// would stall ucp_worker_progress for ~150 ms if it ran in the AM callback.
void UcxCommunicator::materialise_rma_pool() {
    // Re-entrant by design: the pool is built the first time a large message arrives
    // and REBUILT, larger, if a later message turns out not to fit. The build and the
    // advertisement must be atomic with respect to a second grow request, or two
    // threads could interleave and publish a layout that does not match the pool.
    std::lock_guard<std::mutex> lk(rma_build_mu_);

    const size_t want = rma_slot_cap_for(
        rma_pool_hint_bytes_.load(std::memory_order_acquire));
    if (rma_pool_ready_.load(std::memory_order_acquire) &&
        rma_pool_cap_.load(std::memory_order_acquire) >= want) {
        // Another thread already grew it to at least what we need.
        rma_pool_requested_.store(false, std::memory_order_release);
        return;
    }

    init_rx_pool();
    send_rma_setup();  // bumps the epoch; only live (non-retired) slots are published
    rma_pool_ready_.store(true, std::memory_order_release);
    rma_pool_requested_.store(false, std::memory_order_release);
}

void UcxCommunicator::destroy_rx_pool() {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    for (auto &slot : rx_pool_->slots) {
        unmap_slot_from_ucp(context_, slot);
        free_pinned_host(slot.addr, slot.is_cuda_host);
        free_gpu_slot(slot.gpu_addr);  // no-op if nullptr
        slot.gpu_addr = nullptr;
        slot.gpu_capacity = 0;
    }
    rx_pool_->slots.clear();
}

// Find a free slot of at least `needed` bytes. Grows an existing in-use-free
// slot's capacity if the largest is too small, or appends a new slot if all
// are busy. Returns slot index.
size_t UcxCommunicator::acquire_rx_slot(size_t needed) {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);

    // Try to find a free slot big enough. Persistent RMA slots are excluded: a peer
    // can be mid-put into one without the server knowing, so handing it to an incoming
    // eager AM would overwrite the transfer.
    for (size_t i = 0; i < rx_pool_->slots.size(); ++i) {
        if (rx_pool_->slots[i].rma_persistent) continue;
        if (!rx_pool_->slots[i].in_use && rx_pool_->slots[i].capacity >= needed) {
            rx_pool_->slots[i].in_use = true;
            return i;
        }
    }
    // GPUDirect (Step B1): mirror host grow with a GPU shadow grow if active.
    const bool gpudirect_active = g_gpudirect_enabled.load();

    // No free slot big enough — grow the first free one (or append if none free).
    for (size_t i = 0; i < rx_pool_->slots.size(); ++i) {
        if (rx_pool_->slots[i].rma_persistent) continue;  // never repurpose an RMA slot
        if (!rx_pool_->slots[i].in_use) {
            unmap_slot_from_ucp(context_, rx_pool_->slots[i]);
            free_pinned_host(rx_pool_->slots[i].addr, rx_pool_->slots[i].is_cuda_host);
            free_gpu_slot(rx_pool_->slots[i].gpu_addr);
            bool is_cuda = false;
            unsigned char *p = alloc_pinned_host(needed, is_cuda);
            if (p == nullptr) {
                throw std::runtime_error("UcxCommunicator: rx_pool grow failed");
            }
            rx_pool_->slots[i] = PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr};
            if (gpudirect_active) {
                unsigned char *gp = alloc_gpu_slot(needed);
                if (gp != nullptr) {
                    rx_pool_->slots[i].gpu_addr = gp;
                    rx_pool_->slots[i].gpu_capacity = needed;
                }
            }
            map_slot_to_ucp(context_, rx_pool_->slots[i]);
            ucx_debug_log("rx_pool: grew slot %zu to %zu bytes (gpu=%s)",
                          i, needed,
                          rx_pool_->slots[i].gpu_addr ? "yes" : "no");
            return i;
        }
    }
    // All slots in use — append a new one.
    bool is_cuda = false;
    unsigned char *p = alloc_pinned_host(needed, is_cuda);
    if (p == nullptr) {
        throw std::runtime_error("UcxCommunicator: rx_pool append failed");
    }
    rx_pool_->slots.push_back(PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr});
    size_t idx = rx_pool_->slots.size() - 1;
    if (gpudirect_active) {
        unsigned char *gp = alloc_gpu_slot(needed);
        if (gp != nullptr) {
            rx_pool_->slots[idx].gpu_addr = gp;
            rx_pool_->slots[idx].gpu_capacity = needed;
        }
    }
    map_slot_to_ucp(context_, rx_pool_->slots[idx]);
    ucx_debug_log("rx_pool: appended slot %zu (%zu bytes, gpu=%s), total=%zu",
                  idx, needed,
                  rx_pool_->slots[idx].gpu_addr ? "yes" : "no",
                  rx_pool_->slots.size());
    return idx;
}

void UcxCommunicator::release_rx_slot(size_t slot_idx) {
    bool send_ack = false;
    std::uint64_t gen = 0;
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        if (slot_idx >= rx_pool_->slots.size()) return;
        auto &s = rx_pool_->slots[slot_idx];
        s.in_use = false;
        // If this slot was filled by a client RMA put, the client is waiting
        // (or will wait) for an explicit consumption ack before reusing it.
        if (s.rma_origin) {
            send_ack = true;
            gen = s.rma_generation;
            s.rma_origin = false;
        }
    }
    // Sent OUTSIDE rx_pool_->mu: send_slot_consumed takes worker_mutex_, and no
    // release_rx_slot caller holds it, so there is no lock-order inversion.
    if (send_ack) send_slot_consumed(slot_idx, gen);
}

// Client side. The backend confirmed (SlotConsumed) that it finished consuming
// the data we RMA-put into remote slot `slot_idx`. Return the slot to Free so a
// WriteIovRma waiter can reuse it — but ONLY if the generation still matches.
// A stale/duplicate ack (ABA: the slot was already freed and re-acquired for a
// newer op) must be ignored, or it would corrupt an unrelated in-flight write.
void UcxCommunicator::release_remote_slot(size_t server_idx,
                                          std::uint64_t tag) {
    using gvirtus::communicators::ucxam::slot_tag_epoch;
    using gvirtus::communicators::ucxam::slot_tag_generation;

    std::lock_guard<std::mutex> lk(rma_state_mu_);

    const std::uint32_t ack_epoch = slot_tag_epoch(tag);
    const std::uint64_t generation = slot_tag_generation(tag);

    // Epoch guard. An ack minted against a layout we have already replaced must not
    // touch the current one: slot ids are not stable across a regrow, so matching by
    // id alone could mark a NEW slot free while the backend is still consuming it.
    // Epoch 0 means the peer predates this scheme -- accept it, matching the old
    // generation-only behaviour.
    if (ack_epoch != 0 && remote_epoch_ != 0 && ack_epoch != remote_epoch_) {
        ucx_debug_log("SlotConsumed: dropping ack from epoch %u (current %u)",
                      ack_epoch, remote_epoch_);
        return;
    }

    // Slots are addressed by the SERVER's index, not our position in the vector.
    for (auto &s : remote_slots_) {
        if (s.server_idx != static_cast<std::uint16_t>(server_idx)) continue;
        if (s.state == RemoteSlot::State::InFlight && s.generation == generation) {
            s.state = RemoteSlot::State::Free;
        }
        // else: stale/duplicate ack — ignore (ABA guard).
        break;
    }

    // A layout parked by handle_rma_setup_am is installed as soon as the last
    // in-flight transfer drains. Doing it here, rather than making the advertisement
    // wait, keeps the AM callback non-blocking.
    if (rma_swap_pending_) {
        bool any_inflight = false;
        for (const auto &rs : remote_slots_)
            if (rs.state == RemoteSlot::State::InFlight) { any_inflight = true; break; }
        if (!any_inflight) apply_pending_slots_locked();
    }
    rma_slot_cv_.notify_all();
}

// Server side. Notify the client that RMA-origin slot `slot_idx` (at the given
// generation) has been fully consumed and may be reused.
void UcxCommunicator::send_slot_consumed(size_t slot_idx,
                                         std::uint64_t generation) {
    if (endpoint_ == nullptr || worker_ == nullptr) return;
    gvirtus::communicators::ucxam::EnvelopeHeader ack{};
    ack.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
    ack.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
    ack.message_type = static_cast<std::uint16_t>(
        gvirtus::communicators::ucxam::MessageType::SlotConsumed);
    ack.header_size = sizeof(ack);
    ack.reserved0 = static_cast<std::uint16_t>(slot_idx);
    ack.request_id = generation;
    ucp_request_param_t sp{};
    sp.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    sp.datatype = ucp_dt_make_contig(1);
    std::lock_guard<std::mutex> wl(*worker_mutex_);
    void *req = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                &ack, sizeof(ack), &sp);
    try {
        wait_request_completion(req, "slot_consumed_ack");
    } catch (const std::exception &e) {
        ucx_debug_log("send_slot_consumed: %s", e.what());
    }
}

// Async H2D Phase 3 drain point. The MemcpyAsync handler issues a fire-and-
// forget D2D from a GPU shadow slot without synchronizing and sets
// tls_async_gpu_pending. Consecutive such copies overlap; but before we send a
// response-bearing reply (the frontend's flow control treats every sync reply as
// "all prior RMA slots drained"), we must block until those in-flight D2Ds have
// fully read their source slots — otherwise the frontend could reuse a remote
// slot and the NIC would peer-DMA fresh data over a shadow still being read.
// cudaDeviceSynchronize is the coarse-but-correct drain; it only runs at a sync
// point AND only when a fire-and-forget GPU copy is actually pending, so its cost
// is amortized across the whole in-flight batch (typically the ring depth).
void UcxCommunicator::drain_device_if_async_pending() {
    if (!gvirtus::communicators::tls_async_gpu_pending) return;
    gvirtus::communicators::tls_async_gpu_pending = false;
    auto fn = g_cuda_device_sync.load();
    if (fn != nullptr) fn();
}

// True iff the peer advertised RMA slots and at least one has a usable rkey.
// When the peer's RmaSetup rkey failed to unpack (e.g. a native frontend whose
// UCX exposes no RMA-unpackable md), every remote slot has rkey == null and we
// cannot ucp_put into them — the backend must then NOT take the D2H GPU-scratch
// path (its device fragment would fall through to the AM path and error).
bool UcxCommunicator::rma_put_capable() const {
    // Cached at RmaSetup time (handle_rma_setup_am) so this is a single atomic
    // load — Process.cpp queries it before every RPC, so keep it O(1).
    return rma_put_capable_.load();
}

// Server-side: pack rkeys of every rx_slot, build an RmaSetup AM body, and
// send it to the connected client. Called once per accepted connection,
// right after the endpoint is created (so the client receives this before
// any data traffic).
void UcxCommunicator::send_rma_setup() {
    if (endpoint_ == nullptr || context_ == nullptr) return;

    // Snapshot rx slot metadata. With GPUDirect Step B2 each slot may also
    // expose a GPU shadow (gpu_addr / gpu_capacity / gpu_rkey).
    struct PackedSlot {
        std::uint64_t addr;
        std::uint64_t capacity;
        void *rkey_buf{nullptr};
        size_t rkey_len{0};
        // GPU shadow (optional). When gpu_rkey_buf == nullptr the slot
        // advertises host only — matches the pre-B2 wire format byte for byte.
        std::uint64_t gpu_addr{0};
        std::uint64_t gpu_capacity{0};
        bool persistent{false};
        std::uint16_t server_idx{0};
        void *gpu_rkey_buf{nullptr};
        size_t gpu_rkey_len{0};
    };
    std::vector<PackedSlot> packed;
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        packed.reserve(rx_pool_->slots.size());
        for (size_t si = 0; si < rx_pool_->slots.size(); ++si) {
            auto &slot = rx_pool_->slots[si];
            if (slot.memh == nullptr) continue;  // never mapped, or reclaimed
            // Advertise only slots the peer may actually use: the live persistent
            // pool. The message-sized slots the eager AM path appends are ours alone,
            // and a RETIRED slot must not be advertised at all -- a larger pool has
            // superseded it and it is waiting out its grace epoch before being freed.
            //
            // Skipping entries is safe now only because slot identity is explicit:
            // each descriptor carries the server's own index in the top 16 bits of
            // reserved0, and RmaPosted / SlotConsumed travel with it. Under the old
            // positional mapping this loop's `continue` on a rkey_pack failure
            // silently shifted every later slot by one.
            if (!slot.rma_persistent || slot.rma_retired) continue;
            PackedSlot ps{};
            ps.addr = reinterpret_cast<std::uint64_t>(slot.addr);
            ps.capacity = slot.capacity;
            ps.persistent = slot.rma_persistent;
            ps.server_idx = static_cast<std::uint16_t>(si);
            ucs_status_t st = ucp_rkey_pack(context_, slot.memh,
                                            &ps.rkey_buf, &ps.rkey_len);
            if (st != UCS_OK) {
                std::fprintf(stderr,
                             "UCX rma_setup: ucp_rkey_pack failed (%s)\n",
                             ucs_status_string(st));
                continue;
            }
            // Pack GPU shadow rkey if present.
            if (slot.gpu_memh != nullptr && slot.gpu_addr != nullptr) {
                ucs_status_t gst = ucp_rkey_pack(context_, slot.gpu_memh,
                                                 &ps.gpu_rkey_buf, &ps.gpu_rkey_len);
                if (gst == UCS_OK) {
                    ps.gpu_addr = reinterpret_cast<std::uint64_t>(slot.gpu_addr);
                    ps.gpu_capacity = slot.gpu_capacity;
                } else {
                    ucx_debug_log("rma_setup: gpu rkey_pack FAILED (%s) — advertising host only",
                                  ucs_status_string(gst));
                    ps.gpu_rkey_buf = nullptr;
                }
            }
            packed.push_back(ps);
        }
    }

    // An EMPTY advertisement is meaningful and must still be sent: the peer's Connect()
    // blocks until it arrives, so skipping it costs the full 2 s handshake timeout on
    // every connection. num_slots = 0 tells the peer "no usable slots, use the eager
    // path", which is the correct state until the pool is built on demand.
    if (packed.empty()) {
        ucx_debug_log("rma_setup: advertising an empty pool (slots are built on demand)");
    }

    // Assemble AM body: [EnvelopeHeader] [N * RmaSlotDescriptor] [N * rkey blobs]
    using gvirtus::communicators::ucxam::EnvelopeHeader;
    using gvirtus::communicators::ucxam::MessageType;
    using gvirtus::communicators::ucxam::RmaSlotDescriptor;
    using gvirtus::communicators::ucxam::kEnvelopeMagic;
    using gvirtus::communicators::ucxam::kEnvelopeVersion;

    // Wire format (Step B2 extension):
    //   [EnvelopeHeader]
    //   [N * RmaSlotDescriptor]     ← per-slot header; descriptor.reserved0
    //                                  bit 0 = "has_gpu_shadow" flag
    //   For each slot in order:
    //     [host_rkey_blob (rkey_size bytes)]
    //     If has_gpu_shadow:
    //       [u64 gpu_addr][u64 gpu_capacity][u32 gpu_rkey_size][gpu_rkey_blob]
    //
    // Old peers (pre-B2) see descriptor.reserved0=0 always → no GPU block →
    // identical to pre-B2 layout.
    constexpr std::uint32_t kHasGpuShadow = 1u << 0;
    constexpr std::uint32_t kSlotPersistent = 1u << 1;  // safe for the peer to ucp_put into

    size_t descriptors_bytes = packed.size() * sizeof(RmaSlotDescriptor);
    size_t rkeys_bytes = 0;
    size_t gpu_extension_bytes = 0;
    for (auto &p : packed) {
        rkeys_bytes += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            gpu_extension_bytes += sizeof(std::uint64_t)  // gpu_addr
                                 + sizeof(std::uint64_t)  // gpu_capacity
                                 + sizeof(std::uint32_t)  // gpu_rkey_size
                                 + p.gpu_rkey_len;
        }
    }
    size_t total_bytes = sizeof(EnvelopeHeader) + descriptors_bytes + rkeys_bytes + gpu_extension_bytes;

    std::vector<unsigned char> buf(total_bytes);
    auto *hdr = reinterpret_cast<EnvelopeHeader *>(buf.data());
    hdr->magic = kEnvelopeMagic;
    hdr->version = kEnvelopeVersion;
    hdr->message_type = static_cast<std::uint16_t>(MessageType::RmaSetup);
    hdr->header_size = sizeof(EnvelopeHeader);
    hdr->reserved0 = 0;
    // Epoch of THIS layout. Bumped on every advertisement, delivered to the client,
    // and echoed by it on every RmaPosted so an ack minted against a superseded
    // layout can be recognised and dropped instead of freeing a live slot.
    const std::uint32_t epoch =
        rma_pool_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    hdr->status_code = epoch;
    hdr->request_id = 0;
    hdr->routine_size = 0;
    hdr->payload_size = static_cast<std::uint64_t>(packed.size());

    size_t off = sizeof(EnvelopeHeader);
    for (auto &p : packed) {
        RmaSlotDescriptor d{};
        d.remote_addr = p.addr;
        d.slot_capacity = p.capacity;
        d.rkey_size = static_cast<std::uint32_t>(p.rkey_len);
        d.reserved0 = (p.gpu_rkey_buf != nullptr) ? kHasGpuShadow : 0u;
        if (p.persistent) d.reserved0 |= kSlotPersistent;
        // Top 16 bits: the server's own index for this slot (see RemoteSlot::server_idx).
        d.reserved0 |= (static_cast<std::uint32_t>(p.server_idx) << 16);
        std::memcpy(buf.data() + off, &d, sizeof(d));
        off += sizeof(d);
    }
    // Per-slot rkey blobs, interleaved with optional gpu extension.
    for (auto &p : packed) {
        std::memcpy(buf.data() + off, p.rkey_buf, p.rkey_len);
        off += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            std::memcpy(buf.data() + off, &p.gpu_addr, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::memcpy(buf.data() + off, &p.gpu_capacity, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::uint32_t gsz = static_cast<std::uint32_t>(p.gpu_rkey_len);
            std::memcpy(buf.data() + off, &gsz, sizeof(std::uint32_t));
            off += sizeof(std::uint32_t);
            std::memcpy(buf.data() + off, p.gpu_rkey_buf, p.gpu_rkey_len);
            off += p.gpu_rkey_len;
        }
    }

    // Send as a single AM.
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = ucp_dt_make_contig(1);
        void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                        buf.data(), buf.size(), &request_param);
        wait_request_completion(request, "rma_setup_send");
    }

    // Release the packed rkey buffers (host + optional GPU).
    size_t gpu_advertised = 0;
    for (auto &p : packed) {
        if (p.rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.rkey_buf);
        }
        if (p.gpu_rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.gpu_rkey_buf);
            ++gpu_advertised;
        }
    }
    ucx_debug_log("rma_setup: advertised %zu slots (%zu rkey bytes, %zu with gpu shadow)",
                  packed.size(), rkeys_bytes, gpu_advertised);
}

// Client-side: parse an incoming RmaSetup AM body, unpack each rkey, and
// populate remote_slots_. After this returns the data path can use ucp_put.
void UcxCommunicator::handle_rma_setup_am(const void *data, size_t length) {
    using gvirtus::communicators::ucxam::EnvelopeHeader;
    using gvirtus::communicators::ucxam::RmaSlotDescriptor;

    if (length < sizeof(EnvelopeHeader)) {
        std::fprintf(stderr, "RmaSetup: body too short (%zu)\n", length);
        return;
    }
    EnvelopeHeader hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    const size_t num_slots = static_cast<size_t>(hdr.payload_size);
    const size_t descriptors_bytes = num_slots * sizeof(RmaSlotDescriptor);
    if (length < sizeof(hdr) + descriptors_bytes) {
        std::fprintf(stderr, "RmaSetup: descriptors truncated\n");
        return;
    }

    const auto *base = static_cast<const unsigned char *>(data);
    const auto *desc_ptr = reinterpret_cast<const RmaSlotDescriptor *>(
        base + sizeof(hdr));
    const unsigned char *rkey_cursor = base + sizeof(hdr) + descriptors_bytes;
    const unsigned char *rkey_end = base + length;

    constexpr std::uint32_t kHasGpuShadow = 1u << 0;
    constexpr std::uint32_t kSlotPersistent = 1u << 1;  // safe for the peer to ucp_put into

    std::vector<RemoteSlot> new_slots;
    new_slots.reserve(num_slots);
    size_t gpu_received = 0;
    for (size_t i = 0; i < num_slots; ++i) {
        if (rkey_cursor + desc_ptr[i].rkey_size > rkey_end) {
            std::fprintf(stderr, "RmaSetup: rkey blob %zu truncated\n", i);
            break;
        }
        ucp_rkey_h rkey = nullptr;
        ucs_status_t st = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &rkey);
        if (st != UCS_OK) {
            std::fprintf(stderr,
                         "RmaSetup: ucp_ep_rkey_unpack[%zu] failed (%s)\n",
                         i, ucs_status_string(st));
            rkey = nullptr;
        }
        rkey_cursor += desc_ptr[i].rkey_size;

        RemoteSlot rs{desc_ptr[i].remote_addr,
                      desc_ptr[i].slot_capacity, rkey};
        rs.persistent = (desc_ptr[i].reserved0 & kSlotPersistent) != 0;
        rs.server_idx = static_cast<std::uint16_t>(desc_ptr[i].reserved0 >> 16);

        // GPUDirect Step B2: parse optional GPU extension after the host
        // rkey blob if the descriptor's flag bit is set. Old peers don't
        // set this flag → no extension to read → rs.gpu_rkey stays null.
        if ((desc_ptr[i].reserved0 & kHasGpuShadow) != 0u) {
            const size_t kFixedExt = sizeof(std::uint64_t)  // gpu_addr
                                   + sizeof(std::uint64_t)  // gpu_capacity
                                   + sizeof(std::uint32_t); // gpu_rkey_size
            if (rkey_cursor + kFixedExt > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu extension header %zu truncated\n", i);
                break;
            }
            std::uint64_t gpu_addr = 0, gpu_cap = 0;
            std::uint32_t gpu_rkey_size = 0;
            std::memcpy(&gpu_addr,      rkey_cursor + 0,  sizeof(std::uint64_t));
            std::memcpy(&gpu_cap,       rkey_cursor + 8,  sizeof(std::uint64_t));
            std::memcpy(&gpu_rkey_size, rkey_cursor + 16, sizeof(std::uint32_t));
            rkey_cursor += kFixedExt;
            if (rkey_cursor + gpu_rkey_size > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu rkey blob %zu truncated\n", i);
                break;
            }
            ucp_rkey_h gpu_rkey = nullptr;
            ucs_status_t gst = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &gpu_rkey);
            if (gst == UCS_OK) {
                rs.gpu_addr     = gpu_addr;
                rs.gpu_capacity = gpu_cap;
                rs.gpu_rkey     = gpu_rkey;
                ++gpu_received;
            } else {
                std::fprintf(stderr,
                             "RmaSetup: gpu rkey unpack[%zu] failed (%s), skipping gpu path\n",
                             i, ucs_status_string(gst));
            }
            rkey_cursor += gpu_rkey_size;
        }

        new_slots.push_back(rs);
    }

    // Cache RMA-put capability once here (any slot with a usable rkey) so the
    // per-RPC rma_put_capable() query is a single atomic load. Computed from
    // new_slots before the move below.
    bool put_capable = false;
    for (const auto &rs : new_slots)
        if (rs.rkey != nullptr) { put_capable = true; break; }

    const std::uint32_t new_epoch = hdr.status_code;
    bool applied = false;
    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        // Swapping the layout while a slot is InFlight is NOT safe: that slot would
        // come back Free in the new vector although the backend is still consuming
        // it, and the very next WriteIovRma would ucp_put on top of a live transfer.
        // Park the new layout and let release_remote_slot() install it once the last
        // in-flight transfer has been acknowledged. Meanwhile WriteIovRma stops
        // handing out slots, so the drain is guaranteed to finish.
        bool any_inflight = false;
        for (const auto &rs : remote_slots_)
            if (rs.state == RemoteSlot::State::InFlight) { any_inflight = true; break; }

        if (any_inflight) {
            // A second advertisement arriving while an earlier one is still parked
            // supersedes it; drop the older parked rkeys rather than leak them.
            if (rma_swap_pending_) {
                for (auto &rs : pending_slots_) {
                    if (rs.rkey != nullptr) ucp_rkey_destroy(rs.rkey);
                    if (rs.gpu_rkey != nullptr) ucp_rkey_destroy(rs.gpu_rkey);
                }
            }
            pending_slots_ = std::move(new_slots);
            pending_epoch_ = new_epoch;
            rma_swap_pending_ = true;
            ucx_debug_log("rma_setup: epoch %u parked (transfers in flight)", new_epoch);
        } else {
            pending_slots_ = std::move(new_slots);
            pending_epoch_ = new_epoch;
            rma_swap_pending_ = true;
            apply_pending_slots_locked();
            applied = true;
        }
        rma_setup_received_.store(true);
    }
    if (applied) rma_put_capable_.store(put_capable);
    rma_setup_cv_.notify_all();
    rma_slot_cv_.notify_all();
    ucx_debug_log("rma_setup: epoch %u, %zu remote slots (%zu with gpu shadow)%s",
                  new_epoch, remote_slots_.size(), gpu_received,
                  applied ? "" : " [deferred]");
}

// Install the parked layout. Caller holds rma_state_mu_ and must have established
// that no slot is InFlight. Destroys the outgoing rkeys -- the previous code moved a
// new vector over the old one and leaked every rkey it held, which was harmless only
// because a second advertisement never happened.
void UcxCommunicator::apply_pending_slots_locked() {
    if (!rma_swap_pending_) return;
    for (auto &rs : remote_slots_) {
        if (rs.rkey != nullptr) ucp_rkey_destroy(rs.rkey);
        if (rs.gpu_rkey != nullptr) ucp_rkey_destroy(rs.gpu_rkey);
    }
    remote_slots_ = std::move(pending_slots_);
    pending_slots_.clear();
    remote_epoch_ = pending_epoch_;
    rma_swap_pending_ = false;
    next_remote_slot_idx_ = 0;

    bool put_capable = false;
    for (const auto &rs : remote_slots_)
        if (rs.rkey != nullptr) { put_capable = true; break; }
    rma_put_capable_.store(put_capable);
    ucx_debug_log("rma_setup: epoch %u installed (%zu slots)", remote_epoch_,
                  remote_slots_.size());
}

void UcxCommunicator::destroy_rma_state() {
    std::lock_guard<std::mutex> lk(rma_state_mu_);
    for (auto &rs : remote_slots_) {
        if (rs.rkey != nullptr) ucp_rkey_destroy(rs.rkey);
        if (rs.gpu_rkey != nullptr) ucp_rkey_destroy(rs.gpu_rkey);
    }
    remote_slots_.clear();
    rma_setup_received_.store(false);
    rma_put_capable_.store(false);
    next_remote_slot_idx_ = 0;
}

// D2H-via-GET, server side. Register the backend's GPU scratch [gpu_addr,len)
// for remote RDMA-READ (ucp_mem_map CUDA) and pack its rkey so the client can
// ucp_get_nbx from it directly. The registration is cached per device address
// (the TLS gpu scratch is reused / grows monotonically), so steady state pays
// only a cheap ucp_rkey_pack. Passive-responder side: no active send-from-cuda
// proto is constructed here, so this works under the forced rcache-off config
// that blocks the server ucp_put-from-cuda path. Returns false on any failure;
// the caller then keeps the legacy (host-staged / put) response path.
bool UcxCommunicator::PrepareGpuGet(void *gpu_addr, size_t len,
                                    std::uint64_t &out_remote_addr,
                                    std::vector<char> &out_rkey) {
    if (context_ == nullptr || gpu_addr == nullptr || len == 0) return false;

    std::lock_guard<std::mutex> lk(gpu_get_mu_);
    const std::uint64_t key = reinterpret_cast<std::uint64_t>(gpu_addr);
    auto it = gpu_get_regs_.find(key);
    if (it != gpu_get_regs_.end() && it->second.len < len) {
        // Same base address but a larger transfer than we registered: remap.
        ucp_mem_unmap(context_, it->second.memh);
        gpu_get_regs_.erase(it);
        it = gpu_get_regs_.end();
    }
    if (it == gpu_get_regs_.end()) {
        ucp_mem_map_params_t p{};
        p.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        p.address = gpu_addr;
        p.length = len;
        p.memory_type = UCS_MEMORY_TYPE_CUDA;
        ucp_mem_h memh = nullptr;
        ucs_status_t st = ucp_mem_map(context_, &p, &memh);
        if (st != UCS_OK) {
            ucx_debug_log("PrepareGpuGet: ucp_mem_map(CUDA) failed: %s",
                          ucs_status_string(st));
            return false;
        }
        it = gpu_get_regs_.emplace(key, GpuGetReg{len, memh}).first;
    }

    void *rkey_buf = nullptr;
    size_t rkey_size = 0;
    ucs_status_t st = ucp_rkey_pack(context_, it->second.memh, &rkey_buf, &rkey_size);
    if (st != UCS_OK || rkey_buf == nullptr) {
        ucx_debug_log("PrepareGpuGet: ucp_rkey_pack failed: %s", ucs_status_string(st));
        return false;
    }
    out_rkey.assign(reinterpret_cast<char *>(rkey_buf),
                    reinterpret_cast<char *>(rkey_buf) + rkey_size);
    ucp_rkey_buffer_release(rkey_buf);
    out_remote_addr = key;
    return true;
}

// D2H-via-GET, client side. Unpack the server-supplied rkey against our
// endpoint and RDMA-GET `count` bytes from the server's GPU scratch straight
// into `dst_host` (the caller's pinned host buffer — UCX registers it on the
// fly; the client-side rcache works). Single-threaded per connection (same
// contract as WriteIovRma), so no worker_mutex_ needed. Returns false on error.
bool UcxCommunicator::GetFromRemoteGpu(void *dst_host, std::uint64_t remote_addr,
                                       const void *rkey_blob, size_t rkey_len,
                                       size_t count) {
    if (endpoint_ == nullptr || dst_host == nullptr || rkey_blob == nullptr ||
        rkey_len == 0 || count == 0) {
        return false;
    }
    ucp_rkey_h rkey = nullptr;
    ucs_status_t st = ucp_ep_rkey_unpack(endpoint_, rkey_blob, &rkey);
    if (st != UCS_OK) {
        ucx_debug_log("GetFromRemoteGpu: ucp_ep_rkey_unpack failed: %s",
                      ucs_status_string(st));
        return false;
    }

    // Register the destination host buffer ourselves and cache the memh, keyed
    // by address (grow-remap like the server scratch). Passed to ucp_get_nbx as
    // a memh hint so UCX does NOT re-register the dst on every call/fragment —
    // the broken rcache can't cache it (rcache=y errors "Bad address"), so
    // without this the per-op registration dominates and D2H collapses at large
    // sizes (64 MB fell to ~1 GB/s). D2H reuses the same dst, so this registers
    // once and every subsequent GET is a pure line-rate RDMA READ.
    ucp_mem_h dst_memh = nullptr;
    if (context_ != nullptr) {
        std::lock_guard<std::mutex> lk(client_dst_mu_);
        const std::uint64_t key = reinterpret_cast<std::uint64_t>(dst_host);
        auto it = client_dst_regs_.find(key);
        if (it != client_dst_regs_.end() && it->second.len < count) {
            ucp_mem_unmap(context_, it->second.memh);
            client_dst_regs_.erase(it);
            it = client_dst_regs_.end();
        }
        if (it == client_dst_regs_.end()) {
            ucp_mem_map_params_t mp{};
            mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
            mp.address = dst_host;
            mp.length = count;
            ucp_mem_h m = nullptr;
            ucs_status_t mst = ucp_mem_map(context_, &mp, &m);
            if (mst == UCS_OK) {
                it = client_dst_regs_.emplace(key, GpuGetReg{count, m}).first;
            } else {
                ucx_debug_log("GetFromRemoteGpu: dst ucp_mem_map failed: %s "
                              "(falling back to on-the-fly reg)",
                              ucs_status_string(mst));
            }
        }
        if (it != client_dst_regs_.end()) dst_memh = it->second.memh;
    }

    ucp_request_param_t param{};
    if (dst_memh != nullptr) {
        param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        param.memh = dst_memh;
    } else {
        param.op_attr_mask = 0;
    }
    void *req = ucp_get_nbx(endpoint_, dst_host, count, remote_addr, rkey, &param);
    try {
        wait_request_completion(req, "d2h_get");
    } catch (const std::exception &e) {
        ucx_debug_log("GetFromRemoteGpu: %s", e.what());
        ucp_rkey_destroy(rkey);
        return false;
    }
    ucp_rkey_destroy(rkey);
    return true;
}

// RMA-mode send. Two paths, selected by env var GVIRTUS_RMA_ZEROCOPY:
//
//  * "staged" (GVIRTUS_RMA_ZEROCOPY=0): copy ALL iov fragments into the
//     pre-registered local tx_scratch_, then one ucp_put_nbx of the
//     contiguous buffer. Simple, predictable. Pays a host-RAM memcpy of
//     ~2.5ms for a 64MB payload, but no per-call buffer registration.
//
//  * "zerocopy" (default, or GVIRTUS_RMA_ZEROCOPY=1): only the small
//     fragments (header, routine) are staged into tx_scratch_; the large
//     fragment (the user's payload) is ucp_put_nbx'd directly from the
//     caller's buffer. Two puts in flight in parallel. UCX rcache caches
//     the user buffer's registration after first use — steady state is
//     ~2.5ms faster per call. First call against a fresh user buffer pays
//     a one-time registration cost (typically ~10ms for 64MB).
//
// Both paths end with a tiny RmaPosted AM carrying the slot index.
size_t UcxCommunicator::WriteIovRma(const struct iovec *iov, size_t iov_count,
                                    size_t total) {
    // Acquire a remote slot with EXPLICIT ownership. The old round-robin assumed
    // a strictly synchronous request/response so the slot was already consumed
    // by the time we wrapped back to it; that invariant breaks under concurrent
    // / async-dispatched prefill (multiple RMA writes in flight), reusing a slot
    // the backend hasn't finished with -> QP error (rma_put_pre EIO crash). Now:
    // wait for a Free slot (backpressure), flip it InFlight, bump its generation.
    // It returns to Free only on the backend's SlotConsumed ack — a local UCX
    // put completion does NOT imply the remote app released the buffer.
    size_t slot_idx;
    std::uint64_t slot_gen;
    std::uint64_t slot_tag;
    RemoteSlot rs;
    {
        std::unique_lock<std::mutex> lk(rma_state_mu_);
        if (remote_slots_.empty()) return 0;
        size_t found = static_cast<size_t>(-1);
        // Pick a free slot that actually FITS. Taking the first free slot and then
        // giving up on capacity means one undersized slot at a low index sends every
        // large transfer down the eager path for the lifetime of the connection --
        // which is exactly what happened once the pool started holding a mix of
        // full-capacity slots and the message-sized ones acquire_rx_slot() appends.
        const bool got = rma_slot_cv_.wait_for(
            lk, std::chrono::seconds(30), [&] {
                if (endpoint_failed_.load()) return true;
                // A new layout is parked waiting for the in-flight transfers to
                // drain. Handing out another slot from the outgoing layout would
                // keep the drain from ever completing, so stop issuing until the
                // swap has been installed. Slots still in flight will release and
                // install it; if something goes wrong the 30 s timeout falls back
                // to the eager path rather than wedging.
                if (rma_swap_pending_) return false;
                bool any_free = false;
                // If the peer tags persistent slots, restrict puts to those. If it tags
                // none (an older peer), keep the previous behaviour rather than refusing
                // to use RMA at all.
                bool any_persistent = false;
                for (const auto &r : remote_slots_)
                    if (r.persistent) { any_persistent = true; break; }
                for (size_t i = 0; i < remote_slots_.size(); ++i) {
                    if (remote_slots_[i].state != RemoteSlot::State::Free) continue;
                    any_free = true;
                    if (remote_slots_[i].rkey != nullptr &&
                        (!any_persistent || remote_slots_[i].persistent) &&
                        remote_slots_[i].capacity >= total) {
                        found = i;
                        return true;
                    }
                }
                // Every slot is free and none is big enough: waiting cannot help, so
                // stop instead of burning the 30 s timeout before falling back.
                if (any_free) {
                    bool all_free = true;
                    for (const auto &r : remote_slots_)
                        if (r.state != RemoteSlot::State::Free) { all_free = false; break; }
                    if (all_free) return true;
                }
                return false;
            });
        if (!got || endpoint_failed_.load() || found == static_cast<size_t>(-1)) {
            return 0;  // backpressure timeout / endpoint failure -> IOV fallback
        }
        slot_idx = found;
        remote_slots_[slot_idx].state = RemoteSlot::State::InFlight;
        slot_gen = ++remote_slots_[slot_idx].generation;
        rs = remote_slots_[slot_idx];
        // Tag every RmaPosted with the epoch of the layout it was addressed against,
        // so the returning SlotConsumed can be matched against that layout and not
        // whatever has replaced it.
        slot_tag = gvirtus::communicators::ucxam::make_slot_tag(remote_epoch_, slot_gen);
    }

    // RAII: if we bail before success (capacity fallback, or ANY throw during
    // the puts) return the slot to Free so it never leaks (a leak would
    // eventually wedge backpressure). Disarmed on the success path, where the
    // slot is instead freed later by the backend's SlotConsumed ack.
    bool rma_committed = false;
    struct SlotReleaser {
        UcxCommunicator *self; size_t srv_idx; std::uint64_t tag; bool *committed;
        ~SlotReleaser() { if (!*committed) self->release_remote_slot(srv_idx, tag); }
    } slot_releaser{this, rs.server_idx, slot_tag, &rma_committed};

    if (rs.rkey != nullptr && total > rs.capacity) {
        // Falling off the RMA fast path for size is a 3x-class performance cliff, and
        // it used to be completely silent. Say it once, with both numbers, so it is
        // diagnosable from a normal run instead of only from a bandwidth sweep.
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                "[GVS] RMA fast path declined: message %zu B exceeds slot capacity %zu B "
                "-- falling back to eager AM (much slower for large transfers). Raise "
                "GVIRTUS_RMA_SLOT_CAP_MB above %zu MB to keep the fast path.\n",
                total, static_cast<size_t>(rs.capacity),
                (total + 1024u * 1024u - 1u) / (1024u * 1024u));
        }
    }
    if (rs.rkey == nullptr || total > rs.capacity) {
        // Caller will fall back to the IOV path. (slot freed by SlotReleaser)
        return 0;
    }

    // Env-var gated zerocopy: default OFF (set GVIRTUS_RMA_ZEROCOPY=1 to
    // enable). The zerocopy path relies on UCX's registration cache to
    // make repeated ucp_put_nbx(h_user_buf, ...) cheap. In this container
    // build UCX logs "could not create UCP registration cache: Unsupported
    // operation" at init, which means every put re-registers the source
    // buffer (~25ms for 64MB) — regressing write from ~8ms to ~31ms.
    // The staged path uses a pre-mem_map'd tx_scratch and is rcache-
    // independent, so it stays ~8ms warm regardless. Keep zerocopy behind
    // a flag for builds where rcache works (e.g., once nvidia-peermem and
    // UCM event handling are properly set up in the container).
    static const bool zerocopy_enabled = []() {
        const char *v = std::getenv("GVIRTUS_RMA_ZEROCOPY");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();

    // GPUDirect Step B3: set when the big iov fragment is routed to the
    // peer's GPU shadow. Communicated to the peer via RmaPosted:
    //   routine_size = gpu_split_bytes (= big_size routed to GPU)
    //   status_code  = gpu_split_offset (= pre_size in host slot — the
    //                  position where the GPU data belongs in the logical
    //                  message). Allows biggest to be at ANY iov index, not
    //                  just last (Fase 5 puts user_src at index 3 with a
    //                  trailing [count][kind] = 12-byte input_post).
    std::uint64_t gpu_split_bytes  = 0;
    std::uint32_t gpu_split_offset = 0;

    // Find the biggest iov fragment regardless of position. The zerocopy
    // path treats it as the payload to ucp_put directly from caller memory
    // and stages every other fragment through tx_scratch_. With the legacy
    // 3-entry layout [header][routine][payload] the biggest sits at index
    // iov_count-1, matching the prior behavior. With the Fase 5 layout
    // [header][routine][input_pre][user_src][input_post] the biggest sits
    // at an interior index — argmax catches both.
    size_t biggest_idx = 0;
    size_t big_size = 0;
    for (size_t i = 0; i < iov_count; ++i) {
        if (iov[i].iov_len > big_size) {
            big_size = iov[i].iov_len;
            biggest_idx = i;
        }
    }
    const size_t small_size = (big_size <= total) ? (total - big_size) : 0;
    // Bytes from iov fragments that appear BEFORE the biggest one — they
    // get put to rs.addr + 0. Bytes AFTER the biggest go to
    // rs.addr + pre_size + big_size to preserve wire order.
    size_t pre_size = 0;
    for (size_t i = 0; i < biggest_idx; ++i) pre_size += iov[i].iov_len;
    const size_t post_size = small_size - pre_size;
    // Detect GPU mem in the biggest fragment ONCE here so we can both (a)
    // force zerocopy when GPU is present (staged path would memcpy from GPU
    // into tx_scratch → SIGSEGV) and (b) reuse the value for the memh
    // registration further down.
    const bool big_is_gpu = is_gpu_pointer(iov[biggest_idx].iov_base);

    // Control/data-path gate (GPUDirect B3). Consume the per-message hint set
    // by Frontend::Execute::SetNextDeviceFragment (reset it once here so it
    // never leaks into a later message). The big fragment may be peer-DMA'd
    // into the peer GPU shadow ONLY when it is exactly the Fase-5 device-
    // destined direct-input fragment (today: sync cudaMemcpy H2D — the one
    // routine whose backend handler consumes GetGpuPayload()). Control-path
    // buffers (fatbin, cuModuleLoadData, nvrtc, marshaled args) travel in
    // mpInputBuffer, carry no device fragment, and thus stay in the host slot.
    const void  *dev_frag     = next_dev_frag_addr_;
    const size_t dev_frag_len = next_dev_frag_len_;
    next_dev_frag_addr_ = nullptr;
    next_dev_frag_len_  = 0;
    const bool big_is_device_data = (dev_frag != nullptr) &&
                                    (iov[biggest_idx].iov_base == dev_frag) &&
                                    (big_size == dev_frag_len);

    // Only worth splitting when the "big" fragment is genuinely big and the
    // "small" prefix isn't empty (otherwise we'd just be issuing one put).
    // big_is_gpu overrides zerocopy_enabled: with GPU mem we have no choice,
    // the staged fallback can't memcpy device memory through CPU.
    const bool use_zerocopy = (zerocopy_enabled || big_is_gpu) &&
                              iov_count >= 2 &&
                              big_size >= (16u * 1024u) &&
                              small_size > 0;

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    if (use_zerocopy) {
        // Stage all fragments except the biggest into the registered
        // scratch, contiguously in iov order (pre first, then post).
        ensure_tx_scratch_locked(small_size);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                if (i == biggest_idx) continue;
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        ucx_debug_log("WriteIovRma(zerocopy) slot=%zu pre=%zu big=%zu post=%zu biggest_idx=%zu",
                      slot_idx, pre_size, big_size, post_size, biggest_idx);

        // Issue up to three puts non-blocking. UCX progresses them in
        // parallel; their completions are awaited together at the end.
        ucp_request_param_t p_scratch{};
        p_scratch.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        p_scratch.memh = tx_scratch_.memh;

        void *req_pre = nullptr;
        if (pre_size > 0) {
            req_pre = ucp_put_nbx(endpoint_,
                                  tx_scratch_.addr, pre_size,
                                  rs.addr, rs.rkey, &p_scratch);
        }

        void *req_post = nullptr;
        if (post_size > 0) {
            req_post = ucp_put_nbx(endpoint_,
                                   static_cast<char *>(tx_scratch_.addr) + pre_size,
                                   post_size,
                                   rs.addr + pre_size + big_size,
                                   rs.rkey, &p_scratch);
        }

        ucp_request_param_t p_big{};

        // Manual host-buffer registration cache. UCX rcache fails to
        // init in this container ("rcache failed to install UCM event
        // handler: Unsupported operation"), so we mem_map ptr->memh
        // ourselves and pass it explicitly. Saves ~25ms re-register
        // cost per 64MB put on the warm path.
        //
        // SIZE-THRESHOLD GUARD (added for OpenPose-style workloads):
        // The cache keys by virtual address. If the user frees and reallocs
        // at the same address (Caffe blobs, repeated cudaMallocHost cycles)
        // we'd return a stale memh → IB QP Local Protection error. For
        // small buffers (typical of inference frameworks) we skip the cache
        // and pay ucp_mem_map+unmap per call (~ms penalty). Large stable
        // buffers (simple_matrix-style 4 MB+) still cache for big wins.
        //
        // CONC>=8 CRASH FIX (2026-07-23): the 2 MB HOST threshold was too low.
        // llama.cpp's compute pool frees+reallocs a ~2.06 MB host buffer at a
        // fixed virtual address under 8-way concurrent prefill; being >= 2 MB it
        // was cached, and with rcache disabled (UCX_RCACHE_ENABLE=n) the cached
        // addr-keyed memh went stale on the realloc -> ib_mlx5 "Local protection
        // error (synd 0x4)" on the next RDMA_WRITE (observed: identical va+lkey
        // across 48 crash episodes, len 2162688) -> RC QP fatal -> am_send EIO
        // storm -> frontend dies (the intermittent CONC>=8 UNIQUE crash).
        // Registering such a buffer fresh per put (~1 ms/2 MB) always matches the
        // current pages, so it cannot go stale. DEVICE registrations (the GPU
        // shadow / a real device pointer) are STABLE allocations and stay cached
        // at >= 2 MB. HOST buffers are only cached when large enough that the
        // per-put re-registration would actually hurt (>= 16 MB, e.g. the
        // transfer-bench arrays, which are allocated once and do not churn).
        static constexpr size_t kCacheThreshold  = 2u  * 1024u * 1024u;  // 2 MB (device)
        static constexpr size_t kHostCacheMin    = 16u * 1024u * 1024u;  // 16 MB (host)
        const bool use_memh_cache = big_is_gpu ? (big_size >= kCacheThreshold)
                                               : (big_size >= kHostCacheMin);

        static thread_local std::unordered_map<const void *, ucp_mem_h>
            user_memh_cache;

        const void *user_addr = iov[biggest_idx].iov_base;
        ucp_mem_h user_memh = nullptr;
        bool memh_owned = false;  // true iff we own this memh and must unmap after the put

        // big_is_gpu was already computed at WriteIovRma entry (used to force
        // zerocopy when GPU mem is present). Reused here for the mem_map hint
        // — when rcache + memtype-cache are disabled (this container), UCX
        // won't auto-detect CUDA memory and we MUST pass UCS_MEMORY_TYPE_CUDA
        // explicitly.

        auto fill_mp = [&](ucp_mem_map_params_t &mp) {
            mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
            mp.address = const_cast<void *>(user_addr);
            mp.length  = big_size;
            if (big_is_gpu) {
                mp.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
                mp.memory_type  = UCS_MEMORY_TYPE_CUDA;
            }
        };

        if (use_memh_cache) {
            auto cit = user_memh_cache.find(user_addr);

            if (cit != user_memh_cache.end()) {
                user_memh = cit->second;
            } else {
                ucp_mem_map_params_t mp{};
                fill_mp(mp);

                if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                    user_memh_cache.emplace(user_addr, user_memh);
                } else {
                    user_memh = nullptr;
                }
            }
        } else {
            // Small buffer path: register fresh each call, unmap after wait.
            ucp_mem_map_params_t mp{};
            fill_mp(mp);

            if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                memh_owned = true;
            } else {
                user_memh = nullptr;
            }
        }

        if (user_memh != nullptr) {
            p_big.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
            p_big.memh = user_memh;
        } else {
            p_big.op_attr_mask = 0;
        }

        // GPUDirect Step B3: route the big fragment to the peer's GPU shadow
        // when available. Triggers NIC peer-DMA into remote GPU memory via
        // peermem. The biggest fragment can sit at ANY iov index (Fase 5
        // puts user_src at idx 3 with a 12-byte [count][kind] post). We
        // pass the GPU offset (= pre_size) via RmaPosted.status_code so the
        // receiver knows where to fold the GPU bytes back into the host slot.
        //
        // 4 MB threshold (raised from 64 KB after the B4 sweep showed N=256
        // and N=512 regressed): below this size the 3-put orchestration +
        // peer-DMA setup overhead exceeds the savings from skipping the
        // host bounce. 4 MB matches simple_matrix N=1024 (4 MB), the
        // smallest payload where GPUDirect demonstrably wins.
        //
        // Transport gate: ucp_ep_rkey_unpack(gpu_rkey) returns UCS_OK even
        // when the negotiated transport is TCP (the unpack just parses the
        // blob; transport check happens at put time). If we then attempt
        // ucp_put_nbx to GPU memory over a TCP endpoint, UCX errors out or
        // hangs, killing the connection. Defensive check at WriteIovRma init
        // time: query THIS endpoint's negotiated lanes via ucp_ep_query
        // (lazy + cached). Supersedes the previous process-wide UCX_TLS env
        // probe, so a single backend with UCX_TLS=rc_mlx5,ud_mlx5,tcp,self
        // serves mixed RDMA + TCP frontends correctly — GPUDirect activates
        // only on connections that actually negotiated an RDMA lane.
        // CORRECTNESS (2026-07-23): the 4 MB lower bound was a perf heuristic, but
        // this condition is only ever true for DEVICE-source fragments
        // (big_is_device_data). Routing a device-source fragment to the HOST slot
        // (the else branch) asks UCX for a cuda->host RMA put, which cannot be
        // built under the forced rcache-off / memtype-cache-off config -> the
        // ucp_put fails and the RC QP goes fatal (observed: RDMA_WRITE len ~2 MB
        // -> "QP was flushed" -> every subsequent am_send EIO, crashing the
        // frontend under CONC>=8 real prefill). Device data therefore has NO valid
        // host-slot path and MUST go to the GPU shadow (device->device peer-DMA,
        // which works) at ANY size that fits the shadow. Keep only the capacity
        // upper bound; drop the 4 MB lower bound.
        // Lower bound is env-tunable (was a hardcoded 4 MB). Default 0 = route ALL
        // device-source data to the shadow (correctness: the host-slot device path
        // fails under concurrency). Set GVIRTUS_RMA_GPUDIRECT_MIN_BYTES=4194304 to
        // restore the old size-adaptive behavior (host slot below the bound) for
        // A/B measurement of the shadow-vs-host cost curve.
        static const size_t gpudirect_min_bytes = []() {
            const char *v = std::getenv("GVIRTUS_RMA_GPUDIRECT_MIN_BYTES");
            return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : 0u;
        }();
        const bool route_big_to_gpu = (rs.gpu_rkey != nullptr) &&
                                      (rs.gpu_addr != 0) &&
                                      big_is_device_data &&
                                      (big_size >= gpudirect_min_bytes) &&
                                      (big_size <= rs.gpu_capacity) &&
                                      current_connection_supports_cuda();
        std::uint64_t big_target_addr = route_big_to_gpu
                                        ? rs.gpu_addr
                                        : (rs.addr + pre_size);
        ucp_rkey_h    big_target_rkey = route_big_to_gpu ? rs.gpu_rkey : rs.rkey;

        // GUARD DIAGNOSTIC (2026-07-23, unconditional + flushed): a REAL device
        // pointer (big_is_gpu) that did NOT get routed to the GPU shadow is about
        // to be ucp_put into a HOST slot -> cuda->host RMA -> RC QP fatal. This
        // must never happen post-fix; if it fires, print which sub-condition
        // blocked the GPU route. (big_is_device_data alone is NOT a hazard: on
        // the frontend the H2D data-path source is host memory, and host->host-
        // slot is a normal, safe put.)
        if (big_is_gpu && !route_big_to_gpu) {
            std::fprintf(stderr,
                "GVCRASHDIAG dev->host-slot HAZARD big=%zu dev_data=%d is_gpu=%d "
                "gpu_rkey=%d gpu_addr=%d cap=%zu ge_min=%d le_cap=%d supports_cuda=%d\n",
                big_size, (int)big_is_device_data, (int)big_is_gpu,
                (int)(rs.gpu_rkey != nullptr), (int)(rs.gpu_addr != 0), rs.gpu_capacity,
                (int)(big_size >= gpudirect_min_bytes), (int)(big_size <= rs.gpu_capacity),
                (int)current_connection_supports_cuda());
            std::fflush(stderr);
        }

        if (route_big_to_gpu) {
            gpu_split_bytes  = big_size;
            gpu_split_offset = static_cast<std::uint32_t>(pre_size);
            ucx_debug_log("WriteIovRma(B3 gpu-split) slot=%zu pre=%zu big=%zu post=%zu (to gpu_addr=0x%lx)",
                          slot_idx, pre_size, big_size, post_size, big_target_addr);
        }

        void *req_big = ucp_put_nbx(endpoint_,
                                    iov[biggest_idx].iov_base, big_size,
                                    big_target_addr, big_target_rkey, &p_big);

        wait_request_completion(req_pre,  "rma_put_pre");
        wait_request_completion(req_big,  "rma_put_big");
        wait_request_completion(req_post, "rma_put_post");

        // Per-call ownership cleanup: unmap fresh registrations so the next
        // call sees a clean state. Cached memh (>= kCacheThreshold) stays
        // mapped for amortization across calls.
        if (memh_owned && user_memh != nullptr) {
            ucp_mem_unmap(context_, user_memh);
        }
    } else {
        // Staged path: copy everything into the scratch, single put.
        // CRASH-HUNT DIAGNOSTIC: this path CPU-memcpies every fragment. A device
        // pointer here (use_zerocopy was false despite GPU memory: big_size<16KB
        // OR small_size==0 OR iov_count<2) is an imminent SIGSEGV. Flag it first.
        for (size_t i = 0; i < iov_count; ++i) {
            if (is_gpu_pointer(iov[i].iov_base)) {
                std::fprintf(stderr,
                    "GVCRASHDIAG staged-path DEVICE FRAG idx=%zu len=%zu "
                    "big_is_gpu=%d big_is_device_data=%d big_size=%zu small_size=%zu iov_count=%zu "
                    "-> SIGSEGV imminent\n",
                    i, iov[i].iov_len, (int)big_is_gpu, (int)big_is_device_data,
                    big_size, small_size, iov_count);
                std::fflush(stderr);
            }
        }
        ensure_tx_scratch_locked(total);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        ucx_debug_log("WriteIovRma(staged) slot=%zu total=%zu", slot_idx, total);

        ucp_request_param_t put_param{};
        put_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        put_param.memh = tx_scratch_.memh;
        void *put_req = ucp_put_nbx(endpoint_,
                                    tx_scratch_.addr, total,
                                    rs.addr, rs.rkey, &put_param);
        wait_request_completion(put_req, "rma_put");
    }

    // Tiny RmaPosted notification — same protocol bytes regardless of which
    // data path filled the remote slot.
    {
        gvirtus::communicators::ucxam::EnvelopeHeader notif{};
        notif.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
        notif.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
        notif.message_type = static_cast<std::uint16_t>(
            gvirtus::communicators::ucxam::MessageType::RmaPosted);
        notif.header_size = sizeof(notif);
        // The SERVER's slot id, not our vector position (see RemoteSlot::server_idx).
        notif.reserved0 = rs.server_idx;
        // GPUDirect Step B3: status_code carries the gpu_split_offset (=
        // pre_size, the position in the host slot where the GPU data folds in).
        notif.status_code = gpu_split_offset;
        // (epoch << 32) | generation, echoed verbatim in SlotConsumed so the ack can
        // be rejected if it belongs to a layout we have since replaced.
        notif.request_id = slot_tag;
        // GPUDirect Step B3: non-zero routine_size = bytes that landed in
        // slot.gpu_addr (vs slot.host_addr). The receiver uses this together
        // with status_code (offset) to build a dual PooledMsg. Zero = legacy
        // single-region path.
        notif.routine_size = gpu_split_bytes;
        notif.payload_size = static_cast<std::uint64_t>(total);

        ucp_request_param_t send_param{};
        send_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        send_param.datatype = ucp_dt_make_contig(1);
        void *send_req = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                         &notif, sizeof(notif), &send_param);
        wait_request_completion(send_req, "rma_posted_notify");
    }

    rma_committed = true;  // success: slot stays InFlight until backend SlotConsumed ack
    ucx_debug_log("WriteIovRma done slot=%zu total=%zu gen=%lu", slot_idx, total,
                  (unsigned long)slot_gen);
    return total;
}

// Grow the pre-registered TX scratch to at least `needed` bytes. Must be
// called with worker_mutex_ held. Rounds capacity up to a power of two
// (≥4MB) so consecutive WriteIovs of the same size class hit the warm path.
void UcxCommunicator::ensure_tx_scratch_locked(size_t needed) {
    if (tx_scratch_.capacity >= needed) return;

    // Free previous registration + allocation if any.
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;

    // Round up to power of two, minimum 4MB.
    size_t cap = 4u * 1024u * 1024u;
    while (cap < needed) cap <<= 1;

    void *addr = nullptr;
    if (posix_memalign(&addr, 4096, cap) != 0 || addr == nullptr) {
        throw std::runtime_error("UcxCommunicator: posix_memalign failed for tx scratch");
    }

    ucp_mem_map_params_t map_params{};
    map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    map_params.address = addr;
    map_params.length = cap;

    ucp_mem_h memh = nullptr;
    ucs_status_t status = ucp_mem_map(context_, &map_params, &memh);
    if (status != UCS_OK) {
        std::free(addr);
        throw std::runtime_error("UcxCommunicator: ucp_mem_map failed: " +
                                 std::string(ucs_status_string(status)));
    }

    tx_scratch_.addr = addr;
    tx_scratch_.capacity = cap;
    tx_scratch_.memh = memh;
    ucx_debug_log("tx_scratch grown capacity=%zu", cap);
}

void UcxCommunicator::release_tx_scratch_locked() {
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }

    // Send payload as a single UCX Active Message.
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    ucx_debug_log("Write(AM) begin bytes=%zu", size);

    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0, buffer, size,
                                    &request_param);
    wait_request_completion(request, "am_send");
    ucx_debug_log("Write(AM) done bytes=%zu", size);
    return size;
}

// Two-mode gather-send. For small payloads (under kStagingThreshold) the
// fragments are passed straight to UCX via UCP_DATATYPE_IOV — eager AM
// handles short messages efficiently and the iov metadata cost is
// negligible. For large payloads the fragments are concatenated into a
// pre-registered tx_scratch_ buffer and sent as one contiguous chunk with
// the memh hint, which lets UCX bypass its internal RNDV-fragment staging
// (UCX_RNDV_FRAG_SIZE) and DMA directly from the registered memory.
size_t UcxCommunicator::WriteIov(const struct iovec *iov, size_t iov_count) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: WriteIov called without an active endpoint");
    }
    if (iov == nullptr || iov_count == 0) return 0;

    size_t total = 0;
    for (size_t i = 0; i < iov_count; ++i) total += iov[i].iov_len;

    // RMA fast path: if the server advertised its RX slot rkeys (via
    // RmaSetup at connect time) and the payload is large enough to amortise
    // the staging memcpy, push the bytes via ucp_put_nbx directly into the
    // remote slot and notify with a tiny AM. Avoids UCX's per-message
    // rendezvous handshake (which doesn't amortise in our sync pattern).
    // Floor at the measured crossover, not 64 KB: below a few MB the RMA handshake
    // costs a round trip that a small eager AM never pays. Matches the frontend
    // default so both directions agree (commit f0d8c1f).
    static const size_t kRmaMinBytes = ucx_rma_min_bytes();
    static const size_t kRmaMinBytesUnused = []() -> size_t {
        const char *e = std::getenv("GVIRTUS_RMA_MIN_BYTES");
        if (e == nullptr || e[0] == 0) return 4u * 1024u * 1024u;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(e, &end, 10);
        return (end != e) ? static_cast<size_t>(parsed) : 4u * 1024u * 1024u;
    }();

    if (total >= kRmaMinBytes && rma_setup_received_.load()) {
        size_t put = WriteIovRma(iov, iov_count, total);
        if (put == total) return put;  // RMA path completed
        // else: fall through to the IOV/AM path (slot too small or no rkey)
    }

    // Staging via the local TX scratch (with memh hint) regresses on this
    // UCX 1.20 + RoCE combo because it forces true-rendezvous over AM.
    // Kept disabled — see commit history for the measurement campaign.
    constexpr size_t kStagingThreshold = static_cast<size_t>(-1);

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    if (total < kStagingThreshold) {
        // Fast path: IOV directly, eager AM.
        std::vector<ucp_dt_iov_t> ucx_iov(iov_count);
        for (size_t i = 0; i < iov_count; ++i) {
            ucx_iov[i].buffer = iov[i].iov_base;
            ucx_iov[i].length = iov[i].iov_len;
        }
        ucx_debug_log("WriteIov(AM,iov) begin frags=%zu total=%zu", iov_count, total);

        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = UCP_DATATYPE_IOV;

        void *request = ucp_am_send_nbx(endpoint_, am_id_,
                                        nullptr, 0,
                                        ucx_iov.data(), iov_count,
                                        &request_param);
        wait_request_completion(request, "am_send_iov");
        ucx_debug_log("WriteIov(AM,iov) done total=%zu", total);
        return total;
    }

    // Staging path: gather into the pre-registered TX scratch and send
    // as one contiguous, memh-hinted, AM rendezvous message.
    ensure_tx_scratch_locked(total);

    {
        char *dst = static_cast<char *>(tx_scratch_.addr);
        size_t off = 0;
        for (size_t i = 0; i < iov_count; ++i) {
            std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
            off += iov[i].iov_len;
        }
    }

    ucx_debug_log("WriteIov(AM,pool) begin frags=%zu total=%zu cap=%zu",
                  iov_count, total, tx_scratch_.capacity);

    // No memh hint: that flag pushes UCX into the slow true-rendezvous
    // path (RTS/RTR + fragmented RDMA) which doesn't amortize over a
    // single 64MB sync request. Without the hint UCX still picks up the
    // ucp_mem_map'd registration via its rcache. The win we're after here
    // is the contiguous buffer (vs IOV) — same protocol, fewer iov ops.
    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request = ucp_am_send_nbx(endpoint_, am_id_,
                                    nullptr, 0,
                                    tx_scratch_.addr, total,
                                    &request_param);
    wait_request_completion(request, "am_send_pool");
    ucx_debug_log("WriteIov(AM,pool) done total=%zu", total);
    return total;
}

void UcxCommunicator::Sync() {
    if (worker_ == nullptr) {
        return;
    }

    // Flush worker to complete any in-flight sends/receives.
    ucp_request_param_t request_param{};
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    ucx_debug_log("Sync begin (worker flush)");
    void *request = ucp_worker_flush_nbx(worker_, &request_param);
    wait_request_completion(request, "worker_flush");
    ucx_debug_log("Sync done");

    progress_am_rndv();
}

void UcxCommunicator::Close() {
    // Signal shutdown and release UCX resources.
    ucx_debug_log("Close called");
    running_ = false;
    conn_cv_.notify_all();
    destroy_ucx();
}

void UcxCommunicator::run() {
    // Placeholder for compatibility with Communicator interface.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_endpoint = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_endpoint) {
        throw std::runtime_error("UcxCommunicator: endpoint type mismatch");
    }

    return std::make_shared<UcxCommunicator>(ucx_endpoint->address(), ucx_endpoint->port());
}
