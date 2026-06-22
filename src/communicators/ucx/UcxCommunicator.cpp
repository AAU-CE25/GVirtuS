/*
 * UcxCommunicator implementation — all six optimization phases in one file.
 *
 * Phase 1 (Baseline AM): UCX context/worker init, listener/accept/connect,
 *   AM receive callback, stream-style Read()/Write() over active messages.
 *
 * Phase 2 (Protocol): EnvelopeHeader framing parsed in am_recv_handler;
 *   RmaSetup and RmaPosted message types dispatched inline.
 *
 * Phase 4 (Gather-send): WriteIov() with UCP_DATATYPE_IOV passes scatter
 *   fragments natively to ucp_am_send_nbx. TryAcquireFrame()/ReleaseFrame()
 *   expose pinned RX-pool slots zero-copy to the caller.
 *
 * Phase 5 (RMA): WriteIovRma() stages or zero-copies iov fragments into a
 *   pre-registered remote slot via ucp_put_nbx, followed by a tiny RmaPosted
 *   AM notification. send_rma_setup() / handle_rma_setup_am() exchange rkeys
 *   at connection time. Manual memh cache bypasses broken UCX rcache.
 *
 * Phase 6 (GPUDirect): probe_gpudirect() validates cudaMalloc + ucp_mem_map
 *   (CUDA) at startup. init_rx_pool() allocates GPU shadow regions alongside
 *   host slots. WriteIovRma routes the big iov fragment to the peer's GPU
 *   shadow when the connection has an RDMA lane (per-connection transport gate
 *   via current_connection_supports_cuda()). am_recv_handler publishes the GPU
 *   portion via PooledMsg.gpu_data for B4 handler-side zero-copy.
 *
 * Runtime-resolved CUDA symbols (dlopen): cudaHostAlloc, cudaMalloc, cudaFree,
 * cudaMemcpy, cudaPointerGetAttributes — no static link to libcudart.
 */
#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <unordered_map>
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
#include "gvirtus/communicators/Protocol.h"
#include "UcxInternal.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using gvirtus::communicators::UcxCommunicator;
using namespace gvirtus::communicators::ucx_internal;
using namespace log4cplus;

namespace {
constexpr unsigned kUcxAmId = 1;
static Logger ucx_logger = Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator"));
}  // namespace

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

    LOG4CPLUS_DEBUG(::ucx_logger, "am_recv_handler: self=" << (void *)self
                    << " length=" << length
                    << " rndv=" << ((param != nullptr && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) ? 1 : 0));

    if (length == 0) {
        self->enqueue_am_message(PooledMsg{});
        return UCS_OK;
    }

    // Quick peek at the envelope header (small messages only, always eager):
    //   * RmaSetup → handshake message, unpack rkeys inline
    //   * RmaPosted → data already RDMA-put into an RX slot, queue a
    //                 PooledMsg pointing at that slot instead of acquiring
    //                 a fresh one + memcpy'ing
    if (length >= sizeof(gvirtus::communicators::am::EnvelopeHeader) &&
        (param == nullptr || !(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV))) {
        gvirtus::communicators::am::EnvelopeHeader peek;
        std::memcpy(&peek, data, sizeof(peek));
        if (peek.magic == gvirtus::communicators::am::kEnvelopeMagic) {
            using gvirtus::communicators::am::MessageType;
            using gvirtus::communicators::am::RmaPostedBody;
            if (peek.message_type == static_cast<std::uint8_t>(MessageType::RmaSetup)) {
                self->handle_rma_setup_am(data, length);
                return UCS_OK;
            }
            if (peek.message_type == static_cast<std::uint8_t>(MessageType::RmaPosted)) {
                if (length < sizeof(peek) + sizeof(RmaPostedBody)) {
                    std::fprintf(stderr, "RmaPosted: body too short (%zu)\n", length);
                    return UCS_OK;
                }
                RmaPostedBody body;
                std::memcpy(&body, static_cast<const unsigned char *>(data) + sizeof(peek),
                            sizeof(body));
                const size_t slot_idx   = static_cast<size_t>(peek.reserved0);
                const size_t total      = static_cast<size_t>(body.slot_total);
                // GPUDirect Step B3: non-zero gpu_size means the peer routed
                // `gpu_size` bytes into slot.gpu_addr via NIC peer-DMA.
                // `gpu_offset` is the position in the logical message where
                // the GPU data folds in.
                const size_t gpu_size   = static_cast<size_t>(body.gpu_size);
                const size_t gpu_offset = static_cast<size_t>(body.gpu_offset);
                std::lock_guard<std::mutex> lk(self->rx_pool_->mu);
                if (slot_idx >= self->rx_pool_->slots.size()) {
                    std::fprintf(stderr,
                                 "RmaPosted: invalid slot_idx=%zu (pool=%zu)\n",
                                 slot_idx, self->rx_pool_->slots.size());
                    return UCS_OK;
                }
                auto &slot = self->rx_pool_->slots[slot_idx];
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
                    LOG4CPLUS_DEBUG(::ucx_logger, "RmaPosted B4: slot=" << slot_idx
                                    << " host_bytes=" << (total - gpu_size)
                                    << " gpu_bytes=" << gpu_size
                                    << " offset=" << gpu_offset << " (no consolidation)");
                }

                self->enqueue_am_message(msg);
                return UCS_OK;
            }
        }
    }

    // Acquire a pinned slot from the RX pool — slot capacity is pre-allocated,
    // no per-message std::vector zero-init.
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
    if (gpudirect_requested) {
        std::string reason;
        const bool ok = probe_gpudirect(context_, reason);
        set_gpudirect_enabled(ok);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", ok ? "1" : "0", /*overwrite=*/1);
        if (ok) {
            LOG4CPLUS_INFO(::ucx_logger,
                "GPUDirect=enabled (cudaMalloc + ucp_mem_map(CUDA) probe OK, "
                "auto-set UCX_RCACHE_ENABLE=n UCX_MEMTYPE_CACHE=n)");
        } else {
            LOG4CPLUS_INFO(::ucx_logger,
                "GPUDirect=disabled (GVIRTUS_GPUDIRECT=1 but probe FAILED: " << reason << ") "
                "- falling back to host slots, behavior unchanged");
        }
    } else {
        set_gpudirect_enabled(false);
        setenv("GVIRTUS_GPUDIRECT_ACTIVE", "0", /*overwrite=*/1);
        if (gpudirect_env_set && !tls_supports_cuda) {
            LOG4CPLUS_INFO(::ucx_logger,
                "GPUDirect=disabled (UCX_TLS=" << (std::getenv("UCX_TLS") ? std::getenv("UCX_TLS") : "(unset)") << " has no CUDA-capable transport)");
        } else {
            LOG4CPLUS_INFO(::ucx_logger,
                "GPUDirect=disabled (env GVIRTUS_GPUDIRECT not set)");
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
    LOG4CPLUS_DEBUG(::ucx_logger, "init_ucx completed host=" << hostname_ << " port=" << port_ << " mode=am");
}

void UcxCommunicator::destroy_ucx() {
    if (!initialized_) return;

    // Tear down UCX resources in reverse order of creation.
    LOG4CPLUS_DEBUG(::ucx_logger, "destroy_ucx begin endpoint=" << (void *)endpoint_
                    << " listener=" << (void *)listener_
                    << " worker=" << (void *)worker_
                    << " context=" << (void *)context_);

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
    LOG4CPLUS_DEBUG(::ucx_logger, "destroy_ucx completed");
}

void UcxCommunicator::enqueue_connection(ucp_conn_request_h conn_request) {
    // Queue incoming connection requests from the listener callback.
    std::lock_guard<std::mutex> lock(conn_mutex_);
    pending_conn_requests_.push(conn_request);
    LOG4CPLUS_DEBUG(::ucx_logger, "enqueue_connection request=" << (void *)conn_request
                    << " queue_size=" << pending_conn_requests_.size());
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
    LOG4CPLUS_DEBUG(::ucx_logger, op_name << ": wait_request_completion request=" << request);

    if (request == nullptr) {
        LOG4CPLUS_DEBUG(::ucx_logger, op_name << ": immediate completion (null request)");
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

    LOG4CPLUS_DEBUG(::ucx_logger, op_name << ": completed status=" << ucs_status_string(final_status));
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
    LOG4CPLUS_INFO(::ucx_logger,
        "UCX control-plane ready: Serve (" << hostname_ << ":" << port_ << ")");
    LOG4CPLUS_DEBUG(::ucx_logger, "listener created listener=" << (void *)listener_);
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    // Accept a connection and create a UCX endpoint for the new client.
    auto *self = const_cast<UcxCommunicator *>(this);
    if (!self->running_ || self->listener_ == nullptr) {
        return nullptr;
    }

    ucp_conn_request_h req = self->wait_for_connection_request();
    if (req == nullptr) {
        LOG4CPLUS_DEBUG(::ucx_logger, "Accept returned null request (shutdown or no request)");
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
    LOG4CPLUS_DEBUG(::ucx_logger, "server endpoint created endpoint=" << (void *)accepted->endpoint_
                    << " worker=" << (void *)accepted->worker_
                    << " from request=" << (void *)req);

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
    std::thread([accepted]() {
        try {
            accepted->init_rx_pool();
            // Client may disconnect quickly during bring-up; avoid crashing
            // the backend process on a best-effort RmaSetup send.
            accepted->send_rma_setup();
        } catch (const std::exception &e) {
            LOG4CPLUS_DEBUG(::ucx_logger, "Accept setup thread: RmaSetup skipped (" << e.what() << ")");
        }
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
    LOG4CPLUS_DEBUG(::ucx_logger, "client endpoint created endpoint=" << (void *)endpoint_);

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
        LOG4CPLUS_DEBUG(::ucx_logger, "Connect: RmaSetup not received within timeout, RMA path disabled");
    } else {
        LOG4CPLUS_DEBUG(::ucx_logger, "Connect: RMA path enabled with " << remote_slots_.size() << " remote slots (server -> client)");

        // Bidirectional RMA: now that we know the server is RMA-capable
        // (it sent us its rkeys), advertise our own rx_pool's rkeys so it
        // can ucp_put_nbx into our slots for large responses (D2H 64MB
        // etc). Without this the server's WriteIov for the response falls
        // back to the AM-stream path, which is ~1.7s for 64MB without an
        // rcache. With this it becomes a single RDMA write + tiny AM ≈
        // ~10-15ms.
        send_rma_setup();
        LOG4CPLUS_DEBUG(::ucx_logger, "Connect: client rkeys advertised (client -> server done)");
    }
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Read called without an active endpoint");
    }
    if (size == 0) {
        return 0;
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
    LOG4CPLUS_DEBUG(::ucx_logger, "current_connection_supports_cuda: endpoint=" << (void *)endpoint_
                    << " -> " << (supports ? "RDMA (CUDA-capable)" : "non-RDMA (TCP-class)"));
    return supports;
}

// Register `slot.addr/capacity` (host) AND `slot.gpu_addr/gpu_capacity` (if

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }

    // Send payload as a single UCX Active Message.
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    LOG4CPLUS_DEBUG(::ucx_logger, "Write(AM) begin bytes=" << size);

    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0, buffer, size,
                                    &request_param);
    wait_request_completion(request, "am_send");
    LOG4CPLUS_DEBUG(::ucx_logger, "Write(AM) done bytes=" << size);
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
    if (total >= (64u * 1024u) && rma_setup_received_.load()) {
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
        LOG4CPLUS_DEBUG(::ucx_logger, "WriteIov(AM,iov) begin frags=" << iov_count << " total=" << total);

        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = UCP_DATATYPE_IOV;

        void *request = ucp_am_send_nbx(endpoint_, am_id_,
                                        nullptr, 0,
                                        ucx_iov.data(), iov_count,
                                        &request_param);
        wait_request_completion(request, "am_send_iov");
        LOG4CPLUS_DEBUG(::ucx_logger, "WriteIov(AM,iov) done total=" << total);
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

    LOG4CPLUS_DEBUG(::ucx_logger, "WriteIov(AM,pool) begin frags=" << iov_count << " total=" << total << " cap=" << tx_scratch_.capacity);

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
    LOG4CPLUS_DEBUG(::ucx_logger, "WriteIov(AM,pool) done total=" << total);
    return total;
}

void UcxCommunicator::Sync() {
    if (worker_ == nullptr) {
        return;
    }

    // Flush worker to complete any in-flight sends/receives.
    ucp_request_param_t request_param{};
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    LOG4CPLUS_DEBUG(::ucx_logger, "Sync begin (worker flush)");
    void *request = ucp_worker_flush_nbx(worker_, &request_param);
    wait_request_completion(request, "worker_flush");
    LOG4CPLUS_DEBUG(::ucx_logger, "Sync done");

    progress_am_rndv();
}

void UcxCommunicator::Close() {
    // Signal shutdown and release UCX resources.
    LOG4CPLUS_DEBUG(::ucx_logger, "Close called");
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
