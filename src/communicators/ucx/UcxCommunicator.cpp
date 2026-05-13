#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <sys/socket.h>
#include <stdexcept>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

namespace {
constexpr unsigned kUcxAmId = 1;
constexpr size_t kDefaultScratchSizeBytes = 128ull * 1024 * 1024;  // 128 MB
constexpr size_t kSendScratchMinBytes = 8 * 1024;                  // below this, skip copy

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

    if (length == 0) {
        self->enqueue_am_message({});
        return UCS_OK;
    }

    if ((param != nullptr) && (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        // Try to land the rendezvous receive in the pre-registered scratch.
        // That keeps the destination VA stable across calls and lets the UCX
        // registration cache hit, eliminating the per-AM memory pinning that
        // was bottlenecking large RDMA transfers.
        unsigned char *recv_buf = nullptr;
        ByteBuffer delivery(length);  // final landing buffer, default-init
        bool from_scratch = false;
        {
            std::lock_guard<std::mutex> lock(self->am_state_->mutex);
            if (!self->am_state_->recv_scratch_in_use &&
                length <= self->am_state_->recv_scratch.size() &&
                self->am_state_->recv_memh != nullptr) {
                self->am_state_->recv_scratch_in_use = true;
                recv_buf = self->am_state_->recv_scratch.data();
                from_scratch = true;
            }
        }
        if (recv_buf == nullptr) {
            // Either the scratch is busy with a previous (still in-flight)
            // rendezvous receive or this payload is larger than the
            // configured scratch. Fall back to a fresh per-call buffer (the
            // pre-fix behaviour) so correctness is preserved.
            recv_buf = delivery.data();
        }

        ucp_request_param_t recv_param{};
        recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        recv_param.datatype = ucp_dt_make_contig(1);

        void *request = ucp_am_recv_data_nbx(self->worker_, data, recv_buf, length,
                                             &recv_param);
        if (request == nullptr) {
            // Synchronous completion.
            if (from_scratch) {
                std::memcpy(delivery.data(), recv_buf, length);
                std::lock_guard<std::mutex> lock(self->am_state_->mutex);
                self->am_state_->recv_scratch_in_use = false;
            }
            self->enqueue_am_message(std::move(delivery));
            return UCS_OK;
        }
        if (UCS_PTR_IS_ERR(request)) {
            if (from_scratch) {
                std::lock_guard<std::mutex> lock(self->am_state_->mutex);
                self->am_state_->recv_scratch_in_use = false;
            }
            return UCS_PTR_STATUS(request);
        }

        self->enqueue_am_rndv(request, std::move(delivery), from_scratch, length);
        return UCS_INPROGRESS;
    }

    ByteBuffer buffer(length);  // default-init: no zero-fill
    std::memcpy(buffer.data(), data, length);
    self->enqueue_am_message(std::move(buffer));

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

    ucp_params_t ucp_params{};
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_AM;

    ucs_status_t status = ucp_init(&ucp_params, nullptr, &context_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_init failed: " +
                                 std::string(ucs_status_string(status)));
    }

    // parameters for ucp_worker
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    // SERIALIZED: multiple OS threads may call into UCX, but we already
    // serialize them via worker_mutex_, so UCX itself can skip its internal
    // per-call locking (which was the case with UCS_THREAD_MODE_MULTI).
    worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;

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

    initialized_ = true;
    // Allocate and pre-register the receive scratch (shared via AmState so
    // both this listener/client and any accepted communicator see the same
    // pinned region) and the per-endpoint send scratch.
    allocate_recv_scratch();
    allocate_send_scratch();
    ucx_debug_log("init_ucx completed host=%s port=%u mode=am", hostname_.c_str(), port_);
}

size_t UcxCommunicator::scratch_size_bytes() {
    // Optional override via env var so workloads beyond size=4096 (64 MB
    // matrices) can be supported without recompiling. Values are in MiB.
    const char *env = std::getenv("GVIRTUS_UCX_SCRATCH_SIZE_MB");
    if (env != nullptr) {
        char *end = nullptr;
        long mb = std::strtol(env, &end, 10);
        if (end != env && mb > 0) {
            return static_cast<size_t>(mb) * 1024 * 1024;
        }
    }
    return kDefaultScratchSizeBytes;
}

void UcxCommunicator::allocate_recv_scratch() {
    if (context_ == nullptr || !am_state_) return;
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    if (am_state_->recv_memh != nullptr) {
        // Some other communicator that shares this AmState (e.g. the listener
        // when an accepted instance is being set up via init_ucx-style path)
        // already pre-registered the recv scratch. Nothing to do.
        return;
    }
    am_state_->recv_scratch.resize(scratch_size_bytes());

    ucp_mem_map_params_t params{};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = am_state_->recv_scratch.data();
    params.length = am_state_->recv_scratch.size();

    ucs_status_t status = ucp_mem_map(context_, &params, &am_state_->recv_memh);
    if (status != UCS_OK) {
        am_state_->recv_memh = nullptr;
        std::fprintf(stderr,
                     "UcxCommunicator: ucp_mem_map(recv_scratch, %zu B) failed: %s\n",
                     am_state_->recv_scratch.size(), ucs_status_string(status));
        am_state_->recv_scratch.clear();
        am_state_->recv_scratch.shrink_to_fit();
    } else {
        ucx_debug_log("recv scratch mapped size=%zu memh=%p",
                      am_state_->recv_scratch.size(),
                      (void *)am_state_->recv_memh);
    }
}

void UcxCommunicator::allocate_send_scratch() {
    if (context_ == nullptr) return;
    if (send_scratch_memh_ != nullptr) return;
    send_scratch_.resize(scratch_size_bytes());

    ucp_mem_map_params_t params{};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = send_scratch_.data();
    params.length = send_scratch_.size();

    ucs_status_t status = ucp_mem_map(context_, &params, &send_scratch_memh_);
    if (status != UCS_OK) {
        send_scratch_memh_ = nullptr;
        std::fprintf(stderr,
                     "UcxCommunicator: ucp_mem_map(send_scratch, %zu B) failed: %s\n",
                     send_scratch_.size(), ucs_status_string(status));
        send_scratch_.clear();
        send_scratch_.shrink_to_fit();
    } else {
        ucx_debug_log("send scratch mapped size=%zu memh=%p",
                      send_scratch_.size(), (void *)send_scratch_memh_);
    }
}

void UcxCommunicator::release_recv_scratch() {
    if (context_ == nullptr || !am_state_) return;
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    if (am_state_->recv_memh != nullptr) {
        ucp_mem_unmap(context_, am_state_->recv_memh);
        am_state_->recv_memh = nullptr;
    }
    am_state_->recv_scratch.clear();
    am_state_->recv_scratch.shrink_to_fit();
    am_state_->recv_scratch_in_use = false;
}

void UcxCommunicator::release_send_scratch() {
    if (context_ == nullptr) {
        send_scratch_memh_ = nullptr;
        send_scratch_.clear();
        send_scratch_.shrink_to_fit();
        return;
    }
    if (send_scratch_memh_ != nullptr) {
        ucp_mem_unmap(context_, send_scratch_memh_);
        send_scratch_memh_ = nullptr;
    }
    send_scratch_.clear();
    send_scratch_.shrink_to_fit();
}


void UcxCommunicator::destroy_ucx() {
    if (!initialized_) return;

    // Tear down UCX resources in reverse order of creation.
    ucx_debug_log("destroy_ucx begin endpoint=%p listener=%p worker=%p context=%p",
                  (void *)endpoint_, (void *)listener_, (void *)worker_, (void *)context_);

    if (endpoint_ != nullptr) {
        ucp_request_param_t close_params{};
        close_params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        // Always use FORCE close. The previous graceful path would block
        // indefinitely (especially on the frontend at process exit) when the
        // peer doesn't actively drain the remaining protocol handshake, which
        // is exactly what happened between simple_matrix iterations. The
        // counter-party will still notice the close because we configured
        // UCP_ERR_HANDLING_MODE_PEER on both endpoints.
        close_params.flags = UCP_EP_CLOSE_FLAG_FORCE;

        void *close_req;
        {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            close_req = ucp_ep_close_nbx(endpoint_, &close_params);
        }
        if (UCS_PTR_IS_ERR(close_req)) {
            std::fprintf(stderr, "UCX endpoint close failed: %s\n",
                         ucs_status_string(UCS_PTR_STATUS(close_req)));
        } else if (close_req != nullptr) {
            wait_request_completion(close_req, "ep_close");
        }
        endpoint_ = nullptr;
    }

    if (owns_context_worker_listener_ && listener_ != nullptr) {
        ucp_listener_destroy(listener_);
        listener_ = nullptr;
    }

    // Release scratches before destroying the worker/context. Send scratch
    // is per-endpoint and always owned by this instance; recv scratch lives
    // in the shared AmState and only the context owner is allowed to unmap
    // it (other accepted communicators share it via shared_ptr).
    release_send_scratch();
    if (owns_context_worker_listener_) {
        release_recv_scratch();
    }

    if (owns_context_worker_listener_ && worker_ != nullptr) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }

    if (owns_context_worker_listener_ && context_ != nullptr) {
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
    // Progress the worker until the request completes. The worker_mutex_ is
    // acquired only briefly around each progress call so that concurrent
    // Read()/Write() callers (and the rendezvous-receive bookkeeping) are not
    // blocked for the entire duration of a large transfer.
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
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
            ucp_worker_progress(worker_);
        }

        progress_am_rndv();

        // If the endpoint has failed (for example, remote peer reset), cancel
        // the in-flight request so callers can unwind instead of hanging.
        if (!cancel_issued && endpoint_failed_.load() && worker_ != nullptr) {
            std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
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

void UcxCommunicator::enqueue_am_message(ByteBuffer message) {
    // Store a completed AM payload for stream-style Read().
    {
        std::lock_guard<std::mutex> lock(am_state_->mutex);
        am_state_->queue.push_back(std::move(message));
    }
    am_state_->cv.notify_one();
}

void UcxCommunicator::enqueue_am_rndv(void *request, ByteBuffer buffer,
                                      bool from_scratch, size_t scratch_len) {
    // Track a rendezvous receive until UCX reports completion.
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    am_state_->rndv.push_back(
        PendingAmRecv{request, std::move(buffer), from_scratch, scratch_len});
}

void UcxCommunicator::progress_am_rndv() {
    // Check rendezvous receive requests and move completed payloads into the queue.
    std::lock_guard<std::mutex> lock(am_state_->mutex);
    for (auto it = am_state_->rndv.begin(); it != am_state_->rndv.end();) {
        if (it->request == nullptr) {
            if (it->from_scratch) {
                std::memcpy(it->buffer.data(), am_state_->recv_scratch.data(),
                            it->scratch_len);
                am_state_->recv_scratch_in_use = false;
            }
            am_state_->queue.push_back(std::move(it->buffer));
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
            if (it->from_scratch) {
                std::memcpy(it->buffer.data(), am_state_->recv_scratch.data(),
                            it->scratch_len);
                am_state_->recv_scratch_in_use = false;
            }
            am_state_->queue.push_back(std::move(it->buffer));
            am_state_->cv.notify_one();
        } else {
            if (it->from_scratch) {
                am_state_->recv_scratch_in_use = false;
            }
            std::fprintf(stderr, "UCX AM rendezvous receive failed: %s\n",
                         ucs_status_string(status));
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
    accepted->worker_ = self->worker_;
    accepted->initialized_ = true;
    accepted->owns_context_worker_listener_ = false;
    accepted->running_ = true;
    accepted->worker_mutex_ = self->worker_mutex_;
    accepted->am_state_ = self->am_state_;
    accepted->endpoint_failed_.store(false);
    // The accepted communicator shares context_/worker_ with the listener,
    // so the recv scratch (kept in the shared AmState) is already pinned.
    // It does still need its own pre-registered send scratch so Write()
    // payloads originate from a stable VA that UCX can keep in its
    // registration cache across calls.
    accepted->allocate_send_scratch();

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.conn_request = req;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = accepted;
    // PEER mode is required for the error handler to fire when the remote
    // side closes its endpoint. With the default NONE mode, peer disconnect
    // is invisible to us: the read loop blocks forever in
    // ucp_worker_progress() and the per-client thread never exits, which
    // also prevents subsequent simple_matrix benchmark iterations from
    // re-connecting.
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;

    ucs_status_t status = ucp_ep_create(accepted->worker_, &ep_params, &accepted->endpoint_);
    if (status != UCS_OK) {
        delete accepted;
        throw std::runtime_error("UcxCommunicator: server ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    std::printf("UCX control-plane accepted connection\n");
    ucx_debug_log("server endpoint created endpoint=%p from request=%p",
                  (void *)accepted->endpoint_, (void *)req);
    return accepted;
}

void UcxCommunicator::Connect() {
    // Connect to the UCX server and create a client endpoint.
    init_ucx();

    sockaddr_storage ss = make_sockaddr(hostname_, port_);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                           UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&ss);
    ep_params.sockaddr.addrlen = sizeof(sockaddr_in);
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = this;
    // PEER mode lets ucp_ep_close_nbx complete promptly on the client side
    // and ensures the server's error handler fires when we go away (or vice
    // versa). With NONE mode (the UCX default) a graceful close can block
    // indefinitely waiting for a peer-side acknowledgment that NONE mode
    // does not produce.
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;

    ucs_status_t status = ucp_ep_create(worker_, &ep_params, &endpoint_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: client ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
    }

    running_ = true;
    endpoint_failed_.store(false);
    std::printf("UCX control-plane connected: Connect (%s:%u)\n", hostname_.c_str(), port_);
    ucx_debug_log("client endpoint created endpoint=%p", (void *)endpoint_);
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
        if (pending_read_offset_ < pending_read_bytes_.size()) {
            const size_t available = pending_read_bytes_.size() - pending_read_offset_;
            const size_t to_copy = std::min(size - copied, available);
            std::memcpy(buffer + copied,
                        pending_read_bytes_.data() + pending_read_offset_,
                        to_copy);
            copied += to_copy;
            pending_read_offset_ += to_copy;
            if (pending_read_offset_ == pending_read_bytes_.size()) {
                pending_read_bytes_.clear();
                pending_read_offset_ = 0;
            }
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(am_state_->mutex);
            if (!am_state_->queue.empty()) {
                pending_read_bytes_ = std::move(am_state_->queue.front());
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

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }

    // Send payload as a single UCX Active Message. The worker mutex is held
    // only across the ucp_am_send_nbx call itself; wait_request_completion
    // takes the lock briefly per progress tick so other threads (or the
    // rendezvous-receive bookkeeping) can run concurrently with a large send.
    ucx_debug_log("Write(AM) begin bytes=%zu", size);

    // For messages large enough to take UCX's rendezvous path, copy the
    // payload through the pre-registered send scratch so the wire transfer
    // always issues from the same VA. That keeps the UCX registration cache
    // hot and avoids the per-call KSM/MR registration that was costing tens
    // of ms per 16 MB send. Below the small-message threshold UCX bounces
    // through its own internal eager buffer anyway, so the copy would just
    // add latency.
    const char *send_ptr = buffer;
    if (size >= kSendScratchMinBytes && size <= send_scratch_.size() &&
        send_scratch_memh_ != nullptr) {
        std::memcpy(send_scratch_.data(), buffer, size);
        send_ptr = reinterpret_cast<const char *>(send_scratch_.data());
    }

    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    void *request;
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0, send_ptr, size,
                                  &request_param);
    }
    wait_request_completion(request, "am_send");
    ucx_debug_log("Write(AM) done bytes=%zu", size);
    return size;
}

void UcxCommunicator::Sync() {
    // No-op: ucp_am_send_nbx + wait_request_completion already guarantees
    // local completion of every Write(). A ucp_worker_flush_nbx() here would
    // force the whole protocol stack to quiesce on every message, adding
    // measurable per-call latency without changing semantics.
    (void)worker_;
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
