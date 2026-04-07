#include "UcxCommunicator.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <stdexcept>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

namespace {
constexpr ucp_tag_t kGvirtusTag = 0x4746495254555301ULL;
constexpr ucp_tag_t kGvirtusTagMask = UINT64_MAX;
constexpr size_t kFrameHeaderSize = sizeof(uint64_t);

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

    ucp_params_t ucp_params{};
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG;

    ucs_status_t status = ucp_init(&ucp_params, nullptr, &context_);
    if (status != UCS_OK) {
        throw std::runtime_error("UcxCommunicator: ucp_init failed: " +
                                 std::string(ucs_status_string(status)));
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

    initialized_ = true;
    ucx_debug_log("init_ucx completed host=%s port=%u", hostname_.c_str(), port_);
}

void UcxCommunicator::destroy_ucx() {
    if (!initialized_) return;

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

    if (owns_context_worker_listener_ && listener_ != nullptr) {
        ucp_listener_destroy(listener_);
        listener_ = nullptr;
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
    std::lock_guard<std::mutex> lock(conn_mutex_);
    pending_conn_requests_.push(conn_request);
    ucx_debug_log("enqueue_connection request=%p queue_size=%zu", (void *)conn_request,
                  pending_conn_requests_.size());
    conn_cv_.notify_one();
}

ucp_conn_request_h UcxCommunicator::wait_for_connection_request() {
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

    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        if (worker_ != nullptr) {
            ucp_worker_progress(worker_);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

void UcxCommunicator::recv_message_exact(void *buffer, size_t size, const char *op_name) {
    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    if (size == 0) {
        return;
    }

    void *request =
        ucp_tag_recv_nbx(worker_, buffer, size, kGvirtusTag, kGvirtusTagMask, &request_param);
    wait_request_completion(request, op_name);
}

void UcxCommunicator::send_message_exact(const void *buffer, size_t size, const char *op_name) {
    ucp_request_param_t request_param{};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    request_param.datatype = ucp_dt_make_contig(1);

    if (size == 0) {
        return;
    }

    void *request = ucp_tag_send_nbx(endpoint_, buffer, size, kGvirtusTag, &request_param);
    wait_request_completion(request, op_name);
}

void UcxCommunicator::Serve() {
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
    accepted->endpoint_failed_.store(false);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.conn_request = req;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = accepted;

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
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Read called without an active endpoint");
    }

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    if (size == 0) {
        return 0;
    }

    ucx_debug_log("Read begin bytes=%zu tag=0x%llx", size,
                  static_cast<unsigned long long>(kGvirtusTag));

    size_t copied = 0;
    while (copied < size) {
        const size_t available =
            (pending_read_offset_ < pending_read_bytes_.size())
                ? (pending_read_bytes_.size() - pending_read_offset_)
                : 0;

        if (available == 0) {
            uint64_t frame_len = 0;
            recv_message_exact(&frame_len, kFrameHeaderSize, "tag_recv_header");

            pending_read_bytes_.clear();
            pending_read_offset_ = 0;

            if (frame_len > 0) {
                pending_read_bytes_.resize(static_cast<size_t>(frame_len));
                recv_message_exact(pending_read_bytes_.data(), pending_read_bytes_.size(),
                                   "tag_recv_payload");
            }
            continue;
        }

        const size_t to_copy = std::min(size - copied, available);
        std::memcpy(buffer + copied, pending_read_bytes_.data() + pending_read_offset_, to_copy);
        copied += to_copy;
        pending_read_offset_ += to_copy;

        if (pending_read_offset_ == pending_read_bytes_.size()) {
            pending_read_bytes_.clear();
            pending_read_offset_ = 0;
        }
    }

    ucx_debug_log("Read done bytes=%zu", size);
    return size;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr || worker_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    ucx_debug_log("Write begin bytes=%zu tag=0x%llx", size,
                  static_cast<unsigned long long>(kGvirtusTag));

    const uint64_t frame_len = static_cast<uint64_t>(size);
    send_message_exact(&frame_len, kFrameHeaderSize, "tag_send_header");
    if (size > 0) {
        send_message_exact(buffer, size, "tag_send_payload");
    }

    ucx_debug_log("Write done bytes=%zu", size);
    return size;
}

void UcxCommunicator::Sync() {
    if (worker_ == nullptr) {
        return;
    }

    ucp_request_param_t request_param{};
    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
    ucx_debug_log("Sync begin (worker flush)");
    void *request = ucp_worker_flush_nbx(worker_, &request_param);
    wait_request_completion(request, "worker_flush");
    ucx_debug_log("Sync done");
}

void UcxCommunicator::Close() {
    ucx_debug_log("Close called");
    running_ = false;
    conn_cv_.notify_all();
    destroy_ucx();
}

void UcxCommunicator::run() {
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
