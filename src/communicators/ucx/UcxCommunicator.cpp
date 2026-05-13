#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstring>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

namespace {

log4cplus::Logger ucx_logger() {
    return log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator"));
}

sockaddr_storage make_sockaddr(const std::string &host, std::uint16_t port,
                               socklen_t &out_len) {
    sockaddr_storage storage{};
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *result = nullptr;
    int rc = getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (rc != 0 || result == nullptr) {
        throw std::runtime_error("UcxCommunicator: getaddrinfo failed for " + host + ": " +
                                 (rc != 0 ? gai_strerror(rc) : "no result"));
    }

    std::memcpy(&storage, result->ai_addr, result->ai_addrlen);
    out_len = static_cast<socklen_t>(result->ai_addrlen);
    if (storage.ss_family == AF_INET) {
        reinterpret_cast<sockaddr_in *>(&storage)->sin_port = htons(port);
        out_len = sizeof(sockaddr_in);
    } else if (storage.ss_family == AF_INET6) {
        reinterpret_cast<sockaddr_in6 *>(&storage)->sin6_port = htons(port);
        out_len = sizeof(sockaddr_in6);
    }
    freeaddrinfo(result);
    return storage;
}

}  // namespace

UcxCommunicator::UcxShared::~UcxShared() {
    if (worker != nullptr) {
        ucp_worker_destroy(worker);
        worker = nullptr;
    }
    if (context != nullptr) {
        ucp_cleanup(context);
        context = nullptr;
    }
}

UcxCommunicator::UcxCommunicator(const std::string &hostname, std::uint16_t port)
    : hostname_(hostname), port_(port) {
    is_listener_ = true;
    init_shared();
}

UcxCommunicator::UcxCommunicator(std::shared_ptr<UcxShared> shared, std::string hostname,
                                 std::uint16_t port)
    : shared_(std::move(shared)), hostname_(std::move(hostname)), port_(port) {
    is_listener_ = false;
}

UcxCommunicator::~UcxCommunicator() { Close(); }

void UcxCommunicator::init_shared() {
    if (shared_) {
        return;
    }
    shared_ = std::make_shared<UcxShared>();

    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    // Stream is the only feature we need: byte-stream send/recv that maps
    // 1:1 to Communicator::Read / Communicator::Write.
    params.features = UCP_FEATURE_STREAM;

    ucs_status_t status = ucp_init(&params, nullptr, &shared_->context);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_init failed: ") +
                                 ucs_status_string(status));
    }

    ucp_worker_params_t wparams{};
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    // Multiple OS threads may call into the worker, but they are externally
    // serialized via shared_->worker_mutex.
    wparams.thread_mode = UCS_THREAD_MODE_SERIALIZED;

    status = ucp_worker_create(shared_->context, &wparams, &shared_->worker);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_worker_create failed: ") +
                                 ucs_status_string(status));
    }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void UcxCommunicator::listener_conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    {
        std::lock_guard<std::mutex> lock(self->conn_mutex_);
        self->pending_conn_requests_.push(conn_request);
    }
    self->conn_cv_.notify_one();
}

void UcxCommunicator::endpoint_error_handler(void *arg, ucp_ep_h /*ep*/, ucs_status_t status) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    self->endpoint_failed_.store(true);
    LOG4CPLUS_DEBUG(ucx_logger(),
                    "endpoint error: " << ucs_status_string(status));
}

void UcxCommunicator::stream_recv_callback(void * /*request*/, ucs_status_t status,
                                           size_t length, void *user_data) {
    auto *state = static_cast<StreamRecvState *>(user_data);
    state->status = status;
    state->length = length;
    state->done.store(1, std::memory_order_release);
}

void UcxCommunicator::stream_send_callback(void * /*request*/, ucs_status_t status,
                                           void *user_data) {
    auto *flag = static_cast<std::atomic<int> *>(user_data);
    if (status != UCS_OK) {
        // Encode error in the high bit so the waiter can distinguish.
        flag->store(2, std::memory_order_release);
    } else {
        flag->store(1, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// Completion helpers
// ---------------------------------------------------------------------------

void UcxCommunicator::wait_send_completion(void *request, std::atomic<int> *flag) {
    if (request == nullptr) {
        // Immediate completion: callback was NOT invoked.  The operation
        // succeeded synchronously.
        return;
    }
    if (UCS_PTR_IS_ERR(request)) {
        throw std::runtime_error(std::string("UcxCommunicator: stream send error: ") +
                                 ucs_status_string(UCS_PTR_STATUS(request)));
    }

    unsigned idle_iters = 0;
    while (flag->load(std::memory_order_acquire) == 0) {
        unsigned progressed;
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            progressed = ucp_worker_progress(shared_->worker);
        }
        if (progressed == 0) {
            // Adaptive backoff: yield first, then short sleeps. Keeps tight
            // hot paths fast while letting other threads (CUDA, accept
            // listener, etc.) make progress when this stream is idle.
            if (++idle_iters < 64) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        } else {
            idle_iters = 0;
        }
    }

    int value = flag->load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        ucp_request_free(request);
    }
    if (value == 2) {
        throw std::runtime_error("UcxCommunicator: stream send completed with error");
    }
}

size_t UcxCommunicator::wait_stream_recv_completion(void *request, StreamRecvState *state,
                                                    size_t inline_length) {
    // Immediate completion: callback was NOT invoked.  The number of bytes
    // actually received is in the in/out `length` parameter passed to
    // ucp_stream_recv_nbx, which the caller forwards as `inline_length`.
    if (request == nullptr) {
        return inline_length;
    }
    if (UCS_PTR_IS_ERR(request)) {
        ucs_status_t status = UCS_PTR_STATUS(request);
        if (status == UCS_ERR_CANCELED || status == UCS_ERR_CONNECTION_RESET ||
            status == UCS_ERR_ENDPOINT_TIMEOUT) {
            return 0;  // EOF semantics
        }
        throw std::runtime_error(std::string("UcxCommunicator: stream recv error: ") +
                                 ucs_status_string(status));
    }

    unsigned idle_iters = 0;
    while (state->done.load(std::memory_order_acquire) == 0) {
        unsigned progressed;
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            progressed = ucp_worker_progress(shared_->worker);
        }
        if (progressed == 0) {
            if (++idle_iters < 64) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        } else {
            idle_iters = 0;
        }
    }

    ucs_status_t status = state->status;
    size_t length = state->length;
    {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        ucp_request_free(request);
    }

    if (status != UCS_OK) {
        if (status == UCS_ERR_CANCELED || status == UCS_ERR_CONNECTION_RESET ||
            status == UCS_ERR_ENDPOINT_TIMEOUT) {
            return 0;
        }
        throw std::runtime_error(std::string("UcxCommunicator: stream recv error: ") +
                                 ucs_status_string(status));
    }
    return length;
}

ucp_conn_request_h UcxCommunicator::wait_for_connection_request() const {
    std::unique_lock<std::mutex> lock(conn_mutex_);
    while (pending_conn_requests_.empty()) {
        lock.unlock();
        unsigned progressed;
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            progressed = ucp_worker_progress(shared_->worker);
        }
        if (progressed == 0) {
            // Accept is a low-frequency long-idle event.  Use a real sleep
            // so we don't burn a core waiting for the next client.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        lock.lock();
    }
    ucp_conn_request_h cr = pending_conn_requests_.front();
    pending_conn_requests_.pop();
    return cr;
}

// ---------------------------------------------------------------------------
// Server: Serve / Accept
// ---------------------------------------------------------------------------

void UcxCommunicator::Serve() {
    if (listener_ != nullptr) {
        return;
    }
    socklen_t addrlen = 0;
    sockaddr_storage addr = make_sockaddr(hostname_, port_, addrlen);

    ucp_listener_params_t params{};
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = reinterpret_cast<sockaddr *>(&addr);
    params.sockaddr.addrlen = addrlen;
    params.conn_handler.cb = &UcxCommunicator::listener_conn_handler;
    params.conn_handler.arg = this;

    ucs_status_t status = ucp_listener_create(shared_->worker, &params, &listener_);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_listener_create failed: ") +
                                 ucs_status_string(status));
    }

    LOG4CPLUS_INFO(ucx_logger(),
                   "UCX listener bound to " << hostname_ << ":" << port_);
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    if (listener_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Accept called before Serve");
    }

    ucp_conn_request_h conn_request = wait_for_connection_request();

    auto *accepted = new UcxCommunicator(shared_, hostname_, port_);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.conn_request = conn_request;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = accepted;
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;

    ucs_status_t status;
    {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        status = ucp_ep_create(shared_->worker, &ep_params, &accepted->endpoint_);
    }
    if (status != UCS_OK) {
        // Tell the peer it failed so it can give up cleanly instead of
        // dangling waiting for the endpoint we never created.
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            ucp_listener_reject(listener_, conn_request);
        }
        delete accepted;
        throw std::runtime_error(std::string("UcxCommunicator: server ucp_ep_create failed: ") +
                                 ucs_status_string(status));
    }

    LOG4CPLUS_INFO(ucx_logger(), "UCX accepted client (ep=" << accepted->endpoint_ << ")");
    return accepted;
}

void UcxCommunicator::Connect() {
    if (!shared_) {
        init_shared();
    }
    if (endpoint_ != nullptr) {
        return;
    }
    socklen_t addrlen = 0;
    sockaddr_storage addr = make_sockaddr(hostname_, port_, addrlen);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                           UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<sockaddr *>(&addr);
    ep_params.sockaddr.addrlen = addrlen;
    ep_params.err_handler.cb = &UcxCommunicator::endpoint_error_handler;
    ep_params.err_handler.arg = this;
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;

    ucs_status_t status;
    {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        status = ucp_ep_create(shared_->worker, &ep_params, &endpoint_);
    }
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: client ucp_ep_create failed: ") +
                                 ucs_status_string(status));
    }
    LOG4CPLUS_INFO(ucx_logger(),
                   "UCX client connected to " << hostname_ << ":" << port_);
}

// ---------------------------------------------------------------------------
// Data plane: Read / Write
// ---------------------------------------------------------------------------

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    if (endpoint_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Read called without an active endpoint");
    }
    if (size == 0) {
        return 0;
    }

    size_t total = 0;
    while (total < size) {
        if (endpoint_failed_.load()) {
            return total;  // EOF: peer closed / error
        }

        StreamRecvState state;
        size_t length_inout = 0;  // out: bytes copied for the inline-completion path

        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_USER_DATA |
                             UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_FLAGS;
        param.cb.recv_stream = &UcxCommunicator::stream_recv_callback;
        param.user_data = &state;
        param.datatype = ucp_dt_make_contig(1);
        // WAITALL: only deliver completion when all `size - total` bytes have
        // arrived.  This collapses the "fill the buffer" loop in the common
        // case.
        param.flags = UCP_STREAM_RECV_FLAG_WAITALL;

        void *request;
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            request = ucp_stream_recv_nbx(endpoint_, buffer + total, size - total,
                                          &length_inout, &param);
        }
        size_t recvd = wait_stream_recv_completion(request, &state, length_inout);
        if (recvd == 0) {
            // EOF: matches the TCP `Read returns 0` contract; getstring()
            // exits cleanly and the per-client thread tears down.
            return total;
        }
        total += recvd;
    }
    return total;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (endpoint_ == nullptr) {
        throw std::runtime_error("UcxCommunicator: Write called without an active endpoint");
    }
    if (size == 0) {
        return 0;
    }
    if (endpoint_failed_.load()) {
        throw std::runtime_error("UcxCommunicator: Write on failed endpoint");
    }

    std::atomic<int> flag{0};

    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.send = &UcxCommunicator::stream_send_callback;
    param.user_data = &flag;
    param.datatype = ucp_dt_make_contig(1);

    void *request;
    {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        request = ucp_stream_send_nbx(endpoint_, buffer, size, &param);
    }
    wait_send_completion(request, &flag);
    return size;
}

void UcxCommunicator::Sync() {
    // Stream completions are already per-call synchronous from the caller's
    // perspective (Read/Write block until all bytes are in-flight).  Nothing
    // additional to do.
}

void UcxCommunicator::Close() {
    if (endpoint_ != nullptr && shared_) {
        void *request;
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        // FORCE is required to avoid hangs that the prior AM-based
        // implementation discovered.  Peers will see endpoint_error.
        param.flags = UCP_EP_CLOSE_FLAG_FORCE;
        {
            std::lock_guard<std::mutex> lk(shared_->worker_mutex);
            request = ucp_ep_close_nbx(endpoint_, &param);
        }
        if (UCS_PTR_IS_ERR(request)) {
            LOG4CPLUS_WARN(ucx_logger(), "ucp_ep_close_nbx returned error: "
                                             << ucs_status_string(UCS_PTR_STATUS(request)));
        } else if (request != nullptr) {
            unsigned idle_iters = 0;
            for (;;) {
                ucs_status_t s;
                unsigned progressed;
                {
                    std::lock_guard<std::mutex> lk(shared_->worker_mutex);
                    s = ucp_request_check_status(request);
                    if (s == UCS_INPROGRESS) {
                        progressed = ucp_worker_progress(shared_->worker);
                    } else {
                        progressed = 0;
                    }
                }
                if (s != UCS_INPROGRESS) {
                    break;
                }
                if (progressed == 0) {
                    if (++idle_iters < 64) {
                        std::this_thread::yield();
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                } else {
                    idle_iters = 0;
                }
            }
            {
                std::lock_guard<std::mutex> lk(shared_->worker_mutex);
                ucp_request_free(request);
            }
        }
        endpoint_ = nullptr;
    }

    if (is_listener_ && listener_ != nullptr && shared_) {
        std::lock_guard<std::mutex> lk(shared_->worker_mutex);
        ucp_listener_destroy(listener_);
        listener_ = nullptr;
    }
    // shared_ will be released when this owner goes away; ctx+worker are
    // destroyed when the last accepted client + listener are all gone.
}

void UcxCommunicator::run() {
    // Compatibility shim with the Communicator interface: nothing to do.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_endpoint = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_endpoint) {
        throw std::runtime_error("UcxCommunicator: endpoint type mismatch");
    }
    return std::make_shared<UcxCommunicator>(ucx_endpoint->address(), ucx_endpoint->port());
}
