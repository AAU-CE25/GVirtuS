#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;
using gvirtus::communicators::FramedStream;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

UcxCommunicator::UcxCommunicator(const std::string &hostname, std::uint16_t port)
    : hostname_(hostname), port_(port) {}

UcxCommunicator::~UcxCommunicator() {
    Close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Connect — initializes UCX and establishes endpoint
// ─────────────────────────────────────────────────────────────────────────────

void UcxCommunicator::InitUcpContext() {
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_STREAM;  // stream transport for sync path
                                              // + WAKEUP for ProgressLoop arm/wait

    ucp_config_t *config = nullptr;
    ucp_config_read(nullptr, nullptr, &config);

    ucs_status_t status = ucp_init(&params, config, &ucp_context_);
    ucp_config_release(config);

    if (status != UCS_OK)
        throw std::runtime_error("ucp_init failed: " +
                                 std::string(ucs_status_string(status)));
}

void UcxCommunicator::CreateWorker() {
    ucp_worker_params_t params{};
    params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCS_THREAD_MODE_SERIALIZED;

    ucs_status_t status = ucp_worker_create(ucp_context_, &params, &ucp_worker_);
    if (status != UCS_OK)
        throw std::runtime_error("ucp_worker_create failed: " +
                                 std::string(ucs_status_string(status)));
}

void UcxCommunicator::CreateEndpoint() {
    // Get local worker address to send to the peer
    ucp_address_t *local_addr = nullptr;
    size_t         local_addr_len = 0;
    ucp_worker_get_address(ucp_worker_, &local_addr, &local_addr_len);

    // Build sockaddr for the remote backend
    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port   = htons(port_);
    inet_pton(AF_INET, hostname_.c_str(), &server_addr.sin_addr);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = reinterpret_cast<const sockaddr *>(&server_addr);
    ep_params.sockaddr.addrlen = sizeof(server_addr);

    ucs_status_t status = ucp_ep_create(ucp_worker_, &ep_params, &ucp_ep_);
    ucp_worker_release_address(ucp_worker_, local_addr);

    if (status != UCS_OK)
        throw std::runtime_error("ucp_ep_create failed: " +
                                 std::string(ucs_status_string(status)));
}

void UcxCommunicator::Connect() {
    InitUcpContext();
    CreateWorker();
    CreateEndpoint();

    // FramedStream owns the async progress loop and in_flight map
    framed_stream_ = std::make_unique<FramedStream>(ucp_worker_);
    framed_stream_->EnsureDispatchLoop(ucp_ep_);
}

// ─────────────────────────────────────────────────────────────────────────────
// SendAsync — the new async path used by Frontend::Execute(ASYNC)
// Serializes [routine_name \0][buffer bytes] into one payload,
// hands it to FramedStream::SendAsync, returns the PendingRequest handle.
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<FramedStream::PendingRequest>
UcxCommunicator::SendAsync(const char *routine, const Buffer *input_buffer) {
    if (!framed_stream_ || ucp_ep_ == nullptr)
        throw std::runtime_error("UcxCommunicator::SendAsync called before Connect()");

    size_t name_len = std::strlen(routine) + 1;
    size_t buf_size = input_buffer ? input_buffer->GetBufferSize() : 0;
    size_t total    = name_len + buf_size;

    std::vector<uint8_t> payload(total);
    std::memcpy(payload.data(), routine, name_len);

    if (buf_size > 0)
        std::memcpy(payload.data() + name_len,
                    input_buffer->GetBuffer(),   // ← correct method
                    buf_size);

    return framed_stream_->SendAsync(ucp_ep_, MsgType::REQUEST,
                                     payload.data(),
                                     static_cast<uint32_t>(total));
}

// ─────────────────────────────────────────────────────────────────────────────
// Synchronous path — Write/Sync/Read
// Keeps the existing TCP-style call flow working over UCX stream.
// Write() accumulates into write_buffer_, Sync() sends the whole
// batch as one framed message, Read() receives and drains.
// ─────────────────────────────────────────────────────────────────────────────

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    if (!framed_stream_ || ucp_ep_ == nullptr) {
        std::printf("UCX Write called before Connect()\n");
        return size;
    }
    // Accumulate into staging buffer — flushed on Sync()
    const auto *b = reinterpret_cast<const uint8_t *>(buffer);
    write_buffer_.insert(write_buffer_.end(), b, b + size);
    return size;
}

void UcxCommunicator::Sync() {
    if (!framed_stream_ || ucp_ep_ == nullptr || write_buffer_.empty())
        return;

    // Send accumulated write_buffer_ as one framed REQUEST
    // and block until the backend sends a RESPONSE
    auto pending = framed_stream_->SendAsync(
        ucp_ep_, MsgType::REQUEST,
        write_buffer_.data(),
        static_cast<uint32_t>(write_buffer_.size()));

    write_buffer_.clear();

    // Block for the response — this is the synchronous path
    auto response = pending->Wait(std::chrono::milliseconds(5000));

    // Store response payload for subsequent Read() calls
    read_buffer_ = std::move(response);
    read_offset_ = 0;
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    if (read_offset_ + size > read_buffer_.size())
        throw std::runtime_error("UcxCommunicator::Read: not enough data in response buffer");

    std::memcpy(buffer, read_buffer_.data() + read_offset_, size);
    read_offset_ += size;
    return size;
}

// ─────────────────────────────────────────────────────────────────────────────
// Close / Serve / Accept / run — lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void UcxCommunicator::Close() {
    framed_stream_.reset();  // stops progress + dispatch threads first

    if (ucp_ep_ != nullptr) {
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
        auto *req = ucp_ep_close_nbx(ucp_ep_, &param);
        if (UCS_PTR_IS_PTR(req)) {
            while (ucp_request_check_status(req) == UCS_INPROGRESS)
                ucp_worker_progress(ucp_worker_);
            ucp_request_free(req);
        }
        ucp_ep_ = nullptr;
    }

    if (ucp_worker_ != nullptr) {
        ucp_worker_destroy(ucp_worker_);
        ucp_worker_ = nullptr;
    }

    if (ucp_context_ != nullptr) {
        ucp_cleanup(ucp_context_);
        ucp_context_ = nullptr;
    }
}

// Called by UCX when a new connection request arrives on the listener
void UcxCommunicator::OnConnectionRequest(ucp_conn_request_h conn_req, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);

    // Accept the connection request into a new endpoint
    ucp_ep_params_t ep_params{};
    ep_params.field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_req;

    ucp_ep_h new_ep = nullptr;
    ucs_status_t status = ucp_ep_create(self->ucp_worker_, &ep_params, &new_ep);
    if (status != UCS_OK) {
        std::fprintf(stderr, "UcxCommunicator: ucp_ep_create from conn_request failed: %s\n",
                     ucs_status_string(status));
        return;
    }

    {
        std::lock_guard<std::mutex> lock(self->accept_mutex_);
        self->pending_eps_.push(new_ep);
    }
    self->accept_cv_.notify_one();
}

void UcxCommunicator::Serve() {
    InitUcpContext();
    CreateWorker();

    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_port   = htons(port_);
    inet_pton(AF_INET, hostname_.c_str(), &listen_addr.sin_addr);

    ucp_listener_params_t params{};
    params.field_mask        = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                               UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr     = reinterpret_cast<const sockaddr *>(&listen_addr);
    params.sockaddr.addrlen  = sizeof(listen_addr);
    params.conn_handler.cb   = &UcxCommunicator::OnConnectionRequest;
    params.conn_handler.arg  = this;

    ucs_status_t status = ucp_listener_create(ucp_worker_, &params, &ucp_listener_);
    if (status != UCS_OK)
        throw std::runtime_error("ucp_listener_create failed: " +
                                 std::string(ucs_status_string(status)));

    std::printf("[UcxCommunicator] Listening on %s:%u\n", hostname_.c_str(), port_);
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    // Drive the UCX progress engine until a connection arrives
    {
        std::unique_lock<std::mutex> lock(accept_mutex_);
        while (pending_eps_.empty()) {
            lock.unlock();
            ucp_worker_progress(ucp_worker_);
            lock.lock();
        }
    }

    ucp_ep_h new_ep;
    {
        std::lock_guard<std::mutex> lock(accept_mutex_);
        new_ep = pending_eps_.front();
        pending_eps_.pop();
    }

    // Build a connected UcxCommunicator that wraps the accepted endpoint
    auto *peer = new UcxCommunicator();
    peer->ucp_context_ = ucp_context_;   // share context (not owned)
    peer->ucp_worker_  = ucp_worker_;    // share worker  (not owned)
    peer->ucp_ep_      = new_ep;
    peer->framed_stream_ = std::make_unique<FramedStream>(ucp_worker_);
    peer->framed_stream_->EnsureDispatchLoop(new_ep);
    return peer;
}

void UcxCommunicator::run() {
    // Progress loop — keep UCX moving when idle
    while (ucp_worker_ != nullptr)
        ucp_worker_progress(ucp_worker_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory — called by LD_Lib when loading the .so
// ─────────────────────────────────────────────────────────────────────────────

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_ep = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_ep)
        throw std::runtime_error("UcxCommunicator: endpoint type mismatch");
    return std::make_shared<UcxCommunicator>(ucx_ep->address(), ucx_ep->port());
}