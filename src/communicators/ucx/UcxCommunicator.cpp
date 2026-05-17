#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/resource.h>

#include <cstring>
#include <stdexcept>

#include "gvirtus/communicators/Endpoint.h"
#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

// ============================================================================
// Per-operation completion state (passed via UCP_OP_ATTR_FIELD_USER_DATA).
//
// We deliberately do NOT use ucp_params_t.request_init / request_size: the
// completion state lives on the caller's stack and is referenced through
// user_data, which lets the immediate-completion path (request == nullptr)
// be handled trivially without touching the per-request slab.
// ============================================================================
namespace {

struct OpState {
    std::atomic<bool> complete{false};
    ucs_status_t status{UCS_OK};
    size_t length{0};
};

void send_cb(void * /*request*/, ucs_status_t status, void *user_data) {
    auto *s = static_cast<OpState *>(user_data);
    s->status = status;
    s->complete.store(true);
}

void stream_recv_cb(void * /*request*/, ucs_status_t status, size_t length,
                    void *user_data) {
    auto *s = static_cast<OpState *>(user_data);
    s->status = status;
    s->length = length;
    s->complete.store(true);
}

void wait_op(ucp_worker_h worker, void *request, OpState &state, const char *op_name) {
    if (request == nullptr) {
        // Immediate completion: the callback was NOT fired by UCX, so state
        // still has its constructed defaults (status=UCS_OK).
        return;
    }
    if (UCS_PTR_IS_ERR(request)) {
        throw std::runtime_error(std::string(op_name) + " failed: " +
                                 ucs_status_string(UCS_PTR_STATUS(request)));
    }
    while (!state.complete.load()) {
        ucp_worker_progress(worker);
    }
    ucs_status_t st = state.status;
    ucp_request_free(request);
    if (st != UCS_OK) {
        throw std::runtime_error(std::string(op_name) + " completed with error: " +
                                 ucs_status_string(st));
    }
}

// Stream send: blocks until all bytes are accepted by UCX for transmission.
size_t stream_send(ucp_worker_h worker, ucp_ep_h ep, const void *buf, size_t len) {
    OpState state;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.datatype = ucp_dt_make_contig(1);
    param.cb.send = send_cb;
    param.user_data = &state;
    void *req = ucp_stream_send_nbx(ep, buf, len, &param);
    wait_op(worker, req, state, "ucp_stream_send_nbx");
    return len;
}

// Stream recv with WAITALL semantics: loops until exactly `len` bytes have
// been read or the peer closes. Mirrors the working pattern from
// examples/ucx_benchmark/data_copy_bench.cpp.
size_t stream_recv_all(ucp_worker_h worker, ucp_ep_h ep, void *buf, size_t len) {
    char *p = static_cast<char *>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        OpState state;
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_FLAGS;
        param.datatype = ucp_dt_make_contig(1);
        param.cb.recv_stream = stream_recv_cb;
        param.user_data = &state;
        param.flags = UCP_STREAM_RECV_FLAG_WAITALL;
        size_t length = 0;
        void *req = ucp_stream_recv_nbx(ep, p, remaining, &length, &param);
        if (req == nullptr) {
            // Immediate completion: `length` populated synchronously.
            if (length == 0) {
                // No data ready; spin progress and retry.
                ucp_worker_progress(worker);
                continue;
            }
            p += length;
            remaining -= length;
        } else if (UCS_PTR_IS_ERR(req)) {
            throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ") +
                                     ucs_status_string(UCS_PTR_STATUS(req)));
        } else {
            while (!state.complete.load()) {
                ucp_worker_progress(worker);
            }
            ucs_status_t st = state.status;
            size_t got = state.length;
            ucp_request_free(req);
            if (st != UCS_OK) {
                throw std::runtime_error(std::string("ucp_stream_recv_nbx error: ") +
                                         ucs_status_string(st));
            }
            if (got == 0) {
                throw std::runtime_error("ucp_stream_recv_nbx: peer closed connection");
            }
            p += got;
            remaining -= got;
        }
    }
    return len;
}

// Raise RLIMIT_MEMLOCK so UCX can pin large buffers for zero-copy RDMA.
// Without this, transfers exceeding the per-process locked-memory cap
// (commonly 64 KiB) silently fall back to bounce-buffer copies and large
// payloads (e.g. 64 MiB) suffer a 3-4x throughput collapse.
void raise_memlock_limit(log4cplus::Logger &logger) {
    struct rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0) return;
    if (rl.rlim_cur == RLIM_INFINITY) return;

    rlim_t old_cur = rl.rlim_cur;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        LOG4CPLUS_INFO(logger, "Raised RLIMIT_MEMLOCK to unlimited (was "
                                   << (unsigned long)old_cur << " bytes)");
        return;
    }
    // Fall back: bump to hard cap.
    struct rlimit rl2{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl2) == 0 && rl2.rlim_max > rl2.rlim_cur) {
        rl2.rlim_cur = rl2.rlim_max;
        if (setrlimit(RLIMIT_MEMLOCK, &rl2) == 0) {
            LOG4CPLUS_INFO(logger, "Raised RLIMIT_MEMLOCK to hard cap "
                                       << (unsigned long)rl2.rlim_max << " bytes");
            return;
        }
    }
    LOG4CPLUS_WARN(logger, "Could not raise RLIMIT_MEMLOCK (current="
                               << (unsigned long)old_cur
                               << " bytes). Large RDMA transfers may fall back "
                                  "to bounce buffers. Run with `ulimit -l "
                                  "unlimited` or configure /etc/security/limits.conf");
}

}  // namespace

// ============================================================================
// Constructors / Destructor
// ============================================================================

UcxCommunicator::UcxCommunicator(const std::string &hostname, uint16_t port)
    : mHostname(hostname), mPort(port) {}

UcxCommunicator::UcxCommunicator(ucp_context_h ctx, ucp_worker_h worker, ucp_ep_h ep)
    : mContext(ctx), mWorker(worker), mEndpoint(ep), mOwnsContext(false) {}

UcxCommunicator::~UcxCommunicator() { Close(); }

// ============================================================================
// Context & Worker initialization
// ============================================================================

void UcxCommunicator::initContext() {
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    // STREAM only: matches socket-like Read(n)/Write(n) semantics.
    params.features = UCP_FEATURE_STREAM;

    ucs_status_t status = ucp_init(&params, nullptr, &mContext);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_init failed: ") +
                                 ucs_status_string(status));
    }
    mOwnsContext = true;
}

void UcxCommunicator::initWorker() {
    ucp_worker_params_t params{};
    params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status_t status = ucp_worker_create(mContext, &params, &mWorker);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_worker_create failed: ") +
                                 ucs_status_string(status));
    }
}

// ============================================================================
// Endpoint error handler
// ============================================================================

void UcxCommunicator::epErrCallback(void *arg, ucp_ep_h /*ep*/, ucs_status_t status) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    self->mPeerClosed.store(true);
    LOG4CPLUS_INFO(self->logger,
                   "UCX endpoint closed by peer: " << ucs_status_string(status));
}

// ============================================================================
// Wireup handshake (1-byte A/R exchange + flush)
//
// Forces UCX wireup messages to complete on this endpoint BEFORE the first
// real Read/Write. Without it, the first large transfer can stall while
// wireup messages are still in flight on the same endpoint.
// ============================================================================

void UcxCommunicator::wireupServer(ucp_ep_h ep) {
    char ack = 'A';
    stream_send(mWorker, ep, &ack, 1);
    // Flush so the ACK actually leaves before we block on the READY recv.
    OpState fstate;
    ucp_request_param_t fparam{};
    fparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    fparam.cb.send = send_cb;
    fparam.user_data = &fstate;
    void *freq = ucp_ep_flush_nbx(ep, &fparam);
    wait_op(mWorker, freq, fstate, "wireupServer flush");

    char ready = 0;
    stream_recv_all(mWorker, ep, &ready, 1);
    if (ready != 'R') {
        throw std::runtime_error("UcxCommunicator: invalid wireup READY byte");
    }
}

void UcxCommunicator::wireupClient(ucp_ep_h ep) {
    char ack = 0;
    stream_recv_all(mWorker, ep, &ack, 1);
    if (ack != 'A') {
        throw std::runtime_error("UcxCommunicator: invalid wireup ACK byte");
    }
    char ready = 'R';
    stream_send(mWorker, ep, &ready, 1);
    OpState fstate;
    ucp_request_param_t fparam{};
    fparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    fparam.cb.send = send_cb;
    fparam.user_data = &fstate;
    void *freq = ucp_ep_flush_nbx(ep, &fparam);
    wait_op(mWorker, freq, fstate, "wireupClient flush");
}

// ============================================================================
// Server side
// ============================================================================

void UcxCommunicator::listenerCallback(ucp_conn_request_h conn_request, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    self->mConnReq.conn_req = conn_request;
    self->mConnReq.ready.store(true);
}

void UcxCommunicator::Serve() {
    LOG4CPLUS_DEBUG(logger, "Serve() called");
    raise_memlock_limit(logger);

    initContext();
    initWorker();

    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_port = htons(mPort);
    if (mHostname.empty() || mHostname == "0.0.0.0") {
        listen_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, mHostname.c_str(), &listen_addr.sin_addr) != 1) {
            struct hostent *ent = gethostbyname(mHostname.c_str());
            if (!ent) {
                throw std::runtime_error("UcxCommunicator: Can't resolve hostname '" + mHostname +
                                         "'");
            }
            memcpy(&listen_addr.sin_addr, ent->h_addr_list[0], ent->h_length);
        }
    }

    ucp_listener_params_t listener_params{};
    listener_params.field_mask =
        UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listener_params.sockaddr.addr = reinterpret_cast<struct sockaddr *>(&listen_addr);
    listener_params.sockaddr.addrlen = sizeof(listen_addr);
    listener_params.conn_handler.cb = listenerCallback;
    listener_params.conn_handler.arg = const_cast<UcxCommunicator *>(this);

    ucs_status_t status = ucp_listener_create(mWorker, &listener_params, &mListener);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_listener_create failed: ") +
                                 ucs_status_string(status));
    }

    LOG4CPLUS_INFO(logger, "Listening on " << mHostname << ":" << mPort << " (UCX)");
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    LOG4CPLUS_TRACE(logger, "Accept() waiting for connection...");

    mConnReq.ready.store(false);
    while (!mConnReq.ready.load()) {
        ucp_worker_progress(mWorker);
    }

    ucp_worker_h client_worker;
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucs_status_t status = ucp_worker_create(mContext, &worker_params, &client_worker);
    if (status != UCS_OK) {
        throw std::runtime_error(
            std::string("UcxCommunicator::Accept: ucp_worker_create failed: ") +
            ucs_status_string(status));
    }

    // Create the accepted-connection wrapper now so that its `this` is the
    // err_handler arg and the wireup helpers can operate on its worker.
    auto *client = new UcxCommunicator(mContext, client_worker, nullptr);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.conn_request = mConnReq.conn_req;
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = epErrCallback;
    ep_params.err_handler.arg = client;

    ucp_ep_h client_ep;
    status = ucp_ep_create(client_worker, &ep_params, &client_ep);
    if (status != UCS_OK) {
        ucp_worker_destroy(client_worker);
        delete client;
        throw std::runtime_error(std::string("UcxCommunicator::Accept: ucp_ep_create failed: ") +
                                 ucs_status_string(status));
    }
    client->mEndpoint = client_ep;

    try {
        client->wireupServer(client_ep);
    } catch (const std::exception &e) {
        LOG4CPLUS_ERROR(logger, "Accept wireup failed: " << e.what());
        delete client;
        throw;
    }

    LOG4CPLUS_INFO(logger, "Client connected (UCX endpoint wired up)");
    return client;
}

// ============================================================================
// Client side
// ============================================================================

void UcxCommunicator::Connect() {
    LOG4CPLUS_DEBUG(logger, "Connect() to " << mHostname << ":" << mPort);
    raise_memlock_limit(logger);

    initContext();
    initWorker();

    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(mPort);
    if (inet_pton(AF_INET, mHostname.c_str(), &server_addr.sin_addr) != 1) {
        struct hostent *ent = gethostbyname(mHostname.c_str());
        if (!ent) {
            throw std::runtime_error("UcxCommunicator: Can't resolve hostname '" + mHostname + "'");
        }
        memcpy(&server_addr.sin_addr, ent->h_addr_list[0], ent->h_length);
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<struct sockaddr *>(&server_addr);
    ep_params.sockaddr.addrlen = sizeof(server_addr);
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = epErrCallback;
    ep_params.err_handler.arg = this;

    ucs_status_t status = ucp_ep_create(mWorker, &ep_params, &mEndpoint);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_ep_create (connect) failed: ") +
                                 ucs_status_string(status));
    }

    wireupClient(mEndpoint);

    LOG4CPLUS_INFO(logger, "Connected to " << mHostname << ":" << mPort << " (UCX, wired up)");
}

// ============================================================================
// Data transfer
// ============================================================================

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Read() size=" << size);
    // Treat a previously-flagged peer disconnect as EOF so the backend's
    // per-client thread can exit its loop cleanly. Without this, the next
    // Read after disconnect would throw inside a detached std::thread and
    // call std::terminate(), killing the whole Process child.
    if (mPeerClosed.load()) return 0;
    try {
        return stream_recv_all(mWorker, mEndpoint, buffer, size);
    } catch (const std::exception &e) {
        // Convert peer-disconnect errors into EOF (0 bytes). Any other UCX
        // failure mode also surfaces here; logging at INFO keeps the backend
        // log readable on normal client teardown.
        mPeerClosed.store(true);
        LOG4CPLUS_INFO(logger, "Read() EOF: " << e.what());
        return 0;
    }
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Write() size=" << size);
    if (mPeerClosed.load()) return 0;
    try {
        return stream_send(mWorker, mEndpoint, buffer, size);
    } catch (const std::exception &e) {
        mPeerClosed.store(true);
        LOG4CPLUS_INFO(logger, "Write() EOF: " << e.what());
        return 0;
    }
}

void UcxCommunicator::Sync() {
    if (!mEndpoint) return;
    OpState state;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = send_cb;
    param.user_data = &state;
    void *req = ucp_ep_flush_nbx(mEndpoint, &param);
    wait_op(mWorker, req, state, "ucp_ep_flush_nbx");
}

void UcxCommunicator::Close() {
    if (mEndpoint) {
        // Graceful close: drain pending ops before tearing down.
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = 0;  // graceful (not FORCE)
        void *request = ucp_ep_close_nbx(mEndpoint, &param);
        if (request != nullptr && !UCS_PTR_IS_ERR(request)) {
            ucs_status_t st;
            do {
                ucp_worker_progress(mWorker);
                st = ucp_request_check_status(request);
            } while (st == UCS_INPROGRESS);
            ucp_request_free(request);
        }
        mEndpoint = nullptr;
    }

    if (mListener) {
        ucp_listener_destroy(mListener);
        mListener = nullptr;
    }

    if (mWorker) {
        ucp_worker_destroy(mWorker);
        mWorker = nullptr;
    }

    if (mOwnsContext && mContext) {
        ucp_cleanup(mContext);
        mContext = nullptr;
        mOwnsContext = false;
    }
}

// ============================================================================
// Factory (loaded by CommunicatorFactory via dlopen)
// ============================================================================

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_end = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_end) {
        throw std::runtime_error("UcxCommunicator: expected Endpoint_Ucx");
    }
    return std::make_shared<UcxCommunicator>(ucx_end->address(), ucx_end->port());
}
