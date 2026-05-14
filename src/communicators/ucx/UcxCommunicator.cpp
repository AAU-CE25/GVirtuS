#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <netdb.h>

#include <cstring>
#include <stdexcept>

#include "gvirtus/communicators/Endpoint.h"
#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

// ----------------------------------------------------------------------------
// Request completion callback for non-blocking operations
// ----------------------------------------------------------------------------
struct UcxRequest {
    std::atomic<bool> complete{false};
};

static void requestInit(void *request) {
    auto *req = static_cast<UcxRequest *>(request);
    req->complete.store(false);
}

static void sendCallback(void *request, ucs_status_t status, void * /*user_data*/) {
    auto *req = static_cast<UcxRequest *>(request);
    req->complete.store(true);
}

static void recvCallback(void *request, ucs_status_t status, const ucp_tag_recv_info_t * /*info*/,
                          void * /*user_data*/) {
    auto *req = static_cast<UcxRequest *>(request);
    req->complete.store(true);
}

// ----------------------------------------------------------------------------
// Constructors / Destructor
// ----------------------------------------------------------------------------

UcxCommunicator::UcxCommunicator(const std::string &hostname, uint16_t port)
    : mHostname(hostname), mPort(port) {}

UcxCommunicator::UcxCommunicator(ucp_context_h ctx, ucp_worker_h worker, ucp_ep_h ep)
    : mContext(ctx), mWorker(worker), mEndpoint(ep), mOwnsContext(true) {}

UcxCommunicator::~UcxCommunicator() {
    Close();
}

// ----------------------------------------------------------------------------
// Context & Worker initialization
// ----------------------------------------------------------------------------

void UcxCommunicator::initContext() {
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT;
    params.features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
    params.request_size = sizeof(UcxRequest);
    params.request_init = requestInit;

    ucs_status_t status = ucp_init(&params, nullptr, &mContext);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_init failed: ") +
                                 ucs_status_string(status));
    }
    mOwnsContext = true;
}

void UcxCommunicator::initWorker() {
    ucp_worker_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status_t status = ucp_worker_create(mContext, &params, &mWorker);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_worker_create failed: ") +
                                 ucs_status_string(status));
    }
}

// ----------------------------------------------------------------------------
// Wait for a non-blocking request to complete
// ----------------------------------------------------------------------------

void UcxCommunicator::waitForCompletion(void *request) {
    if (request == nullptr) {
        // Immediate completion
        return;
    }
    if (UCS_PTR_IS_ERR(request)) {
        throw std::runtime_error(std::string("UcxCommunicator: UCX operation failed: ") +
                                 ucs_status_string(UCS_PTR_STATUS(request)));
    }

    auto *req = static_cast<UcxRequest *>(request);
    while (!req->complete.load()) {
        ucp_worker_progress(mWorker);
    }
    ucp_request_free(request);
}

// ----------------------------------------------------------------------------
// Server side
// ----------------------------------------------------------------------------

void UcxCommunicator::listenerCallback(ucp_conn_request_h conn_request, void *arg) {
    auto *self = static_cast<UcxCommunicator *>(arg);
    self->mConnReq.conn_req = conn_request;
    self->mConnReq.ready.store(true);
}

void UcxCommunicator::Serve() {
    LOG4CPLUS_DEBUG(logger, "Serve() called");

    initContext();
    initWorker();

    // Resolve the listen address
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

    // Create UCP listener
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

    // Wait for a connection request from the listener callback
    mConnReq.ready.store(false);
    while (!mConnReq.ready.load()) {
        ucp_worker_progress(mWorker);
    }

    // Create a new worker + endpoint for this accepted connection
    ucp_worker_h client_worker;
    ucp_worker_params_t worker_params{};
    worker_params.field_mask = UCP_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status_t status = ucp_worker_create(mContext, &worker_params, &client_worker);
    if (status != UCS_OK) {
        throw std::runtime_error(
            std::string("UcxCommunicator::Accept: ucp_worker_create failed: ") +
            ucs_status_string(status));
    }

    // Create endpoint from the connection request
    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = mConnReq.conn_req;

    ucp_ep_h client_ep;
    status = ucp_ep_create(client_worker, &ep_params, &client_ep);
    if (status != UCS_OK) {
        ucp_worker_destroy(client_worker);
        throw std::runtime_error(std::string("UcxCommunicator::Accept: ucp_ep_create failed: ") +
                                 ucs_status_string(status));
    }

    LOG4CPLUS_INFO(logger, "Client connected (UCX endpoint created)");

    return new UcxCommunicator(mContext, client_worker, client_ep);
}

// ----------------------------------------------------------------------------
// Client side
// ----------------------------------------------------------------------------

void UcxCommunicator::Connect() {
    LOG4CPLUS_DEBUG(logger, "Connect() to " << mHostname << ":" << mPort);

    initContext();
    initWorker();

    // Resolve the server address
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

    // Create endpoint (connect to server)
    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = reinterpret_cast<struct sockaddr *>(&server_addr);
    ep_params.sockaddr.addrlen = sizeof(server_addr);

    ucs_status_t status = ucp_ep_create(mWorker, &ep_params, &mEndpoint);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("UcxCommunicator: ucp_ep_create (connect) failed: ") +
                                 ucs_status_string(status));
    }

    LOG4CPLUS_INFO(logger, "Connected to " << mHostname << ":" << mPort << " (UCX)");
}

// ----------------------------------------------------------------------------
// Data transfer
// ----------------------------------------------------------------------------

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Read() size=" << size);

    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype = ucp_dt_make_contig(1);
    param.cb.recv = recvCallback;

    void *request =
        ucp_tag_recv_nbx(mWorker, buffer, size, kTag, kTagMask, &param);

    if (UCS_PTR_IS_ERR(request)) {
        LOG4CPLUS_ERROR(logger, "Read failed: " << ucs_status_string(UCS_PTR_STATUS(request)));
        return 0;
    }

    if (request != nullptr) {
        auto *req = static_cast<UcxRequest *>(request);
        while (!req->complete.load()) {
            ucp_worker_progress(mWorker);
        }
        ucp_request_free(request);
    }

    return size;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    LOG4CPLUS_TRACE(logger, "Write() size=" << size);

    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype = ucp_dt_make_contig(1);
    param.cb.send = sendCallback;

    void *request =
        ucp_tag_send_nbx(mEndpoint, buffer, size, kTag, &param);

    if (UCS_PTR_IS_ERR(request)) {
        LOG4CPLUS_ERROR(logger, "Write failed: " << ucs_status_string(UCS_PTR_STATUS(request)));
        return 0;
    }

    if (request != nullptr) {
        auto *req = static_cast<UcxRequest *>(request);
        while (!req->complete.load()) {
            ucp_worker_progress(mWorker);
        }
        ucp_request_free(request);
    }

    return size;
}

void UcxCommunicator::Sync() {
    // UCX tag operations are inherently flushed on completion.
    // Explicit flush ensures all pending sends are acked by the remote side.
    if (mEndpoint) {
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        param.cb.send = sendCallback;

        void *request = ucp_ep_flush_nbx(mEndpoint, &param);
        waitForCompletion(request);
    }
}

void UcxCommunicator::Close() {
    if (mEndpoint) {
        // Graceful close: flush + close
        ucp_request_param_t param{};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_EP_CLOSE_FLAG_FORCE;

        void *request = ucp_ep_close_nbx(mEndpoint, &param);
        if (!UCS_PTR_IS_ERR(request) && request != nullptr) {
            // Wait for close to complete
            auto *req = static_cast<UcxRequest *>(request);
            while (!req->complete.load()) {
                ucp_worker_progress(mWorker);
            }
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

// ----------------------------------------------------------------------------
// Factory function (loaded by CommunicatorFactory via dlopen)
// ----------------------------------------------------------------------------

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_end = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_end) {
        throw std::runtime_error("UcxCommunicator: expected Endpoint_Ucx");
    }
    return std::make_shared<UcxCommunicator>(ucx_end->address(), ucx_end->port());
}
