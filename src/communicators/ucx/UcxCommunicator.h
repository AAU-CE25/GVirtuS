#pragma once

#include <ucp/api/ucp.h>

#include <atomic>
#include <string>

#include "gvirtus/communicators/Communicator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

namespace gvirtus::communicators {

/**
 * UcxCommunicator implements the Communicator interface using the UCX UCP layer.
 *
 * UCX automatically selects the optimal transport:
 *  - Eager protocol (TCP/shared-memory) for small messages
 *  - Rendezvous protocol (RDMA zero-copy) for large messages
 *
 * The threshold is controlled by the UCX_RNDV_THRESH environment variable
 * (default ~8KB). Messages above this size will use RDMA if available.
 */
class UcxCommunicator : public Communicator {
   public:
    UcxCommunicator() = default;

    /// Client constructor: will connect to hostname:port
    UcxCommunicator(const std::string &hostname, uint16_t port);

    /// Server-side accepted connection constructor
    UcxCommunicator(ucp_context_h ctx, ucp_worker_h worker, ucp_ep_h ep);

    ~UcxCommunicator() override;

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;

    std::string to_string() override { return "ucxcommunicator"; }

   private:
    log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator"));

    std::string mHostname;
    uint16_t mPort = 0;

    // UCX handles
    ucp_context_h mContext = nullptr;
    ucp_worker_h mWorker = nullptr;
    ucp_listener_h mListener = nullptr;
    ucp_ep_h mEndpoint = nullptr;

    // Whether this instance owns the context (server/client main instances do,
    // accepted connections do not share context but own their own worker+ep)
    bool mOwnsContext = false;

    // Tag for tag-matching sends/receives.
    // All messages on a connection use a single tag since it's point-to-point.
    static constexpr ucp_tag_t kTag = 0x1337;
    static constexpr ucp_tag_t kTagMask = 0xFFFFFFFFFFFFFFFF;

    // Connection request storage for Accept()
    struct ConnRequest {
        ucp_conn_request_h conn_req;
        std::atomic<bool> ready;
    };
    mutable ConnRequest mConnReq = {nullptr, false};

    // Internal helpers
    void initContext();
    void initWorker();
    void waitForCompletion(void *request);

    // UCP listener callback
    static void listenerCallback(ucp_conn_request_h conn_request, void *arg);
};

}  // namespace gvirtus::communicators
