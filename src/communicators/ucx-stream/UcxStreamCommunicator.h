#pragma once

#include <ucp/api/ucp.h>

#include <atomic>
#include <cstddef>
#include <string>

#include "gvirtus/communicators/Communicator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

namespace gvirtus::communicators {

/**
 * UcxStreamCommunicator implements the Communicator interface using the UCX UCP
 * stream API (byte-oriented, in-order, socket-like).
 *
 * Design choices validated by examples/ucx_benchmark/data_copy_bench:
 *  - UCP_FEATURE_STREAM only: matches the Read(n)/Write(n) contract of
 *    GVirtuS Buffer; partial reads loop using UCP_STREAM_RECV_FLAG_WAITALL.
 *  - RLIMIT_MEMLOCK raised at startup: without it RDMA falls back to bounce
 *    buffers above the per-process locked-memory cap (often 64 KiB),
 *    collapsing throughput at large message sizes.
 *  - UCP_ERR_HANDLING_MODE_PEER on every endpoint: prevents silent infinite
 *    spins in ucp_worker_progress() if the peer dies mid-transfer.
 *  - 1-byte wireup handshake right after EP creation: forces UCX wireup
 *    messages to flow before the first real transfer, eliminating
 *    first-transfer hangs on large initial payloads.
 *  - Pre-registered staging buffer for large transfers: avoids per-call
 *    ibv_reg_mr overhead that would otherwise dominate large RDMA sends
 *    from dynamically-allocated GVirtuS Buffers (see data_copy_bench for
 *    the pinned-buffer pattern).
 */
class UcxStreamCommunicator : public Communicator {
   public:
    /// Minimum transfer size (bytes) that routes through the pre-registered
    /// staging buffer. Below this, UCX eager sends work without registration.
    static constexpr size_t kStagingThreshold = 65536;  // 64 KiB

    UcxStreamCommunicator() = default;

    /// Client constructor: will connect to hostname:port
    UcxStreamCommunicator(const std::string &hostname, uint16_t port);

    /// Server-side accepted-connection constructor. Shares the listener's
    /// context but owns its worker + endpoint.
    UcxStreamCommunicator(ucp_context_h ctx, ucp_worker_h worker, ucp_ep_h ep);

    ~UcxStreamCommunicator() override;

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;

    std::string to_string() override { return "ucxstreamcommunicator"; }

   private:
    log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("UcxStreamCommunicator"));

    std::string mHostname;
    uint16_t mPort = 0;

    ucp_context_h mContext = nullptr;
    ucp_worker_h mWorker = nullptr;
    ucp_listener_h mListener = nullptr;
    ucp_ep_h mEndpoint = nullptr;

    // True only for the original Serve()/Connect() instance that called
    // ucp_init(). Accepted-connection instances share the listener context
    // and must NOT cleanup it on destruction.
    bool mOwnsContext = false;

    // Set by epErrCallback when the peer drops the connection. Once true,
    // Read() returns 0 (EOF) immediately, matching TCP communicator semantics
    // so callers (e.g. backend Process::getstring) can exit their loops
    // cleanly instead of catching an exception in a detached thread.
    std::atomic<bool> mPeerClosed{false};

    // Connection-request slot populated by the listener callback.
    struct ConnRequest {
        ucp_conn_request_h conn_req;
        std::atomic<bool> ready;
    };
    mutable ConnRequest mConnReq = {nullptr, false};

    void initContext();
    void initWorker();

    static void listenerCallback(ucp_conn_request_h conn_request, void *arg);
    static void epErrCallback(void *arg, ucp_ep_h ep, ucs_status_t status);

    // Drive the wireup handshake on a freshly created endpoint so the first
    // real Read/Write does not block on UCX wireup.
    void wireupServer(ucp_ep_h ep);
    void wireupClient(ucp_ep_h ep);

    // Pre-registered staging buffer for large transfers. Allocated lazily on
    // first use, kept alive for the lifetime of the communicator to ensure
    // every subsequent send/recv hits the UCX memory registration cache.
    char *mStagingBuf = nullptr;
    size_t mStagingSize = 0;
    ucp_mem_h mStagingMemh = nullptr;

    /// Ensure mStagingBuf is at least `needed` bytes and registered.
    void ensureStaging(size_t needed);
    /// Free and deregister the staging buffer.
    void freeStaging();
};

}  // namespace gvirtus::communicators
