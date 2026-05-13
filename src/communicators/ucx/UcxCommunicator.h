#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include "gvirtus/communicators/Communicator.h"

#include <ucp/api/ucp.h>

namespace gvirtus::communicators {

/**
 * UcxCommunicator implements the gVirtuS Communicator byte-stream contract on
 * top of the UCP Stream API.  Because Stream maps 1:1 to Read/Write, the
 * implementation is small: there is no AM dispatcher, no producer/consumer
 * queue, no rendezvous bookkeeping.  RDMA is enabled transparently when UCX
 * selects an RC/DC transport (e.g. rc_mlx5).
 */
class UcxCommunicator : public Communicator {
   public:
    UcxCommunicator() = default;
    UcxCommunicator(const std::string &hostname, std::uint16_t port);
    ~UcxCommunicator() override;

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;
    void run() override;

    std::string to_string() override { return "ucxcommunicator"; }

   private:
    // Per-process UCP context+worker shared by the listener and every
    // accepted client.  Holding it via shared_ptr means it is destroyed only
    // when the listener and all accepted endpoints are gone -- no dangling
    // raw pointers.
    struct UcxShared {
        ucp_context_h context{nullptr};
        ucp_worker_h worker{nullptr};
        // Serializes every direct UCP call (send_nbx, recv_nbx, ep_close,
        // request_cancel, worker_progress).  UCX_THREAD_MODE_SERIALIZED
        // requires us to do this externally.
        std::mutex worker_mutex;

        UcxShared() = default;
        ~UcxShared();
        UcxShared(const UcxShared &) = delete;
        UcxShared &operator=(const UcxShared &) = delete;
    };

    // State filled by the ucp_stream_recv_nbx callback so the waiter can
    // recover the actual number of bytes received (status alone is not
    // enough -- ucp_request_check_status does not report length).
    struct StreamRecvState {
        std::atomic<int> done{0};
        ucs_status_t status{UCS_OK};
        size_t length{0};
    };

    UcxCommunicator(std::shared_ptr<UcxShared> shared, std::string hostname,
                    std::uint16_t port);

    // Listener-side callback: a peer is asking to connect.  We just push the
    // conn_request onto the queue; Accept() consumes it.
    static void listener_conn_handler(ucp_conn_request_h conn_request, void *arg);
    static void endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);

    // Recv callback used by ucp_stream_recv_nbx (immediate-completion path is
    // handled separately by check_inline_completion).
    static void stream_recv_callback(void *request, ucs_status_t status,
                                     size_t length, void *user_data);
    static void stream_send_callback(void *request, ucs_status_t status,
                                     void *user_data);

    void init_shared();
    void wait_send_completion(void *request, std::atomic<int> *flag);
    size_t wait_stream_recv_completion(void *request, StreamRecvState *state,
                                       size_t inline_length);
    ucp_conn_request_h wait_for_connection_request() const;

    std::shared_ptr<UcxShared> shared_;
    std::string hostname_;
    std::uint16_t port_{};

    // Listener-only state (server-side).
    ucp_listener_h listener_{nullptr};
    mutable std::mutex conn_mutex_;
    mutable std::condition_variable conn_cv_;
    mutable std::queue<ucp_conn_request_h> pending_conn_requests_;

    // Endpoint-only state (per accepted/connected stream).
    ucp_ep_h endpoint_{nullptr};
    mutable std::atomic<bool> endpoint_failed_{false};

    // True for the top-level (listener) instance built directly from the
    // factory.  False for instances minted by Accept().  Used to decide
    // whether to destroy the listener in the destructor.
    bool is_listener_{false};
};

}  // namespace gvirtus::communicators
