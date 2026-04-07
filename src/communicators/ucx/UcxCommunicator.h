#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include "gvirtus/communicators/Buffer.h"
#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/FramedStream.h"
#include <mutex>
#include <queue>
#include <condition_variable>

namespace gvirtus::communicators {

class UcxCommunicator : public Communicator {
   public:
    UcxCommunicator() = default;
    UcxCommunicator(const std::string &hostname, std::uint16_t port);
    ~UcxCommunicator();

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;
    void run() override;

    std::string to_string() override { return "ucxcommunicator"; }

    // Async send — serializes routine name + input_buffer, returns
    // a PendingRequest handle the caller can Wait() on later
    std::shared_ptr<FramedStream::PendingRequest> SendAsync(
        const char *routine, const Buffer *input_buffer);

    FramedStream *GetFramedStream() { return framed_stream_.get(); }

   private:
    std::string   hostname_;
    std::uint16_t port_{};

    // UCX handles — initialized in Connect()
    ucp_context_h ucp_context_{nullptr};
    ucp_worker_h  ucp_worker_{nullptr};
    ucp_ep_h      ucp_ep_{nullptr};

    std::unique_ptr<FramedStream> framed_stream_;

    // Synchronous send/recv buffers — used by Write/Read/Sync
    // to keep the existing TCP-style call path working over UCX
    std::vector<uint8_t> write_buffer_;  // accumulates Write() calls
    std::vector<uint8_t> read_buffer_;   // holds last received payload
    size_t               read_offset_{0};

    mutable std::chrono::steady_clock::time_point last_accept_log_{};

    // Listener state — used by Serve() / Accept()
    ucp_listener_h ucp_listener_{nullptr};
    mutable std::mutex                      accept_mutex_;
    mutable std::condition_variable         accept_cv_;
    mutable std::queue<ucp_ep_h>            pending_eps_;

    // Internal helpers
    void InitUcpContext();
    void CreateWorker();
    void CreateEndpoint();
    static void OnConnectionRequest(ucp_conn_request_h req, void *arg);
};

}  // namespace gvirtus::communicators