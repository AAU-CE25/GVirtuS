#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/FramedStream.h"

namespace gvirtus::communicators {

class UcxCommunicator : public Communicator {
   public:
    UcxCommunicator() = default;
    UcxCommunicator(const std::string &hostname, std::uint16_t port);

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;
    void run() override;

    std::string to_string() override { return "ucxcommunicator"; }

    / Async send, returns a handle to wait on later
    std::shared_ptr<FramedStream::PendingRequest> SendAsync(
        const char *routine, const Buffer *input_buffer);

    // Expose framed stream for direct use if needed
    FramedStream *GetFramedStream() { return framed_stream_.get(); }

   private:
    std::string hostname_;
    std::uint16_t port_{};
    mutable std::chrono::steady_clock::time_point last_accept_log_{};

    ucp_context_h ucp_context_{nullptr};
    ucp_worker_h  ucp_worker_{nullptr};
    ucp_ep_h      ucp_ep_{nullptr};
    std::unique_ptr<FramedStream> framed_stream_;
};

}  // namespace gvirtus::communicators
