#pragma once

#include <chrono>
#include <string>

#include "gvirtus/communicators/Communicator.h"

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

   private:
    std::string hostname_;
    std::uint16_t port_{};
    mutable std::chrono::steady_clock::time_point last_accept_log_{};
};

}  // namespace gvirtus::communicators
