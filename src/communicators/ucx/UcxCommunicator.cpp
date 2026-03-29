#include "UcxCommunicator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "gvirtus/communicators/Endpoint_Ucx.h"

using gvirtus::communicators::UcxCommunicator;

UcxCommunicator::UcxCommunicator(const std::string &hostname, std::uint16_t port)
    : hostname_(hostname), port_(port) {}

void UcxCommunicator::Serve() { std::printf("UCX stub called: Serve (%s:%u)\n", hostname_.c_str(), port_); }

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
    const auto now = std::chrono::steady_clock::now();
    if (last_accept_log_.time_since_epoch().count() == 0 ||
        now - last_accept_log_ >= std::chrono::seconds(10)) {
        std::printf("UCX stub called: Accept (no connection, stub mode)\n");
        last_accept_log_ = now;
    }
    return nullptr;
}

void UcxCommunicator::Connect() {
    std::printf("UCX stub called: Connect (%s:%u)\n", hostname_.c_str(), port_);
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
    std::printf("UCX stub called: Read (%zu bytes)\n", size);
    if (buffer != nullptr && size > 0) {
        std::fill(buffer, buffer + size, 0);
    }
    return size;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
    (void)buffer;
    std::printf("UCX stub called: Write (%zu bytes)\n", size);
    return size;
}

void UcxCommunicator::Sync() { std::printf("UCX stub called: Sync\n"); }

void UcxCommunicator::Close() { std::printf("UCX stub called: Close\n"); }

void UcxCommunicator::run() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

extern "C" std::shared_ptr<UcxCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ucx_endpoint = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ucx_endpoint) {
        throw std::runtime_error("UcxCommunicator: endpoint type mismatch");
    }

    return std::make_shared<UcxCommunicator>(ucx_endpoint->address(), ucx_endpoint->port());
}
