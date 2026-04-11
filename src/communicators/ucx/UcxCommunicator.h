#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/UcxAmProtocol.h"

#include <ucp/api/ucp.h>

namespace gvirtus::communicators {

class UcxCommunicator : public Communicator {
   public:
    enum class UcxDataPath : std::uint8_t { TagFramed = 0, ActiveMessage = 1 };
    using UcxAmMessageType = ucxam::MessageType;
    using UcxAmEnvelopeHeader = ucxam::EnvelopeHeader;

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
    static void listener_conn_handler(ucp_conn_request_h conn_request, void *arg);
    static void endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);

    void init_ucx();
    void destroy_ucx();
    void enqueue_connection(ucp_conn_request_h conn_request);
    ucp_conn_request_h wait_for_connection_request();
    void wait_request_completion(void *request, const char *op_name);
    void recv_message_exact(void *buffer, size_t size, const char *op_name);
    void send_message_exact(const void *buffer, size_t size, const char *op_name);
    static sockaddr_storage make_sockaddr(const std::string &host, std::uint16_t port);
    void configure_data_path_from_env();
    const char *data_path_name() const;
    UcxAmEnvelopeHeader make_am_header(UcxAmMessageType type, std::uint64_t request_id,
                                       std::uint64_t routine_size,
                                       std::uint64_t payload_size,
                                       std::uint32_t status_code) const;
    std::vector<unsigned char> encode_am_envelope(const UcxAmEnvelopeHeader &header,
                                                  const char *routine_data,
                                                  std::uint64_t routine_size,
                                                  const char *payload_data,
                                                  std::uint64_t payload_size) const;
    bool decode_am_envelope(const unsigned char *data, std::size_t size,
                            UcxAmEnvelopeHeader &header, std::string &routine,
                            std::vector<unsigned char> &payload,
                            std::string &error) const;

    std::string hostname_;
    std::uint16_t port_{};
    ucp_context_h context_{nullptr};
    ucp_worker_h worker_{nullptr};
    ucp_listener_h listener_{nullptr};
    ucp_ep_h endpoint_{nullptr};

    std::atomic<bool> running_{false};
    bool owns_context_worker_listener_{true};
    bool initialized_{false};

    mutable std::mutex conn_mutex_;
    mutable std::condition_variable conn_cv_;
    mutable std::queue<ucp_conn_request_h> pending_conn_requests_;
    std::shared_ptr<std::mutex> worker_mutex_{std::make_shared<std::mutex>()};

    std::vector<unsigned char> pending_read_bytes_;
    size_t pending_read_offset_{0};
    std::atomic<bool> endpoint_failed_{false};

    UcxDataPath data_path_{UcxDataPath::TagFramed};
    std::atomic<std::uint64_t> next_request_id_{1};
};

}  // namespace gvirtus::communicators
