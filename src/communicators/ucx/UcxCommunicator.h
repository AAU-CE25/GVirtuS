#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "gvirtus/communicators/Communicator.h"

#include <ucp/api/ucp.h>

namespace gvirtus::communicators {

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
    static void listener_conn_handler(ucp_conn_request_h conn_request, void *arg);
    static void endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);
    static ucs_status_t am_recv_handler(void *arg, const void *header, size_t header_length,
                                        void *data, size_t length,
                                        const ucp_am_recv_param_t *param);

    void init_ucx();
    void destroy_ucx();
    void enqueue_connection(ucp_conn_request_h conn_request);
    ucp_conn_request_h wait_for_connection_request();
    void wait_request_completion(void *request, const char *op_name);
    void enqueue_am_message(std::vector<unsigned char> message);
    void enqueue_am_rndv(void *request, std::vector<unsigned char> buffer);
    void progress_am_rndv();
    static sockaddr_storage make_sockaddr(const std::string &host, std::uint16_t port);

    struct PendingAmRecv {
        void *request{nullptr};
        std::vector<unsigned char> buffer;
    };

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

    unsigned am_id_{1};
    std::mutex am_mutex_;
    std::condition_variable am_cv_;
    std::deque<std::vector<unsigned char>> am_queue_;
    std::vector<PendingAmRecv> am_rndv_;

    std::vector<unsigned char> pending_read_bytes_;
    size_t pending_read_offset_{0};
    std::atomic<bool> endpoint_failed_{false};
};

}  // namespace gvirtus::communicators
