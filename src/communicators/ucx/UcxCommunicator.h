#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gvirtus/communicators/Communicator.h"

#include <ucp/api/ucp.h>

namespace gvirtus::communicators {

// Allocator that default-initializes elements instead of value-initializing
// them, so std::vector<unsigned char, default_init_allocator<unsigned char>>(N)
// does NOT zero-fill its storage. Critical for the AM receive path, where we
// allocate buffers the size of a CUDA D2H/H2D payload (tens of MB) on every
// call and immediately overwrite them.
template <typename T, typename A = std::allocator<T>>
class default_init_allocator : public A {
    using a_t = std::allocator_traits<A>;

   public:
    template <typename U>
    struct rebind {
        using other =
            default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
    };

    using A::A;

    template <typename U>
    void construct(U *ptr) noexcept(
        std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void *>(ptr)) U;  // default-init: no zero-fill
    }
    template <typename U, typename... Args>
    void construct(U *ptr, Args &&...args) {
        a_t::construct(static_cast<A &>(*this), ptr,
                       std::forward<Args>(args)...);
    }
};

using ByteBuffer =
    std::vector<unsigned char, default_init_allocator<unsigned char>>;

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
    void enqueue_am_message(ByteBuffer message);
    void enqueue_am_rndv(void *request, ByteBuffer buffer,
                         bool from_scratch, size_t scratch_len);
    void progress_am_rndv();
    void allocate_recv_scratch();
    void allocate_send_scratch();
    void release_recv_scratch();
    void release_send_scratch();
    static size_t scratch_size_bytes();
    static sockaddr_storage make_sockaddr(const std::string &host, std::uint16_t port);

    struct PendingAmRecv {
        void *request{nullptr};
        ByteBuffer buffer;
        // When `from_scratch` is true, the receive landed in the shared
        // pre-registered scratch buffer instead of `buffer`. The completion
        // path must memcpy `scratch_len` bytes from the scratch buffer into
        // `buffer` before enqueueing it, and clear the busy flag so the
        // next rendezvous receive may reuse the scratch.
        bool from_scratch{false};
        size_t scratch_len{0};
    };

    struct AmState {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<ByteBuffer> queue;
        std::vector<PendingAmRecv> rndv;
        // Pre-registered receive scratch shared across all clients on the
        // same UCX worker. Protected by `mutex`.
        ByteBuffer recv_scratch;
        ucp_mem_h recv_memh{nullptr};
        bool recv_scratch_in_use{false};
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
    std::shared_ptr<AmState> am_state_{std::make_shared<AmState>()};

    ByteBuffer pending_read_bytes_;
    size_t pending_read_offset_{0};
    std::atomic<bool> endpoint_failed_{false};

    // Per-endpoint send scratch. Used by Write() so that the user-visible
    // sequence of large active-message sends always reuses the same VA, which
    // lets the UCX registration cache hit instead of re-pinning fresh pages
    // for every RDMA RNDV transfer.
    ByteBuffer send_scratch_;
    ucp_mem_h send_scratch_memh_{nullptr};
};

}  // namespace gvirtus::communicators
