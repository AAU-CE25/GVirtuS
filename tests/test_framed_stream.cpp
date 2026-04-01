#include <gtest/gtest.h>

#ifdef _WIN32
#include <WinSock2.h>
#else
#include <arpa/inet.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gvirtus/communicators/FramedStream.h"

namespace {

using gvirtus::communicators::FramedStream;

struct UcxConnState {
    std::atomic<bool> ready{false};
    ucp_conn_request_h conn_request{nullptr};
};

void on_conn_request(ucp_conn_request_h conn_request, void* arg) {
    auto* state = static_cast<UcxConnState*>(arg);
    state->conn_request = conn_request;
    state->ready.store(true, std::memory_order_release);
}

void check_status(ucs_status_t status, const char* context) {
    if (status != UCS_OK) {
        throw std::runtime_error(std::string(context) + ": " + ucs_status_string(status));
    }
}

void wait_req_with_progress(ucs_status_ptr_t req, ucp_worker_h worker_a, ucp_worker_h worker_b,
                            std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
    if (req == nullptr) {
        return;
    }
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error("UCX request returned error ptr");
    }
    if (!UCS_PTR_IS_PTR(req)) {
        const ucs_status_t immediate = UCS_PTR_STATUS(req);
        if (immediate != UCS_OK) {
            throw std::runtime_error("UCX immediate status is failure");
        }
        return;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        ucp_worker_progress(worker_a);
        ucp_worker_progress(worker_b);

        const ucs_status_t status = ucp_request_check_status(req);
        if (status == UCS_OK) {
            ucp_request_free(req);
            return;
        }
        if (status != UCS_INPROGRESS) {
            ucp_request_free(req);
            throw std::runtime_error("UCX request completion failed");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ucp_request_free(req);
    throw std::runtime_error("Timed out waiting for UCX request completion");
}

class UcxLoopbackFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        ucp_config_t* config = nullptr;
        check_status(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");

        ucp_params_t params;
        std::memset(&params, 0, sizeof(params));
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features = UCP_FEATURE_STREAM;

        const ucs_status_t init_status = ucp_init(&params, config, &context_);
        ucp_config_release(config);
        check_status(init_status, "ucp_init");

        ucp_worker_params_t worker_params;
        std::memset(&worker_params, 0, sizeof(worker_params));
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

        check_status(ucp_worker_create(context_, &worker_params, &worker_server_),
                     "ucp_worker_create(server)");
        check_status(ucp_worker_create(context_, &worker_params, &worker_client_),
                     "ucp_worker_create(client)");

        sockaddr_in listen_addr;
        std::memset(&listen_addr, 0, sizeof(listen_addr));
        listen_addr.sin_family = AF_INET;
        listen_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        listen_addr.sin_port = 0;

        ucp_listener_params_t listener_params;
        std::memset(&listener_params, 0, sizeof(listener_params));
        listener_params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                     UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        listener_params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&listen_addr);
        listener_params.sockaddr.addrlen = sizeof(listen_addr);
        listener_params.conn_handler.cb = on_conn_request;
        listener_params.conn_handler.arg = &conn_state_;

        check_status(ucp_listener_create(worker_server_, &listener_params, &listener_),
                     "ucp_listener_create");

        ucp_listener_attr_t listener_attr;
        std::memset(&listener_attr, 0, sizeof(listener_attr));
        listener_attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
        check_status(ucp_listener_query(listener_, &listener_attr), "ucp_listener_query");

        ucp_ep_params_t client_ep_params;
        std::memset(&client_ep_params, 0, sizeof(client_ep_params));
        client_ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
        client_ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        client_ep_params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&listener_attr.sockaddr);
        client_ep_params.sockaddr.addrlen = sizeof(listener_attr.sockaddr);

        check_status(ucp_ep_create(worker_client_, &client_ep_params, &ep_client_),
                     "ucp_ep_create(client)");

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!conn_state_.ready.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            ucp_worker_progress(worker_server_);
            ucp_worker_progress(worker_client_);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!conn_state_.ready.load(std::memory_order_acquire)) {
            throw std::runtime_error("Timed out waiting for server connection request");
        }

        ucp_ep_params_t server_ep_params;
        std::memset(&server_ep_params, 0, sizeof(server_ep_params));
        server_ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
        server_ep_params.conn_request = conn_state_.conn_request;

        check_status(ucp_ep_create(worker_server_, &server_ep_params, &ep_server_),
                     "ucp_ep_create(server)");

        progress_running_.store(true, std::memory_order_release);
        progress_thread_ = std::thread([this]() {
            while (progress_running_.load(std::memory_order_acquire)) {
                if (worker_server_) {
                    ucp_worker_progress(worker_server_);
                }
                if (worker_client_) {
                    ucp_worker_progress(worker_client_);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    void TearDown() override {
        progress_running_.store(false, std::memory_order_release);
        if (progress_thread_.joinable()) {
            progress_thread_.join();
        }

        if (ep_client_) {
            ucs_status_ptr_t req = ucp_ep_close_nb(ep_client_, UCP_EP_CLOSE_MODE_FORCE);
            if (!UCS_PTR_IS_ERR(req)) {
                wait_req_with_progress(req, worker_client_, worker_server_);
            }
            ep_client_ = nullptr;
        }

        if (ep_server_) {
            ucs_status_ptr_t req = ucp_ep_close_nb(ep_server_, UCP_EP_CLOSE_MODE_FORCE);
            if (!UCS_PTR_IS_ERR(req)) {
                wait_req_with_progress(req, worker_client_, worker_server_);
            }
            ep_server_ = nullptr;
        }

        if (listener_) {
            ucp_listener_destroy(listener_);
            listener_ = nullptr;
        }
        if (worker_client_) {
            ucp_worker_destroy(worker_client_);
            worker_client_ = nullptr;
        }
        if (worker_server_) {
            ucp_worker_destroy(worker_server_);
            worker_server_ = nullptr;
        }
        if (context_) {
            ucp_cleanup(context_);
            context_ = nullptr;
        }
    }

    ucp_context_h context_{nullptr};
    ucp_worker_h worker_server_{nullptr};
    ucp_worker_h worker_client_{nullptr};
    ucp_listener_h listener_{nullptr};
    ucp_ep_h ep_server_{nullptr};
    ucp_ep_h ep_client_{nullptr};
    UcxConnState conn_state_{};
    std::atomic<bool> progress_running_{false};
    std::thread progress_thread_{};
};

void test_send_cb(void* request, ucs_status_t status) {
    (void)request;
    (void)status;
}

void test_recv_cb(void* request, ucs_status_t status, size_t length) {
    (void)request;
    (void)status;
    (void)length;
}

TEST_F(UcxLoopbackFixture, ContiguousStreamRoundTripWorks) {
    const char* msg = "contig smoke";
    const size_t msg_len = std::strlen(msg);

    std::vector<uint8_t> out(msg, msg + msg_len);
    std::vector<uint8_t> in(msg_len, 0);

    ucs_status_ptr_t send_req =
        ucp_stream_send_nb(ep_client_, out.data(), out.size(), ucp_dt_make_contig(1), test_send_cb, 0);
    wait_req_with_progress(send_req, worker_client_, worker_server_);

    size_t got = 0;
    ucs_status_ptr_t recv_req = ucp_stream_recv_nb(ep_server_, in.data(), in.size(),
                                                   ucp_dt_make_contig(1), test_recv_cb, &got, 0);
    wait_req_with_progress(recv_req, worker_client_, worker_server_);

    EXPECT_EQ(std::memcmp(in.data(), out.data(), out.size()), 0);
}

TEST_F(UcxLoopbackFixture, SendAndRecvFrameRoundTrip) {
    const char* msg = "hello framing";
    const uint32_t msg_len = static_cast<uint32_t>(std::strlen(msg));

    FramedStream::Send(ep_client_, MsgType::REQUEST, 42, msg, msg_len);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(FramedStream::Recv(ep_server_, hdr, payload));

    EXPECT_EQ(hdr.magic, ::GV_MAGIC);
    EXPECT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::REQUEST));
    EXPECT_EQ(hdr.seq, 42);
    EXPECT_EQ(hdr.payload_len, msg_len);
    ASSERT_EQ(payload.size(), msg_len);
    EXPECT_EQ(std::memcmp(payload.data(), msg, msg_len), 0);
}

TEST_F(UcxLoopbackFixture, JunkPrefixThrowsInsteadOfHanging) {
    const uint8_t junk[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    ucs_status_ptr_t req =
        ucp_stream_send_nb(ep_client_, junk, sizeof(junk), ucp_dt_make_contig(1), test_send_cb, 0);
    wait_req_with_progress(req, worker_client_, worker_server_);

    const char* msg = "ok";
    FramedStream::Send(ep_client_, MsgType::REQUEST, 7, msg, static_cast<uint32_t>(std::strlen(msg)));

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    EXPECT_THROW({ (void)FramedStream::Recv(ep_server_, hdr, payload); }, std::runtime_error);
}

}  // namespace
