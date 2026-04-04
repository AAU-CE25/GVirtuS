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
using gvirtus::communicators::CudaRemoteError;

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
        params.features = UCP_FEATURE_STREAM | UCP_FEATURE_WAKEUP;

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
        try {
            progress_running_.store(false, std::memory_order_release);
            if (progress_thread_.joinable()) {
                progress_thread_.join();
            }

            if (ep_client_) {
                ucs_status_ptr_t req = ucp_ep_close_nb(ep_client_, UCP_EP_CLOSE_MODE_FORCE);
                if (!UCS_PTR_IS_ERR(req)) {
                    try {
                        wait_req_with_progress(req, worker_client_, worker_server_);
                    } catch (const std::exception&) {
                        // Fault-injection tests can leave endpoint state inconsistent; teardown must be best-effort.
                    }
                }
                ep_client_ = nullptr;
            }

            if (ep_server_) {
                ucs_status_ptr_t req = ucp_ep_close_nb(ep_server_, UCP_EP_CLOSE_MODE_FORCE);
                if (!UCS_PTR_IS_ERR(req)) {
                    try {
                        wait_req_with_progress(req, worker_client_, worker_server_);
                    } catch (const std::exception&) {
                        // Fault-injection tests can leave endpoint state inconsistent; teardown must be best-effort.
                    }
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
        } catch (...) {
            // Never fail test from teardown of intentionally corrupted transport state.
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

uint32_t test_header_crc32(const FrameHeader& hdr) {
    const auto* data = reinterpret_cast<const uint8_t*>(&hdr);
    constexpr std::size_t crc_input_len = offsetof(FrameHeader, header_crc);

    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < crc_input_len; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-(static_cast<int32_t>(crc & 1u)));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

void send_truncated_payload(ucp_ep_h ep, ucp_worker_h worker_a, ucp_worker_h worker_b,
                            uint32_t request_id, uint32_t declared_len, uint32_t actual_len) {
    FrameHeader hdr{};
    hdr.magic = ::GV_MAGIC;
    hdr.msg_type = static_cast<uint8_t>(MsgType::RESPONSE);
    hdr.request_id = request_id;
    hdr.payload_len = declared_len;
    hdr.header_crc = test_header_crc32(hdr);

    std::vector<uint8_t> frame(sizeof(FrameHeader) + actual_len, 0xAB);
    std::memcpy(frame.data(), &hdr, sizeof(FrameHeader));

    ucs_status_ptr_t req =
        ucp_stream_send_nb(ep, frame.data(), frame.size(), ucp_dt_make_contig(1), test_send_cb, 0);
    wait_req_with_progress(req, worker_a, worker_b);
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

    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    sender.Send(ep_client_, MsgType::REQUEST, 42, msg, msg_len);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));

    EXPECT_EQ(hdr.magic, ::GV_MAGIC);
    EXPECT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::REQUEST));
    EXPECT_EQ(hdr.request_id, 42u);
    EXPECT_EQ(hdr.payload_len, msg_len);
    ASSERT_EQ(payload.size(), msg_len);
    EXPECT_EQ(std::memcmp(payload.data(), msg, msg_len), 0);
}

TEST_F(UcxLoopbackFixture, JunkPrefixResyncsAndRecovers) {
    const uint8_t junk[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    ucs_status_ptr_t req =
        ucp_stream_send_nb(ep_client_, junk, sizeof(junk), ucp_dt_make_contig(1), test_send_cb, 0);
    wait_req_with_progress(req, worker_client_, worker_server_);

    const char* first_msg = "sacrificial";
    const char* second_msg = "recovered";
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    // First frame may be partially consumed by bad alignment + resync scan.
    sender.Send(ep_client_, MsgType::REQUEST, 7, first_msg,
                static_cast<uint32_t>(std::strlen(first_msg)));
    // Second frame is expected to be recovered and delivered.
    sender.Send(ep_client_, MsgType::REQUEST, 8, second_msg,
                static_cast<uint32_t>(std::strlen(second_msg)));

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));

    EXPECT_EQ(hdr.magic, ::GV_MAGIC);
    EXPECT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::REQUEST));
    EXPECT_EQ(hdr.request_id, 8u);
    ASSERT_EQ(payload.size(), std::strlen(second_msg));
    EXPECT_EQ(std::memcmp(payload.data(), second_msg, std::strlen(second_msg)), 0);
}

TEST_F(UcxLoopbackFixture, ErrorFrameRoundTripCarriesCudaCodeAndRequestSeq) {
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    constexpr uint16_t request_seq = 77;
    constexpr uint32_t cuda_error = 999;

    sender.SendError(ep_client_, request_seq, cuda_error);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));

    ASSERT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::ERROR));
    ASSERT_EQ(payload.size(), sizeof(ErrorPayload));

    ErrorPayload err{};
    std::memcpy(&err, payload.data(), sizeof(ErrorPayload));
    EXPECT_EQ(err.cuda_error, cuda_error);
    EXPECT_EQ(err.request_seq, request_seq);
}

TEST_F(UcxLoopbackFixture, ResponseFrameEchoesRequestSeqAndResultLen) {
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    constexpr uint32_t request_id = 33;
    constexpr uint32_t frame_request_id = 90;
    const char* result = "result-bytes";
    const uint32_t result_len = static_cast<uint32_t>(std::strlen(result));

    sender.SendResponse(ep_client_, frame_request_id, request_id, result, result_len);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));

    ASSERT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::RESPONSE));
    const ResponseHeader resp = FramedStream::ParseAndValidateResponseHeader(payload, request_id);
    EXPECT_EQ(resp.request_id, request_id);
    EXPECT_EQ(resp.result_len, result_len);

    const uint8_t* result_ptr = payload.data() + sizeof(ResponseHeader);
    EXPECT_EQ(std::memcmp(result_ptr, result, result_len), 0);
}

TEST_F(UcxLoopbackFixture, ResponseSeqMismatchThrows) {
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    constexpr uint32_t request_id = 12;
    constexpr uint32_t wrong_expected_request_id = 99;

    sender.SendResponse(ep_client_, 1, request_id, nullptr, 0);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));

    EXPECT_THROW({
        (void)FramedStream::ParseAndValidateResponseHeader(payload, wrong_expected_request_id);
    }, std::runtime_error);
}

TEST_F(UcxLoopbackFixture, FaultInjection_ExecutionErrorPropagatesAndStreamRemainsHealthy) {
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    constexpr uint16_t failed_request_seq = 41;
    constexpr uint32_t cuda_error = 2;  // representative non-success cudaError_t value

    sender.SendError(ep_client_, failed_request_seq, cuda_error);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));
    bool caught = false;
    try {
        FramedStream::ThrowIfErrorFrame(hdr, payload);
    } catch (const CudaRemoteError& e) {
        caught = true;
        EXPECT_EQ(e.cuda_error(), cuda_error);
        EXPECT_EQ(e.request_seq(), failed_request_seq);
    }
    ASSERT_TRUE(caught);

    // Stream should still be healthy after an ERROR frame.
    constexpr uint32_t ok_request_id = 42;
    constexpr uint32_t ok_frame_request_id = 100;
    const char* result = "post-error-ok";
    const uint32_t result_len = static_cast<uint32_t>(std::strlen(result));

    sender.SendResponse(ep_client_, ok_frame_request_id, ok_request_id, result, result_len);

    ASSERT_TRUE(receiver.Recv(ep_server_, hdr, payload));
    ASSERT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::RESPONSE));
    const ResponseHeader resp = FramedStream::ParseAndValidateResponseHeader(payload, ok_request_id);
    EXPECT_EQ(resp.result_len, result_len);
}

TEST_F(UcxLoopbackFixture, FaultInjection_CorruptedByteThenResyncAndRecovery) {
    FramedStream sender(worker_client_);
    FramedStream receiver(worker_server_);

    // Inject one corrupt byte to break alignment.
    const uint8_t junk = 0xDE;
    ucs_status_ptr_t req =
        ucp_stream_send_nb(ep_client_, &junk, sizeof(junk), ucp_dt_make_contig(1), test_send_cb, 0);
    wait_req_with_progress(req, worker_client_, worker_server_);

    // First frame may be sacrificed by resync procedure.
    const char* first = "sacrificial";
    sender.Send(ep_client_, MsgType::REQUEST, 501, first, static_cast<uint32_t>(std::strlen(first)));

    // Second frame should be the one we recover to.
    const char* second = "recovered";
    sender.Send(ep_client_, MsgType::REQUEST, 502, second, static_cast<uint32_t>(std::strlen(second)));

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    bool recovered = false;
    try {
        recovered = receiver.Recv(ep_server_, hdr, payload, std::chrono::milliseconds(1000));
    } catch (...) {
        recovered = false;
    }

    ASSERT_TRUE(recovered);
    EXPECT_EQ(hdr.msg_type, static_cast<uint8_t>(MsgType::REQUEST));
    EXPECT_EQ(hdr.request_id, 502u);
    ASSERT_EQ(payload.size(), std::strlen(second));
    EXPECT_EQ(std::memcmp(payload.data(), second, std::strlen(second)), 0);
}

TEST_F(UcxLoopbackFixture, FaultInjection_TruncatedPayloadTimesOutNotHang) {
    FramedStream receiver(worker_server_);

    send_truncated_payload(ep_client_, worker_client_, worker_server_, 700, 1000, 10);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;

    const auto t0 = std::chrono::steady_clock::now();
    bool threw = false;
    try {
        (void)receiver.Recv(ep_server_, hdr, payload, std::chrono::milliseconds(300));
    } catch (...) {
        threw = true;
    }

    ASSERT_TRUE(threw);
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(3500));
}

}  // namespace
