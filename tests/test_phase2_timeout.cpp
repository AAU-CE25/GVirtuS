#include "gvirtus/communicators/FramedStream.h"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <ucp/api/ucp.h>

using namespace gvirtus::communicators;

// ============================================================================
// Test Fixture: UCX Loopback Setup (reused from test_framed_stream)
// ============================================================================

class Phase2TimeoutTest : public ::testing::Test {
   protected:
    ucp_context_h context_a_ = nullptr, context_b_ = nullptr;
    ucp_worker_h worker_a_ = nullptr, worker_b_ = nullptr;
    ucp_listener_h listener_b_ = nullptr;
    ucp_ep_h ep_ab_ = nullptr, ep_ba_ = nullptr;

    std::thread progress_thread_;
    std::atomic<bool> stop_progress_{false};

    void SetUp() override {
        // Context A (client side)
        ucp_params_t params_a{};
        params_a.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE;
        params_a.features = UCP_FEATURE_STREAM;
        params_a.request_size = 0;
        ASSERT_EQ(ucp_init(&params_a, &context_a_), UCS_OK);

        // Worker A
        ucp_worker_params_t wparams_a{};
        wparams_a.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wparams_a.thread_mode = UCS_THREAD_MODE_SINGLE;
        ASSERT_EQ(ucp_worker_create(context_a_, &wparams_a, &worker_a_), UCS_OK);

        // Context B (server side)
        ucp_params_t params_b{};
        params_b.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE;
        params_b.features = UCP_FEATURE_STREAM;
        params_b.request_size = 0;
        ASSERT_EQ(ucp_init(&params_b, &context_b_), UCS_OK);

        // Worker B
        ucp_worker_params_t wparams_b{};
        wparams_b.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wparams_b.thread_mode = UCS_THREAD_MODE_SINGLE;
        ASSERT_EQ(ucp_worker_create(context_b_, &wparams_b, &worker_b_), UCS_OK);

        // Listener on B
        struct sockaddr_storage listen_addr {};
        auto* addr = reinterpret_cast<struct sockaddr*>(&listen_addr);
        addr->sa_family = AF_INET;
        ucp_listener_params_t lparams{};
        lparams.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
        lparams.sockaddr = addr;
        lparams.accept_handler.arg = this;
        lparams.accept_handler.cb = [](ucp_conn_request_h req, void* arg) {
            auto* test = static_cast<Phase2TimeoutTest*>(arg);
            ucp_ep_params_t eparams{};
            eparams.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
            eparams.conn_request = req;
            ucp_ep_create(test->worker_b_, &eparams, &test->ep_ba_);
        };
        ASSERT_EQ(ucp_listener_create(worker_b_, &lparams, &listener_b_), UCS_OK);

        // Establish connection A->B
        struct sockaddr_in addr_in {};
        addr_in.sin_family = AF_INET;
        addr_in.sin_port = htons(12345);
        inet_pton(AF_INET, "127.0.0.1", &addr_in.sin_addr);

        ucp_ep_params_t ep_params{};
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_FLAGS;
        ep_params.address = reinterpret_cast<const struct sockaddr*>(&addr_in);
        ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
        ASSERT_EQ(ucp_ep_create(worker_a_, &ep_params, &ep_ab_), UCS_OK);

        // Drive progress until connection established
        int max_iter = 1000;
        while ((ep_ba_ == nullptr || ep_ab_ == nullptr) && max_iter-- > 0) {
            ucp_worker_progress(worker_a_);
            ucp_worker_progress(worker_b_);
            std::this_thread::usleep(100);
        }
    }

    void TearDown() override {
        if (ep_ba_ != nullptr) {
            ucp_ep_close_nb(ep_ba_, UCP_EP_CLOSE_MODE_FLUSH);
        }
        if (ep_ab_ != nullptr) {
            ucp_ep_close_nb(ep_ab_, UCP_EP_CLOSE_MODE_FLUSH);
        }
        if (listener_b_ != nullptr) {
            ucp_listener_destroy(listener_b_);
        }
        if (worker_a_ != nullptr) {
            ucp_worker_destroy(worker_a_);
        }
        if (worker_b_ != nullptr) {
            ucp_worker_destroy(worker_b_);
        }
        if (context_a_ != nullptr) {
            ucp_cleanup(context_a_);
        }
        if (context_b_ != nullptr) {
            ucp_cleanup(context_b_);
        }
    }
};

// ============================================================================
// Test 1: Send completes quickly without timeout
// ============================================================================
TEST_F(Phase2TimeoutTest, SendCompletesWithoutTimeout) {
    FramedStream fs_a(worker_a_);
    FramedStream fs_b(worker_b_);

    const char* msg = "hello from A";
    auto start = std::chrono::steady_clock::now();

    // Send should succeed quickly
    EXPECT_NO_THROW(fs_a.Send(ep_ab_, MsgType::REQUEST, 1, msg, strlen(msg)));

    auto elapsed = std::chrono::steady_clock::now() - start;
    // Should complete in << 5s timeout, realistically within 100ms
    EXPECT_LT(elapsed, std::chrono::milliseconds(500))
        << "Send took too long: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms";
}

// ============================================================================
// Test 2: Recv times out when peer doesn't send
// ============================================================================
TEST_F(Phase2TimeoutTest, RecvTimeoutWhenNoPeerData) {
    FramedStream fs_a(worker_a_);
    // fs_b is intentionally not created, so B won't send anything

    FrameHeader hdr{};
    std::vector<uint8_t> payload;

    auto start = std::chrono::steady_clock::now();

    // Recv should timeout after 200ms
    EXPECT_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(200)), std::runtime_error);

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Should wait for approximately 200ms, allow ±50ms margin
    EXPECT_GE(elapsed_ms, 150) << "Timeout fired too early";
    EXPECT_LT(elapsed_ms, 400) << "Timeout took too long";
}

// ============================================================================
// Test 3: Recv completes successfully within timeout
// ============================================================================
TEST_F(Phase2TimeoutTest, RecvCompletesWithinTimeout) {
    FramedStream fs_a(worker_a_);
    FramedStream fs_b(worker_b_);

    const char* msg = "hello from B";

    // B sends a message
    std::thread sender_thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        fs_b.Send(ep_ba_, MsgType::RESPONSE, 1, msg, strlen(msg));
    });

    FrameHeader hdr{};
    std::vector<uint8_t> payload;

    auto start = std::chrono::steady_clock::now();

    // A receives with 2s timeout (message arrives in ~50ms)
    EXPECT_NO_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(2000)));

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Should complete quickly (< 200ms), not wait the full 2s
    EXPECT_LT(elapsed_ms, 200) << "Recv took too long even though message arrived";

    sender_thread.join();

    // Validate received data
    EXPECT_EQ(hdr.magic, GV_MAGIC);
    EXPECT_EQ(hdr.msg_type, (uint8_t)MsgType::RESPONSE);
    EXPECT_EQ(hdr.payload_len, strlen(msg));
    EXPECT_EQ(memcmp(payload.data(), msg, strlen(msg)), 0);
}

// ============================================================================
// Test 4: Multiple timeouts don't interfere
// ============================================================================
TEST_F(Phase2TimeoutTest, MultipleTimeoutsSequential) {
    FramedStream fs_a(worker_a_);
    FramedStream fs_b(worker_b_);

    FrameHeader hdr{};
    std::vector<uint8_t> payload;

    // First timeout
    EXPECT_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(100)), std::runtime_error);

    // Second timeout (should work independently)
    EXPECT_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(100)), std::runtime_error);

    // Then successful send/recv
    const char* msg = "recovery";
    fs_b.Send(ep_ba_, MsgType::RESPONSE, 2, msg, strlen(msg));

    EXPECT_NO_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(1000)));
    EXPECT_EQ(hdr.seq, 2);
    EXPECT_EQ(payload.size(), strlen(msg));
}

// ============================================================================
// Test 5: Rapid send/recv with short timeouts
// ============================================================================
TEST_F(Phase2TimeoutTest, RapidExchangeWithShortTimeout) {
    FramedStream fs_a(worker_a_);
    FramedStream fs_b(worker_b_);

    // Exchanger thread on B
    std::thread exchanger([&] {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        // Receive from A
        fs_b.Recv(ep_ba_, hdr, payload, std::chrono::milliseconds(1000));
        // Echo back
        fs_b.Send(ep_ba_, MsgType::RESPONSE, hdr.seq, payload.data(), payload.size());
    });

    // A sends and receives with short 500ms timeout
    const char* msg = "quick test";
    fs_a.Send(ep_ab_, MsgType::REQUEST, 99, msg, strlen(msg));

    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    EXPECT_NO_THROW(fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(500)));

    EXPECT_EQ(hdr.seq, 99);
    EXPECT_EQ(payload.size(), strlen(msg));
    EXPECT_EQ(memcmp(payload.data(), msg, strlen(msg)), 0);

    exchanger.join();
}

// ============================================================================
// Test 6: Verify no indefinite hang (Phase 1 regression check)
// ============================================================================
TEST_F(Phase2TimeoutTest, NoIndefiniteHangOnMissingPeer) {
    FramedStream fs_a(worker_a_);
    // Intentionally don't respond from B

    FrameHeader hdr{};
    std::vector<uint8_t> payload;

    auto start = std::chrono::steady_clock::now();
    bool threw = false;
    try {
        fs_a.Recv(ep_ab_, hdr, payload, std::chrono::milliseconds(100));
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_THAT(std::string(e.what()), testing::HasSubstr("timeout"));
    }

    EXPECT_TRUE(threw) << "Should throw timeout exception";

    auto elapsed = std::chrono::steady_clock::now() - start;
    // Critical: must not hang forever. Should timeout in ~100ms
    EXPECT_LT(elapsed, std::chrono::seconds(2)) << "Recv hung for too long!";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
