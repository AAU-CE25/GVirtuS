#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// Forward declarations for UCX types (implementation in .cpp)
typedef struct ucp_worker* ucp_worker_h;
typedef struct ucp_ep* ucp_ep_h;

static constexpr uint32_t GV_MAGIC = 0xCA7ACAFEu;  // 'CUDA CAFE'

enum class MsgType : uint8_t {
    REQUEST = 0x01,   // frontend -> backend: CUDA call
    RESPONSE = 0x02,  // backend -> frontend: result
    ERROR = 0x03,     // backend -> frontend: execution error
    RESYNC = 0xFF,    // either side: 'I am lost, please resync'
};

struct __attribute__((__packed__)) FrameHeader {
    uint32_t magic;        // always GV_MAGIC
    uint8_t msg_type;      // MsgType enum value
    uint16_t seq;          // sequence number (wraps at 65535)
    uint32_t payload_len;  // bytes that follow this header
    uint32_t header_crc;   // CRC32 of the 11 bytes above
    // total: 4+1+2+4+4 = 15 bytes
};

namespace gvirtus::communicators {

// Internal synchronization context for each request
struct RequestContext {
    std::mutex mtx;
    std::condition_variable cv;
    ucs_status_t status = UCS_INPROGRESS;
    bool complete = false;
};

class FramedStream {
   public:
    explicit FramedStream(ucp_worker_h worker);
    ~FramedStream();

    // Non-blocking APIs with timeout
    void Send(ucp_ep_h ep, MsgType type, uint16_t seq, const void* payload, uint32_t len);
    bool Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
              std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

   private:
    ucp_worker_h worker_;
    std::thread progress_thread_;
    std::atomic<bool> stop_{false};

    // Progress loop — only thread allowed to call ucp_worker_progress
    void ProgressLoop();

    // Synchronization helpers
    static uint32_t header_crc32(const FrameHeader& hdr);
    static bool wait_ctx(RequestContext& ctx, std::chrono::milliseconds timeout);
    static void ucx_completion_cb(void* request, ucs_status_t status, void* user_data);
};

}  // namespace gvirtus::communicators