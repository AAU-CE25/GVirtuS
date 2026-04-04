#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <ucp/api/ucp.h>
#include <vector>

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
    uint32_t request_id;   // request identifier
    uint32_t payload_len;  // bytes that follow this header
    uint32_t header_crc;   // CRC32 of the 13 bytes above
    // total: 4+1+4+4+4 = 17 bytes
};

struct __attribute__((__packed__)) ErrorPayload {
    uint32_t cuda_error;   // cudaError_t cast to uint32_t
    uint16_t request_seq;  // seq from the REQUEST that failed
};

struct __attribute__((__packed__)) ResponseHeader {
    uint32_t request_id;   // echoes REQUEST frame request_id
    uint32_t result_len;   // bytes of result data that follow
};

namespace gvirtus::communicators {

class CudaRemoteError : public std::runtime_error {
     public:
        CudaRemoteError(uint32_t cuda_error, uint16_t request_seq)
                : std::runtime_error("remote CUDA execution failed"),
                    cuda_error_(cuda_error),
                    request_seq_(request_seq) {}

        uint32_t cuda_error() const { return cuda_error_; }
        uint16_t request_seq() const { return request_seq_; }

     private:
        uint32_t cuda_error_;
        uint16_t request_seq_;
};

class FramedStream {
   public:
    explicit FramedStream(ucp_worker_h worker);
    ~FramedStream();
    uint32_t NextRequestId();

    // Non-blocking APIs with timeout
    void Send(ucp_ep_h ep, MsgType type, uint32_t request_id, const void* payload, uint32_t len);
    void SendError(ucp_ep_h ep, uint16_t request_seq, uint32_t cuda_error);
    void SendResponse(ucp_ep_h ep, uint32_t frame_request_id, uint32_t request_id,
                      const void* result_data, uint32_t result_len);
    void SendResync(ucp_ep_h ep);
    bool Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
              std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    static ResponseHeader ParseAndValidateResponseHeader(const std::vector<uint8_t>& payload,
                                                         uint32_t expected_request_id);
    static void ThrowIfErrorFrame(const FrameHeader& hdr, const std::vector<uint8_t>& payload);

   private:
    ucp_worker_h worker_;
    std::thread progress_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<uint32_t> next_request_id_{1};

    // Progress loop — only thread allowed to call ucp_worker_progress
    void ProgressLoop();

    // Synchronization helpers
    static uint32_t header_crc32(const FrameHeader& hdr);
    void wait_request(ucs_status_ptr_t req, std::chrono::milliseconds timeout, const char* op_name);
    bool RecvInternal(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
                      std::chrono::milliseconds timeout, bool allow_resync_retry);
    void RecvExact(ucp_ep_h ep, void* dst, size_t len, std::chrono::milliseconds timeout);
    bool DrainUntilMagic(ucp_ep_h ep, std::chrono::milliseconds timeout);
};

}  // namespace gvirtus::communicators