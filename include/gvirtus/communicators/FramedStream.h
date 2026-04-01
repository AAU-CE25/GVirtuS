#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <ucp/api/ucp.h>

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

class FramedStream {
   public:
    static void Send(ucp_ep_h ep, MsgType type, uint16_t seq, const void* payload, uint32_t len);
    static bool Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out);

   private:
    static uint32_t header_crc32(const FrameHeader& hdr);
    static void wait_for_completion(ucs_status_ptr_t request);
};

}  // namespace gvirtus::communicators