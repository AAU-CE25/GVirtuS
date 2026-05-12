#pragma once

#include <cstdint>

namespace gvirtus::communicators::ucxam {

enum class MessageType : std::uint8_t {
    Request = 1,
    Response = 2,
    Error = 3,
};

// 14 bytes, packed
struct __attribute__((packed)) EnvelopeHeader {
    std::uint8_t  message_type;   // Request/Response/Error
    std::int8_t   status_code;    // Exit code (response only; 0 in request)
    std::uint32_t request_id;     // Sequence number
    std::uint32_t payload_size;   // Payload after routine bytes
    std::uint16_t routine_size;   // Routine name length (0 in response)
    std::uint16_t reserved;       // Padding to 14B
};

static_assert(sizeof(EnvelopeHeader) == 14, "EnvelopeHeader must be 14 bytes");

}  // namespace gvirtus::communicators::ucxam
