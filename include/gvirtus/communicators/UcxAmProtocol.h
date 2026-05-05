#pragma once

#include <cstdint>

namespace gvirtus::communicators::ucxam {

constexpr std::uint32_t kEnvelopeMagic = 0x4756414dU;  // "GVAM"
constexpr std::uint16_t kEnvelopeVersion = 1;

// Header flags stored in EnvelopeHeader::reserved0.
constexpr std::uint16_t kEnvelopeFlagNoResponse = 1u << 0;

enum class MessageType : std::uint16_t {
    Request = 1,
    Response = 2,
    Error = 3,
};

struct EnvelopeHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t message_type;
    std::uint16_t header_size;
    std::uint16_t reserved0;
    std::uint32_t status_code;
    std::uint64_t request_id;
    std::uint64_t routine_size;
    std::uint64_t payload_size;
}; 
}  // namespace gvirtus::communicators::ucxam