#pragma once

#include <cstdint>

namespace gvirtus::communicators::ucxam {

constexpr std::uint32_t kEnvelopeMagic = 0x4756414dU;  // "GVAM"
constexpr std::uint16_t kEnvelopeVersion = 1;

// Request-header flag carried in EnvelopeHeader::reserved0 (bit 0) for
// asynchronous / fire-and-forget dispatch (GVIRTUS_ASYNC_DISPATCH). When set on
// a Request, the backend executes the routine but sends NO response, and the
// frontend does not wait for one. Only valid on MessageType::Request headers;
// RmaSetup/RmaPosted reuse reserved0 for other purposes but are distinct
// message types, and the Request header travels intact (even inside an RMA
// slot), so there is no collision. Failures on a no-response call are latched
// on the backend and reconciled onto the next response-bearing (sync) call,
// matching CUDA's "async errors surface at the next synchronization" semantics.
constexpr std::uint16_t kEnvelopeFlagNoResponse = 1u << 0;

enum class MessageType : std::uint16_t {
    Request = 1,
    Response = 2,
    Error = 3,
    // RMA-mode messages (UCX `ucp_put_nbx` data path).
    //
    // RmaSetup: server -> client at connection time. Carries the metadata
    // needed for the client to issue RDMA writes into the server's
    // pre-mem_map'd RX slots — packed RmaSlotDescriptors followed by
    // serialized rkey blobs.
    //
    // RmaPosted: client -> server per cudaMemcpy. The slot referenced by
    // `reserved0` already holds [header || routine || payload] contiguous
    // (the client wrote it via ucp_put_nbx before sending this AM).
    // header.payload_size carries the TOTAL bytes occupying the slot,
    // matching the AM-stream Request layout so the server's TryAcquireFrame
    // parser works unchanged.
    RmaSetup = 4,
    RmaPosted = 5,
};

struct EnvelopeHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t message_type;
    std::uint16_t header_size;
    std::uint16_t reserved0;        // doubles as `slot_idx` for RmaPosted
    std::uint32_t status_code;
    std::uint64_t request_id;
    std::uint64_t routine_size;
    std::uint64_t payload_size;
};

// Body of an RmaSetup AM: one of these for each slot the server is exposing,
// each followed by `rkey_size` bytes of serialized rkey.
struct RmaSlotDescriptor {
    std::uint64_t remote_addr;
    std::uint64_t slot_capacity;
    std::uint32_t rkey_size;
    std::uint32_t reserved0;
};

}  // namespace gvirtus::communicators::ucxam
