/*
 * GVirtuS RPC wire protocol.
 *
 * Defines the binary envelope framing shared between frontend and backend.
 * Every frame begins with an EnvelopeHeader identifying the message type,
 * request ID, routine name length, and serialized-buffer length. The same
 * envelope is used over every transport — byte-stream communicators
 * length-prefix it, message-oriented communicators carry it as a single
 * frame — so the encode/decode path is identical across transports.
 *
 * Message types:
 *   Request/Response/Error — standard RPC.
 *   RmaSetup  — server advertises RX-slot rkeys at connection time. Only
 *               emitted by transports that expose RDMA write-into-pinned-
 *               slot semantics; transports without RMA simply never send it.
 *   RmaPosted — client notification after an RMA write completes into a
 *               slot. Same constraint as RmaSetup.
 *
 * RmaSlotDescriptor carries per-slot metadata in the RmaSetup body. The
 * reserved0 field doubles as a feature-flag bitfield: bit 0 = has_gpu_shadow
 * (GPUDirect extension, backward-compatible with peers without GPU support).
 */
#pragma once

#include <cstdint>

namespace gvirtus::communicators::am {

constexpr std::uint32_t kEnvelopeMagic = 0x4756414dU;  // "GVAM"
constexpr std::uint16_t kEnvelopeVersion = 1;

enum class MessageType : std::uint16_t {
    Request = 1,
    Response = 2,
    Error = 3,
    // RMA-mode messages — only used by transports that expose RDMA
    // write-into-pinned-slot semantics. Transports without RMA never send
    // them and never need to interpret them.
    //
    // RmaSetup: server -> client at connection time. Carries the metadata
    // needed for the client to issue RDMA writes into the server's
    // pre-pinned RX slots — packed RmaSlotDescriptors followed by
    // serialized rkey blobs.
    //
    // RmaPosted: client -> server per data-path call. The slot referenced
    // by `reserved0` already holds [header || routine || payload] contiguous
    // (the client wrote it via RDMA before sending this notification).
    // header.payload_size carries the TOTAL bytes occupying the slot,
    // matching the inline Request layout so the server's TryAcquireFrame
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

}  // namespace gvirtus::communicators::am
