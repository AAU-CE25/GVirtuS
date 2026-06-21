/*
 * GVirtuS RPC wire protocol.
 *
 * Compact 20-byte envelope shared between frontend and backend, identical
 * across every transport — byte-stream communicators length-prefix it,
 * message-oriented communicators carry it as a single frame — so the
 * encode/decode path is the same for TCP, UCX, and anything that comes
 * later.
 *
 * Wire layout, version 3 (this file):
 *
 *   Request:  [EnvelopeHeader][routine_size bytes routine][rest = payload]
 *   Response: [EnvelopeHeader][double exec_sec][rest = output bytes]
 *   RmaSetup: [EnvelopeHeader][N * RmaSlotDescriptor][N * rkey blobs]
 *   RmaPosted: [EnvelopeHeader][RmaPostedBody]
 *
 * Payload / output / rkey-blob sizes are recovered from the frame size the
 * Communicator hands back (TryAcquireFrame yields whole messages), so the
 * envelope no longer carries redundant length fields. This trims 20 bytes
 * per request and 20 bytes per response versus version 1 — meaningful
 * relief for the very many small CUDA calls (cudaGetLastError, cudaMalloc,
 * cudaSetDevice, ...) where the envelope used to dominate the wire.
 *
 * routine_size is u32 so it cannot realistically be hit by any CUDA routine
 * name; the codec only checks against the absolute u32 limit as a defence
 * against silent truncation of a corrupt std::string size.
 *
 * Message types:
 *   Request/Response/Error — standard RPC.
 *   RmaSetup  — server advertises RX-slot rkeys at connection time. Only
 *               emitted by transports with RDMA write-into-pinned-slot
 *               semantics; other transports never send it.
 *   RmaPosted — client notification after an RMA write completes into a
 *               slot. Same constraint as RmaSetup.
 *
 * Version handshake: peers with mismatched `version` fail validation
 * immediately. v2 → v3 is a hard wire break (routine_size widened to u32,
 * header grew 16 → 20 B).
 */
#pragma once

#include <cstdint>

namespace gvirtus::communicators::am {

constexpr std::uint16_t kEnvelopeMagic = 0x5647;  // "GV"
constexpr std::uint8_t kEnvelopeVersion = 3;

enum class MessageType : std::uint8_t {
    Request = 1,
    Response = 2,
    Error = 3,
    // RMA-mode messages — only used by transports with RDMA
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
    // (the client wrote it via RDMA before sending this notification). The
    // trailing RmaPostedBody carries the 64-bit slot total and GPU-split
    // info that no longer fits in the compact envelope.
    RmaSetup = 4,
    RmaPosted = 5,
};

// Compact envelope. 20 bytes, naturally aligned, no implicit padding.
//
// Field reuse map (same wire shape, different meaning per message_type):
//
//   Request:  routine_size = strlen(routine); status_code = 0
//   Response: routine_size = 0;               status_code = CUDA exit code
//   RmaPosted: routine_size = 0; reserved0 = slot_idx; status_code = 0.
//              The 64-bit total and gpu_size/offset live in the trailing
//              RmaPostedBody.
//   RmaSetup:  routine_size = 0; reserved0 = slot_count.
//
// routine_size is u32 so a pathological caller cannot push a routine name
// past the wire-format limit and trigger an error path. Real CUDA routine
// names are ~50 chars, so the field is wildly over-provisioned by design.
struct EnvelopeHeader {
    std::uint16_t magic;        // kEnvelopeMagic
    std::uint8_t  version;      // kEnvelopeVersion
    std::uint8_t  message_type; // MessageType
    std::uint32_t routine_size; // request name length (0 otherwise)
    std::uint16_t reserved0;    // RmaPosted: slot_idx; RmaSetup: slot_count
    std::uint16_t pad_;         // must be 0 — keeps next field u32-aligned
    std::uint32_t request_id;   // matches Response to Request
    std::uint32_t status_code;  // response exit code
};
static_assert(sizeof(EnvelopeHeader) == 20, "EnvelopeHeader must stay 20 bytes");

// Trailing body for RmaPosted (only). Carries the 64-bit fields the
// compact envelope cannot fit; RmaPosted fires only once per large RMA
// transfer (>= 64 KB), so adding 24 bytes here is irrelevant.
struct RmaPostedBody {
    std::uint64_t slot_total;   // total bytes occupying the slot
    std::uint64_t gpu_size;     // GPU split size; 0 = host-only payload
    std::uint64_t gpu_offset;   // offset of GPU portion within the slot
};

// Body of an RmaSetup AM: one of these for each slot the server is exposing,
// each followed by `rkey_size` bytes of serialized rkey. The `reserved0`
// bitfield doubles as feature flags: bit 0 = has_gpu_shadow (GPUDirect
// extension, backward-compatible with peers that don't set it).
struct RmaSlotDescriptor {
    std::uint64_t remote_addr;
    std::uint64_t slot_capacity;
    std::uint32_t rkey_size;
    std::uint32_t reserved0;
};

}  // namespace gvirtus::communicators::am
