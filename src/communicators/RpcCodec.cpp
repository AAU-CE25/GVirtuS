/*
 * GVirtuS RPC envelope codec — see RpcCodec.h.
 *
 * Both wire-format directions live here so neither the backend nor the
 * frontend touches the envelope: the backend dispatch loop calls
 * ReadRequest / WriteResponse, the frontend stub calls WriteRequest /
 * ReadResponse, and every transport just moves frames. Framing (whole-
 * message delivery) is the Communicator's job — the base class length-
 * prefixes a byte stream and message-oriented transports override
 * WriteFrame/TryAcquireFrame — so this codec has zero per-transport
 * branching.
 */
#include "gvirtus/communicators/RpcCodec.h"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <cstring>
#include <ios>
#include <stdexcept>
#include <vector>
#include <sys/uio.h>

namespace gvirtus::communicators::am {

namespace {
constexpr std::uint8_t kRequest = static_cast<std::uint8_t>(am::MessageType::Request);
constexpr std::uint8_t kResponse = static_cast<std::uint8_t>(am::MessageType::Response);

bool valid_header(const am::EnvelopeHeader &h) {
    return h.magic == am::kEnvelopeMagic && h.version == am::kEnvelopeVersion;
}
}  // namespace

bool ReadRequest(Communicator *c, am::EnvelopeHeader &header, std::string &routine,
                 const unsigned char *&payload_data, std::size_t &payload_size,
                 void *&gpu_payload, std::size_t &gpu_payload_size, bool &owns_frame,
                 std::string &error) {
    owns_frame = false;
    payload_data = nullptr;
    payload_size = 0;
    gpu_payload = nullptr;
    gpu_payload_size = 0;

    const unsigned char *frame = nullptr;
    std::size_t frame_size = 0;
    if (!c->TryAcquireFrame(frame, frame_size)) {
        error = "client disconnected";
        return false;
    }
    owns_frame = true;

    if (frame_size < sizeof(header)) {
        c->ReleaseFrame();
        error = "frame smaller than header";
        return false;
    }
    std::memcpy(&header, frame, sizeof(header));
    if (!valid_header(header)) {
        c->ReleaseFrame();
        // Diagnostic: capture what was actually on the wire, since "invalid
        // envelope header" alone doesn't say whether this is stale/zeroed
        // memory (uninitialized slot) or a plausible-but-wrong value
        // (misaligned/reordered fragment).
        static log4cplus::Logger logger =
            log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS.RpcCodec"));
        LOG4CPLUS_WARN(logger,
            "ReadRequest: invalid envelope header (magic=0x"
                << std::hex << header.magic << std::dec
                << " version=" << static_cast<unsigned>(header.version)
                << " type=" << static_cast<unsigned>(header.message_type)
                << " routine_size=" << header.routine_size
                << ", want magic=0x" << std::hex << am::kEnvelopeMagic << std::dec
                << " version=" << static_cast<unsigned>(am::kEnvelopeVersion)
                << ") frame_size=" << frame_size);
        error = "invalid envelope header";
        return false;
    }
    if (header.message_type != kRequest) {
        c->ReleaseFrame();
        error = "unexpected message type";
        return false;
    }
    const std::size_t routine_len = static_cast<std::size_t>(header.routine_size);
    if (frame_size < sizeof(header) + routine_len) {
        c->ReleaseFrame();
        error = "frame truncated (routine)";
        return false;
    }

    routine.assign(reinterpret_cast<const char *>(frame + sizeof(header)), routine_len);
    // Payload is "whatever is left after header + routine"; the framing
    // layer already delimits the frame, so the envelope doesn't carry a
    // redundant payload_size field.
    payload_data = frame + sizeof(header) + routine_len;
    payload_size = frame_size - sizeof(header) - routine_len;
    // GPUDirect tail (if the transport landed part of the payload on the GPU).
    c->current_frame_gpu(gpu_payload, gpu_payload_size);
    error.clear();
    return true;
}

bool WriteResponse(Communicator *c, const am::EnvelopeHeader &request_header, int exit_code,
                   double server_exec_sec, const std::shared_ptr<Buffer> &output_buffer,
                   std::string &error) {
    am::EnvelopeHeader rh{};
    rh.magic = am::kEnvelopeMagic;
    rh.version = am::kEnvelopeVersion;
    rh.message_type = kResponse;
    rh.routine_size = 0;
    rh.reserved0 = 0;
    rh.pad_ = 0;
    rh.request_id = request_header.request_id;
    rh.status_code = static_cast<std::uint32_t>(exit_code);

    // [header][exec_sec][output buffer's tagged fragments]. Any GPU-resident
    // tail (Buffer::SegKind::GpuRef, set by Add()'s auto-detect — see
    // Buffer.h) is just another fragment output_buffer->GetIov() emits,
    // tagged is_device=true — there is no separate GPU parameter to thread
    // through anymore.
    // wire_out_size is no longer encoded — the receiver derives it from
    // (frame_size - sizeof(header) - sizeof(double)).
    std::vector<IovFrag> frags;
    frags.reserve(4);
    frags.push_back(IovFrag{static_cast<void *>(&rh), sizeof(rh), false});
    frags.push_back(IovFrag{static_cast<void *>(&server_exec_sec), sizeof(double), false});
    if (output_buffer != nullptr) {
        std::vector<IovFrag> body;
        output_buffer->GetIov(body);
        for (const auto &f : body) frags.push_back(f);
    }

    try {
        c->WriteFrame(frags.data(), frags.size());
        c->Sync();
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
    error.clear();
    return true;
}

bool WriteRequest(Communicator *c, std::uint32_t request_id, const std::string &routine,
                  const IovFrag *payload_iov, std::size_t payload_iov_count,
                  std::string &error) {
    // Sanity check, not a real limit. routine_size is u32 (4 GB headroom);
    // any real CUDA routine name is ~50 chars. We bound at u32 max to guard
    // against silently truncating a corrupt/malformed std::string size on
    // 64-bit platforms where size() can in principle exceed 4 GB.
    if (routine.size() > 0xFFFFFFFFu) {
        error = "routine name too long for envelope";
        return false;
    }
    am::EnvelopeHeader rh{};
    rh.magic = am::kEnvelopeMagic;
    rh.version = am::kEnvelopeVersion;
    rh.message_type = kRequest;
    rh.routine_size = static_cast<std::uint32_t>(routine.size());
    rh.reserved0 = 0;
    rh.pad_ = 0;
    rh.request_id = request_id;
    rh.status_code = 0;

    // [header][routine] followed by the caller's payload IoV fragments. The
    // payload iov may contain a single marshaled-arena segment OR an
    // interleaved sequence of inline + borrowed (AddRef/GpuRef) segments —
    // either way the codec gather-sends them in place, never copying. The
    // framing layer carries the total size, so the envelope omits
    // payload_size.
    std::vector<IovFrag> iov;
    iov.reserve(2 + payload_iov_count);
    iov.push_back(IovFrag{static_cast<void *>(&rh), sizeof(rh), false});
    if (!routine.empty())
        iov.push_back(
            IovFrag{const_cast<char *>(routine.data()), routine.size(), false});
    for (std::size_t i = 0; i < payload_iov_count; ++i) iov.push_back(payload_iov[i]);

    try {
        c->WriteFrame(iov.data(), iov.size());
        c->Sync();
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
    error.clear();
    return true;
}

bool ReadResponse(Communicator *c, std::uint32_t expected_request_id, int &exit_code,
                  double &server_exec_sec, const unsigned char *&out_data, std::size_t &out_size,
                  bool &owns_frame, std::string &error) {
    owns_frame = false;
    out_data = nullptr;
    out_size = 0;
    exit_code = 0;
    server_exec_sec = 0.0;

    const unsigned char *frame = nullptr;
    std::size_t frame_size = 0;
    if (!c->TryAcquireFrame(frame, frame_size)) {
        error = "failed to acquire response frame";
        return false;
    }
    owns_frame = true;

    if (frame_size < sizeof(am::EnvelopeHeader) + sizeof(double)) {
        c->ReleaseFrame();
        owns_frame = false;
        error = "response frame smaller than minimum";
        return false;
    }

    am::EnvelopeHeader rh{};
    std::memcpy(&rh, frame, sizeof(rh));
    if (!valid_header(rh) || rh.message_type != kResponse) {
        c->ReleaseFrame();
        owns_frame = false;
        error = "invalid response header";
        return false;
    }
    if (rh.request_id != expected_request_id) {
        c->ReleaseFrame();
        owns_frame = false;
        error = "response request_id mismatch";
        return false;
    }

    exit_code = static_cast<int>(rh.status_code);

    const unsigned char *p = frame + sizeof(rh);
    std::memcpy(&server_exec_sec, p, sizeof(double));
    p += sizeof(double);
    // Output is the tail of the frame; framing already delimits it.
    out_data = p;
    out_size = frame_size - sizeof(rh) - sizeof(double);
    error.clear();
    return true;
}

}  // namespace gvirtus::communicators::am
