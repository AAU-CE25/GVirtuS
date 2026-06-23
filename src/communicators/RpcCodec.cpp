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

#include <cstring>
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
                   void *gpu_payload, std::size_t gpu_payload_size, std::string &error) {
    std::size_t host_out_size = 0;
    const char *out_data = nullptr;
    if (output_buffer != nullptr) {
        host_out_size = output_buffer->GetBufferSize();
        out_data = output_buffer->GetBuffer();
    }

    am::EnvelopeHeader rh{};
    rh.magic = am::kEnvelopeMagic;
    rh.version = am::kEnvelopeVersion;
    rh.message_type = kResponse;
    rh.routine_size = 0;
    rh.reserved0 = 0;
    rh.pad_ = 0;
    rh.request_id = request_header.request_id;
    rh.status_code = static_cast<std::uint32_t>(exit_code);

    // [header][exec_sec][host out bytes][optional GPU tail]
    // wire_out_size is no longer encoded — the receiver derives it from
    // (frame_size - sizeof(header) - sizeof(double)).
    struct iovec iov[4];
    int n = 0;
    iov[n].iov_base = &rh;
    iov[n].iov_len = sizeof(rh);
    ++n;
    iov[n].iov_base = &server_exec_sec;
    iov[n].iov_len = sizeof(double);
    ++n;
    if (host_out_size > 0 && out_data != nullptr) {
        iov[n].iov_base = const_cast<char *>(out_data);
        iov[n].iov_len = host_out_size;
        ++n;
    }
    if (gpu_payload != nullptr && gpu_payload_size > 0) {
        iov[n].iov_base = gpu_payload;
        iov[n].iov_len = gpu_payload_size;
        ++n;
    }

    try {
        c->WriteFrame(iov, static_cast<std::size_t>(n));
        c->Sync();
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
    error.clear();
    return true;
}

bool WriteRequest(Communicator *c, std::uint32_t request_id, const std::string &routine,
                  const struct iovec *payload_iov, std::size_t payload_iov_count,
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
    // interleaved sequence of inline + borrowed (AddRef) segments — either
    // way the codec gather-sends them in place, never copying. The framing
    // layer carries the total size, so the envelope omits payload_size.
    std::vector<struct iovec> iov;
    iov.reserve(2 + payload_iov_count);
    iov.push_back(iovec{static_cast<void *>(&rh), sizeof(rh)});
    if (!routine.empty())
        iov.push_back(
            iovec{const_cast<char *>(routine.data()), routine.size()});
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

    // UCX can interleave control-plane envelopes (RmaSetup/RmaPosted) in the
    // same frame queue used for request/response traffic. Skip any valid
    // non-response envelope until we see the matching response frame.
    std::size_t skipped_frames = 0;
    constexpr std::size_t kMaxSkippedFrames = 64;

    for (;;) {
        const unsigned char *frame = nullptr;
        std::size_t frame_size = 0;
        if (!c->TryAcquireFrame(frame, frame_size)) {
            error = "failed to acquire response frame";
            return false;
        }
        owns_frame = true;

        if (frame_size < sizeof(am::EnvelopeHeader)) {
            c->ReleaseFrame();
            owns_frame = false;
            error = "response frame smaller than header";
            return false;
        }

        am::EnvelopeHeader rh{};
        std::memcpy(&rh, frame, sizeof(rh));
        if (!valid_header(rh)) {
            c->ReleaseFrame();
            owns_frame = false;
            error = "invalid response header";
            return false;
        }

        if (rh.message_type != kResponse) {
            c->ReleaseFrame();
            owns_frame = false;
            if (++skipped_frames > kMaxSkippedFrames) {
                error = "too many non-response frames while waiting for response";
                return false;
            }
            continue;
        }

        if (rh.request_id != expected_request_id) {
            c->ReleaseFrame();
            owns_frame = false;
            if (++skipped_frames > kMaxSkippedFrames) {
                error = "too many mismatched response frames";
                return false;
            }
            continue;
        }

        if (frame_size < sizeof(am::EnvelopeHeader) + sizeof(double)) {
            c->ReleaseFrame();
            owns_frame = false;
            error = "response frame smaller than minimum";
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
}

}  // namespace gvirtus::communicators::am
