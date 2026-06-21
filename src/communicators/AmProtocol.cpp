/*
 * GVirtuS RPC envelope codec — see AmProtocol.h.
 *
 * Extracted from Process.cpp: the backend no longer knows the wire format.
 * Framing (whole-message delivery) is handled by the Communicator base class
 * (length-prefixed for byte streams) or overridden by UCX (native AM), so
 * there is no per-transport branching here.
 */
#include "gvirtus/communicators/AmProtocol.h"

#include <cstring>
#include <stdexcept>
#include <sys/uio.h>

namespace gvirtus::communicators::am {

namespace {
constexpr std::uint16_t kRequest = static_cast<std::uint16_t>(ucxam::MessageType::Request);
constexpr std::uint16_t kResponse = static_cast<std::uint16_t>(ucxam::MessageType::Response);

bool valid_header(const ucxam::EnvelopeHeader &h) {
    return h.magic == ucxam::kEnvelopeMagic && h.version == ucxam::kEnvelopeVersion &&
           h.header_size == sizeof(ucxam::EnvelopeHeader);
}
}  // namespace

bool ReadRequest(Communicator *c, ucxam::EnvelopeHeader &header, std::string &routine,
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
    const std::size_t want = sizeof(header) + static_cast<std::size_t>(header.routine_size) +
                             static_cast<std::size_t>(header.payload_size);
    if (frame_size < want) {
        c->ReleaseFrame();
        error = "frame truncated";
        return false;
    }

    routine.assign(reinterpret_cast<const char *>(frame + sizeof(header)),
                   static_cast<std::size_t>(header.routine_size));
    payload_data = frame + sizeof(header) + header.routine_size;
    payload_size = static_cast<std::size_t>(header.payload_size);
    // GPUDirect tail (if the transport landed part of the payload on the GPU).
    c->current_frame_gpu(gpu_payload, gpu_payload_size);
    error.clear();
    return true;
}

bool WriteResponse(Communicator *c, const ucxam::EnvelopeHeader &request_header, int exit_code,
                   double server_exec_sec, const std::shared_ptr<Buffer> &output_buffer,
                   void *gpu_payload, std::size_t gpu_payload_size, std::string &error) {
    std::size_t host_out_size = 0;
    const char *out_data = nullptr;
    if (output_buffer != nullptr) {
        host_out_size = output_buffer->GetBufferSize();
        out_data = output_buffer->GetBuffer();
    }
    // Frontend reads `wire_out_size` bytes contiguously, regardless of whether
    // a trailing slice is GPU-resident.
    const std::size_t wire_out_size = host_out_size + gpu_payload_size;
    const std::size_t payload_size = sizeof(double) + sizeof(std::size_t) + wire_out_size;

    ucxam::EnvelopeHeader rh{};
    rh.magic = ucxam::kEnvelopeMagic;
    rh.version = ucxam::kEnvelopeVersion;
    rh.message_type = kResponse;
    rh.header_size = static_cast<std::uint16_t>(sizeof(ucxam::EnvelopeHeader));
    rh.reserved0 = 0;
    rh.status_code = static_cast<std::uint32_t>(exit_code);
    rh.request_id = request_header.request_id;
    rh.routine_size = 0;
    rh.payload_size = static_cast<std::uint64_t>(payload_size);

    // [header][exec_sec][wire_out_size][host out bytes][optional GPU tail]
    std::size_t wire_out_size_field = wire_out_size;
    struct iovec iov[5];
    int n = 0;
    iov[n].iov_base = &rh;
    iov[n].iov_len = sizeof(rh);
    ++n;
    iov[n].iov_base = &server_exec_sec;
    iov[n].iov_len = sizeof(double);
    ++n;
    iov[n].iov_base = &wire_out_size_field;
    iov[n].iov_len = sizeof(std::size_t);
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

}  // namespace gvirtus::communicators::am
