#include "gvirtus/communicators/FramedStream.h"

#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void send_completion_cb(void* request, ucs_status_t status) {
    (void)request;
    (void)status;
}

void recv_completion_cb(void* request, ucs_status_t status, size_t length) {
    (void)request;
    (void)status;
    (void)length;
}

}  // namespace

namespace gvirtus::communicators {

uint32_t FramedStream::header_crc32(const FrameHeader& hdr) {
    // CRC32/IEEE over header bytes before header_crc.
    const auto* data = reinterpret_cast<const uint8_t*>(&hdr);
    constexpr std::size_t crc_input_len = offsetof(FrameHeader, header_crc);

    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < crc_input_len; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-(static_cast<int32_t>(crc & 1u)));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

void FramedStream::wait_for_completion(ucs_status_ptr_t request) {
    if (request == nullptr) {
        return;
    }

    if (!UCS_PTR_IS_PTR(request)) {
        const ucs_status_t immediate = UCS_PTR_STATUS(request);
        if (immediate != UCS_OK) {
            throw std::runtime_error("FramedStream immediate completion failed");
        }
        return;
    }

    while (true) {
        const ucs_status_t status = ucp_request_check_status(request);
        if (status == UCS_INPROGRESS) {
            std::this_thread::yield();
            continue;
        }

        if (status != UCS_OK) {
            ucp_request_free(request);
            throw std::runtime_error("FramedStream completion failed");
        }

        ucp_request_free(request);
        return;
    }
}

void FramedStream::Send(ucp_ep_h ep, MsgType type, uint16_t seq, const void* payload, uint32_t len) {
    FrameHeader hdr{};
    hdr.magic = GV_MAGIC;
    hdr.msg_type = static_cast<uint8_t>(type);
    hdr.seq = seq;
    hdr.payload_len = len;
    hdr.header_crc = header_crc32(hdr);

    // UCX 1.12 stream+iov can crash in this environment; send one contiguous frame buffer.
    std::vector<uint8_t> frame(sizeof(FrameHeader) + len);
    std::memcpy(frame.data(), &hdr, sizeof(FrameHeader));
    if (len > 0 && payload != nullptr) {
        std::memcpy(frame.data() + sizeof(FrameHeader), payload, len);
    }

    ucs_status_ptr_t req =
        ucp_stream_send_nb(ep, frame.data(), frame.size(), ucp_dt_make_contig(1), send_completion_cb, 0);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error("FramedStream::Send failed");
    }

    if (req != nullptr) {
        wait_for_completion(req);
    }
}

bool FramedStream::Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out) {
    size_t got = 0;
    ucs_status_ptr_t req = ucp_stream_recv_nb(ep, &hdr_out, sizeof(FrameHeader),
                                              ucp_dt_make_contig(1), recv_completion_cb, &got,
                                              0);

    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error("FramedStream::Recv header read failed");
    }
    if (req != nullptr) {
        wait_for_completion(req);
    }

    if (hdr_out.magic != GV_MAGIC) {
        throw std::runtime_error("magic mismatch -- stream desynced");
    }
    if (hdr_out.header_crc != header_crc32(hdr_out)) {
        throw std::runtime_error("header CRC mismatch");
    }
    if (hdr_out.payload_len > 64u * 1024u * 1024u) {
        throw std::runtime_error("payload_len implausibly large");
    }

    payload_out.resize(hdr_out.payload_len);
    if (hdr_out.payload_len == 0) {
        return true;
    }

    req = ucp_stream_recv_nb(ep, payload_out.data(), hdr_out.payload_len, ucp_dt_make_contig(1),
                             recv_completion_cb, &got, 0);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error("FramedStream::Recv payload read failed");
    }
    if (req != nullptr) {
        wait_for_completion(req);
    }

    return true;
}

}  // namespace gvirtus::communicators
