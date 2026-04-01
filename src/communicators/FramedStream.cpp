#include "gvirtus/communicators/FramedStream.h"

#include <cstring>
#include <stdexcept>
#include <ucp/api/ucp.h>

namespace {

void ucx_send_completion_cb(void* request, ucs_status_t status) {
    (void)request;
    (void)status;
}

void ucx_recv_completion_cb(void* request, ucs_status_t status, size_t length) {
    (void)request;
    (void)status;
    (void)length;
}

}  // namespace

namespace gvirtus::communicators {

FramedStream::FramedStream(ucp_worker_h worker) : worker_(worker) {
    progress_thread_ = std::thread([this] { ProgressLoop(); });
}

FramedStream::~FramedStream() {
    stop_.store(true, std::memory_order_release);
    progress_thread_.join();
}

void FramedStream::ProgressLoop() {
    while (!stop_.load(std::memory_order_acquire)) {
        // Drain all pending completions
        while (ucp_worker_progress(worker_) != 0)
            ;

        // Arm the worker — tells UCX to notify on new events
        ucs_status_t s = ucp_worker_arm(worker_);
        if (s == UCS_ERR_BUSY) {
            // Event arrived between progress+arm, loop back
            continue;
        }
        if (s != UCS_OK) {
            break;  // Fatal error in arm
        }

        // Sleep until UCX signals (eventfd / epoll under the hood)
        ucp_worker_wait(worker_);
    }
}

void FramedStream::ucx_completion_cb(void* request, ucs_status_t status, void* user_data) {
    auto* ctx = static_cast<RequestContext*>(user_data);
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->status = status;
        ctx->complete = true;
    }
    ctx->cv.notify_one();
    ucp_request_free(request);
}

bool FramedStream::wait_ctx(RequestContext& ctx, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(ctx.mtx);
    return ctx.cv.wait_for(lk, timeout, [&ctx] { return ctx.complete; });
}

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

    RequestContext ctx;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = ucx_completion_cb;
    param.user_data = &ctx;

    ucs_status_ptr_t req = ucp_stream_send_nbx(ep, frame.data(), frame.size(),
                                               ucp_dt_make_contig(1), &param);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error("FramedStream::Send failed");
    }

    // Check for inline completion — if nullptr, callback will handle it
    if (req == nullptr) {
        return;  // Completed inline, do NOT wait
    }

    // Wait for the callback to signal completion with timeout
    if (!wait_ctx(ctx, std::chrono::milliseconds(5000))) {
        throw std::runtime_error("FramedStream::Send timeout");
    }

    if (ctx.status != UCS_OK) {
        throw std::runtime_error("FramedStream::Send UCX error");
    }
}

bool FramedStream::Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
                        std::chrono::milliseconds timeout) {
    // Step 1: Read header exactly
    RequestContext hdr_ctx;
    size_t hdr_len = sizeof(FrameHeader);
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.cb.recv_stream = ucx_completion_cb;
    p.user_data = &hdr_ctx;

    ucs_status_ptr_t req = ucp_stream_recv_nbx(ep, &hdr_out, sizeof(FrameHeader), &hdr_len, &p);
    if (!UCS_PTR_IS_ERR(req) && req != nullptr) {
        if (!wait_ctx(hdr_ctx, timeout)) {
            throw std::runtime_error("FramedStream::Recv header timeout");
        }
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

    // Step 2: Read payload (if any)
    payload_out.resize(hdr_out.payload_len);
    if (hdr_out.payload_len == 0) {
        return true;
    }

    RequestContext pay_ctx;
    size_t pay_len = hdr_out.payload_len;
    p.user_data = &pay_ctx;

    req = ucp_stream_recv_nbx(ep, payload_out.data(), hdr_out.payload_len, &pay_len, &p);
    if (!UCS_PTR_IS_ERR(req) && req != nullptr) {
        if (!wait_ctx(pay_ctx, timeout)) {
            throw std::runtime_error("FramedStream::Recv payload timeout");
        }
    }

    return true;
}

}  // namespace gvirtus::communicators
