#include "gvirtus/communicators/FramedStream.h"

#include <cstring>
#include <stdexcept>
#include <ucp/api/ucp.h>
#include <zlib.h>

namespace {

struct SendFrameContext {
    std::vector<uint8_t> frame;
};

void send_completion_cb(void* request, ucs_status_t status, void* user_data) {
    (void)status;
    auto* ctx = static_cast<SendFrameContext*>(user_data);
    delete ctx;
    ucp_request_free(request);
}

void validate_header(const FrameHeader& h, uint32_t (*header_crc32_fn)(const FrameHeader&)) {
    if (h.magic != GV_MAGIC) {
        throw std::runtime_error("bad magic");
    }
    if (h.header_crc != header_crc32_fn(h)) {
        throw std::runtime_error("header CRC mismatch");
    }
    if (h.payload_len > 64u * 1024u * 1024u) {
        throw std::runtime_error("payload_len implausibly large");
    }
}

}  // namespace

namespace gvirtus::communicators {

std::vector<uint8_t> FramedStream::PendingRequest::Wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mtx);
    if (!cv.wait_for(lk, timeout, [this] { return complete; })) {
        throw std::runtime_error("PendingRequest::Wait timed out");
    }
    if (status != UCS_OK) {
        throw std::runtime_error("remote CUDA execution failed (cuda_error=" +
                                 std::to_string(cuda_error) + ")");
    }
    return std::move(response_payload);
}

FramedStream::FramedStream(ucp_worker_h worker) : worker_(worker) {
    progress_thread_ = std::thread([this] { ProgressLoop(); });
}

uint32_t FramedStream::NextRequestId() {
    return next_request_id_.fetch_add(1, std::memory_order_relaxed);
}

FramedStream::~FramedStream() {
    stop_.store(true, std::memory_order_release);
    // Wake the worker so ucp_worker_wait returns and the progress loop can exit promptly.
    (void)ucp_worker_signal(worker_);
    if (dispatch_thread_.joinable()) {
        dispatch_thread_.join();
    }
    if (progress_thread_.joinable()) {
        progress_thread_.join();
    }
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

void FramedStream::EnsureDispatchLoop(ucp_ep_h ep) {
    std::lock_guard<std::mutex> lk(dispatch_mtx_);
    if (dispatch_ep_ == nullptr) {
        dispatch_ep_ = ep;
    } else if (dispatch_ep_ != ep) {
        throw std::invalid_argument("FramedStream::EnsureDispatchLoop endpoint mismatch");
    }

    if (!dispatch_thread_.joinable()) {
        dispatch_thread_ = std::thread([this, ep] { DispatchLoop(ep); });
    }
}

void FramedStream::DispatchLoop(ucp_ep_h ep) {
    while (!stop_.load(std::memory_order_acquire)) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        try {
            (void)Recv(ep, hdr, payload, std::chrono::milliseconds(100));
        } catch (const std::runtime_error&) {
            // Timeout and transient receive failures are expected when idle/closing.
            continue;
        }

        if (hdr.msg_type == static_cast<uint8_t>(MsgType::RESPONSE)) {
            if (payload.size() < sizeof(ResponseHeader)) {
                continue;
            }

            ResponseHeader rh{};
            std::memcpy(&rh, payload.data(), sizeof(ResponseHeader));
            std::vector<uint8_t> result(payload.begin() + sizeof(ResponseHeader), payload.end());
            CompleteRequest(rh.request_id, std::move(result));
            continue;
        }

        if (hdr.msg_type == static_cast<uint8_t>(MsgType::ERROR)) {
            if (payload.size() < sizeof(ErrorPayload)) {
                continue;
            }

            ErrorPayload err{};
            std::memcpy(&err, payload.data(), sizeof(ErrorPayload));
            CompleteRequestWithError(hdr.request_id, err.cuda_error);
            continue;
        }
    }
}

std::shared_ptr<FramedStream::PendingRequest> FramedStream::RegisterRequest(uint32_t id) {
    auto pr = std::make_shared<PendingRequest>();
    std::lock_guard<std::mutex> lk(map_mtx_);
    in_flight_[id] = pr;
    return pr;
}

void FramedStream::CompleteRequest(uint32_t id, std::vector<uint8_t> payload) {
    std::shared_ptr<PendingRequest> pr;
    {
        std::lock_guard<std::mutex> lk(map_mtx_);
        const auto it = in_flight_.find(id);
        if (it == in_flight_.end()) {
            return;
        }
        pr = it->second;
        in_flight_.erase(it);
    }

    {
        std::lock_guard<std::mutex> lk2(pr->mtx);
        pr->response_payload = std::move(payload);
        pr->status = UCS_OK;
        pr->complete = true;
    }
    pr->cv.notify_one();
}

void FramedStream::CompleteRequestWithError(uint32_t id, uint32_t cuda_error) {
    std::shared_ptr<PendingRequest> pr;
    {
        std::lock_guard<std::mutex> lk(map_mtx_);
        const auto it = in_flight_.find(id);
        if (it == in_flight_.end()) {
            return;
        }
        pr = it->second;
        in_flight_.erase(it);
    }

    {
        std::lock_guard<std::mutex> lk2(pr->mtx);
        pr->status = UCS_ERR_IO_ERROR;
        pr->cuda_error = cuda_error;
        pr->complete = true;
    }
    pr->cv.notify_one();
}

void FramedStream::PostFrameNoWait(ucp_ep_h ep, MsgType type, uint32_t request_id,
                                   const void* payload, uint32_t len) {
    FrameHeader hdr{};
    hdr.magic = GV_MAGIC;
    hdr.msg_type = static_cast<uint8_t>(type);
    hdr.request_id = request_id;
    hdr.payload_len = len;
    hdr.header_crc = header_crc32(hdr);

    auto* ctx = new SendFrameContext();
    ctx->frame.resize(sizeof(FrameHeader) + len);
    std::memcpy(ctx->frame.data(), &hdr, sizeof(FrameHeader));
    if (len > 0 && payload != nullptr) {
        std::memcpy(ctx->frame.data() + sizeof(FrameHeader), payload, len);
    }

    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = send_completion_cb;
    param.user_data = ctx;

    ucs_status_ptr_t req = ucp_stream_send_nbx(ep, ctx->frame.data(), ctx->frame.size(), &param);
    if (UCS_PTR_IS_ERR(req)) {
        delete ctx;
        throw std::runtime_error("FramedStream::PostFrameNoWait UCX error");
    }

    if (!UCS_PTR_IS_PTR(req)) {
        const ucs_status_t immediate = UCS_PTR_STATUS(req);
        delete ctx;
        if (immediate != UCS_OK) {
            throw std::runtime_error("FramedStream::PostFrameNoWait immediate status error");
        }
    }
}

void FramedStream::wait_request(ucs_status_ptr_t req, std::chrono::milliseconds timeout,
                                const char* op_name) {
    if (req == nullptr) {
        return;
    }
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string(op_name) + " failed");
    }
    if (!UCS_PTR_IS_PTR(req)) {
        const ucs_status_t immediate = UCS_PTR_STATUS(req);
        if (immediate != UCS_OK) {
            throw std::runtime_error(std::string(op_name) + " immediate status error");
        }
        return;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const ucs_status_t status = ucp_request_check_status(req);
        if (status == UCS_OK) {
            ucp_request_free(req);
            return;
        }
        if (status != UCS_INPROGRESS) {
            ucp_request_free(req);
            throw std::runtime_error(std::string(op_name) + " UCX error");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Bounded timeout: cancel then wait briefly for terminal state.
    ucp_request_cancel(worker_, req);
    const auto cancel_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (std::chrono::steady_clock::now() < cancel_deadline) {
        const ucs_status_t status = ucp_request_check_status(req);
        if (status == UCS_OK || status == UCS_ERR_CANCELED) {
            ucp_request_free(req);
            throw std::runtime_error(std::string(op_name) + " timeout");
        }
        if (status != UCS_INPROGRESS) {
            ucp_request_free(req);
            throw std::runtime_error(std::string(op_name) + " UCX error after cancel");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Last resort: release request to avoid leaking UCX request objects on shutdown paths.
    ucp_request_free(req);
    throw std::runtime_error(std::string(op_name) + " timeout (cancel did not complete)");
}

void FramedStream::RecvExact(ucp_ep_h ep, void* dst, size_t len, std::chrono::milliseconds timeout) {
    auto* out = static_cast<uint8_t*>(dst);
    size_t total = 0;
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (total < len) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw std::runtime_error("FramedStream::RecvExact timeout");
        }

        auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (remaining.count() <= 0) {
            remaining = std::chrono::milliseconds(1);
        }

        size_t got = len - total;

        ucp_request_param_t p{};
        p.op_attr_mask = 0;

        ucs_status_ptr_t req = ucp_stream_recv_nbx(ep, out + total, len - total, &got, &p);
        wait_request(req, remaining, "FramedStream::RecvExact");

        if (got == 0) {
            throw std::runtime_error("FramedStream::RecvExact got 0 bytes");
        }

        total += got;
    }
}

bool FramedStream::DrainUntilMagic(ucp_ep_h ep, std::chrono::milliseconds timeout) {
    constexpr size_t MAGIC_LEN = sizeof(uint32_t);
    uint8_t window[MAGIC_LEN] = {};
    size_t filled = 0;
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline) {
        const auto now = std::chrono::steady_clock::now();
        auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (remaining.count() <= 0) {
            remaining = std::chrono::milliseconds(1);
        }

        uint8_t byte = 0;
        try {
            RecvExact(ep, &byte, 1, remaining);
        } catch (const std::runtime_error&) {
            return false;
        }

        if (filled < MAGIC_LEN) {
            window[filled++] = byte;
        } else {
            std::memmove(window, window + 1, MAGIC_LEN - 1);
            window[MAGIC_LEN - 1] = byte;
        }

        if (filled == MAGIC_LEN) {
            uint32_t candidate = 0;
            std::memcpy(&candidate, window, MAGIC_LEN);
            if (candidate == GV_MAGIC) {
                return true;
            }
        }
    }

    return false;
}

uint32_t FramedStream::header_crc32(const FrameHeader& hdr) {
    // CRC over all header fields except header_crc itself.
    return static_cast<uint32_t>(
        crc32(0, reinterpret_cast<const Bytef*>(&hdr), sizeof(FrameHeader) - sizeof(uint32_t)));
}


std::shared_ptr<FramedStream::PendingRequest> FramedStream::SendAsync(ucp_ep_h ep, MsgType type,
                                                                       const void* payload,
                                                                       uint32_t len) {
    if (type != MsgType::REQUEST) {
        throw std::invalid_argument("FramedStream::SendAsync supports REQUEST only");
    }

    EnsureDispatchLoop(ep);

    const uint32_t request_id = NextRequestId();
    auto pending = RegisterRequest(request_id);

    try {
        PostFrameNoWait(ep, type, request_id, payload, len);
    } catch (...) {
        std::lock_guard<std::mutex> lk(map_mtx_);
        in_flight_.erase(request_id);
        throw;
    }

    return pending;
}

void FramedStream::SendError(ucp_ep_h ep, uint16_t request_seq, uint32_t cuda_error) {
    ErrorPayload payload{};
    payload.cuda_error = cuda_error;
    payload.request_seq = request_seq;
    PostFrameNoWait(ep, MsgType::ERROR, request_seq, &payload,
                    static_cast<uint32_t>(sizeof(ErrorPayload)));
}

void FramedStream::SendResponse(ucp_ep_h ep, uint32_t frame_request_id, uint32_t request_id,
                                const void* result_data, uint32_t result_len) {
    ResponseHeader resp_hdr{};
    resp_hdr.request_id = request_id;
    resp_hdr.result_len = result_len;

    std::vector<uint8_t> payload(sizeof(ResponseHeader) + result_len);
    std::memcpy(payload.data(), &resp_hdr, sizeof(ResponseHeader));
    if (result_len > 0 && result_data != nullptr) {
        std::memcpy(payload.data() + sizeof(ResponseHeader), result_data, result_len);
    }

    PostFrameNoWait(ep, MsgType::RESPONSE, frame_request_id, payload.data(),
                    static_cast<uint32_t>(payload.size()));
}

void FramedStream::SendResync(ucp_ep_h ep) {
    PostFrameNoWait(ep, MsgType::RESYNC, 0, nullptr, 0);
}

bool FramedStream::Recv(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
                        std::chrono::milliseconds timeout) {
    return RecvInternal(ep, hdr_out, payload_out, timeout, true);
}

ResponseHeader FramedStream::ParseAndValidateResponseHeader(const std::vector<uint8_t>& payload,
                                                            uint32_t expected_request_id) {
    if (payload.size() < sizeof(ResponseHeader)) {
        throw std::runtime_error("response payload too short");
    }

    ResponseHeader resp{};
    std::memcpy(&resp, payload.data(), sizeof(ResponseHeader));

    if (resp.request_id != expected_request_id) {
        throw std::runtime_error("request_id mismatch -- stale or wrong response");
    }

    const size_t expected_size = sizeof(ResponseHeader) + static_cast<size_t>(resp.result_len);
    if (payload.size() != expected_size) {
        throw std::runtime_error("response payload length mismatch");
    }

    return resp;
}

void FramedStream::ThrowIfErrorFrame(const FrameHeader& hdr, const std::vector<uint8_t>& payload) {
    if (hdr.msg_type != static_cast<uint8_t>(MsgType::ERROR)) {
        return;
    }

    if (payload.size() < sizeof(ErrorPayload)) {
        throw std::runtime_error("error payload too short");
    }

    ErrorPayload err{};
    std::memcpy(&err, payload.data(), sizeof(ErrorPayload));
    throw CudaRemoteError(err.cuda_error, err.request_seq);
}

bool FramedStream::RecvInternal(ucp_ep_h ep, FrameHeader& hdr_out, std::vector<uint8_t>& payload_out,
                                std::chrono::milliseconds timeout, bool allow_resync_retry) {
    RecvExact(ep, &hdr_out, sizeof(FrameHeader), timeout);

    try {
        validate_header(hdr_out, &FramedStream::header_crc32);
    } catch (const std::runtime_error&) {
        // Ask the peer to resynchronize and try to recover locally.
        SendResync(ep);

        if (!DrainUntilMagic(ep, timeout)) {
            throw std::runtime_error("stream unrecoverable -- close connection");
        }

        hdr_out.magic = GV_MAGIC;
        RecvExact(ep, reinterpret_cast<uint8_t*>(&hdr_out) + sizeof(uint32_t),
                  sizeof(FrameHeader) - sizeof(uint32_t), timeout);
        validate_header(hdr_out, &FramedStream::header_crc32);
    }

    if (hdr_out.msg_type == static_cast<uint8_t>(MsgType::RESYNC)) {
        if (!allow_resync_retry) {
            throw std::runtime_error("FramedStream::Recv repeated RESYNC");
        }
        if (!DrainUntilMagic(ep, timeout)) {
            throw std::runtime_error("stream unrecoverable -- close connection");
        }
        return RecvInternal(ep, hdr_out, payload_out, timeout, false);
    }

    payload_out.resize(hdr_out.payload_len);
    if (hdr_out.payload_len == 0) {
        return true;
    }

    RecvExact(ep, payload_out.data(), hdr_out.payload_len, timeout);

    return true;
}

}  // namespace gvirtus::communicators
