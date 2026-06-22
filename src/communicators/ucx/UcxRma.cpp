/*
 * UcxCommunicator RMA fast path — extracted from UcxCommunicator.cpp.
 *
 * Contains the four methods that implement the RDMA-into-pinned-slot data
 * path and the RmaSetup/RmaPosted handshake that bootstraps it:
 *
 *   send_rma_setup()        — server -> client: pack rkeys of every rx
 *                              slot and send them as an AM at connect time.
 *   handle_rma_setup_am()   — client side: unpack the rkey blobs into
 *                              remote_slots_.
 *   destroy_rma_state()     — teardown helper for the rkeys.
 *   WriteIovRma()           — client data path: stage or zero-copy the iov
 *                              fragments into a remote slot via ucp_put_nbx,
 *                              then send a tiny RmaPosted AM.
 *
 * Lives in its own translation unit because (a) it's ~600 lines of code
 * that exists for a single feature, and (b) it lets UcxCommunicator.cpp
 * stay focused on the Communicator interface. The functions remain
 * UcxCommunicator member functions; the split is purely organisational.
 *
 * Shared internal helpers (ucx_debug_log, is_gpu_pointer) come from
 * UcxInternal.h so this file does not need to re-declare or duplicate the
 * CUDA dlopen state that lives in UcxCommunicator.cpp.
 */
#include "UcxCommunicator.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gvirtus/communicators/Protocol.h"
#include "UcxInternal.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using gvirtus::communicators::UcxCommunicator;
using namespace gvirtus::communicators::ucx_internal;
using namespace log4cplus;

namespace {
static Logger ucx_logger = Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator"));
}  // namespace

// Server-side: pack rkeys of every rx_slot, build an RmaSetup AM body, and
// send it to the connected client. Called once per accepted connection,
// right after the endpoint is created (so the client receives this before
// any data traffic).
void UcxCommunicator::send_rma_setup() {
    if (endpoint_ == nullptr || context_ == nullptr) return;

    // Snapshot rx slot metadata. With GPUDirect Step B2 each slot may also
    // expose a GPU shadow (gpu_addr / gpu_capacity / gpu_rkey).
    struct PackedSlot {
        std::uint64_t addr;
        std::uint64_t capacity;
        void *rkey_buf{nullptr};
        size_t rkey_len{0};
        // GPU shadow (optional). When gpu_rkey_buf == nullptr the slot
        // advertises host only — matches the pre-B2 wire format byte for byte.
        std::uint64_t gpu_addr{0};
        std::uint64_t gpu_capacity{0};
        void *gpu_rkey_buf{nullptr};
        size_t gpu_rkey_len{0};
    };
    std::vector<PackedSlot> packed;
    {
        std::lock_guard<std::mutex> lk(rx_pool_->mu);
        packed.reserve(rx_pool_->slots.size());
        for (auto &slot : rx_pool_->slots) {
            if (slot.memh == nullptr) continue;  // skip slots that failed mem_map
            PackedSlot ps{};
            ps.addr = reinterpret_cast<std::uint64_t>(slot.addr);
            ps.capacity = slot.capacity;
            ucs_status_t st = ucp_rkey_pack(context_, slot.memh,
                                            &ps.rkey_buf, &ps.rkey_len);
            if (st != UCS_OK) {
                std::fprintf(stderr,
                             "UCX rma_setup: ucp_rkey_pack failed (%s)\n",
                             ucs_status_string(st));
                continue;
            }
            // Pack GPU shadow rkey if present.
            if (slot.gpu_memh != nullptr && slot.gpu_addr != nullptr) {
                ucs_status_t gst = ucp_rkey_pack(context_, slot.gpu_memh,
                                                 &ps.gpu_rkey_buf, &ps.gpu_rkey_len);
                if (gst == UCS_OK) {
                    ps.gpu_addr = reinterpret_cast<std::uint64_t>(slot.gpu_addr);
                    ps.gpu_capacity = slot.gpu_capacity;
                } else {
                    LOG4CPLUS_DEBUG(ucx_logger, "rma_setup: gpu rkey_pack FAILED (" << ucs_status_string(gst) << ") — advertising host only");
                    ps.gpu_rkey_buf = nullptr;
                }
            }
            packed.push_back(ps);
        }
    }

    if (packed.empty()) {
        LOG4CPLUS_DEBUG(ucx_logger, "rma_setup: no slots to advertise; skipping");
        return;
    }

    // Assemble AM body: [EnvelopeHeader] [N * RmaSlotDescriptor] [N * rkey blobs]
    using gvirtus::communicators::am::EnvelopeHeader;
    using gvirtus::communicators::am::MessageType;
    using gvirtus::communicators::am::RmaSlotDescriptor;
    using gvirtus::communicators::am::kEnvelopeMagic;
    using gvirtus::communicators::am::kEnvelopeVersion;

    // Wire format (Step B2 extension):
    //   [EnvelopeHeader]
    //   [N * RmaSlotDescriptor]     ← per-slot header; descriptor.reserved0
    //                                  bit 0 = "has_gpu_shadow" flag
    //   For each slot in order:
    //     [host_rkey_blob (rkey_size bytes)]
    //     If has_gpu_shadow:
    //       [u64 gpu_addr][u64 gpu_capacity][u32 gpu_rkey_size][gpu_rkey_blob]
    //
    // Old peers (pre-B2) see descriptor.reserved0=0 always → no GPU block →
    // identical to pre-B2 layout.
    constexpr std::uint32_t kHasGpuShadow = 1u << 0;

    size_t descriptors_bytes = packed.size() * sizeof(RmaSlotDescriptor);
    size_t rkeys_bytes = 0;
    size_t gpu_extension_bytes = 0;
    for (auto &p : packed) {
        rkeys_bytes += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            gpu_extension_bytes += sizeof(std::uint64_t)  // gpu_addr
                                 + sizeof(std::uint64_t)  // gpu_capacity
                                 + sizeof(std::uint32_t)  // gpu_rkey_size
                                 + p.gpu_rkey_len;
        }
    }
    size_t total_bytes = sizeof(EnvelopeHeader) + descriptors_bytes + rkeys_bytes + gpu_extension_bytes;

    std::vector<unsigned char> buf(total_bytes);
    auto *hdr = reinterpret_cast<EnvelopeHeader *>(buf.data());
    hdr->magic = kEnvelopeMagic;
    hdr->version = kEnvelopeVersion;
    hdr->message_type = static_cast<std::uint8_t>(MessageType::RmaSetup);
    // RmaSetup: reserved0 carries the slot count (replaces the old payload_size
    // overload; u16 supports 65k slots, far beyond any realistic pool).
    hdr->routine_size = 0;
    hdr->reserved0 = static_cast<std::uint16_t>(packed.size());
    hdr->pad_ = 0;
    hdr->request_id = 0;
    hdr->status_code = 0;

    size_t off = sizeof(EnvelopeHeader);
    for (auto &p : packed) {
        RmaSlotDescriptor d{};
        d.remote_addr = p.addr;
        d.slot_capacity = p.capacity;
        d.rkey_size = static_cast<std::uint32_t>(p.rkey_len);
        d.reserved0 = (p.gpu_rkey_buf != nullptr) ? kHasGpuShadow : 0u;
        std::memcpy(buf.data() + off, &d, sizeof(d));
        off += sizeof(d);
    }
    // Per-slot rkey blobs, interleaved with optional gpu extension.
    for (auto &p : packed) {
        std::memcpy(buf.data() + off, p.rkey_buf, p.rkey_len);
        off += p.rkey_len;
        if (p.gpu_rkey_buf != nullptr) {
            std::memcpy(buf.data() + off, &p.gpu_addr, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::memcpy(buf.data() + off, &p.gpu_capacity, sizeof(std::uint64_t));
            off += sizeof(std::uint64_t);
            std::uint32_t gsz = static_cast<std::uint32_t>(p.gpu_rkey_len);
            std::memcpy(buf.data() + off, &gsz, sizeof(std::uint32_t));
            off += sizeof(std::uint32_t);
            std::memcpy(buf.data() + off, p.gpu_rkey_buf, p.gpu_rkey_len);
            off += p.gpu_rkey_len;
        }
    }

    // Send as a single AM.
    {
        std::lock_guard<std::mutex> worker_lock(*worker_mutex_);
        ucp_request_param_t request_param{};
        request_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        request_param.datatype = ucp_dt_make_contig(1);
        void *request = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                        buf.data(), buf.size(), &request_param);
        wait_request_completion(request, "rma_setup_send");
    }

    // Release the packed rkey buffers (host + optional GPU).
    size_t gpu_advertised = 0;
    for (auto &p : packed) {
        if (p.rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.rkey_buf);
        }
        if (p.gpu_rkey_buf != nullptr) {
            ucp_rkey_buffer_release(p.gpu_rkey_buf);
            ++gpu_advertised;
        }
    }
    LOG4CPLUS_DEBUG(ucx_logger, "rma_setup: advertised " << packed.size() << " slots ("
                    << rkeys_bytes << " rkey bytes, " << gpu_advertised << " with gpu shadow)");
}

// Client-side: parse an incoming RmaSetup AM body, unpack each rkey, and
// populate remote_slots_. After this returns the data path can use ucp_put.
void UcxCommunicator::handle_rma_setup_am(const void *data, size_t length) {
    using gvirtus::communicators::am::EnvelopeHeader;
    using gvirtus::communicators::am::RmaSlotDescriptor;

    if (length < sizeof(EnvelopeHeader)) {
        std::fprintf(stderr, "RmaSetup: body too short (%zu)\n", length);
        return;
    }
    EnvelopeHeader hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    // RmaSetup carries slot count in reserved0 (see send_rma_setup).
    const size_t num_slots = static_cast<size_t>(hdr.reserved0);
    const size_t descriptors_bytes = num_slots * sizeof(RmaSlotDescriptor);
    if (length < sizeof(hdr) + descriptors_bytes) {
        std::fprintf(stderr, "RmaSetup: descriptors truncated\n");
        return;
    }

    const auto *base = static_cast<const unsigned char *>(data);
    const auto *desc_ptr = reinterpret_cast<const RmaSlotDescriptor *>(
        base + sizeof(hdr));
    const unsigned char *rkey_cursor = base + sizeof(hdr) + descriptors_bytes;
    const unsigned char *rkey_end = base + length;

    constexpr std::uint32_t kHasGpuShadow = 1u << 0;

    std::vector<RemoteSlot> new_slots;
    new_slots.reserve(num_slots);
    size_t gpu_received = 0;
    for (size_t i = 0; i < num_slots; ++i) {
        if (rkey_cursor + desc_ptr[i].rkey_size > rkey_end) {
            std::fprintf(stderr, "RmaSetup: rkey blob %zu truncated\n", i);
            break;
        }
        ucp_rkey_h rkey = nullptr;
        ucs_status_t st = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &rkey);
        if (st != UCS_OK) {
            std::fprintf(stderr,
                         "RmaSetup: ucp_ep_rkey_unpack[%zu] failed (%s)\n",
                         i, ucs_status_string(st));
            rkey = nullptr;
        }
        rkey_cursor += desc_ptr[i].rkey_size;

        RemoteSlot rs{desc_ptr[i].remote_addr,
                      desc_ptr[i].slot_capacity, rkey};

        // GPUDirect Step B2: parse optional GPU extension after the host
        // rkey blob if the descriptor's flag bit is set. Old peers don't
        // set this flag → no extension to read → rs.gpu_rkey stays null.
        if ((desc_ptr[i].reserved0 & kHasGpuShadow) != 0u) {
            const size_t kFixedExt = sizeof(std::uint64_t)  // gpu_addr
                                   + sizeof(std::uint64_t)  // gpu_capacity
                                   + sizeof(std::uint32_t); // gpu_rkey_size
            if (rkey_cursor + kFixedExt > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu extension header %zu truncated\n", i);
                break;
            }
            std::uint64_t gpu_addr = 0, gpu_cap = 0;
            std::uint32_t gpu_rkey_size = 0;
            std::memcpy(&gpu_addr,      rkey_cursor + 0,  sizeof(std::uint64_t));
            std::memcpy(&gpu_cap,       rkey_cursor + 8,  sizeof(std::uint64_t));
            std::memcpy(&gpu_rkey_size, rkey_cursor + 16, sizeof(std::uint32_t));
            rkey_cursor += kFixedExt;
            if (rkey_cursor + gpu_rkey_size > rkey_end) {
                std::fprintf(stderr, "RmaSetup: gpu rkey blob %zu truncated\n", i);
                break;
            }
            ucp_rkey_h gpu_rkey = nullptr;
            ucs_status_t gst = ucp_ep_rkey_unpack(endpoint_, rkey_cursor, &gpu_rkey);
            if (gst == UCS_OK) {
                rs.gpu_addr     = gpu_addr;
                rs.gpu_capacity = gpu_cap;
                rs.gpu_rkey     = gpu_rkey;
                ++gpu_received;
            } else {
                std::fprintf(stderr,
                             "RmaSetup: gpu rkey unpack[%zu] failed (%s), skipping gpu path\n",
                             i, ucs_status_string(gst));
            }
            rkey_cursor += gpu_rkey_size;
        }

        new_slots.push_back(rs);
    }

    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        remote_slots_ = std::move(new_slots);
        next_remote_slot_idx_ = 0;
        rma_setup_received_.store(true);
    }
    rma_setup_cv_.notify_all();
    LOG4CPLUS_DEBUG(ucx_logger, "rma_setup: received " << remote_slots_.size() << " remote slots ("
                    << gpu_received << " with gpu shadow)");
}

void UcxCommunicator::destroy_rma_state() {
    std::lock_guard<std::mutex> lk(rma_state_mu_);
    for (auto &rs : remote_slots_) {
        if (rs.rkey != nullptr) ucp_rkey_destroy(rs.rkey);
        if (rs.gpu_rkey != nullptr) ucp_rkey_destroy(rs.gpu_rkey);
    }
    remote_slots_.clear();
    rma_setup_received_.store(false);
    next_remote_slot_idx_ = 0;
}

// RMA-mode send. Two paths, selected by env var GVIRTUS_RMA_ZEROCOPY:
//
//  * "staged" (GVIRTUS_RMA_ZEROCOPY=0): copy ALL iov fragments into the
//     pre-registered local tx_scratch_, then one ucp_put_nbx of the
//     contiguous buffer. Simple, predictable. Pays a host-RAM memcpy of
//     ~2.5ms for a 64MB payload, but no per-call buffer registration.
//
//  * "zerocopy" (default, or GVIRTUS_RMA_ZEROCOPY=1): only the small
//     fragments (header, routine) are staged into tx_scratch_; the large
//     fragment (the user's payload) is ucp_put_nbx'd directly from the
//     caller's buffer. Two puts in flight in parallel. UCX rcache caches
//     the user buffer's registration after first use — steady state is
//     ~2.5ms faster per call. First call against a fresh user buffer pays
//     a one-time registration cost (typically ~10ms for 64MB).
//
// Both paths end with a tiny RmaPosted AM carrying the slot index.
size_t UcxCommunicator::WriteIovRma(const struct iovec *iov, size_t iov_count,
                                    size_t total) {
    // Pick a remote slot. Round-robin keeps things simple; the synchronous
    // request/response pattern guarantees the previous slot has already
    // been consumed by the server before we get here for the next message.
    size_t slot_idx;
    RemoteSlot rs;
    {
        std::lock_guard<std::mutex> lk(rma_state_mu_);
        if (remote_slots_.empty()) return 0;
        slot_idx = next_remote_slot_idx_;
        next_remote_slot_idx_ = (next_remote_slot_idx_ + 1) % remote_slots_.size();
        rs = remote_slots_[slot_idx];
    }

    if (rs.rkey == nullptr || total > rs.capacity) {
        // Caller will fall back to the IOV path.
        return 0;
    }

    // Env-var gated zerocopy: default OFF (set GVIRTUS_RMA_ZEROCOPY=1 to
    // enable). The zerocopy path relies on UCX's registration cache to
    // make repeated ucp_put_nbx(h_user_buf, ...) cheap. In this container
    // build UCX logs "could not create UCP registration cache: Unsupported
    // operation" at init, which means every put re-registers the source
    // buffer (~25ms for 64MB) — regressing write from ~8ms to ~31ms.
    // The staged path uses a pre-mem_map'd tx_scratch and is rcache-
    // independent, so it stays ~8ms warm regardless. Keep zerocopy behind
    // a flag for builds where rcache works (e.g., once nvidia-peermem and
    // UCM event handling are properly set up in the container).
    static const bool zerocopy_enabled = []() {
        const char *v = std::getenv("GVIRTUS_RMA_ZEROCOPY");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();

    // GPUDirect Step B3: set when the big iov fragment is routed to the
    // peer's GPU shadow. Communicated to the peer via RmaPosted:
    //   routine_size = gpu_split_bytes (= big_size routed to GPU)
    //   status_code  = gpu_split_offset (= pre_size in host slot — the
    //                  position where the GPU data belongs in the logical
    //                  message). Allows biggest to be at ANY iov index, not
    //                  just last (Fase 5 puts user_src at index 3 with a
    //                  trailing [count][kind] = 12-byte input_post).
    std::uint64_t gpu_split_bytes  = 0;
    std::uint32_t gpu_split_offset = 0;

    // Find the biggest iov fragment regardless of position. The zerocopy
    // path treats it as the payload to ucp_put directly from caller memory
    // and stages every other fragment through tx_scratch_. With the legacy
    // 3-entry layout [header][routine][payload] the biggest sits at index
    // iov_count-1, matching the prior behavior. With the Fase 5 layout
    // [header][routine][input_pre][user_src][input_post] the biggest sits
    // at an interior index — argmax catches both.
    size_t biggest_idx = 0;
    size_t big_size = 0;
    for (size_t i = 0; i < iov_count; ++i) {
        if (iov[i].iov_len > big_size) {
            big_size = iov[i].iov_len;
            biggest_idx = i;
        }
    }
    const size_t small_size = (big_size <= total) ? (total - big_size) : 0;
    // Bytes from iov fragments that appear BEFORE the biggest one — they
    // get put to rs.addr + 0. Bytes AFTER the biggest go to
    // rs.addr + pre_size + big_size to preserve wire order.
    size_t pre_size = 0;
    for (size_t i = 0; i < biggest_idx; ++i) pre_size += iov[i].iov_len;
    const size_t post_size = small_size - pre_size;
    // Detect GPU mem in the biggest fragment ONCE here so we can both (a)
    // force zerocopy when GPU is present (staged path would memcpy from GPU
    // into tx_scratch → SIGSEGV) and (b) reuse the value for the memh
    // registration further down.
    const bool big_is_gpu = is_gpu_pointer(iov[biggest_idx].iov_base);
    // Only worth splitting when the "big" fragment is genuinely big and the
    // "small" prefix isn't empty (otherwise we'd just be issuing one put).
    // big_is_gpu overrides zerocopy_enabled: with GPU mem we have no choice,
    // the staged fallback can't memcpy device memory through CPU.
    const bool use_zerocopy = (zerocopy_enabled || big_is_gpu) &&
                              iov_count >= 2 &&
                              big_size >= (16u * 1024u) &&
                              small_size > 0;

    std::lock_guard<std::mutex> worker_lock(*worker_mutex_);

    if (use_zerocopy) {
        // Stage all fragments except the biggest into the registered
        // scratch, contiguously in iov order (pre first, then post).
        ensure_tx_scratch_locked(small_size);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                if (i == biggest_idx) continue;
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        LOG4CPLUS_DEBUG(ucx_logger, "WriteIovRma(zerocopy) slot=" << slot_idx
                        << " pre=" << pre_size << " big=" << big_size
                        << " post=" << post_size << " biggest_idx=" << biggest_idx);

        // Issue up to three puts non-blocking. UCX progresses them in
        // parallel; their completions are awaited together at the end.
        ucp_request_param_t p_scratch{};
        p_scratch.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        p_scratch.memh = tx_scratch_.memh;

        void *req_pre = nullptr;
        if (pre_size > 0) {
            req_pre = ucp_put_nbx(endpoint_,
                                  tx_scratch_.addr, pre_size,
                                  rs.addr, rs.rkey, &p_scratch);
        }

        void *req_post = nullptr;
        if (post_size > 0) {
            req_post = ucp_put_nbx(endpoint_,
                                   static_cast<char *>(tx_scratch_.addr) + pre_size,
                                   post_size,
                                   rs.addr + pre_size + big_size,
                                   rs.rkey, &p_scratch);
        }

        ucp_request_param_t p_big{};

        // Manual host-buffer registration cache. UCX rcache fails to
        // init in this container ("rcache failed to install UCM event
        // handler: Unsupported operation"), so we mem_map ptr->memh
        // ourselves and pass it explicitly. Saves ~25ms re-register
        // cost per 64MB put on the warm path.
        //
        // SIZE-THRESHOLD GUARD (added for OpenPose-style workloads):
        // The cache keys by virtual address. If the user frees and reallocs
        // at the same address (Caffe blobs, repeated cudaMallocHost cycles)
        // we'd return a stale memh → IB QP Local Protection error. For
        // small buffers (typical of inference frameworks) we skip the cache
        // and pay ucp_mem_map+unmap per call (~ms penalty). Large stable
        // buffers (simple_matrix-style 4 MB+) still cache for big wins.
        static constexpr size_t kCacheThreshold = 2u * 1024u * 1024u;  // 2 MB
        const bool use_memh_cache = (big_size >= kCacheThreshold);

        static thread_local std::unordered_map<const void *, ucp_mem_h>
            user_memh_cache;

        const void *user_addr = iov[biggest_idx].iov_base;
        ucp_mem_h user_memh = nullptr;
        bool memh_owned = false;  // true iff we own this memh and must unmap after the put

        // big_is_gpu was already computed at WriteIovRma entry (used to force
        // zerocopy when GPU mem is present). Reused here for the mem_map hint
        // — when rcache + memtype-cache are disabled (this container), UCX
        // won't auto-detect CUDA memory and we MUST pass UCS_MEMORY_TYPE_CUDA
        // explicitly.

        auto fill_mp = [&](ucp_mem_map_params_t &mp) {
            mp.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
            mp.address = const_cast<void *>(user_addr);
            mp.length  = big_size;
            if (big_is_gpu) {
                mp.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
                mp.memory_type  = UCS_MEMORY_TYPE_CUDA;
            }
        };

        if (use_memh_cache) {
            auto cit = user_memh_cache.find(user_addr);

            if (cit != user_memh_cache.end()) {
                user_memh = cit->second;
            } else {
                ucp_mem_map_params_t mp{};
                fill_mp(mp);

                if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                    user_memh_cache.emplace(user_addr, user_memh);
                } else {
                    user_memh = nullptr;
                }
            }
        } else {
            // Small buffer path: register fresh each call, unmap after wait.
            ucp_mem_map_params_t mp{};
            fill_mp(mp);

            if (ucp_mem_map(context_, &mp, &user_memh) == UCS_OK) {
                memh_owned = true;
            } else {
                user_memh = nullptr;
            }
        }

        if (user_memh != nullptr) {
            p_big.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
            p_big.memh = user_memh;
        } else {
            p_big.op_attr_mask = 0;
        }

        // GPUDirect Step B3: route the big fragment to the peer's GPU shadow
        // when available. Triggers NIC peer-DMA into remote GPU memory via
        // peermem. The biggest fragment can sit at ANY iov index (Fase 5
        // puts user_src at idx 3 with a 12-byte [count][kind] post). We
        // pass the GPU offset (= pre_size) via RmaPosted.status_code so the
        // receiver knows where to fold the GPU bytes back into the host slot.
        //
        // 4 MB threshold (raised from 64 KB after the B4 sweep showed N=256
        // and N=512 regressed): below this size the 3-put orchestration +
        // peer-DMA setup overhead exceeds the savings from skipping the
        // host bounce. 4 MB matches simple_matrix N=1024 (4 MB), the
        // smallest payload where GPUDirect demonstrably wins.
        //
        // Transport gate: ucp_ep_rkey_unpack(gpu_rkey) returns UCS_OK even
        // when the negotiated transport is TCP (the unpack just parses the
        // blob; transport check happens at put time). If we then attempt
        // ucp_put_nbx to GPU memory over a TCP endpoint, UCX errors out or
        // hangs, killing the connection. Defensive check at WriteIovRma init
        // time: query THIS endpoint's negotiated lanes via ucp_ep_query
        // (lazy + cached). Supersedes the previous process-wide UCX_TLS env
        // probe, so a single backend with UCX_TLS=rc_mlx5,ud_mlx5,tcp,self
        // serves mixed RDMA + TCP frontends correctly — GPUDirect activates
        // only on connections that actually negotiated an RDMA lane.
        const bool route_big_to_gpu = (rs.gpu_rkey != nullptr) &&
                                      (rs.gpu_addr != 0) &&
                                      (big_size >= (4u * 1024u * 1024u)) &&
                                      current_connection_supports_cuda();
        std::uint64_t big_target_addr = route_big_to_gpu
                                        ? rs.gpu_addr
                                        : (rs.addr + pre_size);
        ucp_rkey_h    big_target_rkey = route_big_to_gpu ? rs.gpu_rkey : rs.rkey;

        if (route_big_to_gpu) {
            gpu_split_bytes  = big_size;
            gpu_split_offset = static_cast<std::uint32_t>(pre_size);
            LOG4CPLUS_DEBUG(ucx_logger, "WriteIovRma(B3 gpu-split) slot=" << slot_idx
                            << " pre=" << pre_size << " big=" << big_size
                            << " post=" << post_size
                            << " (to gpu_addr=" << std::hex << big_target_addr << std::dec << ")");
        }

        void *req_big = ucp_put_nbx(endpoint_,
                                    iov[biggest_idx].iov_base, big_size,
                                    big_target_addr, big_target_rkey, &p_big);

        wait_request_completion(req_pre,  "rma_put_pre");
        wait_request_completion(req_big,  "rma_put_big");
        wait_request_completion(req_post, "rma_put_post");

        // Per-call ownership cleanup: unmap fresh registrations so the next
        // call sees a clean state. Cached memh (>= kCacheThreshold) stays
        // mapped for amortization across calls.
        if (memh_owned && user_memh != nullptr) {
            ucp_mem_unmap(context_, user_memh);
        }
    } else {
        // Staged path: copy everything into the scratch, single put.
        ensure_tx_scratch_locked(total);
        {
            char *dst = static_cast<char *>(tx_scratch_.addr);
            size_t off = 0;
            for (size_t i = 0; i < iov_count; ++i) {
                std::memcpy(dst + off, iov[i].iov_base, iov[i].iov_len);
                off += iov[i].iov_len;
            }
        }

        LOG4CPLUS_DEBUG(ucx_logger, "WriteIovRma(staged) slot=" << slot_idx << " total=" << total);

        ucp_request_param_t put_param{};
        put_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        put_param.memh = tx_scratch_.memh;
        void *put_req = ucp_put_nbx(endpoint_,
                                    tx_scratch_.addr, total,
                                    rs.addr, rs.rkey, &put_param);
        wait_request_completion(put_req, "rma_put");
    }

    // Tiny RmaPosted notification — same protocol bytes regardless of which
    // data path filled the remote slot. Compact envelope (16 B) + body (24 B).
    {
        gvirtus::communicators::am::EnvelopeHeader notif{};
        notif.magic = gvirtus::communicators::am::kEnvelopeMagic;
        notif.version = gvirtus::communicators::am::kEnvelopeVersion;
        notif.message_type = static_cast<std::uint8_t>(
            gvirtus::communicators::am::MessageType::RmaPosted);
        notif.routine_size = 0;
        notif.reserved0 = static_cast<std::uint16_t>(slot_idx);
        notif.pad_ = 0;
        notif.request_id = 0;
        notif.status_code = 0;

        gvirtus::communicators::am::RmaPostedBody body{};
        body.slot_total = static_cast<std::uint64_t>(total);
        body.gpu_size = static_cast<std::uint64_t>(gpu_split_bytes);
        body.gpu_offset = static_cast<std::uint64_t>(gpu_split_offset);

        // Send envelope + body as a single contiguous AM payload.
        unsigned char msg[sizeof(notif) + sizeof(body)];
        std::memcpy(msg, &notif, sizeof(notif));
        std::memcpy(msg + sizeof(notif), &body, sizeof(body));

        ucp_request_param_t send_param{};
        send_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        send_param.datatype = ucp_dt_make_contig(1);
        void *send_req = ucp_am_send_nbx(endpoint_, am_id_, nullptr, 0,
                                         msg, sizeof(msg), &send_param);
        wait_request_completion(send_req, "rma_posted_notify");
    }

    LOG4CPLUS_DEBUG(ucx_logger, "WriteIovRma done slot=" << slot_idx << " total=" << total);
    return total;
}
