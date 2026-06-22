/*
 * UcxCommunicator pinned RX pool + TX scratch — extracted from
 * UcxCommunicator.cpp so the slot lifecycle has its own translation unit.
 *
 * The pool holds pre-registered (cudaHostAlloc'd when CUDA is available)
 * host buffers — and, with GPUDirect active, matching GPU shadow regions —
 * shared between the listener and accepted UcxCommunicators. The TX scratch
 * buffer is a single page-aligned, ucp_mem_map'd region that WriteIovRma
 * stages outbound payloads through.
 *
 * The functions remain UcxCommunicator member functions; the split is
 * purely organisational. Shared helpers come from UcxInternal.h so this
 * file does not duplicate the CUDA dlopen state owned by UcxGpu.cpp.
 */
#include "UcxCommunicator.h"

#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "UcxInternal.h"

using namespace log4cplus;

using gvirtus::communicators::UcxCommunicator;
using namespace gvirtus::communicators::ucx_internal;

namespace {
static Logger ucx_logger = Logger::getInstance(LOG4CPLUS_TEXT("UcxCommunicator"));
}  // namespace

// present) with the UCX context so subsequent ucp_am_recv_data_nbx and
// ucp_put_nbx can use the memh hints and skip on-the-fly IB registration.
// Called with rx_pool_->mu held.
void UcxCommunicator::map_slot_to_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx == nullptr) return;
    if (slot.memh == nullptr && slot.addr != nullptr) {
        ucp_mem_map_params_t map_params{};
        map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        map_params.address = slot.addr;
        map_params.length  = slot.capacity;
        ucs_status_t st = ucp_mem_map(ctx, &map_params, &slot.memh);
        if (st != UCS_OK) {
            slot.memh = nullptr;  // continue without — UCX rcache will register on first use
        }
    }
    // GPUDirect (Step B1): register the GPU shadow if it exists. UCX needs
    // UCS_MEMORY_TYPE_CUDA explicitly here since memtype-cache is disabled.
    if (slot.gpu_memh == nullptr && slot.gpu_addr != nullptr) {
        ucp_mem_map_params_t gpu_params{};
        gpu_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        gpu_params.address     = slot.gpu_addr;
        gpu_params.length      = slot.gpu_capacity;
        gpu_params.memory_type = UCS_MEMORY_TYPE_CUDA;
        ucs_status_t st = ucp_mem_map(ctx, &gpu_params, &slot.gpu_memh);
        if (st != UCS_OK) {
            slot.gpu_memh = nullptr;
            LOG4CPLUS_DEBUG(ucx_logger, "map_slot_to_ucp: gpu_addr map FAILED (" << ucs_status_string(st) << ") — slot will keep host-only path");
        }
    }
}

void UcxCommunicator::unmap_slot_from_ucp(ucp_context_h ctx, PinnedSlot &slot) {
    if (ctx != nullptr && slot.memh != nullptr) {
        ucp_mem_unmap(ctx, slot.memh);
        slot.memh = nullptr;
    }
    if (ctx != nullptr && slot.gpu_memh != nullptr) {
        ucp_mem_unmap(ctx, slot.gpu_memh);
        slot.gpu_memh = nullptr;
    }
}

// Pre-allocate N RX slots of an initial size. Slots grow on demand later if
// a message arrives that's bigger than the current capacity.
void UcxCommunicator::init_rx_pool() {
    // 2 slots is enough for the current synchronous request/response pattern
    // (the request occupies slot 0 while the response is in flight; once the
    // app receives the response, slot 0 is free for the next request). Was
    // 4 originally but each cudaHostAlloc(64MB) + ucp_mem_map costs ~75ms;
    // halving the count halves per-connection setup time.
    constexpr size_t kInitialSlotCount = 2;
    // 64MB payload + 1MB headroom: leaves room for the envelope header
    // (~40B), the routine name (variable), and any pad bytes the protocol
    // might tack on. Without this slack a 64MB cudaMemcpy lands at 64MB+N
    // bytes total, exceeds rs.capacity, and WriteIovRma falls back to the
    // AM-stream IOV path (silently losing the RMA fast path).
    constexpr size_t kInitialSlotCap   = (1024u * 1024u + 1024u) * 1024u;  // 1025MB

    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    if (!rx_pool_->slots.empty()) return;  // already initialized

    // GPUDirect (Step B1): when active, each slot ALSO gets a GPU shadow
    // region of the same capacity, mem_map'd as CUDA. The shadow is unused
    // in B1 — purely additive. Step B2 will advertise its rkey to peers;
    // Step B3 will route big H2D payloads here via NIC peer-DMA.
    const bool gpudirect_active = gpudirect_enabled();
    size_t gpu_allocated_count = 0;

    rx_pool_->slots.resize(kInitialSlotCount);
    for (size_t i = 0; i < kInitialSlotCount; ++i) {
        bool is_cuda = false;
        unsigned char *p = alloc_pinned_host(kInitialSlotCap, is_cuda);
        if (p == nullptr) {
            throw std::runtime_error("UcxCommunicator: failed to allocate RX pool slot");
        }
        rx_pool_->slots[i] = PinnedSlot{p, kInitialSlotCap, /*in_use*/false, is_cuda, nullptr};

        if (gpudirect_active) {
            unsigned char *gp = alloc_gpu_slot(kInitialSlotCap);
            if (gp != nullptr) {
                rx_pool_->slots[i].gpu_addr = gp;
                rx_pool_->slots[i].gpu_capacity = kInitialSlotCap;
                ++gpu_allocated_count;
            } else {
                LOG4CPLUS_DEBUG(ucx_logger, "rx_pool: slot " << i << " gpu shadow alloc FAILED — host-only");
            }
        }

        map_slot_to_ucp(context_, rx_pool_->slots[i]);
    }
    LOG4CPLUS_INFO(ucx_logger,
        "rx_pool: initialized " << kInitialSlotCount << " slots x " << kInitialSlotCap
        << " bytes (gpu_shadows=" << gpu_allocated_count << "/" << kInitialSlotCount << ")");
}

void UcxCommunicator::destroy_rx_pool() {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    for (auto &slot : rx_pool_->slots) {
        unmap_slot_from_ucp(context_, slot);
        free_pinned_host(slot.addr, slot.is_cuda_host);
        free_gpu_slot(slot.gpu_addr);  // no-op if nullptr
        slot.gpu_addr = nullptr;
        slot.gpu_capacity = 0;
    }
    rx_pool_->slots.clear();
}

// Find a free slot of at least `needed` bytes. Grows an existing in-use-free
// slot's capacity if the largest is too small, or appends a new slot if all
// are busy. Returns slot index.
size_t UcxCommunicator::acquire_rx_slot(size_t needed) {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);

    // Try to find a free slot big enough.
    for (size_t i = 0; i < rx_pool_->slots.size(); ++i) {
        if (!rx_pool_->slots[i].in_use && rx_pool_->slots[i].capacity >= needed) {
            rx_pool_->slots[i].in_use = true;
            return i;
        }
    }
    // GPUDirect (Step B1): mirror host grow with a GPU shadow grow if active.
    const bool gpudirect_active = gpudirect_enabled();

    // No free slot big enough — grow the first free one (or append if none free).
    for (size_t i = 0; i < rx_pool_->slots.size(); ++i) {
        if (!rx_pool_->slots[i].in_use) {
            unmap_slot_from_ucp(context_, rx_pool_->slots[i]);
            free_pinned_host(rx_pool_->slots[i].addr, rx_pool_->slots[i].is_cuda_host);
            free_gpu_slot(rx_pool_->slots[i].gpu_addr);
            bool is_cuda = false;
            unsigned char *p = alloc_pinned_host(needed, is_cuda);
            if (p == nullptr) {
                throw std::runtime_error("UcxCommunicator: rx_pool grow failed");
            }
            rx_pool_->slots[i] = PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr};
            if (gpudirect_active) {
                unsigned char *gp = alloc_gpu_slot(needed);
                if (gp != nullptr) {
                    rx_pool_->slots[i].gpu_addr = gp;
                    rx_pool_->slots[i].gpu_capacity = needed;
                }
            }
            map_slot_to_ucp(context_, rx_pool_->slots[i]);
            LOG4CPLUS_DEBUG(ucx_logger, "rx_pool: grew slot " << i << " to " << needed
                            << " bytes (gpu=" << (rx_pool_->slots[i].gpu_addr ? "yes" : "no") << ")");
            return i;
        }
    }
    // All slots in use — append a new one.
    bool is_cuda = false;
    unsigned char *p = alloc_pinned_host(needed, is_cuda);
    if (p == nullptr) {
        throw std::runtime_error("UcxCommunicator: rx_pool append failed");
    }
    rx_pool_->slots.push_back(PinnedSlot{p, needed, /*in_use*/true, is_cuda, nullptr});
    size_t idx = rx_pool_->slots.size() - 1;
    if (gpudirect_active) {
        unsigned char *gp = alloc_gpu_slot(needed);
        if (gp != nullptr) {
            rx_pool_->slots[idx].gpu_addr = gp;
            rx_pool_->slots[idx].gpu_capacity = needed;
        }
    }
    map_slot_to_ucp(context_, rx_pool_->slots[idx]);
    LOG4CPLUS_DEBUG(ucx_logger, "rx_pool: appended slot " << idx << " (" << needed << " bytes, gpu="
                    << (rx_pool_->slots[idx].gpu_addr ? "yes" : "no") << "), total=" << rx_pool_->slots.size());
    return idx;
}

void UcxCommunicator::release_rx_slot(size_t slot_idx) {
    std::lock_guard<std::mutex> lk(rx_pool_->mu);
    if (slot_idx >= rx_pool_->slots.size()) return;
    rx_pool_->slots[slot_idx].in_use = false;
}


// Grow the pre-registered TX scratch to at least `needed` bytes. Must be
// called with worker_mutex_ held. Rounds capacity up to a power of two
// (≥4MB) so consecutive WriteIovs of the same size class hit the warm path.
void UcxCommunicator::ensure_tx_scratch_locked(size_t needed) {
    if (tx_scratch_.capacity >= needed) return;

    // Free previous registration + allocation if any.
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;

    // Round up to power of two, minimum 4MB.
    size_t cap = 4u * 1024u * 1024u;
    while (cap < needed) cap <<= 1;

    void *addr = nullptr;
    if (posix_memalign(&addr, 4096, cap) != 0 || addr == nullptr) {
        throw std::runtime_error("UcxCommunicator: posix_memalign failed for tx scratch");
    }

    ucp_mem_map_params_t map_params{};
    map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    map_params.address = addr;
    map_params.length = cap;

    ucp_mem_h memh = nullptr;
    ucs_status_t status = ucp_mem_map(context_, &map_params, &memh);
    if (status != UCS_OK) {
        std::free(addr);
        throw std::runtime_error("UcxCommunicator: ucp_mem_map failed: " +
                                 std::string(ucs_status_string(status)));
    }

    tx_scratch_.addr = addr;
    tx_scratch_.capacity = cap;
    tx_scratch_.memh = memh;
    LOG4CPLUS_DEBUG(ucx_logger, "tx_scratch grown capacity=" << cap);
}

void UcxCommunicator::release_tx_scratch_locked() {
    if (tx_scratch_.memh != nullptr && context_ != nullptr) {
        ucp_mem_unmap(context_, tx_scratch_.memh);
        tx_scratch_.memh = nullptr;
    }
    if (tx_scratch_.addr != nullptr) {
        std::free(tx_scratch_.addr);
        tx_scratch_.addr = nullptr;
    }
    tx_scratch_.capacity = 0;
}
