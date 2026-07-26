#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "gvirtus/communicators/Communicator.h"

#include <ucp/api/ucp.h>

namespace gvirtus::communicators {

class UcxCommunicator : public Communicator {
   public:
    UcxCommunicator() = default;
    UcxCommunicator(const std::string &hostname, std::uint16_t port);
    ~UcxCommunicator() override;

    void Serve() override;
    const Communicator *const Accept() const override;
    void Connect() override;
    size_t Read(char *buffer, size_t size) override;
    size_t Write(const char *buffer, size_t size) override;
    void Sync() override;
    void Close() override;
    void run() override;

    // Override of Communicator::WriteIov — uses UCX's UCP_DATATYPE_IOV so
    // ucp_am_send_nbx gathers the fragments natively without staging.
    size_t WriteIov(const struct iovec *iov, size_t iov_count) override;

    // Zero-copy frame handoff: backends that read a whole AM message at once
    // can call this to get a pointer into the pinned RX pool slot directly,
    // skipping the per-message std::vector allocation in the Read() byte
    // stream. Must be paired with ReleaseFrame() once the caller is done.
    bool TryAcquireFrame(const unsigned char *&data, size_t &size) override;

    // GPUDirect (Variant B Step B4). After a successful TryAcquireFrame the
    // current frame may have an associated GPU-resident payload (set by
    // am_recv_handler when the peer used the Step B3 GPU-split wire format).
    // Process.cpp reads these via the virtual Communicator::current_frame_gpu
    // and attaches them to the input Buffer; GPU-aware handlers route via
    // cudaMemcpyDeviceToDevice instead of HostToDevice, skipping the host bounce.
    void current_frame_gpu(void *&gpu, std::size_t &size) const override {
        gpu  = current_frame_.gpu_data;
        size = current_frame_.gpu_size;
    }
    void ReleaseFrame() override;

    // Per-connection transport capability (Option 2 — per-connection
    // GPUDirect gate). Lazy: on the first call after wire-up has produced
    // a usable endpoint, ucp_ep_query(TRANSPORTS) returns the negotiated
    // lanes; we scan for an RDMA-class transport name. Result is cached
    // and reused for the lifetime of this UcxCommunicator (one connection).
    //
    // Returns false until the global GVIRTUS_GPUDIRECT probe has succeeded
    // (g_gpudirect_enabled). This means: even if the endpoint has an RDMA
    // lane, we won't try GPUDirect ops unless the process was started with
    // GVIRTUS_GPUDIRECT=1 and the probe passed.
    bool current_connection_supports_cuda() const override;

    // See Communicator::SetNextDeviceFragment. Records the address+length of
    // the next WriteIov's Fase-5 device-destined direct-input fragment;
    // WriteIovRma consumes it (once) to gate B3 GPU-shadow routing.
    void SetNextDeviceFragment(const void *addr, size_t len) override {
        next_dev_frag_addr_ = addr;
        next_dev_frag_len_  = len;
    }

    // See Communicator::drain_device_if_async_pending. Blocks on
    // cudaDeviceSynchronize iff a fire-and-forget async H2D D2D is pending on
    // this thread; called by Process.cpp before every response-bearing reply.
    void drain_device_if_async_pending() override;
    void NoteDeviceDestinedPayload(size_t bytes) override;

    // See Communicator::rma_put_capable. True iff RmaSetup was received AND at
    // least one remote slot has a usable (non-null) rkey we can ucp_put into.
    bool rma_put_capable() const override;

    // See Communicator::PrepareGpuGet / GetFromRemoteGpu — D2H-via-GET.
    bool PrepareGpuGet(void *gpu_addr, size_t len, std::uint64_t &out_remote_addr,
                       std::vector<char> &out_rkey) override;
    bool GetFromRemoteGpu(void *dst_host, std::uint64_t remote_addr,
                          const void *rkey_blob, size_t rkey_len,
                          size_t count) override;

    // RMA flow-control introspection for the async dispatcher (Phase 2).
    size_t rma_slot_count() const override {
        return rma_setup_received_.load() ? remote_slots_.size() : 0;
    }
    bool rma_uses_slots(size_t bytes) const override {
        // Mirror WriteIov's RMA-path gate: large enough payload + setup done.
        return bytes >= kRmaThresholdBytes && rma_setup_received_.load() &&
               !remote_slots_.empty();
    }

    std::string to_string() override { return "ucxcommunicator"; }

   private:
    // === Nested types — defined first so member declarations below can use them ===

    // Pre-registered TX scratch — page-aligned host buffer mapped with
    // ucp_mem_map. Large WriteIov payloads stage into it so the subsequent
    // ucp_am_send_nbx can pass the memh hint and let UCX skip its internal
    // rndv-fragment staging (UCX_RNDV_FRAG_SIZE chunking). Grows monotonically
    // by size class. Protected by worker_mutex_.
    struct TxScratch {
        void *addr{nullptr};
        size_t capacity{0};
        ucp_mem_h memh{nullptr};
    };

    // Slot in the RX pinned pool. addr is allocated via cudaHostAlloc (when
    // libcudart is dlopen-able) so that the subsequent cudaMemcpy on the
    // backend runs at full PCIe rate, and never zero-initialized — payload
    // memcpy from UCX overwrites only the used prefix. memh is the UCX
    // registration so that true-rendezvous RDMA into this slot doesn't pay
    // an on-the-fly registration cost (and ucp_am_recv_data_nbx can be
    // hinted with the handle).
    struct PinnedSlot {
        unsigned char *addr{nullptr};
        size_t capacity{0};
        bool in_use{false};
        bool is_cuda_host{false};
        ucp_mem_h memh{nullptr};

        // Explicit-ownership bookkeeping for RMA-origin slots (filled by a
        // remote ucp_put + RmaPosted). When this slot is released (fully
        // consumed by the backend), we send a SlotConsumed ack back to the
        // client carrying rma_generation so it can free the matching remote
        // slot only if the generation still matches (ABA guard).
        bool rma_origin{false};
        std::uint64_t rma_generation{0};

        // True for slots provisioned as the RMA pool (init_rx_pool). These are the only
        // slots a peer may ucp_put into, and acquire_rx_slot() must never hand one to an
        // incoming eager AM: the peer's data lands by peer-DMA without the server being
        // involved, so an AM given the same slot would overwrite it.
        bool rma_persistent{false};

        // Epoch of the advertisement this slot was published under, and whether a
        // later, larger pool has superseded it. A retired slot is advertised to
        // nobody and handed to nobody -- not to the RMA path (the client has a new
        // layout) and not to eager AMs (a client that has not yet seen the new
        // advertisement may still have a put in flight to this address). It is
        // freed one epoch later, in retire_and_free_locked().
        std::uint32_t rma_epoch{0};
        bool rma_retired{false};

        // Optional GPU shadow region for GPUDirect (Variant B, Step B1).
        // Allocated only when GVIRTUS_GPUDIRECT=1 and probe passed. Lives
        // alongside the host `addr`. Frontend will eventually be told the
        // gpu_addr+rkey via RmaSetup (Step B2) and route big payloads here
        // directly via NIC peer-DMA (Step B3). Step B1 only allocates +
        // registers — nothing reads/writes gpu_addr yet.
        unsigned char *gpu_addr{nullptr};
        size_t gpu_capacity{0};
        ucp_mem_h gpu_memh{nullptr};
    };

    // Message handed off from the AM handler to consumers (Read or
    // TryAcquireFrame). `slot_idx` is the index back into rx_slots_ used to
    // release the slot when the message is fully consumed.
    struct PooledMsg {
        unsigned char *data{nullptr};
        size_t size{0};
        size_t slot_idx{static_cast<size_t>(-1)};  // -1 if not from pool (e.g., empty)

        // GPUDirect Step B3: when the peer split the payload across host +
        // GPU shadows, this points into the slot's GPU region. The byte range
        // [data + (size - gpu_size), data + size) of the LOGICAL message lives
        // in gpu_data[0 .. gpu_size). Consumers can either: (a) consolidate
        // via cudaMemcpy D2H into the host hole (B3 default — preserves the
        // existing parser path at a cost of ~3ms warm), or (b) read GPU
        // directly via Buffer's dual-aware AssignAll (B4 optimization).
        unsigned char *gpu_data{nullptr};
        size_t gpu_size{0};
    };

    struct PendingAmRecv {
        void *request{nullptr};
        PooledMsg msg;
    };

    struct AmState {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<PooledMsg> queue;
        std::vector<PendingAmRecv> rndv;
    };

    // === Methods ===
    static void listener_conn_handler(ucp_conn_request_h conn_request, void *arg);
    static void endpoint_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);
    static ucs_status_t am_recv_handler(void *arg, const void *header, size_t header_length,
                                        void *data, size_t length,
                                        const ucp_am_recv_param_t *param);

    void init_ucx();
    void destroy_ucx();
    void enqueue_connection(ucp_conn_request_h conn_request);
    ucp_conn_request_h wait_for_connection_request();
    void wait_request_completion(void *request, const char *op_name);
    void enqueue_am_message(PooledMsg message);
    void enqueue_am_rndv(void *request, PooledMsg msg);
    void progress_am_rndv();
    static sockaddr_storage make_sockaddr(const std::string &host, std::uint16_t port);

    TxScratch tx_scratch_;
    void ensure_tx_scratch_locked(size_t needed);
    void release_tx_scratch_locked();

    // RX pool of pre-allocated, pinned (cudaHostAlloc'd when libcudart is
    // available) host buffers. The AM receive handler grabs a slot and
    // memcpys the incoming payload into it; the slot travels through the
    // queue and is released back to the pool when the consumer finishes
    // with it. Avoids the per-message std::vector(N) zero-init (~30ms for
    // 64MB) and ensures cudaMemcpy from the slot is at PCIe line rate.
    //
    // Shared between the listener UcxCommunicator and the accepted ones it
    // hands out from Accept() — the AM receive callback is registered on
    // the listener's worker but messages are consumed by the accepted's
    // Read/TryAcquireFrame, so both sides must see the same slot identity
    // to acquire and release correctly.
    struct RxPool {
        // deque, not vector: the AM receive handler takes a PinnedSlot& and uses it
        // (slot.addr, slot.memh, slot.rma_origin) after acquire_rx_slot() has released
        // the pool mutex, while other threads may append -- acquire_rx_slot's on-demand
        // append, and the on-demand pool build. A vector reallocation invalidates every
        // reference into it, which is undefined behaviour and shows up as intermittent
        // data corruption at unpredictable transfers. deque never invalidates references
        // to existing elements on push_back/resize, and indexing is unchanged.
        std::deque<PinnedSlot> slots;
        std::mutex mu;
    };
    std::shared_ptr<RxPool> rx_pool_{std::make_shared<RxPool>()};
    PooledMsg current_frame_;  // held between TryAcquireFrame/ReleaseFrame

    void init_rx_pool();
    // Build the pool and advertise it, from a context where worker_mutex_ is NOT held.
    void materialise_rma_pool();
    // Set when a message large enough for the RMA path arrives on a connection whose
    // pool has not been built yet; consumed at the top of Read().
    std::atomic<bool> rma_pool_requested_{false};
    std::atomic<bool> rma_pool_ready_{false};
    void destroy_rx_pool();
    size_t acquire_rx_slot(size_t needed);   // returns slot_idx, grows pool if all busy
    void release_rx_slot(size_t slot_idx);   // marks slot free

    // ucp_mem_map / unmap helpers that need access to PinnedSlot (private nested
    // type), hence static members rather than file-scope free functions.
    static void map_slot_to_ucp(ucp_context_h ctx, PinnedSlot &slot);
    static void unmap_slot_from_ucp(ucp_context_h ctx, PinnedSlot &slot);

    // === RMA mode ===
    // Server -> client at connection time: pack rx_slots_ rkeys and send to
    // the client so it can issue ucp_put_nbx directly to them. Client unpacks
    // and populates remote_slots_. After this exchange, large WriteIov calls
    // can take the RMA data path (ucp_put + tiny RmaPosted AM) instead of
    // the AM-stream path.
    struct RemoteSlot {
        std::uint64_t addr{0};       // remote address (server's view)
        std::uint64_t capacity{0};
        ucp_rkey_h rkey{nullptr};    // unpacked on this side

        // The server's OWN index for this slot, carried in the top 16 bits of
        // RmaSlotDescriptor::reserved0. Slot identity used to be positional -- the
        // client's slot i had to be the server's slot i -- which forced the server
        // to advertise every slot it owned, including the message-sized ones the
        // eager path appends and the ones it has retired, and silently broke if
        // ucp_rkey_pack failed for any slot (send_rma_setup `continue`s, shifting
        // every later index). RmaPosted/SlotConsumed now travel with this id, so
        // the server can advertise exactly the slots that are usable.
        std::uint16_t server_idx{0};

        // Set when the peer tagged this slot as part of its persistent RMA pool
        // (descriptor.reserved0 bit 1). Only those are safe to ucp_put into: the
        // peer hands its temporary slots to incoming eager AMs, which would land on
        // top of our transfer.
        bool persistent{false};

        // Optional GPU shadow advertised by peer (Variant B, Step B2). When
        // gpu_rkey != nullptr the client can ucp_put_nbx big payloads to
        // gpu_addr (with this rkey) and the NIC will peer-DMA into the
        // server's GPU memory (peermem). nullptr / 0 → peer didn't advertise
        // a GPU shadow → keep host-only path.
        std::uint64_t gpu_addr{0};
        std::uint64_t gpu_capacity{0};
        ucp_rkey_h gpu_rkey{nullptr};

        // Explicit ownership state machine (client side), placed AFTER the wire
        // fields so the positional aggregate-init in handle_rma_setup_am still
        // maps 1:1 (state/generation take their defaults there). A slot is Free
        // until WriteIovRma ACQUIRES it (Free->InFlight, generation bumped) and
        // returns to Free only on the backend's SlotConsumed ack — NOT on the
        // local UCX put completion.
        enum class State : std::uint8_t { Free, InFlight };
        State state{State::Free};
        std::uint64_t generation{0};
    };

    std::vector<RemoteSlot> remote_slots_;
    std::mutex rma_state_mu_;

    // --- Re-advertisement (epoch + quiesce), client side -------------------
    // The server may publish a new slot layout mid-connection: the pool is built
    // on demand and grown again when a message does not fit. Swapping
    // remote_slots_ the moment a new RmaSetup lands is NOT safe -- a slot that is
    // InFlight would silently come back Free in the new vector while the backend
    // is still consuming it, and the client would ucp_put on top of a live
    // transfer. So a layout that arrives while anything is InFlight is parked
    // here and applied by release_remote_slot() once the last one drains.
    std::uint32_t remote_epoch_{0};        // epoch of the layout in remote_slots_
    std::vector<RemoteSlot> pending_slots_;
    std::uint32_t pending_epoch_{0};
    bool rma_swap_pending_{false};
    // How many advertisements had to be parked. Reported unconditionally so a
    // concurrency test can prove the quiesce path executed.
    std::uint64_t rma_swap_parked_count_{0};
    // Apply pending_slots_ to remote_slots_. Caller must hold rma_state_mu_ AND
    // must have established that no slot is InFlight.
    void apply_pending_slots_locked();

    // --- On-demand sizing, server side -------------------------------------
    // Largest payload seen on this connection at or above the RMA floor. The pool
    // is sized from this instead of from the GVIRTUS_RMA_SLOT_CAP_MB knob, which
    // becomes a ceiling rather than a target: a connection whose biggest message
    // is 8 MB should not pin 2 x 1025 MB (plus the same again in GPU shadow) and
    // pay ~2.2 s of connect for buffers it can never fill.
    std::atomic<size_t> rma_pool_hint_bytes_{0};
    // Whether this connection has PROVEN it moves device-destined bulk data, i.e.
    // whether its slots are worth a GPU shadow. A shadow doubles the pool's cost in
    // DEVICE memory, and on a multi-tenant backend allocating one speculatively for
    // every connection is what exhausts the GPU and takes down other tenants. Set by
    // NoteDeviceDestinedPayload the first time such a payload has to be staged through
    // the host slot; the pool then regrows with shadows through the ordinary
    // materialise/re-advertise path. GVIRTUS_RMA_GPU_SHADOW_EAGER=1 restores the old
    // always-allocate behaviour.
    std::atomic<bool> rma_want_gpu_shadow_{false};
    // Whether the live persistent slots actually have shadows, so a regrow is
    // triggered when the answer changes rather than only when the size does.
    std::atomic<bool> rma_pool_has_shadow_{false};
    // Capacity the persistent slots were actually built at, and the epoch under
    // which they were last advertised.
    std::atomic<size_t> rma_pool_cap_{0};
    std::atomic<std::uint32_t> rma_pool_epoch_{0};
    // Serializes materialise_rma_pool() against itself (build + advertise must be
    // atomic with respect to a second grow request).
    std::mutex rma_build_mu_;
    // Persistent slots that a previous epoch advertised and that have since been
    // superseded. They are NOT freed at the moment they are retired: a client that
    // has not yet seen the new advertisement may still have a put in flight to
    // them. They are freed one epoch later, by which point every put was addressed
    // against a layout the client has acknowledged by using it.
    std::vector<size_t> retired_slot_idx_;
    void retire_and_free_locked(std::uint32_t now_epoch);
    // Highest layout epoch we have seen the peer actually USE (the epoch tag on an
    // incoming RmaPosted). Once the peer posts against epoch N, it demonstrably holds
    // that advertisement, so everything retired when N was published can be released --
    // no in-flight put can still be addressed at it. Strictly better than waiting a
    // whole extra epoch: that only reclaimed on the NEXT regrow, so a connection that
    // regrew once and then ran steady kept the superseded generation allocated for its
    // entire life.
    std::atomic<std::uint32_t> rma_epoch_acked_{0};
    std::atomic<bool> rma_reclaim_requested_{false};
    // Reclaim at a safe point. NOT done in the AM callback that observes the epoch:
    // that runs under worker_mutex_ and freeing a slot calls cudaFreeHost/cudaFree,
    // which can block and would stall worker progress.
    void reclaim_retired_slots();
    // Slot capacity to build for a given observed payload size.
    static size_t rma_slot_cap_for(size_t hint_bytes);
    // Backpressure: WriteIovRma waits here when every remote slot is InFlight,
    // and release_remote_slot (driven by the backend's SlotConsumed ack) wakes
    // a waiter. Guards against slot reuse-before-consumption (the old
    // round-robin race that crashed under concurrent prefill).
    std::condition_variable rma_slot_cv_;
    // Client side: free a remote slot on backend SlotConsumed ack (ABA-guarded
    // by generation). Server side: notify the client a consumed RMA slot is free.
    void release_remote_slot(size_t slot_idx, std::uint64_t generation);
    void send_slot_consumed(size_t slot_idx, std::uint64_t generation);
    // Cached at RmaSetup time (handle_rma_setup_am): true iff a remote slot has
    // a usable rkey. rma_put_capable() returns this with zero per-RPC cost.
    std::atomic<bool> rma_put_capable_{false};
    std::condition_variable rma_setup_cv_;
    std::atomic<bool> rma_setup_received_{false};
    size_t next_remote_slot_idx_{0};
    // Payload size (bytes) at/above which WriteIov takes the RMA slot path.
    static constexpr size_t kRmaThresholdBytes = 64u * 1024u;

    void send_rma_setup();                                   // server side
    void handle_rma_setup_am(const void *data, size_t length); // client side
    void destroy_rma_state();                                 // teardown

    // NOTE: gpu_get_regs_ (server GPU scratch) and client_dst_regs_ (client D2H
    // destination) used to live here, each a private address-keyed ucp_mem_map cache
    // with its own mutex and its own comment saying it was deliberately "not unmapped
    // at teardown (process-lifetime)". Both were wrong on both counts: neither could be
    // invalidated when the buffer was freed, and surviving into ucp_cleanup is what
    // corrupted the heap at exit. They are gone, replaced by one process-global
    // registry with an invalidation hook (RunRegistrationInvalidate) and a teardown
    // sweep (release_registrations_for_context), shared with the H2D source cache.

    // Client-side data path: stage iov fragments into tx_scratch_, RDMA-put
    // into the next remote slot, then send a small RmaPosted AM with the
    // slot index. Returns total bytes (== sum of iov lengths) on success.
    // Falls back to the IOV path if remote_slots_ is empty or the message
    // doesn't fit in any remote slot.
    size_t WriteIovRma(const struct iovec *iov, size_t iov_count, size_t total);

    std::string hostname_;
    std::uint16_t port_{};
    ucp_context_h context_{nullptr};
    ucp_worker_h worker_{nullptr};
    ucp_listener_h listener_{nullptr};
    ucp_ep_h endpoint_{nullptr};

    std::atomic<bool> running_{false};
    // Per-resource ownership flags. Listener owns all three; each Accept()
    // creates a NEW instance that owns its own worker but shares the
    // context with the listener (and doesn't own the listener at all).
    // Without this split, accepted connections would either leak (don't
    // destroy own worker) or double-free (try to destroy listener's).
    bool owns_context_{true};
    bool owns_worker_{true};
    bool owns_listener_{true};
    bool initialized_{false};

    mutable std::mutex conn_mutex_;
    mutable std::condition_variable conn_cv_;
    mutable std::queue<ucp_conn_request_h> pending_conn_requests_;
    std::shared_ptr<std::mutex> worker_mutex_{std::make_shared<std::mutex>()};

    unsigned am_id_{1};
    std::shared_ptr<AmState> am_state_{std::make_shared<AmState>()};

    PooledMsg pending_msg_;
    size_t pending_read_offset_{0};
    std::atomic<bool> endpoint_failed_{false};

    // Per-connection GPUDirect gate cache. -1 = not yet queried (lazy on
    // first current_connection_supports_cuda() call), 0 = TCP/non-RDMA
    // negotiated, 1 = RDMA-class transport negotiated. mutable because
    // current_connection_supports_cuda() is logically const.
    mutable std::atomic<int> supports_cuda_cached_{-1};

    // Control/data-path gate for B3. Set per-message by Frontend::Execute via
    // SetNextDeviceFragment (nullptr for control-path messages), consumed once
    // in WriteIovRma. Only the fragment matching this addr+len may be routed to
    // the peer GPU shadow. Not atomic: one frontend connection == one thread.
    const void *next_dev_frag_addr_ = nullptr;
    size_t      next_dev_frag_len_  = 0;
};

}  // namespace gvirtus::communicators
