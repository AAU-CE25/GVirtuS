#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <sys/uio.h>

#include "Endpoint.h"

namespace gvirtus::communicators {
/**
 * Communicator is an abstract class that implements a simple stream oriented
 * mechanism for communicating with two end points.
 * Communicator use a client/server approach, for having a Communicator server
 * the application must call Serve() and the Accept() for accepting the
 * connection by clients and communicating to them.
 * The client has to use just the Connect() method.
 * For sending and receiving data through the communicator is possible the use
 * the input and output stream. Warning: _never_ try to communicate through the
 * streams of a server Communicator, for communicating with the client the
 * Communicator returned from the Accept() must be used.
 */
class Communicator {
   public:
    /**
     * Creates a new communicator. The real type of the communicator and his
     * parameters are obtained from the ConfigFile::Element @arg config.
     *
     * @param config the ConfigFile::Element that stores the configuration.
     *
     * @return a new Communicator.
     */

    virtual ~Communicator() = default;

    /**
     * Sets the communicator as a server.
     */
    virtual void Serve() = 0;

    /**
     * Accepts a new connection. The call to the first Accept() must follow a
     * call to Serve().
     *
     * @return a Communicator to the connected peer.
     */
    virtual const Communicator *const Accept() const = 0;

    /**
     * Sets the communicator as a client and connects it to the end point
     * specified in the ConfigFile::Element used to build this Communicator.
     */
    virtual void Connect() = 0;

    virtual size_t Read(char *buffer, size_t size) = 0;
    virtual size_t Write(const char *buffer, size_t size) = 0;

    // Gather-send: allows callers to send a logically-contiguous message
    // assembled from N non-contiguous fragments without concatenating them
    // first. Concrete UCX-style transports can map this to a single
    // ucp_am_send_nbx with UCP_DATATYPE_IOV, avoiding host-RAM staging for
    // large payloads (e.g. cudaMemcpy of 64MB). The default fallback below
    // preserves correctness for transports that don't support scatter — at
    // the cost of one concatenation memcpy.
    virtual size_t WriteIov(const struct iovec *iov, size_t iov_count) {
        if (iov == nullptr || iov_count == 0) return 0;
        size_t total = 0;
        for (size_t i = 0; i < iov_count; ++i) total += iov[i].iov_len;
        std::vector<char> buf(total);
        size_t off = 0;
        for (size_t i = 0; i < iov_count; ++i) {
            std::memcpy(buf.data() + off, iov[i].iov_base, iov[i].iov_len);
            off += iov[i].iov_len;
        }
        return Write(buf.data(), total);
    }

    // Zero-copy frame handoff for transports that buffer entire messages
    // internally (e.g. UCX active messages). If the implementation can
    // expose the next received message as a contiguous buffer it owns,
    // it returns true and sets `data`/`size`. The caller must then call
    // ReleaseFrame() when done to return the buffer to the underlying pool.
    // Default no-op fallback: returns false, forces callers to use the
    // byte-stream Read() path. Stream-oriented transports (TCP, etc.) keep
    // working with the default.
    virtual bool TryAcquireFrame(const unsigned char *&data, size_t &size) {
        (void)data; (void)size;
        return false;
    }
    virtual void ReleaseFrame() {}

    // A backend plugin may leave device work in flight when it finishes consuming a
    // frame (the GPUDirect shadow -> destination copy). The transport must not tell
    // the peer that the frame's slot is free until that work completes, or the peer
    // overwrites the slot while the copy engine is still reading it. The plugin
    // registers a drain function here and the transport calls it immediately before
    // releasing a frame -- which is *after* the response has been sent, so the wait
    // is off the client's critical path.

    // GPUDirect (Variant B Step B4): after a successful TryAcquireFrame, a
    // transport that supports GPU-resident payload landing (UCX with
    // GPUDirect) may have an additional GPU pointer + size associated with
    // the current frame. Default implementation returns no GPU payload —
    // stream-oriented transports never have one.
    virtual void current_frame_gpu(void *&gpu, std::size_t &size) const {
        gpu = nullptr;
        size = 0;
    }

    // Per-connection transport capability: true iff this specific endpoint
    // negotiated an RDMA-class transport (rc_mlx5 / dc_mlx5 / ud_mlx5 / ib)
    // capable of peer-DMA from CUDA memory. Default false is safe for all
    // non-UCX transports and for UCX endpoints whose wire-up has not yet
    // completed. UcxCommunicator overrides with a lazy ucp_ep_query.
    //
    // Supersedes the process-wide GVIRTUS_GPUDIRECT_ACTIVE env gate for
    // per-call activation decisions: a single backend with
    // UCX_TLS=rc_mlx5,ud_mlx5,tcp,self can now accept both RDMA and TCP
    // frontends concurrently, enabling GPUDirect only on the connections
    // that actually negotiated an RDMA lane.
    virtual bool current_connection_supports_cuda() const { return false; }

    // RMA data-path flow control (async dispatcher, Phase 2). Number of remote
    // RX slots available for the zero-copy WriteIovRma path (0 if the transport
    // has no RMA fast path or it isn't set up yet). The frontend uses this to
    // bound outstanding fire-and-forget large-H2D copies so it never reuses a
    // remote slot the backend hasn't consumed yet.
    virtual size_t rma_slot_count() const { return 0; }
    // True iff a request whose wire payload is `bytes` would travel on the RMA
    // slot path (rather than the AM path), i.e. it consumes a remote slot.
    virtual bool rma_uses_slots(size_t bytes) const {
        (void)bytes;
        return false;
    }

    virtual void Sync() = 0;

    /**
     * Closes the connection with the end point.
     */
    virtual void Close() = 0;

    virtual std::string to_string() { return "communicator"; }

    virtual void run() {};

    // Per-message hint set by Frontend::Execute right before WriteIov: the
    // address+length of a Fase-5 device-destined "data-path" fragment
    // (mDirectInputSrc). Only this fragment may be peer-DMA'd into the peer
    // GPU shadow (GPUDirect Step B3); everything else — fatbin, module blobs,
    // nvrtc, marshaled args (control-path) — must land in the host slot,
    // because only the shadow-aware backend handler (sync cudaMemcpy H2D)
    // reads GetGpuPayload(). Default no-op → control-path only. UcxCommunicator
    // overrides; TCP/Hybrid ignore it.
    //
    // Declared LAST among the virtuals on purpose: appending (rather than
    // inserting mid-class) keeps every pre-existing vtable slot index stable,
    // so a binary built against the older header stays ABI-compatible with one
    // built against this header — important given the mixed baked/mounted libs
    // in this project's docker + native runs.
    virtual void SetNextDeviceFragment(const void * /*addr*/, size_t /*len*/) {}

    // Async H2D (Phase 3): when a fire-and-forget cudaMemcpyAsync H2D peer-DMAs
    // into a GPU shadow slot, the backend handler issues the D2D on the client
    // stream WITHOUT synchronizing (it sets tls_async_gpu_pending) so consecutive
    // copies overlap. Because the frontend's RMA-slot flow control treats ANY
    // synchronous reply as "all prior slots drained" (Frontend.cpp resets
    // mAsyncRmaInflight on every sync response), the backend must drain the
    // device before writing a response-bearing reply, or a reused slot could be
    // peer-DMA'd over while an in-flight D2D still reads it. Process.cpp calls
    // this right before write_ucx_am_response; it is a no-op unless a fire-and-
    // forget GPU copy is pending on this thread. Default no-op (non-CUDA
    // transports never set the flag). Appended LAST to preserve vtable ABI.
    virtual void drain_device_if_async_pending() {}

    // True iff this connection's peer advertised RMA slots whose rkey this side
    // could unpack (so ucp_put into them will succeed). The backend reads it to
    // decide whether the D2H GPUDirect response path (device fragment -> peer
    // slot) is deliverable, or it must fall back to the host path. Default false
    // (non-RMA transports). Appended LAST to preserve vtable ABI.
    virtual bool rma_put_capable() const { return false; }

    // D2H-via-GET (see UcxAmProtocol kEnvelopeFlagD2HGet). Server side: register
    // `gpu_addr[0..len)` for remote RDMA-READ and pack its rkey into `out_rkey`,
    // returning the device address in `out_remote_addr`. Lets the client issue an
    // RDMA GET instead of the server ucp_put-ing from cuda (which UCX can't build
    // under the forced rcache-off config). Returns false when unsupported (non-UCX
    // transport, GPUDirect off, or registration fails) — caller keeps the legacy
    // put/host path. Appended LAST to preserve vtable ABI.
    virtual bool PrepareGpuGet(void * /*gpu_addr*/, size_t /*len*/,
                               std::uint64_t & /*out_remote_addr*/,
                               std::vector<char> & /*out_rkey*/) {
        return false;
    }

    // Client side of D2H-via-GET: RDMA-GET `count` bytes from the server's
    // registered GPU scratch (`remote_addr` + packed rkey blob) straight into the
    // caller's host buffer `dst_host`. Returns false on failure. Appended LAST to
    // preserve vtable ABI.
    virtual bool GetFromRemoteGpu(void * /*dst_host*/, std::uint64_t /*remote_addr*/,
                                  const void * /*rkey_blob*/, size_t /*rkey_len*/,
                                  size_t /*count*/) {
        return false;
    }

   private:
};

using create_t = std::shared_ptr<Communicator>(std::shared_ptr<Endpoint>);

// Per-thread flag set by Process.cpp's UCX-AM dispatch loop immediately
// before invoking a handler's Execute() — captures whether the active
// connection's negotiated transport supports CUDA peer-DMA. Plugins
// (e.g. libgvirtus-plugin-cudart's CudaRtHandler_memory) read it via a
// plain extern, decoupled from any specific Communicator subclass.
//
// Definition lives in CommunicatorFactory.cpp (part of libgvirtus-
// communicators which both backend and plugins link against).
extern thread_local bool tls_connection_supports_cuda;

// Per-thread flag set by libgvirtus-plugin-cudart's MemcpyAsync handler when it
// issues a fire-and-forget async H2D D2D from a GPU shadow slot without
// synchronizing. Consumed (drained + cleared) by drain_device_if_async_pending()
// before the backend writes a response-bearing reply. Definition lives in
// CommunicatorFactory.cpp alongside tls_connection_supports_cuda.
extern thread_local bool tls_async_gpu_pending;

// Per-thread flag set by Process.cpp before each handler Execute() to
// client_comm->rma_put_capable(). The cudart D2H handler reads it to gate the
// GPU-scratch response path (deliverable only when the client can receive a
// ucp_put). Definition in CommunicatorFactory.cpp.
extern thread_local bool tls_client_rma_put_capable;

}  // namespace gvirtus::communicators

namespace gvirtus {
namespace communicators {
using FrameDrainFn = void (*)();
void SetFrameDrainHook(FrameDrainFn fn);
void RunFrameDrainHook();

// --- Registration lifetime -------------------------------------------------
// The UCX transport caches ucp_mem_h registrations for large H2D SOURCE buffers,
// keyed by address, because re-registering a 64 MB source per call halves H2D
// throughput (measured 23.4 -> 11.5 GB/s). An address-keyed cache is only safe if
// something invalidates it when the application frees that address -- otherwise the
// allocator hands the same address back, the stale handle still describes the old
// mapping, and the transfer silently reads or writes the wrong pages. That is
// exactly the defect fixed on the D2H destination side (a86b1ec).
//
// UCX would normally do this itself through rcache/UCM memory hooks, but this
// deployment forces UCX_RCACHE_ENABLE=n for the CUDA memtype workaround, so the
// invalidation has to come from us. Two hooks, both installed by the UCX
// communicator and called by the CUDA frontend:
//
//   InvalidateRegistration  - called from cudaFree / cudaFreeHost, drops any cached
//                             registration covering that address.
//   RegistrationCacheable   - asked before caching. Only buffers whose free we will
//                             actually observe may be cached; a plain malloc'd host
//                             buffer must not be, because free() is invisible to us
//                             and glibc mmaps large allocations at repeatable
//                             addresses.
using RegistrationInvalidateFn = void (*)(const void *addr);
void SetRegistrationInvalidateHook(RegistrationInvalidateFn fn);
void RunRegistrationInvalidate(const void *addr);

using RegistrationCacheableFn = bool (*)(const void *addr, size_t len);
void SetRegistrationCacheableHook(RegistrationCacheableFn fn);
bool RegistrationCacheable(const void *addr, size_t len);
}  // namespace communicators
}  // namespace gvirtus
