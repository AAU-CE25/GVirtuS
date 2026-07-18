#pragma once

#include <cstddef>
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

}  // namespace gvirtus::communicators
