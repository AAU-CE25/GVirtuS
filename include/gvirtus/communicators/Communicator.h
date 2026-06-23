/*
 * Communicator — abstract transport interface for GVirtuS.
 *
 * Extended for the UCX communicator feature with the following virtual methods:
 *
 *   WriteIov()  — gather-send: pass N non-contiguous iov fragments in a single
 *     call. UCX maps this to ucp_am_send_nbx with UCP_DATATYPE_IOV; other
 *     transports use the default concatenate-and-Write fallback. (Phase 4)
 *
 *   TryAcquireFrame() / ReleaseFrame()  — zero-copy frame handoff for message-
 *     oriented transports. Returns a pointer into the pinned RX-pool slot;
 *     caller parses in-place and releases when done. (Phase 4)
 *
 *   current_frame_gpu()  — exposes the GPU-resident portion of the current
 *     frame (GPUDirect B4). Handlers can route via cudaMemcpyDeviceToDevice
 *     instead of bouncing through host. (Phase 6)
 *
 *   current_connection_supports_cuda()  — per-connection RDMA transport gate.
 *     Enables mixed RDMA + TCP frontends on a single backend. (Phase 6)
 *
 *   tls_connection_supports_cuda (extern thread_local)  — per-thread flag set
 *     by Process.cpp before handler dispatch so plugins can query transport
 *     capability without coupling to UcxCommunicator. (Phase 3/6)
 *
 * Optimization phases: 3, 4, 6
 */
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

    // Send one complete, self-delimited message. Default framing for a
    // byte-stream transport: a [uint64 length][body] prefix so the peer can
    // read it back as a single frame via TryAcquireFrame(). Message-oriented
    // transports (UCX active messages) override this to use their native
    // delimiting (no length prefix) and the zero-copy send path.
    virtual size_t WriteFrame(const struct iovec *iov, size_t iov_count) {
        std::uint64_t total = 0;
        for (size_t i = 0; i < iov_count; ++i) total += iov[i].iov_len;
        Write(reinterpret_cast<const char *>(&total), sizeof(total));
        return WriteIov(iov, iov_count);
    }

    // Receive one complete message and expose it as a contiguous buffer the
    // communicator owns; the caller parses in place and calls ReleaseFrame()
    // when done. Default framing for a byte-stream transport: read the
    // [uint64 length][body] frame written by WriteFrame() into an internal
    // buffer. Returns false on a clean connection close (Read() returns 0 at
    // a message boundary). Message-oriented transports (UCX) override this to
    // hand back a pointer into their pinned receive pool (true zero-copy).
    virtual bool TryAcquireFrame(const unsigned char *&data, size_t &size) {
        std::uint64_t total = 0;
        if (!ReadFull(reinterpret_cast<char *>(&total), sizeof(total))) return false;
        _frame_buf.resize(static_cast<size_t>(total));
        if (total > 0 &&
            !ReadFull(reinterpret_cast<char *>(_frame_buf.data()), static_cast<size_t>(total)))
            return false;
        data = _frame_buf.data();
        size = static_cast<size_t>(total);
        return true;
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

   private:
    // Loop Read() until `n` bytes are received; false on clean close (0 read).
    bool ReadFull(char *buf, size_t n) {
        size_t got = 0;
        while (got < n) {
            size_t r = Read(buf + got, n - got);
            if (r == 0) return false;
            got += r;
        }
        return true;
    }

    // Backing store for the base-class (stream) TryAcquireFrame().
    std::vector<unsigned char> _frame_buf;
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
