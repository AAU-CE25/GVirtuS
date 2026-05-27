//
// Created by gco on 3/21/20.
//

/*
 * CommunicatorFactory — transport selection and shared state.
 *
 * UCX additions:
 *   tls_connection_supports_cuda: thread-local flag set by Process.cpp's
 *   dispatch loop before each handler invocation. Allows GPU-aware plugins
 *   (e.g. CudaRtHandler_memory) to query whether the current connection's
 *   negotiated transport supports CUDA peer-DMA, without linking against the
 *   UCX communicator library. Decouples handler code from transport internals.
 *
 * Optimization phases: 3 (handler TLS slot), 6 (per-connection GPUDirect gate)
 */

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/Communicator.h>

namespace gvirtus::communicators {

// Per-thread per-connection transport capability flag. Set by
// Process.cpp's UCX-AM dispatch loop before each handler Execute(), read
// by GPU-aware handlers (CudaRtHandler_memory) to gate Variant A
// activation. Declared extern in Communicator.h. Default false is the
// safe value for stream-oriented transports and for the brief window
// before Process.cpp sets it on each dispatch.
thread_local bool tls_connection_supports_cuda = false;

}  // namespace gvirtus::communicators
