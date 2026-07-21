//
// Created by gco on 3/21/20.
//

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

// Per-thread flag: a fire-and-forget async H2D D2D was issued from a GPU shadow
// slot without synchronizing (see Communicator::drain_device_if_async_pending).
// Set by the cudart MemcpyAsync handler, drained + cleared before a response.
thread_local bool tls_async_gpu_pending = false;

}  // namespace gvirtus::communicators
