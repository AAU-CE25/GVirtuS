/*
 * GVirtuS RPC envelope codec.
 *
 * Both directions of the RPC wire format (Protocol.h EnvelopeHeader) live
 * here, so neither the frontend nor the backend touches raw bytes: the
 * backend dispatch loop only routes requests to handlers, the frontend only
 * orchestrates marshal/unmarshal, and every Communicator just moves frames
 * (TryAcquireFrame / WriteFrame).
 *
 * The codec is transport-agnostic: it never inspects which Communicator it is
 * given. Zero-copy active-message framing vs length-prefixed byte-stream
 * framing is entirely behind the Communicator interface.
 */
#pragma once

#include <sys/uio.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "Buffer.h"
#include "Communicator.h"
#include "Protocol.h"

namespace gvirtus::communicators::am {

// ---- Backend side --------------------------------------------------------

// Receive one request. On success: `header` is filled, `routine` holds a copy
// of the routine name, and payload_data/payload_size are VIEWS into the
// communicator's frame (valid until ReleaseFrame()). `owns_frame` is true and
// the caller MUST call ReleaseFrame() once the handler is done with the
// payload. gpu_payload/gpu_payload_size surface a GPU-resident tail when the
// transport supports it (GPUDirect); null/0 otherwise.
//
// Returns false on a clean disconnect or a protocol error (`error` set).
bool ReadRequest(Communicator *c, am::EnvelopeHeader &header, std::string &routine,
                 const unsigned char *&payload_data, std::size_t &payload_size,
                 void *&gpu_payload, std::size_t &gpu_payload_size, bool &owns_frame,
                 std::string &error);

// Build and send the response frame for a completed request. The response
// body is [exec_sec][wire_out_size][output bytes (+ optional GPU tail)].
// Returns false on a transport error (`error` set).
bool WriteResponse(Communicator *c, const am::EnvelopeHeader &request_header, int exit_code,
                   double server_exec_sec, const std::shared_ptr<Buffer> &output_buffer,
                   void *gpu_payload, std::size_t gpu_payload_size, std::string &error);

// ---- Frontend side -------------------------------------------------------

// Build the request envelope (header + routine) and gather-send it together
// with the caller-provided payload IoV via Communicator::WriteFrame(). The
// codec owns the header; `payload_logical_size` is the byte count to record
// in the header (may differ from the sum of iov lens when the buffer carries
// borrowed/zero-copy segments — Buffer::GetLogicalSize() supplies it).
// Returns false on a transport error (`error` set).
bool WriteRequest(Communicator *c, std::uint64_t request_id, const std::string &routine,
                  const struct iovec *payload_iov, std::size_t payload_iov_count,
                  std::size_t payload_logical_size, std::string &error);

// Acquire the next frame, validate that it is a Response for
// `expected_request_id`, and decode the body prefix. On success `exit_code`
// and `server_exec_sec` are filled and `out_data`/`out_size` are VIEWS into
// the communicator's frame. `owns_frame` is true and the caller MUST call
// ReleaseFrame() once it has consumed `out_data`. Returns false on a
// transport or protocol error (`error` set).
bool ReadResponse(Communicator *c, std::uint64_t expected_request_id, int &exit_code,
                  double &server_exec_sec, const unsigned char *&out_data, std::size_t &out_size,
                  bool &owns_frame, std::string &error);

}  // namespace gvirtus::communicators::am
