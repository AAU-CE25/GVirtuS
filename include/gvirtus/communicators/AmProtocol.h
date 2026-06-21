/*
 * GVirtuS RPC envelope codec.
 *
 * The request/response wire format (UcxAmProtocol.h EnvelopeHeader) used to be
 * hand-rolled inside the backend's dispatch loop (Process.cpp). It lives here
 * instead so the backend only does dispatch and the transport only moves
 * frames: every Communicator delivers a whole message via TryAcquireFrame /
 * WriteFrame (base class frames a byte stream; UCX uses native AM delimiting),
 * and this codec turns frames into requests/responses.
 *
 * The codec is transport-agnostic: it never inspects which Communicator it is
 * given. UCX zero-copy vs TCP length-framing is entirely behind the
 * Communicator interface.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "Buffer.h"
#include "Communicator.h"
#include "UcxAmProtocol.h"

namespace gvirtus::communicators::am {

// Receive one request. On success: `header` is filled, `routine` holds a copy
// of the routine name, and payload_data/payload_size are VIEWS into the
// communicator's frame (valid until ReleaseFrame()). `owns_frame` is true and
// the caller MUST call ReleaseFrame() once the handler is done with the
// payload. gpu_payload/gpu_payload_size surface a GPU-resident tail when the
// transport supports it (GPUDirect); null/0 otherwise.
//
// Returns false on a clean disconnect or a protocol error (`error` set).
bool ReadRequest(Communicator *c, ucxam::EnvelopeHeader &header, std::string &routine,
                 const unsigned char *&payload_data, std::size_t &payload_size,
                 void *&gpu_payload, std::size_t &gpu_payload_size, bool &owns_frame,
                 std::string &error);

// Build and send the response frame for a completed request. The response
// body is [exec_sec][wire_out_size][output bytes (+ optional GPU tail)].
// Returns false on a transport error (`error` set).
bool WriteResponse(Communicator *c, const ucxam::EnvelopeHeader &request_header, int exit_code,
                   double server_exec_sec, const std::shared_ptr<Buffer> &output_buffer,
                   void *gpu_payload, std::size_t gpu_payload_size, std::string &error);

}  // namespace gvirtus::communicators::am
