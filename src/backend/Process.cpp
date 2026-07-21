/**
 * @mainpage gVirtuS - A GPGPU transparent virtualization component
 *
 * @section Introduction
 * gVirtuS tries to fill the gap between in-house hosted computing clusters,
 * equipped with GPGPUs devices, and pay-for-use high performance virtual
 * clusters deployed  via public or private computing clouds. gVirtuS allows an
 * instanced virtual machine to access GPGPUs in a transparent way, with an
 * overhead  slightly greater than a real machine/GPGPU setup. gVirtuS is
 * hypervisor independent, and, even though it currently virtualizes nVIDIA CUDA
 * based GPUs, it is not limited to a specific brand technology. The performance
 * of the components of gVirtuS is assessed through a suite of tests in
 * different deployment scenarios, such as providing GPGPU power to cloud
 * computing based HPC clusters and sharing remotely hosted GPGPUs among HPC
 * nodes.
 *
 * Written By: Carlo Palmieri <carlo.palmieri@uniparthenope.it>,
 *             Department of Applied Science
 *             Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *             Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 *             Antonio Mentone <antonio.mentone@uniparthenope.it>,
 *             Department of Science and Technologies
 * Edited By: Mariano Aponte <aponte2001@gmail.com>,
 *            Department of Science and Technologies, University of Naples Parthenope
 *            Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            Department of Computer Science, University College Dublin
 */

#include <gvirtus/backend/Process.h>
#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>
#include <pthread.h>
#include <signal.h>
#include <sys/uio.h>
#include <unistd.h>

#include <functional>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "communicators/hybrid/HybridCommunicator.h"
#include "gvirtus/communicators/UcxAmProtocol.h"

// DEBUG replaced with log4cplus, so that all diagnostics respect GVIRTUS_LOGLEVEL and share the unified format.

using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;

using namespace std;

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator,
                 vector<string> &plugins)
    : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));

    signal(SIGCHLD, SIG_IGN);
    _communicator = communicator;
    mPlugins = plugins;
}

// File-scope logger for the free function getstring(), which has no access to
// the Process class member.  Using log4cplus instead of raw printf so every
// diagnostic message respects GVIRTUS_LOGLEVEL and shares the unified format.
static log4cplus::Logger gs_logger =
    log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process.getstring"));

namespace {
bool read_exact(Communicator *c, char *buffer, size_t size) {
    size_t copied = 0;
    while (copied < size) {
        size_t n = c->Read(buffer + copied, size - copied);
        if (n == 0) {
            return false;
        }
        copied += n;
    }
    return true;
}

// Reads one AM request from the communicator. Two modes:
//
//   1. Frame mode (UcxCommunicator): TryAcquireFrame returns a pointer to
//      the pinned RX-pool slot containing [header][routine][payload]. We
//      parse those views in-place. `owns_frame` is set to true and the
//      caller MUST call client_comm->ReleaseFrame() once it's done with
//      payload_data (i.e., after the handler returns and the response is
//      built — we already copied the routine string out, so only payload
//      stays in the slot).
//
//   2. Stream fallback (TCP, IB, hybrid, …): byte-stream Read() into
//      `fallback_storage`, exposed via payload_data/payload_size.
//      `owns_frame` is set to false.
bool read_ucx_am_request(Communicator *client_comm,
                         gvirtus::communicators::ucxam::EnvelopeHeader &header,
                         std::string &routine,
                         std::vector<unsigned char> &fallback_storage,
                         const unsigned char *&payload_data,
                         size_t &payload_size,
                         void *&gpu_payload,
                         size_t &gpu_payload_size,
                         bool &owns_frame,
                         std::string &error) {
    owns_frame = false;
    payload_data = nullptr;
    payload_size = 0;
    gpu_payload = nullptr;
    gpu_payload_size = 0;

    const unsigned char *frame = nullptr;
    size_t frame_size = 0;
    if (client_comm->TryAcquireFrame(frame, frame_size)) {
        // Frame mode: validate frame is big enough for the header.
        if (frame_size < sizeof(header)) {
            client_comm->ReleaseFrame();
            error = "AM frame smaller than header";
            return false;
        }
        std::memcpy(&header, frame, sizeof(header));

        if (header.magic != gvirtus::communicators::ucxam::kEnvelopeMagic ||
            header.version != gvirtus::communicators::ucxam::kEnvelopeVersion ||
            header.header_size != sizeof(header)) {
            client_comm->ReleaseFrame();
            error = "invalid AM header";
            return false;
        }
        if (header.message_type !=
            static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Request)) {
            client_comm->ReleaseFrame();
            error = "unexpected AM message type";
            return false;
        }

        const size_t want = sizeof(header) +
                            static_cast<size_t>(header.routine_size) +
                            static_cast<size_t>(header.payload_size);
        if (frame_size < want) {
            client_comm->ReleaseFrame();
            error = "AM frame truncated";
            return false;
        }

        // Copy out the routine name (small, simpler than dealing with pool lifetime).
        routine.assign(reinterpret_cast<const char *>(frame + sizeof(header)),
                       static_cast<size_t>(header.routine_size));

        // Payload stays in the pool slot — caller releases when done.
        payload_data = frame + sizeof(header) + header.routine_size;
        payload_size = static_cast<size_t>(header.payload_size);
        owns_frame = true;

        // GPUDirect (Variant B Step B4): if the UCX peer used the GPU-split
        // wire format, the trailing portion of the LOGICAL payload lives in
        // slot.gpu_addr. Surface it via out-params so the caller can attach
        // it to the input Buffer; GPU-aware handlers route via D2D instead
        // of H2D-from-host. Base Communicator default returns null/0 for
        // transports that don't support GPU payloads.
        client_comm->current_frame_gpu(gpu_payload, gpu_payload_size);

        error.clear();
        return true;
    }

    // Stream fallback (non-UCX transports).
    if (!read_exact(client_comm, reinterpret_cast<char *>(&header), sizeof(header))) {
        error = "unable to read AM header";
        return false;
    }

    if (header.magic != gvirtus::communicators::ucxam::kEnvelopeMagic ||
        header.version != gvirtus::communicators::ucxam::kEnvelopeVersion ||
        header.header_size != sizeof(gvirtus::communicators::ucxam::EnvelopeHeader)) {
        error = "invalid AM header";
        return false;
    }

    if (header.message_type !=static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Request)) {
        error = "unexpected AM message type";
        return false;
    }

    routine.assign(static_cast<size_t>(header.routine_size), '\0');
    if (!routine.empty() &&
        !read_exact(client_comm, routine.data(), static_cast<size_t>(header.routine_size))) {
        error = "unable to read AM routine bytes";
        return false;
    }

    fallback_storage.assign(static_cast<size_t>(header.payload_size), 0);
    if (!fallback_storage.empty() &&
        !read_exact(client_comm, reinterpret_cast<char *>(fallback_storage.data()),
                    fallback_storage.size())) {
        error = "unable to read AM payload bytes";
        return false;
    }
    payload_data = fallback_storage.data();
    payload_size = fallback_storage.size();

    error.clear();
    return true;
}

bool write_ucx_am_response(Communicator *client_comm,
                           const gvirtus::communicators::ucxam::EnvelopeHeader &request_header,
                           int exit_code, double server_exec_sec,
                           const std::shared_ptr<Buffer> &output_buffer,
                           void *gpu_payload, size_t gpu_payload_size,
                           std::string &error) {
    // Host-side portion of the response. With GPUDirect, this is just the
    // protocol prefix (size_t count); without, it's [size_t count][count bytes].
    size_t host_out_size = 0;
    const char *out_data = nullptr;
    if (output_buffer != nullptr) {
        host_out_size = output_buffer->GetBufferSize();
        out_data = output_buffer->GetBuffer();
    }

    // Wire out_size = host prefix + (optional) GPU payload. Frontend reads
    // this many bytes contiguously, regardless of split origin.
    const size_t wire_out_size = host_out_size + gpu_payload_size;
    const size_t payload_size  = sizeof(double) + sizeof(size_t) + wire_out_size;

    gvirtus::communicators::ucxam::EnvelopeHeader response_header{};
    response_header.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
    response_header.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
    response_header.message_type = static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Response);
    response_header.header_size = static_cast<uint16_t>(sizeof(gvirtus::communicators::ucxam::EnvelopeHeader));
    response_header.reserved0 = 0;
    response_header.status_code = static_cast<uint32_t>(exit_code);
    response_header.request_id = request_header.request_id;
    response_header.routine_size = 0;
    response_header.payload_size = static_cast<uint64_t>(payload_size);

    // Gather-send response via WriteIov — eliminates the std::vector staging
    // copy of the entire output payload (was the dual of the frontend's
    // request-side ~27ms marshal). UCX backend maps this to a single
    // ucp_am_send_nbx with UCP_DATATYPE_IOV.
    //
    // With GPUDirect, iov[4] carries a pointer into GPU memory. WriteIovRma
    // detects this via cudaPointerGetAttributes and registers it with
    // UCS_MEMORY_TYPE_CUDA before ucp_put_nbx — NIC peer-DMAs from GPU
    // directly to frontend host slot.
    struct iovec iov[5];
    int n = 0;
    iov[n].iov_base = &response_header;
    iov[n].iov_len  = sizeof(response_header);
    ++n;
    iov[n].iov_base = &server_exec_sec;
    iov[n].iov_len  = sizeof(double);
    ++n;
    // Note: we send wire_out_size on the wire (host prefix + gpu payload),
    // so frontend's AssignAll/memmove sees a single contiguous logical buffer.
    size_t wire_out_size_field = wire_out_size;
    iov[n].iov_base = &wire_out_size_field;
    iov[n].iov_len  = sizeof(size_t);
    ++n;
    if (host_out_size > 0 && out_data != nullptr) {
        iov[n].iov_base = const_cast<char *>(out_data);
        iov[n].iov_len  = host_out_size;
        ++n;
    }
    if (gpu_payload != nullptr && gpu_payload_size > 0) {
        iov[n].iov_base = gpu_payload;
        iov[n].iov_len  = gpu_payload_size;
        ++n;
    }

    try {
        client_comm->WriteIov(iov, static_cast<size_t>(n));
        client_comm->Sync();
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }

    error.clear();
    return true;
}
}

bool getstring(Communicator *c, string &s) {
    // TRACE: fires on every routine call, too noisy for DEBUG
    // RTTI diagnostics merged into one TRACE log.
    // Only fires when GVIRTUS_LOGLEVEL=0 (TRACE).
    if (gs_logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
        const char *rtti = "<no-rtti>";
        try {
            rtti = typeid(*c).name();
        } catch (...) {
        }
        std::string name;
        try {
            name = c->to_string();
        } catch (...) {
            name = "<no to_string()>";
        }
        LOG4CPLUS_TRACE(gs_logger,
                        "[getstring] c=" << (void *)c
                        << " rtti=" << rtti << " to_string()=" << name);
    }

    // TODO: FIX LISKOV SUBSTITUTION AND DIPENDENCE INVERSION!!!!!
    if (c->to_string() == "tcpcommunicator") {
        s = "";
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            // If reading is ended, return true
            if (ch == 0) {
                return true;
            }
            s += ch;
        }
        return false;
    } else if (c->to_string() == "rdmacommunicator") {
        try {
            s = "";
            size_t size = 256;
            char *buf = (char *)malloc(size);
            size = c->Read(buf, size);

            // if read, return true
            if (size > 0) {
                s += std::string(buf);
                return true;
            }
        } catch (const std::exception &e) {
            cerr << e.what() << endl;
        }
        return false;
    } else if (c->to_string() == "hybridcommunicator") {
        s.clear();
        char ch = 0;
        // same as tcp/ip, and stop until read /0
        while (c->Read(&ch, 1) == 1) {
            if (ch == 0) {
                return true;  // take the complete routine name
            }
            s += ch;
        }
        return false;
    } else if (c->to_string() == "ucxcommunicator") {
        throw runtime_error("Not available for UCX anymore. Delete later this else if.");
    }

    throw runtime_error("Communicator getstring read error... Unknown communicator type...");
}

extern std::string getEnvVar(std::string const &key);

std::string getGVirtuSHome() {
    std::string gvirtus_home = getEnvVar("GVIRTUS_HOME");
    return gvirtus_home;
}

void Process::Start() {
    LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] Process::Start() called.");

    for_each(mPlugins.begin(), mPlugins.end(), [this](const std::string &plug) {
        std::string gvirtus_home = getGVirtuSHome();

        std::string to_append = "libgvirtus-plugin-" + plug + ".so";
        LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] appending " << to_append << ".");

        auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);

        try {
            auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
            dl->build_obj();
            _handlers.push_back(dl);
        } catch (const std::exception &e) {
            LOG4CPLUS_ERROR(logger, e.what());
        }
    });

    // inserisci i sym dei plugin in h
    std::function<void(Communicator *)> execute = [this](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "]"
                                            << "Process::Start()'s \"execute\" lambda called");
        // carica i puntatori ai simboli dei moduli in mHandlers

        string routine;
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();
        const bool ucx_am_mode = client_comm != nullptr && client_comm->to_string() == "ucxcommunicator";

        if (ucx_am_mode) {
            // Async dispatch (GVIRTUS_ASYNC_DISPATCH): fire-and-forget requests
            // carry kEnvelopeFlagNoResponse and get no response. A failure in
            // one is latched here and reconciled onto the next response-bearing
            // (sync) call, mirroring CUDA's "async errors surface at the next
            // synchronization point" semantics. Per-connection (this worker
            // thread owns one client), so no locking is needed.
            int deferred_async_error = 0;
            try {
                for (;;) {
                    gvirtus::communicators::ucxam::EnvelopeHeader request_header{};
                    std::string am_routine;
                    std::vector<unsigned char> am_payload_fallback;
                    const unsigned char *am_payload_data = nullptr;
                    size_t am_payload_size = 0;
                    void *am_gpu_payload = nullptr;
                    size_t am_gpu_payload_size = 0;
                    bool owns_frame = false;
                    std::string read_error;

                    if (!read_ucx_am_request(client_comm, request_header, am_routine,
                                             am_payload_fallback, am_payload_data,
                                             am_payload_size,
                                             am_gpu_payload, am_gpu_payload_size,
                                             owns_frame, read_error)) {
                        LOG4CPLUS_INFO(logger,
                                       "Client disconnected (UCX AM): " << read_error);
                        break;
                    }

                    std::shared_ptr<Buffer> am_input = std::make_shared<Buffer>();
                    if (am_payload_size > 0 && am_payload_data != nullptr) {
                        // Non-owning Buffer wrap — the bytes live either in
                        // am_payload_fallback (stream mode) or in the
                        // communicator's pinned RX-pool slot (frame mode).
                        // Either way, lifetime extends past h->Execute().
                        am_input = std::make_shared<Buffer>(
                            reinterpret_cast<char *>(const_cast<unsigned char *>(am_payload_data)),
                            am_payload_size);
                        // GPUDirect Step B4: attach the GPU-resident tail
                        // (if any) so GPU-aware handlers can route via D2D.
                        if (am_gpu_payload != nullptr && am_gpu_payload_size > 0) {
                            am_input->SetGpuPayload(am_gpu_payload, am_gpu_payload_size);
                        }
                    }

                    std::shared_ptr<Handler> h = nullptr;
                    for (auto &ptr_el : _handlers) {
                        if (ptr_el->obj_ptr()->CanExecute(am_routine)) {
                            h = ptr_el->obj_ptr();
                            break;
                        }
                    }

                    std::shared_ptr<communicators::Result> result;
                    if (h == nullptr) {
                        LOG4CPLUS_ERROR(logger, "[Process " << getpid() << "]: Requested unknown routine '" << am_routine << "'.");
                        result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
                    } else {
                        // Per-connection GPUDirect gate (Option 2): publish
                        // this endpoint's transport capability into a
                        // thread-local that GPU-aware handlers read in lieu
                        // of the process-wide GVIRTUS_GPUDIRECT_ACTIVE env.
                        // Reset to false after Execute() to avoid leaking
                        // across dispatches (defensive — in the current
                        // single-threaded per-connection model the worker
                        // thread is reused for the next request on the
                        // same connection, so the flag would be re-set
                        // identically anyway).
                        gvirtus::communicators::tls_connection_supports_cuda =
                            client_comm->current_connection_supports_cuda();
                        gvirtus::communicators::tls_client_rma_put_capable =
                            client_comm->rma_put_capable();
                        auto start = steady_clock::now();
                        result = h->Execute(am_routine, am_input);
                        result->TimeTaken(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                steady_clock::now() - start)
                                .count() /
                            1000.0);
                        gvirtus::communicators::tls_connection_supports_cuda = false;
                        gvirtus::communicators::tls_client_rma_put_capable = false;
                    }

                    const bool no_response =
                        (request_header.reserved0 &
                         gvirtus::communicators::ucxam::kEnvelopeFlagNoResponse) != 0;
                    const int call_exit = result->GetExitCode();

                    if (no_response) {
                        // Fire-and-forget: the frontend did not wait for a
                        // response, so we must NOT write one (it would desync
                        // the strictly in-order request/response stream). Latch
                        // any failure for the next sync call to report.
                        if (call_exit != 0) deferred_async_error = call_exit;

                        // Release the pinned RX-pool slot (handler is done with
                        // am_input, which points into it).
                        if (owns_frame) client_comm->ReleaseFrame();

                        LOG4CPLUS_DEBUG(logger,
                                        "[Process " << getpid() << "]: AM async routine '"
                                                    << am_routine << "' exit=" << call_exit
                                                    << " [req_id=" << request_header.request_id
                                                    << "], no response sent.");
                        continue;
                    }

                    // Reconcile: if this (sync) call succeeded but a prior async
                    // op failed, report the async error here; then clear it (it
                    // is now surfaced). If this call has its own error, that is
                    // the latest error and takes precedence.
                    int effective_exit = call_exit;
                    if (effective_exit == 0 && deferred_async_error != 0) {
                        effective_exit = deferred_async_error;
                    }
                    deferred_async_error = 0;

                    // Phase 3 async H2D drain: if fire-and-forget GPU copies are
                    // still in flight reading their shadow slots, block until the
                    // device drains BEFORE replying. The frontend treats every
                    // synchronous reply as "all prior RMA slots are free" and may
                    // reuse them; draining here keeps that contract correct. No-op
                    // unless tls_async_gpu_pending was set by the MemcpyAsync
                    // handler on this thread.
                    client_comm->drain_device_if_async_pending();

                    std::string write_error;
                    bool response_ok = write_ucx_am_response(client_comm, request_header,
                                                             effective_exit, result->TimeTaken(),
                                                             result->GetOutputBuffer(),
                                                             result->GetGpuPayload(),
                                                             result->GetGpuPayloadSize(),
                                                             write_error);

                    // Release the pinned RX-pool slot now that the handler is
                    // done with it and the response has been sent. Must happen
                    // AFTER h->Execute() returns (am_input points into the slot).
                    if (owns_frame) client_comm->ReleaseFrame();

                    if (!response_ok) {
                        LOG4CPLUS_WARN(logger,
                                       "UCX AM response write failed: " << write_error);
                        break;
                    }

                    LOG4CPLUS_DEBUG(logger,
                                    "[Process " << getpid() << "]: AM routine '" << am_routine
                                                << "' returned " << effective_exit
                                                << " [req_id=" << request_header.request_id
                                                << "].");
                }
            } catch (const std::exception &e) {
                LOG4CPLUS_WARN(logger, "UCX AM client loop exception: " << e.what());
            }

            LOG4CPLUS_INFO(logger, "Client disconnected");
            Notify("process-ended");
            return;
        }

        try {
            while (getstring(client_comm, routine)) {
                LOG4CPLUS_TRACE(logger, "Received routine " << routine);

                // === before reading buffer, chose the protocol of this round by rountine ===
                gvirtus::communicators::HybridCommunicator *hybrid = nullptr;
                if (client_comm && client_comm->to_string() == "hybridcommunicator") {
                    hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator *>(client_comm);
                }
                if (hybrid) {
                    // all those function payload will transfer by RDMA
                    const bool use_rdma = routine.rfind("cudaRegisterFatBinary", 0) == 0 || routine.rfind("cudaRegisterFatBinaryEnd", 0) == 0 || routine.rfind("cudaMemcpyAsync", 0) == 0 || routine.rfind("cudaMemcpy", 0) == 0;

                    if (use_rdma) {
                        // bytes_hint if >0 ,then trigger the first 8B under TCP moniter.
                        // real payload size after 8B head.
                        hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, /*bytes_hint*/ 1);
                    } else {
                        hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, 0);
                    }
                }

                // now reading buffer：8B from TCP, payload will transfer by the selected protocol
                input_buffer->Reset(client_comm);

                std::shared_ptr<Handler> h = nullptr;
                for (auto &ptr_el : _handlers) {
                    if (ptr_el->obj_ptr()->CanExecute(routine)) {
                        h = ptr_el->obj_ptr();
                        break;
                    }
                }

                std::shared_ptr<communicators::Result> result;
                if (h == nullptr) {
                    LOG4CPLUS_ERROR(logger, "[Process " << getpid() << "]: Requested unknown routine '"
                                                        << routine << "'.");
                    result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
                } else {
                    auto start = steady_clock::now();
                    result = h->Execute(routine, input_buffer);
                    result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          steady_clock::now() - start)
                                          .count() /
                                      1000.0);
                }

                // return info：control the head transfer by TCP，then payload RDMA
                result->Dump(client_comm);

                // stop this round, and clean all context
                if (hybrid) {
                    hybrid->end_call();
                }

                LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "]: Routine '" << routine
                                                     << "' returned " << result->GetExitCode()
                                                     << ".");
            }
        } catch (const std::exception &e) {
            LOG4CPLUS_WARN(logger, "Client stream closed with exception: " << e.what());
        }

        LOG4CPLUS_INFO(logger, "Client disconnected");
        Notify("process-ended");
    };

    /*
    common::SignalState sig_hand;
    sig_hand.setup_signal_state(SIGINT);
*/

    try {
        _communicator->obj_ptr()->Serve();

        int pid = 0;
        while (true) {
            Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());

            // Client connect is already logged at INFO by TcpCommunicator::Accept() with IP
            LOG4CPLUS_TRACE(logger,
                            "Accept raw: client=" << (void *)client
                            << ", comm=" << (client ? client->to_string() : "<null>"));

            if (client != nullptr) {
                //      if ((pid = fork()) == 0) {
                std::thread(execute, client).detach();
                //        exit(0);
                //      }

            } else
                _communicator->obj_ptr()->run();

            // check if process received SIGINT

            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(
                    logger, "SIGINT received, killing server on [Process " << getpid() << "]...");
                break;
            }
        }
    } catch (const std::exception &exc) {
        LOG4CPLUS_ERROR(logger, "[Process " << getpid() << "]: " << exc.what());
    }

    LOG4CPLUS_DEBUG(logger, "Process::Start() returned [Process " << getpid() << "].");
    // exit(EXIT_SUCCESS);
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}