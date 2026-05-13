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

bool read_ucx_am_request(Communicator *client_comm,
                         gvirtus::communicators::ucxam::EnvelopeHeader &header,
                         std::string &routine, std::vector<unsigned char> &payload,
                         std::string &error) {
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

    payload.assign(static_cast<size_t>(header.payload_size), 0);
    if (!payload.empty() &&
        !read_exact(client_comm, reinterpret_cast<char *>(payload.data()), payload.size())) {
        error = "unable to read AM payload bytes";
        return false;
    }

    error.clear();
    return true;
}

bool write_ucx_am_response(Communicator *client_comm,
                           const gvirtus::communicators::ucxam::EnvelopeHeader &request_header,
                           int exit_code, double server_exec_sec,
                           const std::shared_ptr<Buffer> &output_buffer,
                           std::string &error) {
    size_t out_size = 0;
    const char *out_data = nullptr;
    if (output_buffer != nullptr) {
        out_size = output_buffer->GetBufferSize();
        out_data = output_buffer->GetBuffer();
    }

    // Wire layout (preserved for backwards compatibility with the frontend):
    //   [EnvelopeHeader][prefix: double server_exec_sec][prefix: size_t out_size][bulk out_data]
    // Previously these three regions were concatenated into a single
    // std::vector and written as one AM, which forced a zero-fill + memcpy
    // of the entire output (up to tens of MB for cudaMemcpy D2H). We now
    // emit the small prefix and the bulk payload as separate AMs so the
    // bulk payload can be sent directly from `out_data` with no copy.
    constexpr size_t kPrefixSize = sizeof(double) + sizeof(size_t);
    const size_t total_payload = kPrefixSize + out_size;

    gvirtus::communicators::ucxam::EnvelopeHeader response_header{};
    response_header.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
    response_header.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
    response_header.message_type = static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Response);
    response_header.header_size = static_cast<uint16_t>(sizeof(gvirtus::communicators::ucxam::EnvelopeHeader));
    response_header.reserved0 = 0;
    response_header.status_code = static_cast<uint32_t>(exit_code);
    response_header.request_id = request_header.request_id;
    response_header.routine_size = 0;
    response_header.payload_size = static_cast<uint64_t>(total_payload);

    unsigned char prefix[kPrefixSize];
    std::memcpy(prefix, &server_exec_sec, sizeof(double));
    std::memcpy(prefix + sizeof(double), &out_size, sizeof(size_t));

    try {
        client_comm->Write(reinterpret_cast<const char *>(&response_header), sizeof(response_header));
        client_comm->Write(reinterpret_cast<const char *>(prefix), sizeof(prefix));
        if (out_size > 0 && out_data != nullptr) {
            client_comm->Write(out_data, out_size);
        }
        // Sync() intentionally omitted: each Write() above completes locally
        // via wait_request_completion inside the UCX communicator, so an
        // additional ucp_worker_flush_nbx would just add latency.
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
            size_t size = 30;
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
            try {
                for (;;) {
                    gvirtus::communicators::ucxam::EnvelopeHeader request_header{};
                    std::string am_routine;
                    std::vector<unsigned char> am_payload;
                    std::string read_error;

                    if (!read_ucx_am_request(client_comm, request_header, am_routine, am_payload, read_error)) {
                        LOG4CPLUS_INFO(logger,
                                       "Client disconnected (UCX AM): " << read_error);
                        break;
                    }

                    std::shared_ptr<Buffer> am_input = std::make_shared<Buffer>();
                    if (!am_payload.empty()) {
                        am_input = std::make_shared<Buffer>(reinterpret_cast<char *>(am_payload.data()), am_payload.size());
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
                        auto start = steady_clock::now();
                        result = h->Execute(am_routine, am_input);
                        result->TimeTaken(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                steady_clock::now() - start)
                                .count() /
                            1000.0);
                    }

                    std::string write_error;
                    if (!write_ucx_am_response(client_comm, request_header, result->GetExitCode(), result->TimeTaken(),
                                               result->GetOutputBuffer(), write_error)) {
                        LOG4CPLUS_WARN(logger,
                                       "UCX AM response write failed: " << write_error);
                        break;
                    }

                    LOG4CPLUS_DEBUG(logger,
                                    "[Process " << getpid() << "]: AM routine '" << am_routine
                                                << "' returned " << result->GetExitCode()
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