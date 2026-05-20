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

#include <sys/wait.h>
#include <unistd.h>
#include <gvirtus/backend/Process.h>
#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "communicators/hybrid/HybridCommunicator.h"
#include "gvirtus/communicators/UcxAmProtocol.h"
#include <gvirtus/common/Protocol.h>

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

    // We fork one worker per accepted frontend session and waitpid() it,
    // so SIGCHLD must remain waitable.
    signal(SIGCHLD, SIG_DFL);
    _communicator = communicator;
    mPlugins = plugins;
}

// File-scope logger for the free function getstring(), which has no access to
// the Process class member. Using log4cplus instead of raw printf so every
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

    if (header.message_type !=
        static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Request)) {
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

    const size_t payload_size = sizeof(double) + sizeof(size_t) + out_size;
    std::vector<unsigned char> payload(payload_size);

    size_t off = 0;
    std::memcpy(payload.data() + off, &server_exec_sec, sizeof(double));
    off += sizeof(double);
    std::memcpy(payload.data() + off, &out_size, sizeof(size_t));
    off += sizeof(size_t);
    if (out_size > 0 && out_data != nullptr) {
        std::memcpy(payload.data() + off, out_data, out_size);
    }

    gvirtus::communicators::ucxam::EnvelopeHeader response_header{};
    response_header.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
    response_header.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
    response_header.message_type =
        static_cast<uint16_t>(gvirtus::communicators::ucxam::MessageType::Response);
    response_header.header_size =
        static_cast<uint16_t>(sizeof(gvirtus::communicators::ucxam::EnvelopeHeader));
    response_header.reserved0 = 0;
    response_header.status_code = static_cast<uint32_t>(exit_code);
    response_header.request_id = request_header.request_id;
    response_header.routine_size = 0;
    response_header.payload_size = static_cast<uint64_t>(payload_size);

    try {
        client_comm->Write(reinterpret_cast<const char *>(&response_header),
                           sizeof(response_header));
        if (!payload.empty()) {
            client_comm->Write(reinterpret_cast<const char *>(payload.data()), payload.size());
        }
        client_comm->Sync();
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }

    error.clear();
    return true;
}
}  // namespace

bool getstring(Communicator *c, string &s) {
    // TRACE: fires on every routine call, too noisy for DEBUG.
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
                                         << " rtti=" << rtti
                                         << " to_string()=" << name);
    }

    // TODO: FIX LISKOV SUBSTITUTION AND DEPENDENCY INVERSION.
    if (c->to_string() == "tcpcommunicator") {
        s = "";
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            if (ch == 0) {
                return true;
            }
            s += ch;
        }
        return false;
    } else if (c->to_string() == "rdmacommunicator") {
        try {
            s = "";
            constexpr size_t max_routine_name = 256;
            size_t size = max_routine_name;
            char *buf = static_cast<char *>(calloc(size, 1));
            if (buf == nullptr) {
                throw std::runtime_error("getstring(rdmacommunicator): calloc failed");
            }

            size = c->Read(buf, size);
            buf[max_routine_name - 1] = '\0';

            if (size > 0) {
                s += std::string(buf);
                std::cerr << "[RDMA getstring] routine=[" << s << "] bytes=" << size << std::endl;
                free(buf);
                return true;
            }

            free(buf);
        } catch (const std::exception &e) {
            cerr << e.what() << endl;
        }
        return false;
    } else if (c->to_string() == "hybridcommunicator") {
        s.clear();
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            if (ch == 0) {
                return true;
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
        LOG4CPLUS_DEBUG(logger,
                        "[Process " << getpid() << "] appending " << to_append << ".");

        auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);

        try {
            auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
            dl->build_obj();
            _handlers.push_back(dl);
        } catch (const std::exception &e) {
            LOG4CPLUS_ERROR(logger, e.what());
        }
    });

    std::function<void(Communicator *)> execute = [this](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "]"
                                            << "Process::Start()'s \"execute\" lambda called");

        // The accepted client communicator is allocated by Accept().
        // Do not own/delete it here. Some accepted communicator instances do not
        // fully initialize all fields used by their destructor. In the current
        // benchmark mode each client is handled in a short-lived worker process,
        // so process exit releases the accepted socket safely.
        auto close_client = [&]() {
            if (!client_comm) {
                return;
            }

            try {
                client_comm->Close();
            } catch (const std::exception &e) {
                LOG4CPLUS_WARN(logger, "Client close failed: " << e.what());
            } catch (...) {
                LOG4CPLUS_WARN(logger, "Client close failed with unknown exception.");
            }
        };

        string routine;
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();
        const bool ucx_am_mode =
            client_comm != nullptr && client_comm->to_string() == "ucxcommunicator";

        if (ucx_am_mode) {
            try {
                for (;;) {
                    gvirtus::communicators::ucxam::EnvelopeHeader request_header{};
                    std::string am_routine;
                    std::vector<unsigned char> am_payload;
                    std::string read_error;

                    if (!read_ucx_am_request(client_comm, request_header, am_routine, am_payload,
                                             read_error)) {
                        LOG4CPLUS_INFO(logger,
                                       "Client disconnected (UCX AM): " << read_error);
                        break;
                    }

                    std::shared_ptr<Buffer> am_input = std::make_shared<Buffer>();
                    if (!am_payload.empty()) {
                        am_input = std::make_shared<Buffer>(
                            reinterpret_cast<char *>(am_payload.data()), am_payload.size());
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
                        LOG4CPLUS_ERROR(logger,
                                        "[Process " << getpid()
                                                    << "]: Requested unknown routine '"
                                                    << am_routine << "'.");
                        result = std::make_shared<communicators::Result>(
                            -1, std::make_shared<Buffer>());
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
                    if (!write_ucx_am_response(client_comm, request_header,
                                               result->GetExitCode(), result->TimeTaken(),
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
            close_client();
            Notify("process-ended");
            return;
        }

        try {
            while (getstring(client_comm, routine)) {
                if (routine == gvirtus::common::protocol::kShutdownRoutine) {
                    LOG4CPLUS_INFO(logger,
                                   "[Process " << getpid()
                                               << "]: Received explicit RDMA shutdown routine.");
                    break;
                }

                LOG4CPLUS_TRACE(logger, "Received routine " << routine);

                // Before reading buffer, choose protocol for this round if hybrid.
                gvirtus::communicators::HybridCommunicator *hybrid = nullptr;
                if (client_comm && client_comm->to_string() == "hybridcommunicator") {
                    hybrid =
                        dynamic_cast<gvirtus::communicators::HybridCommunicator *>(client_comm);
                }

                if (hybrid) {
                    const bool use_rdma =
                        routine.rfind("cudaRegisterFatBinary", 0) == 0 ||
                        routine.rfind("cudaRegisterFatBinaryEnd", 0) == 0 ||
                        routine.rfind("cudaMemcpyAsync", 0) == 0 ||
                        routine.rfind("cudaMemcpy", 0) == 0;

                    if (use_rdma) {
                        hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA,
                                           /*bytes_hint*/ 1);
                    } else {
                        hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, 0);
                    }
                }

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
                    LOG4CPLUS_ERROR(logger,
                                    "[Process " << getpid()
                                                << "]: Requested unknown routine '"
                                                << routine << "'.");
                    result =
                        std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
                } else {
                    auto start = steady_clock::now();
                    result = h->Execute(routine, input_buffer);
                    result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          steady_clock::now() - start)
                                          .count() /
                                      1000.0);
                }

                result->Dump(client_comm);

                if (hybrid) {
                    hybrid->end_call();
                }

                LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "]: Routine '" << routine
                                                     << "' returned " << result->GetExitCode()
                                                     << ".");

                if (client_comm &&
                    client_comm->to_string() == "rdmacommunicator" &&
                    routine == "cudaUnregisterFatBinary") {
                    setenv("GVIRTUS_RDMA_ROUTINE_RECV_TIMEOUT_MS", "3000", 1);
                    LOG4CPLUS_INFO(logger,
                                   "[Process " << getpid()
                                               << "]: Enabled RDMA routine receive timeout after "
                                                  "cudaUnregisterFatBinary.");
                }

                // Do not end the RDMA client loop on cudaUnregisterFatBinary.
                // OpenCV/Python workloads may still issue GVirtuS calls during shutdown.
            }
        } catch (const std::exception &e) {
            LOG4CPLUS_WARN(logger, "Client stream closed with exception: " << e.what());
        }

        LOG4CPLUS_INFO(logger, "Client disconnected");
        close_client();
        Notify("process-ended");
    };

    /*
    common::SignalState sig_hand;
    sig_hand.setup_signal_state(SIGINT);
    */

    try {
        _communicator->obj_ptr()->Serve();

        while (true) {
            Communicator *client =
                const_cast<Communicator *>(_communicator->obj_ptr()->Accept());

            LOG4CPLUS_TRACE(logger,
                            "Accept raw: client=" << (void *)client
                                                  << ", comm="
                                                  << (client ? client->to_string() : "<null>"));

            if (client != nullptr) {
                if (client->to_string() == "rdmacommunicator" ||
                    client->to_string() == "ucxcommunicator") {
                    LOG4CPLUS_DEBUG(logger,
                        "Handling " << client->to_string()
                                    << " client synchronously to avoid fork-after-transport state.");
                    execute(client);
                    delete client;
                    continue;
                }

                pid_t worker_pid = fork();

                if (worker_pid == 0) {
                    execute(client);
                    _exit(EXIT_SUCCESS);
                }

                if (worker_pid < 0) {
                    LOG4CPLUS_ERROR(logger,
                                    "Failed to fork client worker: " << strerror(errno));
                    execute(client);
                } else {
                    int worker_status = 0;
                    pid_t waited = 0;

                    do {
                        waited = waitpid(worker_pid, &worker_status, 0);
                    } while (waited < 0 && errno == EINTR);

                    if (waited < 0) {
                        LOG4CPLUS_ERROR(logger,
                                        "waitpid(" << worker_pid << ") failed: "
                                                    << strerror(errno));
                    } else if (WIFEXITED(worker_status)) {
                        LOG4CPLUS_DEBUG(logger,
                                        "Client worker " << worker_pid
                                                         << " exited with code "
                                                         << WEXITSTATUS(worker_status));
                    } else if (WIFSIGNALED(worker_status)) {
                        LOG4CPLUS_ERROR(logger,
                                        "Client worker " << worker_pid
                                                         << " killed by signal "
                                                         << WTERMSIG(worker_status));
                    }
                }

                // Do not delete client in the listener process.
                // The worker process owns the active client session lifetime.
            } else {
                _communicator->obj_ptr()->run();
            }

            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(
                    logger,
                    "SIGINT received, killing server on [Process " << getpid() << "]...");
                break;
            }
        }
    } catch (const std::exception &exc) {
        LOG4CPLUS_ERROR(logger, "[Process " << getpid() << "]: " << exc.what());
    }

    LOG4CPLUS_DEBUG(logger, "Process::Start() returned [Process " << getpid() << "].");
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}
