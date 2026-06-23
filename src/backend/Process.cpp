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

#include "gvirtus/communicators/RpcCodec.h"
#include "gvirtus/communicators/Protocol.h"

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
        LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "] dispatch loop started");

        // Single transport-agnostic request/response loop. Framing (whole-
        // message delivery) is the Communicator's job — the base class
        // length-prefixes a byte stream and message-oriented transports
        // override it with native frame delimiting — and the envelope codec
        // (communicators::am) turns frames into requests/responses. This
        // loop only dispatches; it never touches the wire format.
        try {
            for (;;) {
                gvirtus::communicators::am::EnvelopeHeader request_header{};
                std::string routine;
                const unsigned char *payload_data = nullptr;
                size_t payload_size = 0;
                void *gpu_payload = nullptr;
                size_t gpu_payload_size = 0;
                bool owns_frame = false;
                std::string err;

                if (!communicators::am::ReadRequest(client_comm, request_header, routine,
                                                    payload_data, payload_size, gpu_payload,
                                                    gpu_payload_size, owns_frame, err)) {
                    LOG4CPLUS_DEBUG(logger, "Client disconnected: " << err);
                    break;
                }

                // Non-owning wrap of the payload view (valid until ReleaseFrame).
                std::shared_ptr<Buffer> input = std::make_shared<Buffer>();
                if (payload_size > 0 && payload_data != nullptr) {
                    input = std::make_shared<Buffer>(
                        reinterpret_cast<char *>(const_cast<unsigned char *>(payload_data)),
                        payload_size);
                    if (gpu_payload != nullptr && gpu_payload_size > 0)
                        input->SetGpuPayload(gpu_payload, gpu_payload_size);
                }

                std::shared_ptr<Handler> h = nullptr;
                for (auto &ptr_el : _handlers) {
                    if (ptr_el->obj_ptr()->CanExecute(routine)) {
                        h = ptr_el->obj_ptr();
                        break;
                    }
                }

                std::shared_ptr<communicators::Result> result;
                if (h == nullptr) {
                    LOG4CPLUS_ERROR(logger, "[Process " << getpid()
                                    << "]: Requested unknown routine '" << routine << "'.");
                    result = std::make_shared<communicators::Result>(
                        -1, std::make_shared<Buffer>());
                } else {
                    // Per-connection GPUDirect gate: GPU-aware handlers read this
                    // thread-local instead of coupling to a Communicator subclass.
                    gvirtus::communicators::tls_connection_supports_cuda =
                        client_comm->current_connection_supports_cuda();
                    auto start = steady_clock::now();
                    result = h->Execute(routine, input);
                    result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          steady_clock::now() - start).count() / 1000.0);
                    gvirtus::communicators::tls_connection_supports_cuda = false;
                }

                std::string write_error;
                bool response_ok = communicators::am::WriteResponse(
                    client_comm, request_header, result->GetExitCode(), result->TimeTaken(),
                    result->GetOutputBuffer(), result->GetGpuPayload(),
                    result->GetGpuPayloadSize(), write_error);

                // Release the frame only AFTER Execute + response, since `input`
                // points into it.
                if (owns_frame) client_comm->ReleaseFrame();

                if (!response_ok) {
                    LOG4CPLUS_WARN(logger, "Response write failed: " << write_error);
                    break;
                }

                LOG4CPLUS_DEBUG(logger, "[Process " << getpid() << "]: routine '" << routine
                                << "' returned " << result->GetExitCode()
                                << " [req_id=" << request_header.request_id << "].");
            }
        } catch (const std::exception &e) {
            LOG4CPLUS_WARN(logger, "Client loop exception: " << e.what());
        }

        LOG4CPLUS_DEBUG(logger, "Client disconnected");
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