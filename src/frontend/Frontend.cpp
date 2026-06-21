/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
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

/*
 * Frontend::Execute() — single, transport-agnostic RPC path.
 *
 * One envelope protocol for every transport. Framing is the Communicator's
 * job: WriteFrame() length-prefixes a byte stream or hands off a native
 * message frame; TryAcquireFrame() yields the whole reply for in-place
 * parsing (an internal buffer for stream transports, a pinned RX slot for
 * message-oriented transports).
 *
 * The codec (communicators::am::WriteRequest / ReadResponse) owns the
 * envelope; this function only marshals the input Buffer's IoV, calls the
 * codec, and unmarshals the response. There is no per-transport branching.
 *
 * Zero-copy: the request IoV is the Buffer's GetIov() output, so large
 *   payloads added via AddHostPointerForArgumentsDirect (Buffer::AddRef) are
 *   referenced in place and never staged. SetOutputDestination() lets the
 *   response payload land directly in the caller's dst buffer.
 *
 * Reentrancy guard: some transports' init paths can trigger CUDA probe calls
 *   that reach Execute() before mpInitialized is set; we return
 *   CUDA_ERROR_NOT_INITIALIZED so the probe concludes "no local CUDA"
 *   gracefully and lets the transport finish bringing itself up.
 */

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <gvirtus/frontend/Frontend.h>
#include <pthread.h>
#include <stdlib.h> /* getenv */
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <atomic>
#include <vector>

#include "gvirtus/communicators/Protocol.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::CommunicatorFactory;
using gvirtus::communicators::EndpointFactory;
using gvirtus::frontend::Frontend;

static Frontend msFrontend;
std::mutex gFrontendMutex;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;
static std::atomic<std::uint64_t> gRequestId{1};

Logger logger;


std::string getEnvVar(std::string const &key) {
    char *env_var = getenv(key.c_str());
    return (env_var == nullptr) ? std::string("") : std::string(env_var);
}

void Frontend::Init(Communicator *c) {
    // Logger configuration
    BasicConfigurator basicConfigurator;
    basicConfigurator.configure();

    // Set the logging level
    std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
    LogLevel logLevel = INFO_LOG_LEVEL;
    if (!logLevelString.empty()) {
        try {
            logLevel = static_cast<LogLevel>(std::stoi(logLevelString));
        } catch (const std::exception &e) {
            std::cerr << "[GVIRTUS WARNING] Invalid GVIRTUS_LOGLEVEL value: '" << logLevelString
                      << "'. Using default INFO_LOG_LEVEL. (" << e.what() << ")\n";
            logLevel = INFO_LOG_LEVEL;
        }
    }

    Logger root = Logger::getRoot();
    root.setLogLevel(logLevel);

    logger = Logger::getInstance(LOG4CPLUS_TEXT("Frontend"));

    pid_t tid = syscall(SYS_gettid);

    // Get the GVIRTUS_CONFIG environment varibale
    std::string config_path = getEnvVar("GVIRTUS_CONFIG");

    // Check if the configuration file is defined
    if (config_path.empty()) {
        // Check if the configuration file is in the GVIRTUS_HOME directory
        config_path = getEnvVar("GVIRTUS_HOME") + "/etc/properties.json";
        if (config_path.empty()) {
            // Finally consider the current directory
            config_path = "./properties.json";
        }
    }

    std::unique_ptr<char> default_endpoint;

    // no frontend found
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends->find(tid) == mpFrontends->end()) {
            Frontend *f = new Frontend();
            mpFrontends->insert(make_pair(tid, f));
        }
    }

    LOG4CPLUS_INFO(logger, "Using properties file: " + config_path);

    // Allocate buffers BEFORE Connect(). Some communicators run CUDA
    // probe code during their init path, which re-enters this frontend
    // (LD_LIBRARY_PATH puts our stub first) and calls Frontend::Prepare()
    // -> Buffer::Reset(). If the buffers haven't been allocated yet, that
    // derefs nullptr and SIGSEGVs. mpInitialized stays false here and is
    // set to true only at the end — Execute() reads it as a reentrancy
    // guard to short-circuit RPC calls coming from a transport's own init.
    mpFrontends->find(tid)->second->mpInputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpOutputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpLaunchBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mExitCode = -1;

    try {
        auto endpoint = EndpointFactory::get_endpoint(config_path);

        mpFrontends->find(tid)->second->_communicator =
            CommunicatorFactory::get_communicator(endpoint);
        mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();
    } catch (const std::exception &e) {
        LOG4CPLUS_FATAL(logger, fs::path(__FILE__).filename()
                                    << ":" << __LINE__ << ":"
                                    << " Exception occurred: " << e.what());
        exit(EXIT_FAILURE);
    }

    mpFrontends->find(tid)->second->mpInitialized = true;
}

Frontend::~Frontend() {
    static bool destroying = false;
    if (destroying || mpFrontends == nullptr) return;
    destroying = true;

    std::lock_guard<std::mutex> lock(gFrontendMutex);
    {
        pid_t tid = syscall(SYS_gettid);

        auto env = getenv("GVIRTUS_DUMP_STATS");
        bool dump_stats = env && (strcasecmp(env, "on") == 0 || strcasecmp(env, "true") == 0 ||
                                  strcmp(env, "1") == 0);

        // Safe iteration while erasing entries
        for (auto it = mpFrontends->begin(); it != mpFrontends->end(); /* no increment here */) {
            if (it->second == this) {
                it = mpFrontends->erase(it);
                continue;
            }

            if (dump_stats) {
                std::cerr << "[GVIRTUS_STATS] Executed " << it->second->mRoutinesExecuted
                          << " routine(s) in " << it->second->mRoutineExecutionTime
                          << " second(s)\n"
                          << "[GVIRTUS_STATS] Sent " << it->second->mDataSent / (1024 * 1024.0)
                          << " Mb(s) in " << it->second->mSendingTime << " second(s)\n"
                          << "[GVIRTUS_STATS] Received "
                          << it->second->mDataReceived / (1024 * 1024.0) << " Mb(s) in "
                          << it->second->mReceivingTime << " second(s)\n";
            }

            delete it->second;
            it = mpFrontends->erase(it);
        }

        // Delete the map itself and set pointer to nullptr
        delete mpFrontends;
        mpFrontends = nullptr;
    }
}

Frontend *Frontend::GetFrontend(Communicator *c) {
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends == nullptr) mpFrontends = new map<pthread_t, Frontend *>();
    }

    pid_t tid = syscall(SYS_gettid);  // getting frontend's tid

    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it != mpFrontends->end()) return it->second;
    }

    Frontend *f = new Frontend();
    try {
        f->Init(c);
        {
            std::lock_guard<std::mutex> lock(gFrontendMutex);
            mpFrontends->insert(make_pair(tid, f));
        }
    } catch (const std::exception &e) {
        LOG4CPLUS_ERROR(logger, "Error initializing Frontend: " << e.what());
        delete f;  // Clean up on failure
        return nullptr;
    }

    return f;
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
    // Reentrancy guard for the case where a transport's init path fires
    // cu* probe calls during Frontend::Init -> Connect. At that point
    // _communicator->obj_ptr() is set but not yet Connected; if we tried
    // to send we would throw on the null endpoint. Return a harmless error
    // so the probe concludes "no local CUDA support" and the transport
    // bring-up proceeds.
    if (!mpInitialized) {
        // 3 == CUDA_ERROR_NOT_INITIALIZED (driver API) == cudaErrorInitializationError
        // (runtime API). Same numeric value on both APIs.
        mExitCode = 3;
        return;
    }

    if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();

    pid_t tid = syscall(SYS_gettid);
    pid_t pid = getpid();
    size_t in_size = input_buffer->GetBufferSize();
    int exit_code = 0;
    double server_exec_sec = 0.0;
    double send_sec = 0.0;
    double recv_sec = 0.0;

    Frontend *frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end()) {
            LOG4CPLUS_ERROR(logger, "Cannot send any job request");
            return;
        }
        frontend = it->second;
    }

    LOG4CPLUS_DEBUG(logger, "DEBUG - Received routine " << routine << " [pid=" << pid
                                                        << ", tid=" << tid << "]");

    frontend->mRoutinesExecuted++;

    {
        auto start_send = steady_clock::now();

        const std::uint64_t request_id = gRequestId.fetch_add(1);
        // Logical payload size = marshaled arena + any borrowed AddRef
        // segments. Equals GetBufferSize() for a plain marshaled call.
        const std::size_t payload_size = input_buffer->GetLogicalSize();

        // PROFILE: timing breakdown for transfers >= 1MB. payload_size already
        // includes any zero-copy AddRef bytes (GetLogicalSize). The D2H output
        // count pre-registered via SetOutputDestination is added so large
        // device-to-host transfers (payload in the RESPONSE) also trip the gate.
        const std::size_t effective_payload =
            payload_size + frontend->mDirectOutputCount;
        const bool profile = effective_payload >= (1u << 20);
        auto tA = steady_clock::now();

        // Gather-send: the codec wraps [header][routine] in front of the
        // input Buffer's ordered IoV fragments. GetIov() returns one arena
        // fragment for a plain marshaled call, or interleaved inline +
        // borrowed fragments when the caller used
        // AddHostPointerForArgumentsDirect (Buffer::AddRef) — the big user
        // payload is then referenced in place and never copied. The
        // transport's WriteFrame then delivers the whole message atomically.
        std::vector<struct iovec> payload_iov;
        input_buffer->GetIov(payload_iov);
        auto tB = steady_clock::now();

        frontend->mDataSent += payload_size;
        std::string err;
        if (!gvirtus::communicators::am::WriteRequest(
                frontend->_communicator->obj_ptr(), request_id, routine,
                payload_iov.data(), payload_iov.size(), payload_size, err)) {
            throw std::runtime_error("Frontend: WriteRequest failed: " + err);
        }
        auto tD = steady_clock::now();
        auto tC = tD;  // WriteRequest does WriteFrame+Sync internally

        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;

        frontend->mpOutputBuffer->Reset();
        auto start_recv = steady_clock::now();
        auto tE = steady_clock::now();

        size_t out_buffer_size = 0;
        const unsigned char *out_data = nullptr;
        bool owns_frame = false;
        if (!gvirtus::communicators::am::ReadResponse(
                frontend->_communicator->obj_ptr(), request_id, exit_code, server_exec_sec,
                out_data, out_buffer_size, owns_frame, err)) {
            if (owns_frame) frontend->_communicator->obj_ptr()->ReleaseFrame();
            throw std::runtime_error("Frontend: ReadResponse failed: " + err);
        }
        auto tF = steady_clock::now();
        auto tG = tF;
        auto tH = tF;

        frontend->mExitCode = exit_code;

        if (out_buffer_size > 0) {
            tG = steady_clock::now();
            frontend->mDataReceived += out_buffer_size;
            // Zero-copy fast path: when the caller pre-registered a dst via
            // SetOutputDestination() AND the response Buffer layout is
            // exactly [size_t prefix == count][count bytes payload], memcpy
            // the payload straight into the caller's buffer. Eliminates one
            // of the two large memcpys in the D2H path.
            bool direct_ok = false;
            if (frontend->mDirectOutputDst != nullptr &&
                out_buffer_size == sizeof(size_t) + frontend->mDirectOutputCount) {
                size_t payload_prefix = 0;
                std::memcpy(&payload_prefix, out_data, sizeof(size_t));
                if (payload_prefix == frontend->mDirectOutputCount) {
                    std::memcpy(frontend->mDirectOutputDst,
                                out_data + sizeof(size_t),
                                frontend->mDirectOutputCount);
                    frontend->mDirectOutputConsumed = true;
                    direct_ok = true;
                }
            }
            if (!direct_ok) {
                // Single bulk memcpy into mpOutputBuffer.
                frontend->mpOutputBuffer->AppendBytes(
                    reinterpret_cast<const char *>(out_data), out_buffer_size);
            }
            tH = steady_clock::now();
        }

        if (owns_frame) frontend->_communicator->obj_ptr()->ReleaseFrame();

        recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;

        if (profile) {
            auto us = [](auto a, auto b) {
                return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                    "[GVS PROFILE] %s payload=%zuMB | marshal=%ldus write=%ldus sync=%ldus "
                    "read_hdr=%ldus read_payload=%ldus append=%ldus | total_send=%ldus total_recv=%ldus\n",
                    routine, effective_payload >> 20,
                    us(tA, tB), us(tB, tC), us(tC, tD),
                    us(tE, tF), us(tF, tG), us(tG, tH),
                    us(tA, tD), us(tE, tH));
            fflush(stderr);
        }

        frontend->mRoutineExecutionTime += server_exec_sec;
        frontend->mSendingTime += send_sec;
        frontend->mReceivingTime += recv_sec;

        LOG4CPLUS_DEBUG(logger, "Routine '" << routine << "' returned " << exit_code
                                            << " | server_exec=" << server_exec_sec << "s"
                                            << " | send=" << send_sec << "s"
                                            << " | recv=" << recv_sec << "s"
                                            << " | in=" << in_size << "B"
                                            << " | out=" << out_buffer_size << "B"
                                            << " | pid=" << pid << " tid=" << tid
                                            << " | req_id=" << request_id);
        LOG4CPLUS_DEBUG(logger, "DEBUG - Called: " << routine);
        return;
    }
}
