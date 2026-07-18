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
 * UCX communicator changes to Frontend::Execute():
 *
 * The UCX AM path (ucx_am_mode branch) replaces the original sequential
 * Write(routine) + Dump(buffer) + Read(exit_code) + Read(output) flow with a
 * single-message envelope protocol:
 *
 *   Send: [EnvelopeHeader][routine_name][serialized_buffer] via WriteIov()
 *   Recv: TryAcquireFrame() → parse header + payload in-place from pinned slot
 *
 * Fase 4: gather-send via WriteIov avoids staging the 64 MB payload into a
 *   contiguous buffer before sending. Frame-based receive skips the per-byte
 *   Add<char> loop (was ~1.3s for 64 MB).
 *
 * Fase 5: AddHostPointerForArgumentsDirect() splices the user's buffer into
 *   the iov at the recorded offset — the payload never touches mpInputBuffer.
 *   SetOutputDestination() lets the response path write directly into the
 *   caller's dst buffer.
 *
 * Reentrancy guard: UCX's libuct_cuda fires cu* calls during ucp_init. These
 *   reach Execute() before mpInitialized is set; we return CUDA_ERROR_NOT_-
 *   INITIALIZED so UCX's probe concludes "no local CUDA" gracefully.
 *
 * Optimization phases: 2 (protocol), 4 (gather-send + frame recv), 5 (zero-copy)
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
#include <fstream>
#include <iostream>
#include <mutex>
#include <atomic>
#include <vector>

#include "communicators/hybrid/HybridCommunicator.h"
#include "gvirtus/communicators/UcxAmProtocol.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

// --- Per-RPC latency tracing (GUSTO benchmarking, INFOCOM) --------------------
// Env-gated: when GVIRTUS_LATENCY_TRACE=<file> is set, every UCX Active-Message
// RPC records its total round-trip latency (µs), the server-side execution time
// (µs), routine name and effective payload size into a per-thread buffer. All
// buffers are flushed to <file> as CSV at process exit. Zero cost when the env
// var is unset (a single atomic-free bool check on the hot path). Enables
// p50/p90/p99/p99.9 + CDF + tail analysis without perturbing steady-state means.
namespace gvirtus_lattrace {
struct Sample {
    long rt_us;          // total client-observed round-trip
    long server_us;      // server-reported execution time
    unsigned long bytes; // effective payload size
    const char *routine; // string literal (static lifetime) from Execute()
};

class Tracer {
 public:
    static Tracer &instance() {
        static Tracer t;
        return t;
    }
    bool enabled() const { return enabled_; }
    void registerBuffer(std::vector<Sample> *buf) {
        std::lock_guard<std::mutex> lk(mtx_);
        buffers_.push_back(buf);
    }
    ~Tracer() { flush(); }
    void flush() {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lk(mtx_);
        std::ofstream ofs(path_);
        if (!ofs) return;
        ofs << "routine,payload_bytes,rt_us,server_us\n";
        for (auto *b : buffers_)
            for (const auto &s : *b)
                ofs << (s.routine ? s.routine : "?") << ',' << s.bytes << ','
                    << s.rt_us << ',' << s.server_us << '\n';
    }

 private:
    Tracer() {
        const char *p = std::getenv("GVIRTUS_LATENCY_TRACE");
        if (p && p[0]) {
            enabled_ = true;
            path_ = p;
        }
    }
    bool enabled_ = false;
    std::string path_;
    std::mutex mtx_;
    std::vector<std::vector<Sample> *> buffers_;
};

// Per-thread sample buffer (heap-allocated, intentionally never freed so it
// survives thread exit and is still valid when Tracer flushes at process exit).
inline thread_local std::vector<Sample> *tls_buf = nullptr;

inline void record(const char *routine, unsigned long bytes, long rt_us, long server_us) {
    Tracer &tr = Tracer::instance();
    if (!tr.enabled()) return;
    if (tls_buf == nullptr) {
        tls_buf = new std::vector<Sample>();
        tls_buf->reserve(1u << 17);
        tr.registerBuffer(tls_buf);
    }
    tls_buf->push_back(Sample{rt_us, server_us, bytes, routine});
}
}  // namespace gvirtus_lattrace

using std::chrono::microseconds;

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
static std::atomic<std::uint64_t> gUcxAmRequestId{1};

Logger logger;

namespace {
bool read_exact(gvirtus::communicators::Communicator *c, char *buffer, size_t size) {
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
}

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

    // Allocate buffers BEFORE Connect(). During Connect() the UCX
    // communicator runs ucp_init, which dlopens libuct_cuda; libuct_cuda's
    // module init then calls cuInit and other cu* probing functions on the
    // GVirtuS frontend (because LD_LIBRARY_PATH puts ours first). Those
    // functions reach Frontend::Prepare() -> Buffer::Reset(). If buffers
    // haven't been allocated yet, that derefs nullptr and SIGSEGVs.
    // mpInitialized stays false here and is set to true only at the end —
    // Frontend::Execute() reads it as a reentrancy guard to short-circuit
    // RPC calls coming from libuct_cuda's init.
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
    ExecuteInternal(routine, input_buffer, /*force_fire_and_forget=*/false);
}

// Asynchronous / fire-and-forget dispatch (GVIRTUS_ASYNC_DISPATCH). Only the
// UCX Active-Message transport honours this; on other transports it degrades to
// a normal synchronous Execute (the stream protocol always expects a response).
// The caller must only route stream-ordered, output-less CUDA calls here (see
// the frontend stub allowlist); any failure surfaces at the next sync point.
void Frontend::ExecuteAsync(const char *routine, const Buffer *input_buffer) {
    ExecuteInternal(routine, input_buffer, /*force_fire_and_forget=*/true);
}

void Frontend::ExecuteInternal(const char *routine, const Buffer *input_buffer,
                               bool force_fire_and_forget) {
    // Reentrancy guard for libuct_cuda's module init firing cu* calls during
    // ucp_init() (which itself runs inside Frontend::Init -> Connect). At
    // that point _communicator->obj_ptr() is set but not yet Connected; if
    // we tried to send we would throw on the null endpoint. Return a
    // harmless error so libuct_cuda concludes "no CUDA support" and lets
    // UCX continue picking rc_mlx5 for the data path.
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

    const bool ucx_am_mode = frontend->_communicator->obj_ptr()->to_string() == "ucxcommunicator";
    // Fire-and-forget only applies to the UCX AM transport; other transports
    // must still round-trip (their stream protocol expects a response).
    const bool fire_and_forget = force_fire_and_forget && ucx_am_mode;

    if (ucx_am_mode) {
        auto start_send = steady_clock::now();

        const std::uint64_t request_id = gUcxAmRequestId.fetch_add(1);
        const std::size_t routine_size = std::strlen(routine);
        const std::size_t payload_size = input_buffer->GetBufferSize();

        gvirtus::communicators::ucxam::EnvelopeHeader req_header{};
        req_header.magic = gvirtus::communicators::ucxam::kEnvelopeMagic;
        req_header.version = gvirtus::communicators::ucxam::kEnvelopeVersion;
        req_header.message_type = static_cast<std::uint16_t>(gvirtus::communicators::ucxam::MessageType::Request);
        req_header.header_size = static_cast<std::uint16_t>(sizeof(gvirtus::communicators::ucxam::EnvelopeHeader));
        req_header.reserved0 = fire_and_forget
                                   ? gvirtus::communicators::ucxam::kEnvelopeFlagNoResponse
                                   : 0;
        req_header.status_code = 0;
        req_header.request_id = request_id;
        req_header.routine_size = static_cast<std::uint64_t>(routine_size);
        req_header.payload_size = static_cast<std::uint64_t>(payload_size);

        // PROFILE: timing breakdown for transfers >= 1MB.
        // Fase 5: when the caller uses AddHostPointerForArgumentsDirect, the
        // user payload bypasses mpInputBuffer entirely — payload_size alone
        // becomes tiny (just the marshaled headers / scalars). Include the
        // direct chunk so big transfers still trip the gate.
        // Fase 4 (D2H): the big payload is in the RESPONSE, not the request.
        // mDirectOutputCount is the count the caller pre-registered via
        // SetOutputDestination — include it so D2H also gets profiled.
        const std::size_t effective_payload =
            payload_size + frontend->mDirectInputBytes
                         + frontend->mDirectOutputCount;
        const bool profile = effective_payload >= (1u << 20);
        auto tA = steady_clock::now();

        // Gather-send via Communicator::WriteIov - UCX backend maps this to
        // ucp_am_send_nbx with UCP_DATATYPE_IOV, avoiding the staging memcpy
        // of the entire payload (was ~27ms per 64MB on Connect-7). Fase 5
        // zero-copy: when the caller used AddHostPointerForArgumentsDirect,
        // splice the user buffer at the recorded offset so the 64MB never
        // touches mpInputBuffer. Wire payload bytes are identical either way.
        const bool has_direct = frontend->HasDirectInput();
        if (has_direct) {
            req_header.payload_size =
                static_cast<std::uint64_t>(payload_size + frontend->mDirectInputBytes);
        }
        struct iovec iov[5];
        int n = 0;
        iov[n].iov_base = &req_header;
        iov[n].iov_len  = sizeof(req_header);
        ++n;
        if (routine_size > 0) {
            iov[n].iov_base = const_cast<char *>(routine);
            iov[n].iov_len  = routine_size;
            ++n;
        }
        if (payload_size > 0) {
            if (has_direct) {
                const size_t split = frontend->mDirectInputBufferOffset;
                char *base = const_cast<char *>(input_buffer->GetBuffer());
                iov[n].iov_base = base;
                iov[n].iov_len  = split;
                ++n;
                iov[n].iov_base = const_cast<void *>(frontend->mDirectInputSrc);
                iov[n].iov_len  = frontend->mDirectInputBytes;
                ++n;
                if (payload_size > split) {
                    iov[n].iov_base = base + split;
                    iov[n].iov_len  = payload_size - split;
                    ++n;
                }
            } else {
                iov[n].iov_base = const_cast<char *>(input_buffer->GetBuffer());
                iov[n].iov_len  = payload_size;
                ++n;
            }
        }
        auto tB = steady_clock::now();

        frontend->mDataSent += payload_size;
        frontend->_communicator->obj_ptr()->WriteIov(iov, static_cast<size_t>(n));
        auto tC = steady_clock::now();

        if (fire_and_forget) {
            // Fire-and-forget: WriteIov already waited for LOCAL send completion
            // (wait_request_completion inside the communicator), so the stack
            // envelope header + iov are safe to release now, and the request is
            // in-order on the wire. We deliberately skip Sync() (worker flush)
            // and the entire response read — that is where the round-trip
            // latency we are eliminating lives. Errors are reconciled by the
            // backend onto the next synchronous call.
            send_sec =
                duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;
            frontend->ClearDirectInput();
            frontend->mpOutputBuffer->Reset();
            frontend->mExitCode = 0;  // cudaSuccess; async errors surface at next sync point
            frontend->mSendingTime += send_sec;

            if (gvirtus_lattrace::Tracer::instance().enabled()) {
                const long rt_us =
                    duration_cast<microseconds>(steady_clock::now() - start_send).count();
                gvirtus_lattrace::record(routine,
                                         static_cast<unsigned long>(effective_payload), rt_us, 0);
            }

            LOG4CPLUS_DEBUG(logger, "[UCX AM] fire-and-forget '"
                                        << routine << "' [req_id=" << request_id
                                        << ", send=" << send_sec << "s]");
            return;
        }

        frontend->_communicator->obj_ptr()->Sync();
        auto tD = steady_clock::now();

        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;

        // Fase 5: direct input chunk has been consumed by WriteIov+Sync above;
        // drop the pointer so the next Execute() doesn't accidentally inherit it.
        frontend->ClearDirectInput();

        frontend->mpOutputBuffer->Reset();
        auto start_recv = steady_clock::now();
        auto tE = steady_clock::now();

        gvirtus::communicators::ucxam::EnvelopeHeader resp_header{};
        size_t out_buffer_size = 0;
        auto tF = steady_clock::now();
        auto tG = tF;
        auto tH = tF;

        // Try the zero-copy frame path first: TryAcquireFrame returns a
        // pointer to the communicator's pinned RX-pool slot containing the
        // entire response (header || exec_sec || out_size || out_data). We
        // parse everything in place and AppendBytes(out_data, out_size) does
        // a single bulk memcpy into mpOutputBuffer instead of the previous
        // per-byte Add<char> loop (~67M calls for 64MB = ~1.3s wasted).
        const unsigned char *frame_data = nullptr;
        size_t frame_size = 0;
        const bool owns_frame =
            frontend->_communicator->obj_ptr()->TryAcquireFrame(frame_data, frame_size);

        if (owns_frame) {
            if (frame_size < sizeof(resp_header)) {
                frontend->_communicator->obj_ptr()->ReleaseFrame();
                throw std::runtime_error("Frontend UCX AM: response frame smaller than header");
            }
            std::memcpy(&resp_header, frame_data, sizeof(resp_header));
            tF = steady_clock::now();

            if (resp_header.magic != gvirtus::communicators::ucxam::kEnvelopeMagic ||
                resp_header.version != gvirtus::communicators::ucxam::kEnvelopeVersion ||
                resp_header.header_size != sizeof(gvirtus::communicators::ucxam::EnvelopeHeader)) {
                frontend->_communicator->obj_ptr()->ReleaseFrame();
                throw std::runtime_error("Frontend UCX AM: invalid response header");
            }
            if (resp_header.request_id != request_id) {
                frontend->_communicator->obj_ptr()->ReleaseFrame();
                throw std::runtime_error("Frontend UCX AM: response request_id mismatch");
            }

            frontend->mExitCode = static_cast<int>(resp_header.status_code);
            exit_code = frontend->mExitCode;

            const size_t payload_len = static_cast<size_t>(resp_header.payload_size);
            const size_t fixed_prefix = sizeof(double) + sizeof(size_t);
            if (payload_len > 0) {
                if (frame_size < sizeof(resp_header) + fixed_prefix) {
                    frontend->_communicator->obj_ptr()->ReleaseFrame();
                    throw std::runtime_error("Frontend UCX AM: response payload too small");
                }
                const unsigned char *p = frame_data + sizeof(resp_header);
                std::memcpy(&server_exec_sec, p, sizeof(double));
                p += sizeof(double);
                std::memcpy(&out_buffer_size, p, sizeof(size_t));
                p += sizeof(size_t);
                if (sizeof(resp_header) + fixed_prefix + out_buffer_size > frame_size) {
                    frontend->_communicator->obj_ptr()->ReleaseFrame();
                    throw std::runtime_error("Frontend UCX AM: output payload size mismatch");
                }
                tG = steady_clock::now();
                frontend->mDataReceived += out_buffer_size;
                // Fase 4 zero-copy: when the caller pre-registered a dst via
                // SetOutputDestination() AND the response Buffer layout is
                // exactly [size_t prefix == count][count bytes payload],
                // memcpy the payload straight into the caller's buffer.
                // Eliminates one of the two 64MB memcpys in the D2H path.
                bool direct_ok = false;
                if (frontend->mDirectOutputDst != nullptr &&
                    out_buffer_size == sizeof(size_t) + frontend->mDirectOutputCount) {
                    size_t payload_prefix = 0;
                    std::memcpy(&payload_prefix, p, sizeof(size_t));
                    if (payload_prefix == frontend->mDirectOutputCount) {
                        std::memcpy(frontend->mDirectOutputDst,
                                    p + sizeof(size_t),
                                    frontend->mDirectOutputCount);
                        frontend->mDirectOutputConsumed = true;
                        direct_ok = true;
                    }
                }
                if (!direct_ok) {
                    // Single bulk memcpy (~3ms for 64MB) replacing 67M Add<char> calls.
                    frontend->mpOutputBuffer->AppendBytes(
                        reinterpret_cast<const char *>(p), out_buffer_size);
                }
                tH = steady_clock::now();
            } else {
                tG = steady_clock::now();
                tH = tG;
            }

            frontend->_communicator->obj_ptr()->ReleaseFrame();
        } else {
            // Stream fallback (non-UCX transports).
            if (!read_exact(frontend->_communicator->obj_ptr().get(),
                            reinterpret_cast<char *>(&resp_header), sizeof(resp_header))) {
                throw std::runtime_error("Frontend UCX AM: failed to read response header");
            }
            tF = steady_clock::now();

            if (resp_header.magic != gvirtus::communicators::ucxam::kEnvelopeMagic ||
                resp_header.version != gvirtus::communicators::ucxam::kEnvelopeVersion ||
                resp_header.header_size != sizeof(gvirtus::communicators::ucxam::EnvelopeHeader)) {
                throw std::runtime_error("Frontend UCX AM: invalid response header");
            }
            if (resp_header.request_id != request_id) {
                throw std::runtime_error("Frontend UCX AM: response request_id mismatch");
            }

            std::vector<unsigned char> resp_payload(static_cast<std::size_t>(resp_header.payload_size));
            if (!resp_payload.empty() &&
                !read_exact(frontend->_communicator->obj_ptr().get(),
                            reinterpret_cast<char *>(resp_payload.data()), resp_payload.size())) {
                throw std::runtime_error("Frontend UCX AM: failed to read response payload");
            }

            frontend->mExitCode = static_cast<int>(resp_header.status_code);
            exit_code = frontend->mExitCode;

            const std::size_t fixed_prefix = sizeof(double) + sizeof(size_t);
            if (resp_payload.size() < fixed_prefix) {
                throw std::runtime_error("Frontend UCX AM: response payload too small");
            }
            std::size_t parse_off = 0;
            std::memcpy(&server_exec_sec, resp_payload.data() + parse_off, sizeof(double));
            parse_off += sizeof(double);
            std::memcpy(&out_buffer_size, resp_payload.data() + parse_off, sizeof(size_t));
            parse_off += sizeof(size_t);
            if (resp_payload.size() < parse_off + out_buffer_size) {
                throw std::runtime_error("Frontend UCX AM: output payload size mismatch");
            }
            tG = steady_clock::now();
            frontend->mDataReceived += out_buffer_size;
            // Fase 4 zero-copy (stream fallback): same direct-dst shortcut as
            // the frame path above. See comment there for the contract.
            bool direct_ok = false;
            if (frontend->mDirectOutputDst != nullptr &&
                out_buffer_size == sizeof(size_t) + frontend->mDirectOutputCount) {
                const unsigned char *sp = resp_payload.data() + parse_off;
                size_t payload_prefix = 0;
                std::memcpy(&payload_prefix, sp, sizeof(size_t));
                if (payload_prefix == frontend->mDirectOutputCount) {
                    std::memcpy(frontend->mDirectOutputDst,
                                sp + sizeof(size_t),
                                frontend->mDirectOutputCount);
                    frontend->mDirectOutputConsumed = true;
                    direct_ok = true;
                }
            }
            if (!direct_ok) {
                // Single bulk memcpy here too — same fix as the frame path.
                frontend->mpOutputBuffer->AppendBytes(
                    reinterpret_cast<const char *>(resp_payload.data() + parse_off),
                    out_buffer_size);
            }
            tH = steady_clock::now();
        }

        recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;

        if (gvirtus_lattrace::Tracer::instance().enabled()) {
            const long rt_us =
                duration_cast<microseconds>(steady_clock::now() - start_send).count();
            gvirtus_lattrace::record(routine,
                                     static_cast<unsigned long>(effective_payload), rt_us,
                                     static_cast<long>(server_exec_sec * 1e6));
        }

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

        LOG4CPLUS_DEBUG(logger, "[UCX AM] Routine '" << routine << "' returned " << exit_code
                                                      << " | server_exec=" << server_exec_sec
                                                      << "s"
                                                      << " | send=" << send_sec << "s"
                                                      << " | recv=" << recv_sec << "s"
                                                      << " | in=" << in_size << "B"
                                                      << " | out=" << out_buffer_size << "B"
                                                      << " | pid=" << pid << " tid=" << tid
                                                      << " | req_id=" << request_id);
        LOG4CPLUS_DEBUG(logger, "DEBUG - Called: " << routine);
        return;
    }

    // ===== send routine info first（under TCP）=====
    auto start_send = steady_clock::now();
    frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);

    // ===== chose protocol by different routine =====
    if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
        auto *hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator *>(
            frontend->_communicator->obj_ptr().get());
        if (hybrid) {
            if (std::string(routine).find("cudaMemcpy") != std::string::npos ||
                std::string(routine).find("cudaRegisterFatBinary") != std::string::npos ||
                std::string(routine).find("cudaRegisterFatBinaryEnd") != std::string::npos ||
                std::string(routine).find("cudaMemcpyAsync") != std::string::npos) {
                hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, in_size);
            } else {
                hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, in_size);
            }
        }
    }

    // ===== send paramemter data =====
    frontend->mDataSent += in_size;
    LOG4CPLUS_DEBUG(logger, "Write " << in_size << " bytes to the buffer");
    input_buffer->Dump(frontend->_communicator->obj_ptr().get());

    // ===== sync by chosen channel =====
    frontend->_communicator->obj_ptr()->Sync();

    send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;

    frontend->mpOutputBuffer->Reset();

    // ===== receive exit_code =====
    auto start_recv = steady_clock::now();
    frontend->_communicator->obj_ptr()->Read((char *)&exit_code, sizeof(int));
    frontend->mExitCode = exit_code;

    // ===== receive backend time cost =====
    frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&server_exec_sec),
                                             sizeof(server_exec_sec));

    // ===== receive output buffer =====
    size_t out_buffer_size = 0;
    frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
    frontend->mDataReceived += out_buffer_size;
    LOG4CPLUS_DEBUG(logger, "Read " << out_buffer_size << " bytes from the buffer");
    if (out_buffer_size > 0) {
        LOG4CPLUS_DEBUG(logger, "Output buffer size is greater than 0, reading...");
        frontend->mpOutputBuffer->Read<char>(frontend->_communicator->obj_ptr().get(),
                                             out_buffer_size);
    }
    recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;

    // ===== update info =====
    frontend->mRoutineExecutionTime += server_exec_sec;
    frontend->mSendingTime += send_sec;
    frontend->mReceivingTime += recv_sec;

    // ===== print log =====
    LOG4CPLUS_DEBUG(logger, "Routine '" << routine << "' returned " << exit_code
                                        << " | server_exec=" << server_exec_sec << "s"
                                        << " | send=" << send_sec << "s"
                                        << " | recv=" << recv_sec << "s"
                                        << " | in=" << in_size << "B"
                                        << " | out=" << out_buffer_size << "B"
                                        << " | pid=" << pid << " tid=" << tid);

    LOG4CPLUS_DEBUG(logger, "DEBUG - Called: " << routine);

    // ===== stop this call，clean HybridCommunicator status =====
    if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
        auto hybrid = std::dynamic_pointer_cast<gvirtus::communicators::HybridCommunicator>(
            frontend->_communicator->obj_ptr());
        if (hybrid) {
            hybrid->end_call();
        }
    }
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}
