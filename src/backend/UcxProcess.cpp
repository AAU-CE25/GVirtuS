#include "gvirtus/backend/UcxProcess.h"
#include "gvirtus/communicators/FramedStream.h"
#include "gvirtus/communicators/Result.h"
#include "gvirtus/communicators/Buffer.h"
#include "communicators/ucx/UcxCommunicator.h"

#include <arpa/inet.h>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <filesystem>

#include "log4cplus/loggingmacros.h"

namespace fs = std::filesystem;
using namespace gvirtus::backend;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::FramedStream;

extern std::string getGVirtuSHome();

// ─── Listener callback ────────────────────────────────────────────────────────
// UCX calls this on the server worker when a new client connects.
struct ConnRequest { ucp_conn_request_h req; };
static thread_local std::vector<ConnRequest> g_pending_conns;

static void listener_cb(ucp_conn_request_h conn_req, void *arg) {
    g_pending_conns.push_back({conn_req});
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────
UcxProcess::UcxProcess(uint16_t port, std::vector<std::string> plugins)
    : port_(port), plugin_names_(std::move(plugins)) {
    logger_ = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("UcxProcess"));
}

UcxProcess::~UcxProcess() {
    if (listener_)    ucp_listener_destroy(listener_);
    if (ucp_worker_)  ucp_worker_destroy(ucp_worker_);
    if (ucp_context_) ucp_cleanup(ucp_context_);
}

// ─── Start ────────────────────────────────────────────────────────────────────
void UcxProcess::Start() {
    // Load plugins
    for (auto &plug : plugin_names_) {
        auto path = fs::path(getGVirtuSHome() + "/lib/libgvirtus-plugin-" + plug + ".so");
        try {
            auto dl = std::make_shared<common::LD_Lib<Handler>>(path, "create_t");
            dl->build_obj();
            handlers_.push_back(dl);
        } catch (const std::exception &e) {
            LOG4CPLUS_ERROR(logger_, "Plugin load failed: " << e.what());
        }
    }

    // Init UCX context
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_STREAM;
    ucp_config_t *cfg = nullptr;
    ucp_config_read(nullptr, nullptr, &cfg);
    ucp_init(&params, cfg, &ucp_context_);
    ucp_config_release(cfg);

    // Create worker
    ucp_worker_params_t wparams{};
    wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SERIALIZED;
    ucp_worker_create(ucp_context_, &wparams, &ucp_worker_);

    // Start listener
    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port_);
    addr.sin_addr.s_addr = INADDR_ANY;

    ucp_listener_params_t lparams{};
    lparams.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lparams.sockaddr.addr    = reinterpret_cast<const sockaddr *>(&addr);
    lparams.sockaddr.addrlen = sizeof(addr);
    lparams.conn_handler.cb  = listener_cb;
    lparams.conn_handler.arg = nullptr;

    ucp_listener_create(ucp_worker_, &lparams, &listener_);
    LOG4CPLUS_INFO(logger_, "UCX backend listening on port " << port_);

    // Accept loop
    while (true) {
        ucp_worker_progress(ucp_worker_);
        for (auto &cr : g_pending_conns) {
            // Accept the connection request into a new endpoint
            ucp_ep_params_t ep_params{};
            ep_params.field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST;
            ep_params.conn_request = cr.req;
            ucp_ep_h ep = nullptr;
            ucp_ep_create(ucp_worker_, &ep_params, &ep);

            std::thread([this, ep]() {
                ServeConnection(ucp_worker_, ep);
            }).detach();
        }
        g_pending_conns.clear();
    }
}

// ─── Per-connection handler ───────────────────────────────────────────────────
void UcxProcess::ServeConnection(ucp_worker_h worker, ucp_ep_h ep) {
    FramedStream fs(worker);
    LOG4CPLUS_INFO(logger_, "UCX client connected");

    while (true) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;

        if (!fs.Recv(ep, hdr, payload, std::chrono::milliseconds(30000)))
            break;  // timeout or disconnect

        if (static_cast<MsgType>(hdr.msg_type) != MsgType::REQUEST)
            continue;

        // Deserialize [routine\0][buffer_bytes]
        const char *routine = reinterpret_cast<const char *>(payload.data());
        size_t name_len = std::strlen(routine) + 1;

        auto input_buf = std::make_shared<Buffer>();
        if (payload.size() > name_len) {
            // Wrap remaining bytes in a Buffer
            char *raw = const_cast<char *>(
                reinterpret_cast<const char *>(payload.data() + name_len));
            *input_buf = Buffer(raw, payload.size() - name_len);
        }

        // Dispatch to handler
        std::shared_ptr<Handler> h;
        for (auto &ptr : handlers_) {
            if (ptr->obj_ptr()->CanExecute(routine)) {
                h = ptr->obj_ptr();
                break;
            }
        }

        std::shared_ptr<communicators::Result> result;
        if (!h) {
            LOG4CPLUS_ERROR(logger_, "Unknown routine: " << routine);
            result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
        } else {
            auto t0 = std::chrono::steady_clock::now();
            result = h->Execute(routine, input_buf);
            result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0).count() / 1000.0);
        }

        // Serialize result into wire format matching Frontend::Execute(SYNC) reader:
        // [int exit_code][double time_taken][size_t out_size][bytes...]
        auto resp = SerializeResult(result);
        fs.SendResponse(ep, hdr.request_id, hdr.request_id,
                        resp.data(), static_cast<uint32_t>(resp.size()));
    }

    LOG4CPLUS_INFO(logger_, "UCX client disconnected");
}

// ─── Serialize Result to wire bytes ──────────────────────────────────────────
std::vector<uint8_t> UcxProcess::SerializeResult(std::shared_ptr<communicators::Result> result) {
    // Capture Result::Dump() output into a vector using a VecWriter adapter.
    // Dump() writes: [int exit_code][double time_taken][size_t out_size][bytes...]
    // which matches exactly what Frontend::Execute(SYNC) reads back.

    struct VecWriter : public communicators::Communicator {
        std::vector<uint8_t> data;
        size_t Write(const char *buf, size_t n) override {
            data.insert(data.end(), buf, buf + n); return n;
        }
        void Serve() override {}
        const Communicator *const Accept() const override { return nullptr; }
        void Connect() override {}
        size_t Read(char *, size_t n) override { return n; }
        void Sync() override {}
        void Close() override {}
    } writer;

    result->Dump(&writer);
    return std::move(writer.data);
}
