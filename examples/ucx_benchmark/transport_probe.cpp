/**
 * Transport Probe — Verifies TCP and UCX/RDMA connectivity between two hosts.
 *
 * Tests each transport by sending a small ping-pong and reports success/failure.
 * Use this to diagnose which transports actually work on your network.
 *
 * Usage:
 *   Server:  ./transport_probe server <port>
 *   Client:  ./transport_probe client <server_ip> <port>
 *
 * Build:
 *   g++ -O2 -DUSE_UCX -o transport_probe transport_probe.cpp -lucp -lucs -lpthread
 *
 * The probe tests (in order):
 *   1. TCP socket echo (baseline)
 *   2. UCX with tcp transport only
 *   3. UCX with rc_v (RC verbs / RoCE)
 *   4. UCX with rc_mlx5 (RC Mellanox accelerated)
 *   5. UCX with all available transports (auto)
 */

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_UCX
#include <ucp/api/ucp.h>
#endif

// ============================================================================
// Helpers
// ============================================================================

static std::string ts() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000000;
    char buf[32];
    snprintf(buf, sizeof(buf), "[%06ld.%03ld] ", ms / 1000, ms % 1000);
    return std::string(buf);
}

static constexpr uint32_t MAGIC_PING = 0xDEADBEEF;
static constexpr uint32_t MAGIC_PONG = 0xCAFEBABE;
static constexpr int PROBE_PORT_OFFSET = 100;  // UCX probes use port+100+N

// ============================================================================
// TCP Probe
// ============================================================================

namespace tcp_probe {

static bool test_server(int port) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return false;
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(srv);
        return false;
    }
    listen(srv, 1);

    struct timeval tv{10, 0};  // 10s timeout
    setsockopt(srv, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in cli_addr{};
    socklen_t cli_len = sizeof(cli_addr);
    int client = accept(srv, (struct sockaddr *)&cli_addr, &cli_len);
    if (client < 0) {
        close(srv);
        return false;
    }

    // Receive ping
    uint32_t ping = 0;
    ssize_t n = recv(client, &ping, sizeof(ping), MSG_WAITALL);
    if (n != sizeof(ping) || ping != MAGIC_PING) {
        close(client);
        close(srv);
        return false;
    }

    // Send pong
    uint32_t pong = MAGIC_PONG;
    n = send(client, &pong, sizeof(pong), 0);
    close(client);
    close(srv);
    return n == sizeof(pong);
}

static bool test_client(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;

    struct timeval tv{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        struct hostent *ent = gethostbyname(host);
        if (!ent) { close(fd); return false; }
        memcpy(&addr.sin_addr, ent->h_addr_list[0], ent->h_length);
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return false;
    }

    uint32_t ping = MAGIC_PING;
    ssize_t n = send(fd, &ping, sizeof(ping), 0);
    if (n != sizeof(ping)) { close(fd); return false; }

    uint32_t pong = 0;
    n = recv(fd, &pong, sizeof(pong), MSG_WAITALL);
    close(fd);
    return (n == sizeof(pong) && pong == MAGIC_PONG);
}

}  // namespace tcp_probe

// ============================================================================
// UCX Probe
// ============================================================================

#ifdef USE_UCX

namespace ucx_probe {

static constexpr ucp_tag_t TAG_PROBE = 0x200;
static constexpr ucp_tag_t TAG_MASK  = 0xFFFFFFFFFFFFFFFF;

struct UcxReq {
    std::atomic<bool> complete{false};
    ucs_status_t status{UCS_OK};
};

static void req_init(void *r) {
    auto *req = static_cast<UcxReq *>(r);
    req->complete.store(false);
    req->status = UCS_OK;
}

static void send_cb(void *request, ucs_status_t status, void *) {
    auto *req = static_cast<UcxReq *>(request);
    req->status = status;
    req->complete.store(true);
}

static void recv_cb(void *request, ucs_status_t status, const ucp_tag_recv_info_t *, void *) {
    auto *req = static_cast<UcxReq *>(request);
    req->status = status;
    req->complete.store(true);
}

struct UcxCtx {
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
};

static bool init_ucx(UcxCtx &ctx) {
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT;
    params.features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
    params.request_size = sizeof(UcxReq);
    params.request_init = req_init;

    ucs_status_t st = ucp_init(&params, nullptr, &ctx.context);
    if (st != UCS_OK) {
        std::cerr << "  ucp_init failed: " << ucs_status_string(st) << std::endl;
        return false;
    }

    ucp_worker_params_t wparams{};
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

    st = ucp_worker_create(ctx.context, &wparams, &ctx.worker);
    if (st != UCS_OK) {
        std::cerr << "  ucp_worker_create failed: " << ucs_status_string(st) << std::endl;
        ucp_cleanup(ctx.context);
        ctx.context = nullptr;
        return false;
    }
    return true;
}

static void cleanup_ucx(UcxCtx &ctx) {
    if (ctx.worker) ucp_worker_destroy(ctx.worker);
    if (ctx.context) ucp_cleanup(ctx.context);
}

static bool wait_req(ucp_worker_h worker, void *request, int timeout_ms = 10000) {
    if (request == nullptr) return true;
    if (UCS_PTR_IS_ERR(request)) {
        std::cerr << "  UCX op failed: " << ucs_status_string(UCS_PTR_STATUS(request)) << std::endl;
        return false;
    }
    auto *req = static_cast<UcxReq *>(request);
    auto start = std::chrono::steady_clock::now();
    while (!req->complete.load()) {
        ucp_worker_progress(worker);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeout_ms) {
            std::cerr << "  UCX op timed out after " << timeout_ms << "ms" << std::endl;
            ucp_request_cancel(worker, request);
            ucp_request_free(request);
            return false;
        }
    }
    bool ok = (req->status == UCS_OK);
    if (!ok) std::cerr << "  UCX op error: " << ucs_status_string(req->status) << std::endl;
    ucp_request_free(request);
    return ok;
}

// Connection callback context for listener
struct ConnCtx {
    ucp_conn_request_h conn_req = nullptr;
    bool have_conn = false;
};

static void conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *c = static_cast<ConnCtx *>(arg);
    c->conn_req = conn_request;
    c->have_conn = true;
}

static bool test_server(int port, const char *tls_name) {
    std::cerr << "  [server] Testing UCX transport: " << tls_name << " on port " << port << std::endl;

    // Set UCX_TLS for this test
    setenv("UCX_TLS", tls_name, 1);

    UcxCtx ctx{};
    if (!init_ucx(ctx)) return false;

    ConnCtx conn_ctx{};
    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(port);

    ucp_listener_params_t lparams{};
    lparams.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                         UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lparams.sockaddr.addr = (struct sockaddr *)&listen_addr;
    lparams.sockaddr.addrlen = sizeof(listen_addr);
    lparams.conn_handler.cb = conn_handler;
    lparams.conn_handler.arg = &conn_ctx;

    ucp_listener_h listener;
    ucs_status_t st = ucp_listener_create(ctx.worker, &lparams, &listener);
    if (st != UCS_OK) {
        std::cerr << "  [server] listener_create failed: " << ucs_status_string(st) << std::endl;
        cleanup_ucx(ctx);
        return false;
    }

    // Wait for connection (10s timeout)
    auto start = std::chrono::steady_clock::now();
    while (!conn_ctx.have_conn) {
        ucp_worker_progress(ctx.worker);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 10000) {
            std::cerr << "  [server] No connection within 10s" << std::endl;
            ucp_listener_destroy(listener);
            cleanup_ucx(ctx);
            return false;
        }
    }

    // Accept
    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_ctx.conn_req;
    ucp_ep_h ep;
    st = ucp_ep_create(ctx.worker, &ep_params, &ep);
    if (st != UCS_OK) {
        std::cerr << "  [server] ep_create failed: " << ucs_status_string(st) << std::endl;
        ucp_listener_destroy(listener);
        cleanup_ucx(ctx);
        return false;
    }

    // Receive ping
    uint32_t ping = 0;
    ucp_request_param_t rparam{};
    rparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    rparam.datatype = ucp_dt_make_contig(1);
    rparam.cb.recv = recv_cb;
    void *req = ucp_tag_recv_nbx(ctx.worker, &ping, sizeof(ping), TAG_PROBE, TAG_MASK, &rparam);
    if (!wait_req(ctx.worker, req, 10000)) {
        ucp_listener_destroy(listener);
        cleanup_ucx(ctx);
        return false;
    }

    std::cerr << "  [server] Received ping: 0x" << std::hex << ping << std::dec << std::endl;
    bool ping_ok = (ping == MAGIC_PING);

    // Send pong
    uint32_t pong = MAGIC_PONG;
    ucp_request_param_t sparam{};
    sparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    sparam.datatype = ucp_dt_make_contig(1);
    sparam.cb.send = send_cb;
    req = ucp_tag_send_nbx(ep, &pong, sizeof(pong), TAG_PROBE, &sparam);
    bool send_ok = wait_req(ctx.worker, req, 5000);

    // Flush
    ucp_request_param_t fparam{};
    fparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    fparam.cb.send = send_cb;
    req = ucp_ep_flush_nbx(ep, &fparam);
    wait_req(ctx.worker, req, 5000);

    // Close
    ucp_request_param_t cparam{};
    cparam.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    cparam.flags = UCP_EP_CLOSE_FLAG_FORCE;
    void *close_req = ucp_ep_close_nbx(ep, &cparam);
    if (!UCS_PTR_IS_ERR(close_req) && close_req != nullptr) {
        auto *cr = static_cast<UcxReq *>(close_req);
        while (!cr->complete.load()) ucp_worker_progress(ctx.worker);
        ucp_request_free(close_req);
    }

    ucp_listener_destroy(listener);
    cleanup_ucx(ctx);
    return ping_ok && send_ok;
}

static bool test_client(const char *host, int port, const char *tls_name) {
    std::cerr << "  [client] Testing UCX transport: " << tls_name << " on port " << port << std::endl;

    setenv("UCX_TLS", tls_name, 1);

    UcxCtx ctx{};
    if (!init_ucx(ctx)) return false;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        struct hostent *ent = gethostbyname(host);
        if (!ent) { cleanup_ucx(ctx); return false; }
        memcpy(&addr.sin_addr, ent->h_addr_list[0], ent->h_length);
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr *)&addr;
    ep_params.sockaddr.addrlen = sizeof(addr);

    ucp_ep_h ep;
    ucs_status_t st = ucp_ep_create(ctx.worker, &ep_params, &ep);
    if (st != UCS_OK) {
        std::cerr << "  [client] ep_create failed: " << ucs_status_string(st) << std::endl;
        cleanup_ucx(ctx);
        return false;
    }

    // Send ping
    uint32_t ping = MAGIC_PING;
    ucp_request_param_t sparam{};
    sparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    sparam.datatype = ucp_dt_make_contig(1);
    sparam.cb.send = send_cb;
    void *req = ucp_tag_send_nbx(ep, &ping, sizeof(ping), TAG_PROBE, &sparam);
    if (!wait_req(ctx.worker, req, 5000)) {
        cleanup_ucx(ctx);
        return false;
    }

    // Flush to ensure delivery
    ucp_request_param_t fparam{};
    fparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    fparam.cb.send = send_cb;
    req = ucp_ep_flush_nbx(ep, &fparam);
    if (!wait_req(ctx.worker, req, 5000)) {
        cleanup_ucx(ctx);
        return false;
    }

    // Receive pong
    uint32_t pong = 0;
    ucp_request_param_t rparam{};
    rparam.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    rparam.datatype = ucp_dt_make_contig(1);
    rparam.cb.recv = recv_cb;
    req = ucp_tag_recv_nbx(ctx.worker, &pong, sizeof(pong), TAG_PROBE, TAG_MASK, &rparam);
    if (!wait_req(ctx.worker, req, 10000)) {
        cleanup_ucx(ctx);
        return false;
    }

    std::cerr << "  [client] Received pong: 0x" << std::hex << pong << std::dec << std::endl;

    // Close
    ucp_request_param_t cparam{};
    cparam.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    cparam.flags = UCP_EP_CLOSE_FLAG_FORCE;
    void *close_req = ucp_ep_close_nbx(ep, &cparam);
    if (!UCS_PTR_IS_ERR(close_req) && close_req != nullptr) {
        auto *cr = static_cast<UcxReq *>(close_req);
        while (!cr->complete.load()) ucp_worker_progress(ctx.worker);
        ucp_request_free(close_req);
    }

    cleanup_ucx(ctx);
    return (pong == MAGIC_PONG);
}

}  // namespace ucx_probe

#endif  // USE_UCX

// ============================================================================
// UCX Info (print available transports)
// ============================================================================

#ifdef USE_UCX
static void print_ucx_info() {
    std::cout << "\n--- UCX Configuration ---" << std::endl;

    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_TAG;

    ucp_context_h ctx;
    ucs_status_t st = ucp_init(&params, nullptr, &ctx);
    if (st != UCS_OK) {
        std::cout << "  Cannot initialize UCX: " << ucs_status_string(st) << std::endl;
        return;
    }

    ucp_context_attr_t attr{};
    attr.field_mask = UCP_ATTR_FIELD_THREAD_MODE;
    ucp_context_query(ctx, &attr);
    std::cout << "  UCX initialized OK" << std::endl;
    std::cout << "  Thread mode: " << attr.thread_mode << std::endl;

    // Print worker address size as proxy for available transports
    ucp_worker_params_t wparams{};
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_h worker;
    st = ucp_worker_create(ctx, &wparams, &worker);
    if (st == UCS_OK) {
        ucp_worker_attr_t wattr{};
        wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
        st = ucp_worker_query(worker, &wattr);
        if (st == UCS_OK) {
            std::cout << "  Worker address size: " << wattr.address_length << " bytes" << std::endl;
            ucp_worker_release_address(worker, wattr.address);
        }
        ucp_worker_destroy(worker);
    }

    ucp_cleanup(ctx);

    // Print env vars
    const char *tls = getenv("UCX_TLS");
    const char *net = getenv("UCX_NET_DEVICES");
    const char *rndv = getenv("UCX_RNDV_THRESH");
    std::cout << "  UCX_TLS=" << (tls ? tls : "(not set, auto)") << std::endl;
    std::cout << "  UCX_NET_DEVICES=" << (net ? net : "(not set, auto)") << std::endl;
    std::cout << "  UCX_RNDV_THRESH=" << (rndv ? rndv : "(not set, default)") << std::endl;
    std::cout << "---" << std::endl;
}
#endif

// ============================================================================
// Main
// ============================================================================

struct ProbeConfig {
    const char *name;
    const char *tls;
    int port_offset;
};

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "  Server: " << argv[0] << " server <base_port>" << std::endl;
        std::cerr << "  Client: " << argv[0] << " client <server_ip> <base_port>" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Tests TCP + multiple UCX transport variants." << std::endl;
        std::cerr << "Run server first, then client." << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    // Define probes: name, UCX_TLS value, port offset
    std::vector<ProbeConfig> ucx_probes = {
        {"UCX (tcp only)",   "tcp",         1},
        {"UCX (rc verbs)",   "rc_v,tcp",    2},
        {"UCX (rc mlx5)",    "rc_mlx5,tcp", 3},
        {"UCX (auto/all)",   "all",         4},
    };

    if (mode == "server") {
        int base_port = std::stoi(argv[2]);

        std::cout << "=== Transport Probe Server ===" << std::endl;
        std::cout << "Base port: " << base_port << std::endl;
        std::cout << std::endl;

#ifdef USE_UCX
        print_ucx_info();
        std::cout << std::endl;
#endif

        // Test 1: TCP
        std::cout << "[1] TCP (port " << base_port << ")... " << std::flush;
        bool tcp_ok = tcp_probe::test_server(base_port);
        std::cout << (tcp_ok ? "OK" : "FAILED") << std::endl;

#ifdef USE_UCX
        // Tests 2-5: UCX variants
        int test_num = 2;
        for (auto &p : ucx_probes) {
            int port = base_port + p.port_offset;
            std::cout << "[" << test_num << "] " << p.name << " (port " << port << ")... " << std::flush;
            bool ok = ucx_probe::test_server(port, p.tls);
            std::cout << (ok ? "OK" : "FAILED") << std::endl;
            test_num++;
        }
#else
        std::cout << "\n(UCX not compiled in — rebuild with -DUSE_UCX)" << std::endl;
#endif

        std::cout << "\n=== Server Done ===" << std::endl;

    } else if (mode == "client") {
        if (argc < 4) {
            std::cerr << "Client needs: <server_ip> <base_port>" << std::endl;
            return 1;
        }
        const char *host = argv[2];
        int base_port = std::stoi(argv[3]);

        std::cout << "=== Transport Probe Client ===" << std::endl;
        std::cout << "Server: " << host << ":" << base_port << std::endl;
        std::cout << std::endl;

#ifdef USE_UCX
        print_ucx_info();
        std::cout << std::endl;
#endif

        // Test 1: TCP
        std::cout << "[1] TCP... " << std::flush;
        bool tcp_ok = tcp_probe::test_client(host, base_port);
        std::cout << (tcp_ok ? "OK" : "FAILED") << std::endl;

#ifdef USE_UCX
        // Tests 2-5: UCX variants
        int test_num = 2;
        for (auto &p : ucx_probes) {
            int port = base_port + p.port_offset;
            std::cout << "[" << test_num << "] " << p.name << "... " << std::flush;
            bool ok = ucx_probe::test_client(host, port, p.tls);
            std::cout << (ok ? "OK" : "FAILED") << std::endl;
            test_num++;
        }
#else
        std::cout << "\n(UCX not compiled in — rebuild with -DUSE_UCX)" << std::endl;
#endif

        std::cout << "\n=== Summary ===" << std::endl;

    } else {
        std::cerr << "Unknown mode: " << mode << " (use 'server' or 'client')" << std::endl;
        return 1;
    }

    return 0;
}
