/**
 * Test 1: Raw Data Copy Benchmark (NO CUDA, NO GVirtuS)
 *
 * Pure transport-level benchmark. Sends N bytes from client → server,
 * server echoes them back, client measures roundtrip latency.
 *
 * Supports two transports: TCP sockets and UCX (UCP tag matching).
 * This directly measures how much faster UCX/RDMA is vs TCP on your setup.
 *
 * Usage:
 *   Server:  ./data_copy_bench server <transport> <port>
 *   Client:  ./data_copy_bench client <transport> <server_ip> <port> <n_bytes> <num_runs>
 *
 *   transport: "tcp" or "ucx"
 *
 * Example:
 *   # Terminal 1 (server on GPU node):
 *   ./data_copy_bench server tcp 5555
 *
 *   # Terminal 2 (client):
 *   ./data_copy_bench client tcp 24.24.24.1 5555 1048576 10
 *
 * Client output: CSV lines to stdout
 *
 * Build:
 *   TCP only:  g++ -O2 -o data_copy_bench data_copy_bench.cpp -lpthread
 *   TCP + UCX: g++ -O2 -DUSE_UCX -o data_copy_bench data_copy_bench.cpp -lucp -lucs -lpthread
 */

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_UCX
#include <ucp/api/ucp.h>
#endif

// ============================================================================
// TCP Transport
// ============================================================================

namespace tcp_transport {

static void send_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, buf + sent, len - sent, 0);
        if (n <= 0) throw std::runtime_error("TCP send failed");
        sent += n;
    }
}

static void recv_all(int fd, char *buf, size_t len) {
    size_t received = 0;
    while (received < len) {
        ssize_t n = ::recv(fd, buf + received, len - received, 0);
        if (n <= 0) throw std::runtime_error("TCP recv failed");
        received += n;
    }
}

static void run_server(uint16_t port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) throw std::runtime_error("socket() failed");

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("bind() failed");
    if (listen(server_fd, 1) < 0)
        throw std::runtime_error("listen() failed");

    std::cerr << "[TCP server] Listening on port " << port << std::endl;

    while (true) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) continue;

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        std::cerr << "[TCP server] Client connected" << std::endl;

        // Protocol: read 8-byte size header, then payload, echo payload back
        while (true) {
            uint64_t payload_size = 0;
            ssize_t n = ::recv(client_fd, &payload_size, sizeof(payload_size), MSG_WAITALL);
            if (n <= 0) break;
            if (payload_size == 0) break;

            std::vector<char> buf(payload_size);
            recv_all(client_fd, buf.data(), payload_size);
            send_all(client_fd, buf.data(), payload_size);
        }

        close(client_fd);
        std::cerr << "[TCP server] Client disconnected" << std::endl;
    }
    close(server_fd);
}

static void run_client(const std::string &host, uint16_t port, size_t n_bytes, int num_runs) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct hostent *ent = gethostbyname(host.c_str());
        if (!ent) throw std::runtime_error("Can't resolve host");
        memcpy(&addr.sin_addr, ent->h_addr_list[0], ent->h_length);
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        throw std::runtime_error("connect() failed");

    std::vector<char> send_buf(n_bytes);
    std::vector<char> recv_buf(n_bytes);
    for (size_t i = 0; i < n_bytes; i++) send_buf[i] = static_cast<char>(i & 0xFF);

    // Warmup
    uint64_t size_header = n_bytes;
    send_all(fd, (char *)&size_header, sizeof(size_header));
    send_all(fd, send_buf.data(), n_bytes);
    recv_all(fd, recv_buf.data(), n_bytes);

    // CSV header
    std::cout << "run,n_bytes,send_us,recv_us,roundtrip_us" << std::endl;

    for (int run = 0; run < num_runs; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        send_all(fd, (char *)&size_header, sizeof(size_header));
        send_all(fd, send_buf.data(), n_bytes);

        auto t1 = std::chrono::high_resolution_clock::now();

        recv_all(fd, recv_buf.data(), n_bytes);

        auto t2 = std::chrono::high_resolution_clock::now();

        double send_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double recv_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double roundtrip_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();

        std::cout << run << "," << n_bytes << "," << send_us << "," << recv_us << ","
                  << roundtrip_us << std::endl;
    }

    // Verify last roundtrip
    if (memcmp(send_buf.data(), recv_buf.data(), n_bytes) != 0) {
        std::cerr << "ERROR: data mismatch!" << std::endl;
    }

    // Signal done
    uint64_t zero = 0;
    send_all(fd, (char *)&zero, sizeof(zero));

    close(fd);
}

}  // namespace tcp_transport

// ============================================================================
// UCX Transport
// ============================================================================

#ifdef USE_UCX

namespace ucx_transport {

static constexpr ucp_tag_t TAG_HANDSHAKE = 0x100;
static constexpr ucp_tag_t TAG_DATA     = 0x200;
static constexpr ucp_tag_t TAG_MASK = 0xFFFFFFFFFFFFFFFF;

// Timestamp helper for debug logs
static std::string ts() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000000;
    char buf[32];
    snprintf(buf, sizeof(buf), "[%06ld.%03ld] ", ms / 1000, ms % 1000);
    return std::string(buf);
}

struct UcxRequest {
    std::atomic<bool> complete{false};
    ucs_status_t status{UCS_OK};
};

static void request_init(void *request) {
    auto *req = static_cast<UcxRequest *>(request);
    req->complete.store(false);
    req->status = UCS_OK;
}

static void send_cb(void *request, ucs_status_t status, void *) {
    auto *req = static_cast<UcxRequest *>(request);
    req->status = status;
    req->complete.store(true);
}

static void recv_cb(void *request, ucs_status_t status, const ucp_tag_recv_info_t *info, void *) {
    auto *req = static_cast<UcxRequest *>(request);
    req->status = status;
    req->complete.store(true);
}

static void wait_request(ucp_worker_h worker, void *request, const char *op_name = "op") {
    if (request == nullptr) {
        std::cerr << ts() << op_name << ": immediate completion" << std::endl;
        return;
    }
    if (UCS_PTR_IS_ERR(request)) {
        std::string err = std::string(op_name) + " failed: " +
                          ucs_status_string(UCS_PTR_STATUS(request));
        std::cerr << ts() << err << std::endl;
        throw std::runtime_error(err);
    }
    auto *req = static_cast<UcxRequest *>(request);
    uint64_t spins = 0;
    while (!req->complete.load()) {
        ucp_worker_progress(worker);
        spins++;
        if (spins % 10000000 == 0) {
            std::cerr << ts() << op_name << ": still waiting (spins=" << spins << ")" << std::endl;
        }
    }
    if (req->status != UCS_OK) {
        std::string err = std::string(op_name) + " completed with error: " +
                          ucs_status_string(req->status);
        std::cerr << ts() << err << std::endl;
        ucp_request_free(request);
        throw std::runtime_error(err);
    }
    std::cerr << ts() << op_name << ": done (spins=" << spins << ")" << std::endl;
    ucp_request_free(request);
}

static void ucx_send(ucp_worker_h worker, ucp_ep_h ep, const void *buf, size_t len,
                     ucp_tag_t tag = TAG_DATA) {
    std::cerr << ts() << "ucx_send: len=" << len << " tag=0x" << std::hex << tag << std::dec << std::endl;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype = ucp_dt_make_contig(1);
    param.cb.send = send_cb;
    void *req = ucp_tag_send_nbx(ep, buf, len, tag, &param);
    wait_request(worker, req, "send");
}

static void ucx_flush(ucp_worker_h worker, ucp_ep_h ep) {
    std::cerr << ts() << "ucx_flush: start" << std::endl;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    param.cb.send = send_cb;
    void *req = ucp_ep_flush_nbx(ep, &param);
    wait_request(worker, req, "flush");
}

static void ucx_recv(ucp_worker_h worker, void *buf, size_t len,
                     ucp_tag_t tag = TAG_DATA) {
    std::cerr << ts() << "ucx_recv: len=" << len << " tag=0x" << std::hex << tag << std::dec << std::endl;
    ucp_request_param_t param{};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype = ucp_dt_make_contig(1);
    param.cb.recv = recv_cb;
    void *req = ucp_tag_recv_nbx(worker, buf, len, tag, TAG_MASK, &param);
    wait_request(worker, req, "recv");
}

struct UcxContext {
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;

    void init() {
        ucp_params_t params{};
        params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
        params.features = UCP_FEATURE_TAG;
        params.request_size = sizeof(UcxRequest);
        params.request_init = request_init;

        ucs_status_t st = ucp_init(&params, nullptr, &context);
        if (st != UCS_OK) throw std::runtime_error("ucp_init failed");

        ucp_worker_params_t wp{};
        wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wp.thread_mode = UCS_THREAD_MODE_SINGLE;
        st = ucp_worker_create(context, &wp, &worker);
        if (st != UCS_OK) throw std::runtime_error("ucp_worker_create failed");
    }

    ~UcxContext() {
        if (worker) ucp_worker_destroy(worker);
        if (context) ucp_cleanup(context);
    }
};

struct ConnRequest {
    ucp_conn_request_h req = nullptr;
    std::atomic<bool> ready{false};
};

static void listener_cb(ucp_conn_request_h conn_request, void *arg) {
    auto *cr = static_cast<ConnRequest *>(arg);
    cr->req = conn_request;
    cr->ready.store(true);
}

static void run_server(uint16_t port) {
    UcxContext ctx;
    ctx.init();

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    ConnRequest conn_req;

    ucp_listener_params_t lp{};
    lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lp.sockaddr.addr = (struct sockaddr *)&addr;
    lp.sockaddr.addrlen = sizeof(addr);
    lp.conn_handler.cb = listener_cb;
    lp.conn_handler.arg = &conn_req;

    ucp_listener_h listener;
    ucs_status_t st = ucp_listener_create(ctx.worker, &lp, &listener);
    if (st != UCS_OK) throw std::runtime_error("ucp_listener_create failed");

    std::cerr << "[UCX server] Listening on port " << port << std::endl;
    std::cerr << "[UCX server] UCX will auto-select transport (set UCX_TLS, UCX_RNDV_THRESH)"
              << std::endl;

    while (true) {
        // Wait for connection
        conn_req.ready.store(false);
        while (!conn_req.ready.load()) ucp_worker_progress(ctx.worker);

        // Create EP for this client
        ucp_ep_params_t ep_params{};
        ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
        ep_params.conn_request = conn_req.req;

        ucp_ep_h ep;
        st = ucp_ep_create(ctx.worker, &ep_params, &ep);
        if (st != UCS_OK) {
            std::cerr << "[UCX server] ep_create failed" << std::endl;
            continue;
        }

        std::cerr << "[UCX server] Client connected" << std::endl;

        // Wireup handshake: server sends ACK, flushes, then waits for client READY.
        // This ensures both sides have fully established the connection.
        char ack = 'A';
        try {
            ucx_send(ctx.worker, ep, &ack, 1, TAG_HANDSHAKE);
            ucx_flush(ctx.worker, ep);
            std::cerr << "[UCX server] ACK sent, waiting for client READY..." << std::endl;
            char ready = 0;
            ucx_recv(ctx.worker, &ready, 1, TAG_HANDSHAKE);
            if (ready != 'R') {
                std::cerr << "[UCX server] Invalid READY byte: " << (int)ready << std::endl;
                continue;
            }
        } catch (const std::exception &e) {
            std::cerr << "[UCX server] Wireup handshake failed: " << e.what() << std::endl;
            continue;
        }
        std::cerr << "[UCX server] Wireup complete, starting echo loop" << std::endl;

        // Echo loop: recv size header, recv payload, send payload back
        while (true) {
            uint64_t payload_size = 0;
            try {
                ucx_recv(ctx.worker, &payload_size, sizeof(payload_size));
            } catch (const std::exception &e) {
                std::cerr << "[UCX server] recv error: " << e.what() << std::endl;
                break;
            }

            std::cerr << ts() << "[UCX server] payload_size=" << payload_size << std::endl;
            if (payload_size == 0) break;  // client signals done

            std::vector<char> buf(payload_size);
            ucx_recv(ctx.worker, buf.data(), payload_size);
            ucx_send(ctx.worker, ep, buf.data(), payload_size);
            ucx_flush(ctx.worker, ep);
        }

        // Close endpoint
        ucp_request_param_t close_param{};
        close_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        close_param.flags = UCP_EP_CLOSE_FLAG_FORCE;
        void *close_req = ucp_ep_close_nbx(ep, &close_param);
        if (!UCS_PTR_IS_ERR(close_req) && close_req != nullptr) {
            auto *r = static_cast<UcxRequest *>(close_req);
            while (!r->complete.load()) ucp_worker_progress(ctx.worker);
            ucp_request_free(close_req);
        }

        std::cerr << "[UCX server] Client disconnected" << std::endl;
    }

    ucp_listener_destroy(listener);
}

static void run_client(const std::string &host, uint16_t port, size_t n_bytes, int num_runs) {
    UcxContext ctx;
    ctx.init();

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct hostent *ent = gethostbyname(host.c_str());
        if (!ent) throw std::runtime_error("Can't resolve host");
        memcpy(&addr.sin_addr, ent->h_addr_list[0], ent->h_length);
    }

    ucp_ep_params_t ep_params{};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr *)&addr;
    ep_params.sockaddr.addrlen = sizeof(addr);

    ucp_ep_h ep;
    ucs_status_t st = ucp_ep_create(ctx.worker, &ep_params, &ep);
    if (st != UCS_OK) throw std::runtime_error("ucp_ep_create failed");

    // Wireup: wait for the server's ACK to confirm connection is established
    std::cerr << "[UCX client] Waiting for server wireup ACK..." << std::endl;
    char ack = 0;
    ucx_recv(ctx.worker, &ack, 1, TAG_HANDSHAKE);
    if (ack != 'A') throw std::runtime_error("Invalid wireup ACK");

    // Send READY back so server knows we're synchronized
    char ready = 'R';
    ucx_send(ctx.worker, ep, &ready, 1, TAG_HANDSHAKE);
    ucx_flush(ctx.worker, ep);
    std::cerr << "[UCX client] Wireup complete" << std::endl;

    std::vector<char> send_buf(n_bytes);
    std::vector<char> recv_buf(n_bytes);
    for (size_t i = 0; i < n_bytes; i++) send_buf[i] = static_cast<char>(i & 0xFF);

    // Warmup — flush after header to ensure ordering
    uint64_t size_header = n_bytes;
    ucx_send(ctx.worker, ep, &size_header, sizeof(size_header));
    ucx_flush(ctx.worker, ep);
    ucx_send(ctx.worker, ep, send_buf.data(), n_bytes);
    ucx_flush(ctx.worker, ep);
    ucx_recv(ctx.worker, recv_buf.data(), n_bytes);

    // CSV header
    std::cout << "run,n_bytes,send_us,recv_us,roundtrip_us" << std::endl;

    for (int run = 0; run < num_runs; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        ucx_send(ctx.worker, ep, &size_header, sizeof(size_header));
        ucx_flush(ctx.worker, ep);
        ucx_send(ctx.worker, ep, send_buf.data(), n_bytes);
        ucx_flush(ctx.worker, ep);

        auto t1 = std::chrono::high_resolution_clock::now();

        ucx_recv(ctx.worker, recv_buf.data(), n_bytes);

        auto t2 = std::chrono::high_resolution_clock::now();

        double send_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double recv_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double roundtrip_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();

        std::cout << run << "," << n_bytes << "," << send_us << "," << recv_us << ","
                  << roundtrip_us << std::endl;
    }

    // Verify last roundtrip
    if (memcmp(send_buf.data(), recv_buf.data(), n_bytes) != 0) {
        std::cerr << "ERROR: data mismatch!" << std::endl;
    }

    // Signal done
    uint64_t zero = 0;
    ucx_send(ctx.worker, ep, &zero, sizeof(zero));

    // Close
    ucp_request_param_t close_param{};
    close_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    close_param.flags = UCP_EP_CLOSE_FLAG_FORCE;
    void *close_req = ucp_ep_close_nbx(ep, &close_param);
    if (!UCS_PTR_IS_ERR(close_req) && close_req != nullptr) {
        auto *r = static_cast<UcxRequest *>(close_req);
        while (!r->complete.load()) ucp_worker_progress(ctx.worker);
        ucp_request_free(close_req);
    }
}

}  // namespace ucx_transport

#endif  // USE_UCX

// ============================================================================
// Main
// ============================================================================

static void usage() {
    std::cerr << "Usage:\n"
              << "  Server: ./data_copy_bench server <tcp|ucx> <port>\n"
              << "  Client: ./data_copy_bench client <tcp|ucx> <host> <port> <n_bytes> <num_runs>\n"
              << "\n"
              << "Build:\n"
              << "  TCP only:  g++ -O2 -o data_copy_bench data_copy_bench.cpp -lpthread\n"
              << "  TCP + UCX: g++ -O2 -DUSE_UCX -o data_copy_bench data_copy_bench.cpp"
              << " -lucp -lucs -lpthread\n";
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        usage();
        return 1;
    }

    std::string mode = argv[1];
    std::string transport = argv[2];

    try {
        if (mode == "server") {
            uint16_t port = static_cast<uint16_t>(std::stoi(argv[3]));

            if (transport == "tcp") {
                tcp_transport::run_server(port);
            }
#ifdef USE_UCX
            else if (transport == "ucx") {
                ucx_transport::run_server(port);
            }
#endif
            else {
                std::cerr << "Unknown transport: " << transport << std::endl;
                std::cerr << "(UCX requires building with -DUSE_UCX)" << std::endl;
                return 1;
            }
        } else if (mode == "client") {
            if (argc < 7) {
                usage();
                return 1;
            }
            std::string host = argv[3];
            uint16_t port = static_cast<uint16_t>(std::stoi(argv[4]));
            size_t n_bytes = std::stoull(argv[5]);
            int num_runs = std::stoi(argv[6]);

            if (transport == "tcp") {
                tcp_transport::run_client(host, port, n_bytes, num_runs);
            }
#ifdef USE_UCX
            else if (transport == "ucx") {
                ucx_transport::run_client(host, port, n_bytes, num_runs);
            }
#endif
            else {
                std::cerr << "Unknown transport: " << transport << std::endl;
                std::cerr << "(UCX requires building with -DUSE_UCX)" << std::endl;
                return 1;
            }
        } else {
            usage();
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
