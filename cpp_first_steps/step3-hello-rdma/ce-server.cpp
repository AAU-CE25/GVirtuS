/*
 * RDMA Hello World — SERVER (C++)
 *
 * Listens for a TCP control connection, exchanges QP info with the
 * client, then waits to receive the "Hello, RDMA World!" message over
 * a Reliable-Connected (RC) Queue Pair.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PHYSICAL TOPOLOGY — Single NVIDIA BlueField-3 Link
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   CLIENT MACHINE                         SERVER MACHINE
 *  ┌──────────────────────┐               ┌──────────────────────┐
 *  │   Application        │               │   Application        │
 *  │   (ce-client.cpp)    │               │   (ce-server.cpp)    │
 *  │                      │               │                      │
 *  │   ┌──────────────┐   │               │   ┌──────────────┐   │
 *  │   │ libibverbs   │   │               │   │ libibverbs   │   │
 *  │   └──────┬───────┘   │               │   └──────┬───────┘   │
 *  │          │           │               │          │           │
 *  │   ┌──────▼───────────────────────────────────────▼───────┐  │
 *  │   │          NVIDIA BlueField-3 DPU  (single NIC)        │  │
 *  │   │                                                      │  │
 *  │   │   Port 0 ◄──────── 1x physical cable ──────► Port 0 │  │
 *  │   │   (client side)     RoCE / IB link    (server side)  │  │
 *  │   └──────────────────────────────────────────────────────┘  │
 *  └──────────────────────┘               └──────────────────────┘
 *
 *  The BlueField-3 is an NVIDIA DPU (Data Processing Unit) — a SmartNIC
 *  with an onboard ARM CPU. It exposes standard RDMA verbs to the host via
 *  libibverbs. Both endpoints share the same physical device but communicate
 *  over the single RoCE/IB link as if they were separate machines.
 *
 *  ┌────────────────────────────────────────────────────────────┐
 *  │              BlueField-3 DPU internals                     │
 *  │                                                            │
 *  │  ┌──────────────┐      ┌──────────────────────────────┐    │
 *  │  │  Host (x86)  │      │  DPU ARM cores (Cortex-A78)  │    │
 *  │  │  PCIe attach │      │  Runs OS, offload programs   │   │
 *  │  └──────┬───────┘      └────────────┬─────────────────┘   │
 *  │         │                           │                      │
 *  │  ┌──────▼───────────────────────────▼─────────────────┐   │
 *  │  │           ConnectX-7 RDMA NIC core                  │   │
 *  │  │    2 x 400 GbE / HDR200 IB ports (hardware)         │   │
 *  │  └────────────────────────┬───────────────────────────┘   │
 *  └───────────────────────────┼────────────────────────────────┘
 *                              │
 *                    Physical cable (this demo uses 1 port)
 *
 * Build:  see Makefile
 * Run:    ./ce-server
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <iostream>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "rdma_common.h"

// Exchange ConnInfo over the TCP control socket
static void exchange_info(int sock, ConnInfo &local, ConnInfo &remote)
{
    if (write(sock, &local, sizeof(local)) != static_cast<ssize_t>(sizeof(local)))
        die("write conn_info");
    if (read(sock, &remote, sizeof(remote)) != static_cast<ssize_t>(sizeof(remote)))
        die("read conn_info");
}

int main()
{
    // 1. Open RDMA device and allocate resources
    RdmaRes r{};
    r.ctx = open_device();
    alloc_resources(r);

    // 2. Query local LID and GID
    ibv_port_attr port_attr{};
    if (ibv_query_port(r.ctx, IB_PORT, &port_attr))
        die("ibv_query_port");

    ibv_gid local_gid{};
    if (ibv_query_gid(r.ctx, IB_PORT, GID_INDEX, &local_gid))
        die("ibv_query_gid");

    ConnInfo local_info{};
    local_info.qp_num = r.qp->qp_num;
    local_info.lid    = port_attr.lid;
    std::memcpy(local_info.gid, local_gid.raw, 16);

    // 3. TCP handshake — wait for client to connect
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) die("socket");

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(RDMA_PORT));

    if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        die("bind");
    if (listen(server_fd, 1) < 0)
        die("listen");

    std::cout << "[server] Listening on port " << RDMA_PORT << " …\n";

    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) die("accept");
    std::cout << "[server] Client connected\n";

    // 4. Exchange QP info
    ConnInfo remote_info{};
    exchange_info(client_fd, local_info, remote_info);
    close(client_fd);
    close(server_fd);

    // 5. Bring QP to RTS
    connect_qp(r, remote_info);
    std::cout << "[server] QP connected\n";

    // 6. Post a RECV and wait
    post_recv(r);
    std::cout << "[server] Waiting for message …\n";
    poll_cq(r);

    std::cout << "[server] Received: \"" << std::string(r.buf, MSG_SIZE) << "\"\n";

    // 7. Cleanup
    free_resources(r);
    return 0;
}
