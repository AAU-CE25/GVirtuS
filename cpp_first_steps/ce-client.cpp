/*
 * RDMA Hello World — CLIENT (C++)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SYSTEM OVERVIEW
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   CLIENT MACHINE                            SERVER MACHINE
 *  ┌───────────────────────┐                ┌───────────────────────┐
 *  │  Application          │                │  Application          │
 *  │  (ce-client.cpp)      │                │  (ce-server.cpp)      │
 *  │                       │                │                       │
 *  │  ┌─────────────────┐  │                │  ┌─────────────────┐  │
 *  │  │  libibverbs API │  │                │  │  libibverbs API │  │
 *  │  └────────┬────────┘  │                │  └────────┬────────┘  │
 *  └───────────┼───────────┘                └───────────┼───────────┘
 *              │                                        │
 *  ┌───────────▼────────────────────────────────────────▼───────────┐
 *  │               NVIDIA BlueField-3 DPU                           │
 *  │                                                                │
 *  │   [ Port 0 — client side ] ──1 cable── [ Port 0 — server side ]│
 *  │              RoCEv2 / IB link  (single physical connection)    │
 *  └────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * MEMORY LAYOUT (CLIENT SIDE)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Virtual Address Space
 *  ┌──────────────────────────────────────────────┐
 *  │              RdmaRes::buf  [64 bytes]        │  ← registered as MR
 *  │  ┌────────────────────────────────────────┐  │
 *  │  │ "Hello, RDMA World!\0................" │  │
 *  │  └────────────────────────────────────────┘  │
 *  │         ▲ lkey = mr->lkey (NIC token)        │
 *  └──────────────────────────────────────────────┘
 *                    │ DMA
 *             ┌──────▼──────┐
 *             │  RDMA  NIC  │  ← NIC reads directly, no CPU copy
 *             └─────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * COMMUNICATION SEQUENCE
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   CLIENT                                          SERVER
 *     │                                               │
 *     │  [both sides: open device, alloc PD/MR/CQ/QP] │
 *     │                                               │
 *     │◄──────── TCP connect (port 18515) ───────────►│  out-of-band channel
 *     │                                               │
 *     │◄──── TCP recv: server ConnInfo (QPN,LID,GID)──│  server sends first
 *     │───── TCP send: client ConnInfo (QPN,LID,GID)─►│  client sends second
 *     │                                               │
 *     │◄──────────── TCP socket closed ──────────────►│
 *     │                                               │
 *     │  [both sides: QP  RESET → INIT → RTR → RTS]   │
 *     │                                               │
 *     │                                               │  post_recv() ← server
 *     │                                               │  arms its RQ
 *     │                                               │
 *     │══════ RDMA SEND "Hello, RDMA World!" ════════►│  NIC-to-NIC, no CPU
 *     │                                               │
 *  poll_cq()                                       poll_cq()
 *  SEND complete                                   RECV complete
 *     │                                               │
 *     │                                    print received message
 *     │                                               │
 *     │  [both sides: destroy QP/CQ/MR/PD, close dev] │
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * QP STATE MACHINE
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─────────┐   ibv_create_qp()   ┌───────┐
 *   │  RESET  │ ──────────────────► │ INIT  │  pkey, port, access flags set
 *   └─────────┘                     └───┬───┘
 *                                       │  ibv_modify_qp() → RTR
 *                                       │  (remote QPN, LID, GID configured)
 *                                   ┌───▼───┐
 *                                   │  RTR  │  Ready To Receive — RQ armed
 *                                   └───┬───┘
 *                                       │  ibv_modify_qp() → RTS
 *                                       │  (PSN, timeout, retry configured)
 *                                   ┌───▼───┐
 *                                   │  RTS  │  Ready To Send — full duplex
 *                                   └───────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SEND WORK REQUEST FLOW
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Application          SQ (Send Queue)         NIC Hardware          CQ
 *       │                      │                      │                 │
 *       │  post_send()         │                      │                 │
 *       │─── ibv_post_send()──►│                      │                 │
 *       │                      │── WR dequeued ──────►│                 │
 *       │                      │                      │ DMA from MR     │
 *       │                      │                      │ transmit packet  │
 *       │                      │                      │── WC posted ───►│
 *       │  poll_cq()           │                      │                 │
 *       │◄── ibv_poll_cq() ────────────────────────────────────────────│
 *       │  (WC status = SUCCESS)                                        │
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Build:  see Makefile   (g++ -std=c++17, links -libverbs)
 * Run:    ./ce-client <server_ip>
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <iostream>          // std::cout, std::cerr
#include <string>            // std::string, std::to_string
#include <unistd.h>          // read(), write(), close()
#include <netinet/in.h>      // sockaddr_in, htons()
#include <sys/socket.h>      // socket(), connect(), AF_INET, SOCK_STREAM
#include <netdb.h>           // getaddrinfo(), freeaddrinfo(), addrinfo
#include "rdma_common.h"     // RdmaRes, ConnInfo, and all RDMA helper functions

// ─────────────────────────────────────────────────────────────────────────────
// exchange_info()
//
// Performs the out-of-band (TCP) QP handshake:
//   • First READS the server's ConnInfo (QP number, LID, GID) from the socket.
//   • Then WRITES our own ConnInfo to the socket so the server can address us.
//
// Both sides must call this in the correct order:
//   server: write first, then read
//   client: read first, then write
//
// Parameters:
//   sock   – connected TCP socket to the server
//   local  – our own connection info (filled before the call)
//   remote – will be populated with the server's connection info
// ─────────────────────────────────────────────────────────────────────────────
static void exchange_info(int sock, ConnInfo &local, ConnInfo &remote)
{
    // Block until we receive the full ConnInfo struct from the server.
    // static_cast ensures the signed/unsigned size comparison is clean.
    if (read(sock, &remote, sizeof(remote)) != static_cast<ssize_t>(sizeof(remote)))
        die("read conn_info");   // fatal if we get a short read or error

    // Send our own ConnInfo to the server so it can set up its RTR state.
    if (write(sock, &local, sizeof(local)) != static_cast<ssize_t>(sizeof(local)))
        die("write conn_info");  // fatal if the write fails or is incomplete
}

// ─────────────────────────────────────────────────────────────────────────────
// main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    // Require exactly one argument: the server's IP address or hostname.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_ip>\n";
        return EXIT_FAILURE;       // non-zero exit signals failure to the shell
    }
    const std::string server_ip = argv[1]; // store the server address as a C++ string

    // ── Step 1: Open RDMA device & allocate all necessary resources ──────────
    RdmaRes r{};                   // zero-initialise the resource bundle
    r.ctx = open_device();         // open the first available IB/RoCE device
    alloc_resources(r);            // create PD, register MR, create CQ and RC QP

    // ── Step 2: Query local port attributes (LID + GID) ─────────────────────
    ibv_port_attr port_attr{};             // will hold link-layer port attributes
    if (ibv_query_port(r.ctx, IB_PORT, &port_attr))
        die("ibv_query_port");             // fails if port doesn't exist or is down

    ibv_gid local_gid{};                   // 128-bit GID used for RoCE routing
    if (ibv_query_gid(r.ctx, IB_PORT, GID_INDEX, &local_gid))
        die("ibv_query_gid");              // GID_INDEX 0 = RoCEv1, 3 = RoCEv2

    // Pack our addressing info into ConnInfo so it can be sent to the server.
    ConnInfo local_info{};
    local_info.qp_num = r.qp->qp_num;     // our QP number — server sends here
    local_info.lid    = port_attr.lid;    // Local IDentifier (used in IB networks)
    std::memcpy(local_info.gid, local_gid.raw, 16); // GID is 16 bytes (IPv6-style)

    // ── Step 3: TCP handshake — resolve server address & connect ────────────
    addrinfo hints{};                      // criteria for address resolution
    hints.ai_family   = AF_INET;           // IPv4 only
    hints.ai_socktype = SOCK_STREAM;       // TCP (reliable, connection-oriented)

    addrinfo *res = nullptr;               // getaddrinfo fills this linked list
    const std::string port_str = std::to_string(RDMA_PORT); // convert port int → string
    if (getaddrinfo(server_ip.c_str(), port_str.c_str(), &hints, &res) != 0)
        die("getaddrinfo");                // fails if hostname cannot be resolved

    // Create the TCP socket using the first result returned by getaddrinfo.
    int sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) die("socket");           // fails if OS cannot allocate a socket fd

    // Initiate the TCP three-way handshake to the server.
    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0)
        die("connect");                    // fails if server is not listening yet
    freeaddrinfo(res);                     // release the address list — no longer needed
    std::cout << "[client] Connected to server " << server_ip << '\n';

    // ── Step 4: Exchange QP connection info over the TCP socket ─────────────
    ConnInfo remote_info{};                // will be filled with server's QP info
    exchange_info(sock, local_info, remote_info); // bidirectional swap over TCP
    close(sock);                           // TCP channel no longer needed after swap

    // ── Step 5: Transition our QP from INIT → RTR → RTS ────────────────────
    // RTR (Ready To Receive) configures the remote address handle using
    // remote_info (QP num, LID, GID). RTS (Ready To Send) arms the send queue.
    connect_qp(r, remote_info);            // defined in rdma_common.h
    std::cout << "[client] QP connected\n";

    // ── Step 6: Fill the message buffer and post an RDMA SEND ───────────────
    std::snprintf(r.buf, MSG_SIZE, "Hello, RDMA World!"); // write payload into the MR
    post_send(r);                          // enqueue a SEND work request on the SQ
    poll_cq(r);                            // spin on the CQ until the send completes
    std::cout << "[client] Sent: \"" << r.buf << "\"\n";

    // ── Step 7: Tear down all RDMA resources in reverse-allocation order ────
    free_resources(r);                     // destroy QP → CQ → dereg MR → dealloc PD → close device
    return 0;                              // clean exit
}

// ─────────────────────────────────────────────────────────────────────────────
// GLOSSARY — RDMA / InfiniBand / RoCE terminology used in this file
// ─────────────────────────────────────────────────────────────────────────────
//
// RDMA (Remote Direct Memory Access)
//   A technology that allows a host to read from or write to the memory of a
//   remote host without involving the remote CPU or OS. Data moves directly
//   between NICs, bypassing kernel networking stacks.
//
// IB (InfiniBand)
//   A high-speed, low-latency interconnect standard. The libibverbs API was
//   originally designed for IB but also supports RoCE and iWARP.
//
// RoCE (RDMA over Converged Ethernet)
//   Runs the InfiniBand transport protocol over standard Ethernet hardware.
//   RoCEv1 uses EtherType framing; RoCEv2 encapsulates in UDP/IP.
//
// libibverbs
//   The userspace C library that exposes the verbs API for RDMA programming.
//   Linked with -libverbs. All ibv_* functions come from this library.
//
// PD (Protection Domain)  [ibv_pd]
//   A scope that groups MRs and QPs together. Only resources in the same PD
//   can communicate with each other, providing a basic isolation boundary.
//
// MR (Memory Region)  [ibv_mr]
//   A range of virtual memory that has been registered with the RDMA NIC.
//   Registration pins the pages in RAM and grants the NIC DMA access.
//   Each MR has an lkey (for local access) and rkey (for remote access).
//
// lkey (Local Key)
//   A 32-bit token associated with an MR. Must be provided in every Scatter/
//   Gather Element (SGE) that references the MR for local send/receive ops.
//
// rkey (Remote Key)
//   A token that authorises a remote peer to perform one-sided READ/WRITE
//   operations on the MR. Not needed for basic SEND/RECV.
//
// CQ (Completion Queue)  [ibv_cq]
//   A queue where the NIC posts Work Completions (WCs) after finishing a
//   send or receive operation. Applications poll the CQ to detect completion.
//
// WC (Work Completion)  [ibv_wc]
//   An entry posted to the CQ by the NIC indicating that a WR has finished.
//   Contains a status field (IBV_WC_SUCCESS on success) and opcode details.
//
// QP (Queue Pair)  [ibv_qp]
//   The fundamental RDMA communication endpoint. Consists of:
//     • SQ (Send Queue)   – holds outgoing Work Requests
//     • RQ (Receive Queue) – holds pre-posted receive buffers
//   QPs are always paired; this client's SQ posts to the server's RQ.
//
// RC (Reliable Connected) QP type  [IBV_QPT_RC]
//   A QP transport mode providing reliable, in-order, connection-oriented
//   delivery. The NIC handles retransmission automatically. This is the most
//   commonly used type for RDMA applications.
//
// WR (Work Request)  [ibv_send_wr / ibv_recv_wr]
//   A descriptor posted to the SQ or RQ that describes an operation to
//   perform (SEND, RECV, RDMA READ, RDMA WRITE, etc.) and the memory buffer.
//
// SGE (Scatter/Gather Element)  [ibv_sge]
//   Describes one contiguous memory buffer (addr + length + lkey) inside a
//   WR. Multiple SGEs allow scatter/gather I/O across non-contiguous buffers.
//
// LID (Local IDentifier)
//   A 16-bit address assigned to each port in a pure InfiniBand subnet.
//   Analogous to a MAC address. Not used for routing in RoCEv2 (GID is used
//   instead), but must still be populated in the address handle struct.
//
// GID (Global IDentifier)
//   A 128-bit address (structured like an IPv6 address) that uniquely
//   identifies an RDMA port globally. Used for inter-subnet routing and
//   mandatory for RoCE. GID index 0 = RoCEv1, GID index 3 = RoCEv2 (common).
//
// AH (Address Handle)  [ibv_ah_attr inside ibv_qp_attr]
//   Encodes all addressing information needed to reach the remote port:
//   LID, GID, traffic class, hop limit, etc. Set during the RTR transition.
//
// QP State Machine
//   A QP must be moved through these states before it can transfer data:
//     RESET  → initial state after creation (no traffic possible)
//     INIT   → QP is configured but not yet able to communicate
//     RTR    → Ready To Receive: remote address is configured, RQ is armed
//     RTS    → Ready To Send: SQ is armed, full bidirectional traffic allowed
//   State transitions are performed with ibv_modify_qp().
//
// MTU (Maximum Transfer Unit)  [IBV_MTU_*]
//   The largest packet payload the QP will use. IBV_MTU_1024 = 1024 bytes.
//   Must match or be smaller than the physical link MTU.
//
// PSN (Packet Sequence Number)
//   A 24-bit counter used for in-order delivery and duplicate detection.
//   sq_psn is the starting sequence number for outgoing packets;
//   rq_psn is the expected starting sequence number for incoming packets.
//
// Out-of-band (OOB) connection
//   A separate communication channel (TCP in this example) used only to
//   exchange addressing metadata (QP numbers, LIDs, GIDs) before the RDMA
//   data path is established. Not used for actual data transfer.
//
// ─────────────────────────────────────────────────────────────────────────────
