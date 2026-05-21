//
// Created by Mariano Aponte on 07/12/23.
//

#include "RdmaCommunicator.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <sstream>
#include <chrono>
#include <stdexcept>

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Rdma.h>
#include <gvirtus/communicators/Endpoint_Tcp.h>

using gvirtus::communicators::RdmaCommunicator;

namespace {
constexpr size_t kPreRegisteredBufferSize = 1024 * 5;
// Keep SEND/RECV messages comfortably below the observed 128 MiB limit.
// Both frontend and backend use this communicator, so rebuild both sides
// after changing this value.
constexpr size_t kRdmaChunkSize = 64 * 1024 * 1024;
}  // namespace

RdmaCommunicator::RdmaCommunicator(const std::string &hostname, const std::string &port,
                                   bool isRoce)
    : isRoce(isRoce) {
#ifdef DEBUG
    std::cout << "Called RdmaCommunicator(" << hostname << ", " << port
              << ", isRoce=" << isRoce << ")" << std::endl;
#endif

    if (port.empty()) {
        throw std::runtime_error("RdmaCommunicator: Port not specified...");
    }

    hostent *ent = gethostbyname(hostname.c_str());
    if (ent == nullptr) {
        std::ostringstream oss;
        oss << "RdmaCommunicator: Can't resolve hostname \"" << hostname << "\"...";
        throw std::runtime_error(oss.str());
    }

    strcpy(this->hostname, hostname.c_str());
    strcpy(this->port, port.c_str());

    rdmaCmId = nullptr;
    rdmaCmListenId = nullptr;
}

// Constructor used on the server side when a connection is accepted.
RdmaCommunicator::RdmaCommunicator(rdma_cm_id *rdmaCmId) : isRoce(false) {
#ifdef DEBUG
    std::cout << "Called RdmaCommunicator(rdma_cm_id *rdmaCmId)" << std::endl;
#endif

    this->rdmaCmId = rdmaCmId;
    this->rdmaCmListenId = nullptr;

    preregisteredMr = ktm_rdma_reg_msgs(rdmaCmId, preregisteredBuffer,
                                        kPreRegisteredBufferSize);
    if (preregisteredMr == nullptr) {
        throw std::runtime_error("RdmaCommunicator: failed to register small-message buffer");
    }
}

RdmaCommunicator::~RdmaCommunicator() {
#ifdef DEBUG
    std::cout << "Called ~RdmaCommunicator()" << std::endl;
#endif

    Close();
}

void RdmaCommunicator::Close() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Close(): called." << std::endl;
#endif

    if (rdmaCmId) {
        rdma_disconnect(rdmaCmId);
        rdma_destroy_id(rdmaCmId);
        rdmaCmId = nullptr;
    }

    if (rdmaCmListenId) {
        rdma_destroy_id(rdmaCmListenId);
        rdmaCmListenId = nullptr;
    }
}

void RdmaCommunicator::Serve() {
#ifdef DEBUG
    std::cout << "Called Serve()" << std::endl;
#endif

    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));

    hints.ai_port_space = isRoce ? RDMA_PS_TCP : RDMA_PS_IB;
    hints.ai_flags = RAI_PASSIVE;

    rdma_addrinfo *rdmaAddrinfo = nullptr;
    ktm_rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);

    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    qpInitAttr.cap.max_send_wr = 64;
    qpInitAttr.cap.max_recv_wr = 64;
    qpInitAttr.cap.max_send_sge = 1;
    qpInitAttr.cap.max_recv_sge = 1;
    qpInitAttr.sq_sig_all = 1;
    qpInitAttr.qp_type = IBV_QPT_RC;

    ktm_rdma_create_ep(&rdmaCmListenId, rdmaAddrinfo, nullptr, &qpInitAttr);
    rdma_freeaddrinfo(rdmaAddrinfo);

    ktm_rdma_listen(rdmaCmListenId, BACKLOG);
}

void RdmaCommunicator::PrePostInitialRecv() {
#ifdef DEBUG
    std::cout << "RDMA_DEBUG: pre-posting initial recv size=256" << std::endl;
#endif

    if (preregisteredMr == nullptr) {
        preregisteredMr = ktm_rdma_reg_msgs(rdmaCmId, preregisteredBuffer,
                                            kPreRegisteredBufferSize);
        if (preregisteredMr == nullptr) {
            throw std::runtime_error(
                "RdmaCommunicator: failed to register initial receive buffer");
        }
    }

    ktm_rdma_post_recv(rdmaCmId, nullptr, preregisteredBuffer, 256, preregisteredMr);
    hasPrepostedRecv = true;

#ifdef DEBUG
    std::cout << "RDMA_DEBUG: initial recv posted" << std::endl;
#endif
}

const gvirtus::communicators::Communicator *const RdmaCommunicator::Accept() const {
#ifdef DEBUG
    std::cout << "Called Accept()" << std::endl;
#endif

    rdma_cm_id *clientRdmaCmId = nullptr;
    ktm_rdma_get_request(rdmaCmListenId, &clientRdmaCmId);

    auto *client = new RdmaCommunicator(clientRdmaCmId);

    // Important for RDMA SEND/RECV:
    // post the first receive before accepting the connection, otherwise
    // the frontend can send immediately after connect and get stuck/RNR.
    client->PrePostInitialRecv();

    ktm_rdma_accept(clientRdmaCmId, nullptr);

    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(calloc(1, sizeof(ibv_qp_attr)));
    if (ibvQpAttr != nullptr) {
        ibvQpAttr->min_rnr_timer = 12;
        // Chunked SEND/RECV can briefly make the sender arrive before the
        // receiver has posted the next chunk. Infinite RNR retries keep the
        // RC QP alive during that normal scheduling gap.
        ibvQpAttr->rnr_retry = 7;

        if (ibv_modify_qp(clientRdmaCmId->qp, ibvQpAttr,
                          IBV_QP_MIN_RNR_TIMER | IBV_QP_RNR_RETRY)) {
            fprintf(stderr, "ibv_modify_qp() failed: %s\n", strerror(errno));
        }

        free(ibvQpAttr);
    }

    return client;
}

void RdmaCommunicator::Connect() {
#ifdef DEBUG
    std::cout << "Called Connect()" << std::endl;
#endif

    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_port_space = isRoce ? RDMA_PS_TCP : RDMA_PS_IB;

    rdma_addrinfo *rdmaAddrinfo = nullptr;
    ktm_rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);

    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    qpInitAttr.cap.max_send_wr = 10;
    qpInitAttr.cap.max_recv_wr = 10;
    qpInitAttr.cap.max_send_sge = 10;
    qpInitAttr.cap.max_recv_sge = 10;
    qpInitAttr.sq_sig_all = 1;
    qpInitAttr.qp_type = IBV_QPT_RC;

    ktm_rdma_create_ep(&rdmaCmId, rdmaAddrinfo, nullptr, &qpInitAttr);
    rdma_freeaddrinfo(rdmaAddrinfo);

    ktm_rdma_connect(rdmaCmId, nullptr);

    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    if (!ibvQpAttr) {
        throw std::runtime_error("RdmaCommunicator::Connect(): malloc failed");
    }

    memset(ibvQpAttr, 0, sizeof(ibv_qp_attr));
    ibvQpAttr->min_rnr_timer = 12;
    // Same reasoning as the server side: tolerate transient RNR while
    // chunked payloads are exchanged.
    ibvQpAttr->rnr_retry = 7;

    if (ibv_modify_qp(rdmaCmId->qp, ibvQpAttr,
                      IBV_QP_MIN_RNR_TIMER | IBV_QP_RNR_RETRY)) {
        fprintf(stderr, "ibv_modify_attr() failed: %s\n", strerror(errno));
    }

    free(ibvQpAttr);

    preregisteredMr = ktm_rdma_reg_msgs(rdmaCmId, preregisteredBuffer,
                                        kPreRegisteredBufferSize);
    if (preregisteredMr == nullptr) {
        throw std::runtime_error("RdmaCommunicator: failed to register small-message buffer");
    }
}

size_t RdmaCommunicator::Read(char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "Called Read(char *buffer, size_t size) - Size: " << size << std::endl;
#endif

    if (size == 0) {
        return 0;
    }

    auto poll_recv_completion = [&](long timeout_ms) -> size_t {
        int num_comp;
        auto poll_start = std::chrono::steady_clock::now();

        do {
            num_comp = ibv_poll_cq(rdmaCmId->recv_cq, 1, &workCompletion);

            if (num_comp == 0 && timeout_ms > 0) {
                auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - poll_start)
                        .count();

                if (elapsed_ms >= timeout_ms) {
#ifdef DEBUG
                    std::cout << "RDMA_DEBUG: routine recv timeout after "
                              << elapsed_ms << " ms" << std::endl;
#endif
                    throw std::runtime_error("RDMA routine receive timeout");
                }
            }
        } while (num_comp == 0);

        if (num_comp < 0) {
            throw std::runtime_error("ibv_poll_cq() failed");
        }

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: recv CQ completion status=" << workCompletion.status
                  << " opcode=" << workCompletion.opcode
                  << " byte_len=" << workCompletion.byte_len << std::endl;
#endif

        if (workCompletion.status != IBV_WC_SUCCESS) {
            throw std::runtime_error("RDMA recv completion failed: " +
                                     std::string(ibv_wc_status_str(workCompletion.status)));
        }

        return static_cast<size_t>(workCompletion.byte_len);
    };

    if (size < kPreRegisteredBufferSize) {
        if (hasPrepostedRecv) {
#ifdef DEBUG
            std::cout << "RDMA_DEBUG: using pre-posted recv for size=" << size << std::endl;
#endif
            hasPrepostedRecv = false;
        } else {
#ifdef DEBUG
            std::cout << "RDMA_DEBUG: posting recv small size=" << size << std::endl;
#endif
            ktm_rdma_post_recv(rdmaCmId, nullptr, preregisteredBuffer,
                               kPreRegisteredBufferSize, preregisteredMr);
        }

        long routine_recv_timeout_ms = 0;
        if (size == 256) {
            const char *timeout_env = std::getenv("GVIRTUS_RDMA_ROUTINE_RECV_TIMEOUT_MS");
            if (timeout_env != nullptr) {
                routine_recv_timeout_ms = std::strtol(timeout_env, nullptr, 10);
            }
        }

        const size_t received = poll_recv_completion(routine_recv_timeout_ms);

        if (received > size) {
            throw std::runtime_error("RDMA small read received more bytes than requested");
        }

        memcpy(buffer, preregisteredBuffer, received);

        if (received < size) {
            // getstring() reads a fixed 256-byte routine buffer, while the
            // frontend sends only strlen(routine)+1 bytes. Zero-fill the rest
            // so stale bytes from preregisteredBuffer cannot leak into the
            // routine string.
            if (size == 256) {
                memset(buffer + received, 0, size - received);
            } else {
                throw std::runtime_error("RDMA small read short completion: expected " +
                                         std::to_string(size) + " bytes, got " +
                                         std::to_string(received));
            }
        }

        // Preserve the old blocking-read contract used by Buffer/getstring:
        // once this method succeeds, the caller can treat the requested
        // destination range as fully initialized.
        return size;
    }

#ifdef DEBUG
    std::cout << "RDMA_DEBUG: posting recv large total size=" << size
              << " chunk_size=" << kRdmaChunkSize << std::endl;
#endif

    size_t offset = 0;
    while (offset < size) {
        const size_t remaining = size - offset;
        const size_t chunk_capacity =
            remaining > kRdmaChunkSize ? kRdmaChunkSize : remaining;
        char *recvAddr = buffer + offset;

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: posting recv chunk offset=" << offset
                  << " capacity=" << chunk_capacity << std::endl;
#endif

        ibv_mr *chunkMr = ktm_rdma_reg_msgs(rdmaCmId, recvAddr, chunk_capacity);
        if (chunkMr == nullptr) {
            throw std::runtime_error("RDMA large read failed to register receive chunk");
        }

        ktm_rdma_post_recv(rdmaCmId, nullptr, recvAddr, chunk_capacity, chunkMr);

        size_t received = 0;
        try {
            received = poll_recv_completion(0);
        } catch (...) {
            rdma_dereg_mr(chunkMr);
            throw;
        }

        rdma_dereg_mr(chunkMr);

        if (received == 0) {
            throw std::runtime_error("RDMA large read received zero-byte completion");
        }

        if (received > chunk_capacity || received > remaining) {
            throw std::runtime_error("RDMA large read received invalid chunk length: got " +
                                     std::to_string(received) + " bytes, capacity " +
                                     std::to_string(chunk_capacity) + ", remaining " +
                                     std::to_string(remaining));
        }

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: received chunk offset=" << offset
                  << " bytes=" << received << std::endl;
#endif

        // Advance by the actual CQ byte_len, not by the requested receive
        // capacity. The final chunk can be smaller, and this also prevents
        // silent truncation if the peer sends smaller chunks.
        offset += received;
    }

    return size;
}
size_t RdmaCommunicator::Write(const char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "Called Write(const char *buffer, size_t size) - Size: " << size << std::endl;
#endif

    if (size == 0) {
        return 0;
    }

    auto poll_send_completion = [&]() {
        int num_comp;
        do {
            num_comp = ibv_poll_cq(rdmaCmId->send_cq, 1, &workCompletion);
        } while (num_comp == 0);

        if (num_comp < 0) {
            throw std::runtime_error("ibv_poll_cq() failed");
        }

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: send CQ completion status=" << workCompletion.status
                  << " opcode=" << workCompletion.opcode
                  << " byte_len=" << workCompletion.byte_len << std::endl;
#endif

        if (workCompletion.status != IBV_WC_SUCCESS) {
            throw std::runtime_error("RDMA send completion failed: " +
                                     std::string(ibv_wc_status_str(workCompletion.status)));
        }
    };

    if (size < kPreRegisteredBufferSize) {
        if (preregisteredMr == nullptr) {
            throw std::runtime_error("RDMA small write has no registered buffer");
        }

        memcpy(preregisteredBuffer, buffer, size);

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: posting send small size=" << size << std::endl;
#endif

        ktm_rdma_post_send(rdmaCmId, nullptr, preregisteredBuffer, size,
                           preregisteredMr, IBV_SEND_SIGNALED);

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: post_send returned small size=" << size << std::endl;
#endif

        poll_send_completion();
        return size;
    }

#ifdef DEBUG
    std::cout << "RDMA_DEBUG: posting send large total size=" << size
              << " chunk_size=" << kRdmaChunkSize << std::endl;
#endif

    size_t offset = 0;
    while (offset < size) {
        size_t remaining = size - offset;
        size_t chunk = remaining > kRdmaChunkSize ? kRdmaChunkSize : remaining;

        char *sendAddr = static_cast<char *>(malloc(chunk));
        if (sendAddr == nullptr) {
            throw std::runtime_error("malloc failed in RdmaCommunicator::Write chunk");
        }

        memcpy(sendAddr, buffer + offset, chunk);

#ifdef DEBUG
        std::cout << "RDMA_DEBUG: posting send chunk offset=" << offset
                  << " size=" << chunk << std::endl;
#endif

        ibv_mr *chunkMr = ktm_rdma_reg_msgs(rdmaCmId, sendAddr, chunk);
        if (chunkMr == nullptr) {
            free(sendAddr);
            throw std::runtime_error("RDMA large write failed to register send chunk");
        }

        try {
            ktm_rdma_post_send(rdmaCmId, nullptr, sendAddr, chunk,
                               chunkMr, IBV_SEND_SIGNALED);

#ifdef DEBUG
            std::cout << "RDMA_DEBUG: post_send returned chunk offset=" << offset
                      << " size=" << chunk << std::endl;
#endif

            poll_send_completion();
        } catch (...) {
            if (chunkMr != nullptr) {
                rdma_dereg_mr(chunkMr);
            }
            free(sendAddr);
            throw;
        }

        if (chunkMr != nullptr) {
            rdma_dereg_mr(chunkMr);
        }
        free(sendAddr);

        offset += chunk;
    }

    return size;
}

void RdmaCommunicator::Sync() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Sync(): called." << std::endl;
#endif
}

// Factory function to create an RDMA communicator.
extern "C" std::shared_ptr<RdmaCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    std::string hostname =
        std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->address();

    std::string port = std::to_string(
        std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->port());

    bool isRoce =
        std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->suite() ==
        "roce-rdma";

    return std::make_shared<RdmaCommunicator>(hostname, port, isRoce);
}