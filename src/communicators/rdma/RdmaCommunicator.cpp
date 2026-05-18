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
#include <stdexcept>

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Rdma.h>
#include <gvirtus/communicators/Endpoint_Tcp.h>

using gvirtus::communicators::RdmaCommunicator;

namespace {
constexpr size_t kPreRegisteredBufferSize = 1024 * 5;
constexpr size_t kRdmaChunkSize = 128 * 1024 * 1024;
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

const gvirtus::communicators::Communicator *const RdmaCommunicator::Accept() const {
#ifdef DEBUG
    std::cout << "Called Accept()" << std::endl;
#endif

    rdma_cm_id *clientRdmaCmId = nullptr;
    ktm_rdma_get_request(rdmaCmListenId, &clientRdmaCmId);
    ktm_rdma_accept(clientRdmaCmId, nullptr);

    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    if (!ibvQpAttr) {
        throw std::runtime_error("RdmaCommunicator::Accept(): malloc failed");
    }

    memset(ibvQpAttr, 0, sizeof(ibv_qp_attr));
    ibvQpAttr->min_rnr_timer = 1;

    if (ibv_modify_qp(clientRdmaCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "ibv_modify_attr() failed: %s\n", strerror(errno));
    }

    free(ibvQpAttr);

    return new RdmaCommunicator(clientRdmaCmId);
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
    ibvQpAttr->min_rnr_timer = 1;

    if (ibv_modify_qp(rdmaCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "ibv_modify_attr() failed: %s\n", strerror(errno));
    }

    free(ibvQpAttr);

    preregisteredMr = ktm_rdma_reg_msgs(rdmaCmId, preregisteredBuffer,
                                        kPreRegisteredBufferSize);
}

size_t RdmaCommunicator::Read(char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "Called Read(char *buffer, size_t size) - Size: " << size << std::endl;
#endif

    if (size == 0) {
        return 0;
    }

    // Small/control messages are single RDMA SEND messages.
    // Important: the posted receive size may be larger than the actual sent size.
    // Example: backend posts 30 bytes for a routine name, frontend sends strlen(routine)+1.
    if (size <= kRdmaChunkSize) {
        ibv_mr *chunkMr = nullptr;
        void *recvAddr = nullptr;

        if (size <= kPreRegisteredBufferSize) {
            recvAddr = preregisteredBuffer;
            chunkMr = preregisteredMr;
        } else {
            recvAddr = buffer;
            chunkMr = ktm_rdma_reg_msgs(rdmaCmId, recvAddr, size);
        }

        ktm_rdma_post_recv(rdmaCmId, nullptr, recvAddr, size, chunkMr);

        int num_comp;
        do {
            num_comp = ibv_poll_cq(rdmaCmId->recv_cq, 1, &workCompletion);
        } while (num_comp == 0);

        if (num_comp < 0) {
            if (chunkMr && chunkMr != preregisteredMr) {
                ibv_dereg_mr(chunkMr);
            }
            throw std::runtime_error("ibv_poll_cq() failed");
        }

        if (workCompletion.status == IBV_WC_WR_FLUSH_ERR) {
            if (chunkMr && chunkMr != preregisteredMr) {
                ibv_dereg_mr(chunkMr);
            }
            return 0;
        }

        if (workCompletion.status != IBV_WC_SUCCESS) {
            if (chunkMr && chunkMr != preregisteredMr) {
                ibv_dereg_mr(chunkMr);
            }
            throw std::runtime_error("Failed status: " +
                                     std::string(ibv_wc_status_str(workCompletion.status)));
        }

#ifdef DEBUG
        std::cout << "  -> recv completed byte_len=" << workCompletion.byte_len
                  << " requested=" << size << std::endl;
#endif

        const size_t actual = std::min(size, static_cast<size_t>(workCompletion.byte_len));

        if (size <= kPreRegisteredBufferSize) {
            memcpy(buffer, preregisteredBuffer, actual);
        }

        if (chunkMr && chunkMr != preregisteredMr) {
            ibv_dereg_mr(chunkMr);
        }

        return actual;
    }

#ifdef DEBUG
    std::cout << "  -> RDMA recv chunked transfer total=" << size
              << " chunk_size=" << kRdmaChunkSize
              << " full_mr=1" << std::endl;
#endif

    // Large payload optimization:
    // Register the full destination buffer once, then post chunked receives
    // using the same MR. This avoids register/deregister per chunk.
    ibv_mr *fullMr = ktm_rdma_reg_msgs(rdmaCmId, buffer, size);

    size_t total = 0;

    try {
        while (total < size) {
            const size_t remaining = size - total;
            const size_t chunk = std::min(remaining, kRdmaChunkSize);

            void *recvAddr = buffer + total;

            ktm_rdma_post_recv(rdmaCmId, nullptr, recvAddr, chunk, fullMr);

            int num_comp;
            do {
                num_comp = ibv_poll_cq(rdmaCmId->recv_cq, 1, &workCompletion);
            } while (num_comp == 0);

            if (num_comp < 0) {
                throw std::runtime_error("ibv_poll_cq() failed");
            }

            if (workCompletion.status == IBV_WC_WR_FLUSH_ERR) {
                ibv_dereg_mr(fullMr);
                return total;
            }

            if (workCompletion.status != IBV_WC_SUCCESS) {
                throw std::runtime_error("Failed status: " +
                                         std::string(ibv_wc_status_str(workCompletion.status)));
            }

            const size_t actual = static_cast<size_t>(workCompletion.byte_len);

            if (actual == 0) {
                ibv_dereg_mr(fullMr);
                return total;
            }

            total += actual;
        }
    } catch (...) {
        ibv_dereg_mr(fullMr);
        throw;
    }

    ibv_dereg_mr(fullMr);
    return total;
}

size_t RdmaCommunicator::Write(const char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "Called Write(const char *buffer, size_t size) - Size: " << size << std::endl;
#endif

    if (size == 0) {
        return 0;
    }

    // Small/control messages stay as before.
    if (size <= kRdmaChunkSize) {
#ifdef DEBUG
        std::cout << "  -> posting send size=" << size << std::endl;
#endif

        ibv_mr *chunkMr = nullptr;
        void *sendAddr = nullptr;

        if (size <= kPreRegisteredBufferSize) {
            memcpy(preregisteredBuffer, buffer, size);
            sendAddr = preregisteredBuffer;
            chunkMr = preregisteredMr;
        } else {
            sendAddr = const_cast<char *>(buffer);
            chunkMr = ktm_rdma_reg_msgs(rdmaCmId, sendAddr, size);
        }

        ktm_rdma_post_send(rdmaCmId, nullptr, sendAddr, size, chunkMr, IBV_SEND_SIGNALED);

        int num_comp;
        do {
            num_comp = ibv_poll_cq(rdmaCmId->send_cq, 1, &workCompletion);
        } while (num_comp == 0);

        if (num_comp < 0) {
            if (chunkMr && chunkMr != preregisteredMr) {
                ibv_dereg_mr(chunkMr);
            }
            throw std::runtime_error("ibv_poll_cq() failed");
        }

        if (workCompletion.status != IBV_WC_SUCCESS) {
            if (chunkMr && chunkMr != preregisteredMr) {
                ibv_dereg_mr(chunkMr);
            }
            throw std::runtime_error("Failed status: " +
                                     std::string(ibv_wc_status_str(workCompletion.status)));
        }

        if (chunkMr && chunkMr != preregisteredMr) {
            ibv_dereg_mr(chunkMr);
        }

        return size;
    }

#ifdef DEBUG
    std::cout << "  -> RDMA send chunked transfer total=" << size
              << " chunk_size=" << kRdmaChunkSize
              << " full_mr=1" << std::endl;
#endif

    // Large payload optimization:
    // Register the full source buffer once, then post chunked sends
    // using the same MR. This avoids register/deregister per chunk.
    void *fullSendAddr = const_cast<char *>(buffer);
    ibv_mr *fullMr = ktm_rdma_reg_msgs(rdmaCmId, fullSendAddr, size);

    size_t total = 0;

    try {
        while (total < size) {
            const size_t remaining = size - total;
            const size_t chunk = std::min(remaining, kRdmaChunkSize);

            void *sendAddr = const_cast<char *>(buffer + total);

            ktm_rdma_post_send(rdmaCmId, nullptr, sendAddr, chunk, fullMr, IBV_SEND_SIGNALED);

            int num_comp;
            do {
                num_comp = ibv_poll_cq(rdmaCmId->send_cq, 1, &workCompletion);
            } while (num_comp == 0);

            if (num_comp < 0) {
                throw std::runtime_error("ibv_poll_cq() failed");
            }

            if (workCompletion.status != IBV_WC_SUCCESS) {
                throw std::runtime_error("Failed status: " +
                                         std::string(ibv_wc_status_str(workCompletion.status)));
            }

            total += chunk;
        }
    } catch (...) {
        ibv_dereg_mr(fullMr);
        throw;
    }

    ibv_dereg_mr(fullMr);
    return total;
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