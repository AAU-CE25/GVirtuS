#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <infiniband/verbs.h>
#include <arpa/inet.h>

// ── tunables ──────────────────────────────────────────────────────────────
//
// DEVICE_NAME: the mlx5 device that corresponds to the BlueField-3 (82:00).
//   Run `ibv_devinfo` to list devices and match to BF3 PCI address:
//     for d in /sys/class/infiniband/mlx5_*; do
//         echo "$d -> $(cat $d/device/uevent | grep PCI_SLOT)"; done
//   Set this to whichever mlx5_X maps to PCI slot 82:00.
//
constexpr const char *DEVICE_NAME = "mlx5_2"; // BF3 at PCI 82:00 — change if needed

constexpr int     RDMA_PORT  = 18515;
constexpr int     MSG_SIZE   = 64;
constexpr uint8_t IB_PORT    = 1;      // physical port on the BF3 NIC
constexpr int     GID_INDEX  = 3;      // 3 = RoCEv2 (required for BlueField-3)
constexpr int     MAX_WR     = 10;
constexpr int     CQ_SIZE    = 10;

// ── connection info exchanged out-of-band via TCP ─────────────
struct ConnInfo {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t  gid[16];
} __attribute__((packed));

// ── per-connection RDMA resources ────────────────────────────
struct RdmaRes {
    ibv_context *ctx = nullptr;
    ibv_pd      *pd  = nullptr;
    ibv_mr      *mr  = nullptr;
    ibv_cq      *cq  = nullptr;
    ibv_qp      *qp  = nullptr;
    char         buf[MSG_SIZE] = {};
};

// ── helpers ───────────────────────────────────────────────────
[[noreturn]] inline void die(const std::string &msg)
{
    std::perror(msg.c_str());
    std::exit(EXIT_FAILURE);
}

// Open the BF3 RDMA device by name (DEVICE_NAME).
// Avoids accidentally opening the ConnectX-6 Lx which is also present.
inline ibv_context *open_device()
{
    int num_devs = 0;
    ibv_device **dev_list = ibv_get_device_list(&num_devs);  // list all RDMA devices
    if (!dev_list || num_devs == 0) {
        std::cerr << "No RDMA devices found\n";
        std::exit(EXIT_FAILURE);
    }

    ibv_context *ctx = nullptr;
    for (int i = 0; i < num_devs; ++i) {
        // Match device by name — DEVICE_NAME must point to the BF3 (82:00)
        if (std::string(ibv_get_device_name(dev_list[i])) == DEVICE_NAME) {
            ctx = ibv_open_device(dev_list[i]);  // open only the BF3
            break;
        }
    }
    ibv_free_device_list(dev_list);  // always free the list

    if (!ctx) {
        std::cerr << "Device '" << DEVICE_NAME
                  << "' not found. Run ibv_devinfo to list available devices.\n";
        std::exit(EXIT_FAILURE);
    }
    return ctx;
}

// Allocate PD, MR, CQ and an RC QP in INIT state
inline void alloc_resources(RdmaRes &r)
{
    r.pd = ibv_alloc_pd(r.ctx);
    if (!r.pd) die("ibv_alloc_pd");

    r.mr = ibv_reg_mr(r.pd, r.buf, MSG_SIZE,
                      IBV_ACCESS_LOCAL_WRITE |
                      IBV_ACCESS_REMOTE_READ |
                      IBV_ACCESS_REMOTE_WRITE);
    if (!r.mr) die("ibv_reg_mr");

    r.cq = ibv_create_cq(r.ctx, CQ_SIZE, nullptr, nullptr, 0);
    if (!r.cq) die("ibv_create_cq");

    ibv_qp_init_attr qp_attr{};
    qp_attr.send_cq          = r.cq;
    qp_attr.recv_cq          = r.cq;
    qp_attr.qp_type          = IBV_QPT_RC;
    qp_attr.cap.max_send_wr  = MAX_WR;
    qp_attr.cap.max_recv_wr  = MAX_WR;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    r.qp = ibv_create_qp(r.pd, &qp_attr);
    if (!r.qp) die("ibv_create_qp");
}

// INIT → RTR → RTS
inline void connect_qp(RdmaRes &r, const ConnInfo &remote)
{
    // INIT
    ibv_qp_attr attr{};
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.port_num        = IB_PORT;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_LOCAL_WRITE;

    if (ibv_modify_qp(r.qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX |
            IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS))
        die("modify_qp INIT");

    // RTR
    ibv_gid remote_gid{};
    std::memcpy(remote_gid.raw, remote.gid, 16);

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state                  = IBV_QPS_RTR;
    attr.path_mtu                  = IBV_MTU_1024;
    attr.dest_qp_num               = remote.qp_num;
    attr.rq_psn                    = 0;
    attr.max_dest_rd_atomic        = 1;
    attr.min_rnr_timer             = 12;
    attr.ah_attr.is_global         = 1;
    attr.ah_attr.dlid              = remote.lid;
    attr.ah_attr.sl                = 0;
    attr.ah_attr.src_path_bits     = 0;
    attr.ah_attr.port_num          = IB_PORT;
    attr.ah_attr.grh.dgid          = remote_gid;
    attr.ah_attr.grh.flow_label    = 0;
    attr.ah_attr.grh.sgid_index    = GID_INDEX;
    attr.ah_attr.grh.hop_limit     = 1;
    attr.ah_attr.grh.traffic_class = 0;

    if (ibv_modify_qp(r.qp, &attr,
            IBV_QP_STATE             | IBV_QP_AV                 |
            IBV_QP_PATH_MTU          | IBV_QP_DEST_QPN           |
            IBV_QP_RQ_PSN            | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER))
        die("modify_qp RTR");

    // RTS
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout       = 14;
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;
    attr.sq_psn        = 0;
    attr.max_rd_atomic = 1;

    if (ibv_modify_qp(r.qp, &attr,
            IBV_QP_STATE     | IBV_QP_TIMEOUT    |
            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY  |
            IBV_QP_SQ_PSN    | IBV_QP_MAX_QP_RD_ATOMIC))
        die("modify_qp RTS");
}

// Post one RECV work request
inline void post_recv(RdmaRes &r)
{
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uintptr_t>(r.buf);
    sge.length = MSG_SIZE;
    sge.lkey   = r.mr->lkey;

    ibv_recv_wr rwr{};
    rwr.sg_list = &sge;
    rwr.num_sge = 1;

    ibv_recv_wr *bad = nullptr;
    if (ibv_post_recv(r.qp, &rwr, &bad)) die("ibv_post_recv");
}

// Post one SEND work request
inline void post_send(RdmaRes &r)
{
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uintptr_t>(r.buf);
    sge.length = MSG_SIZE;
    sge.lkey   = r.mr->lkey;

    ibv_send_wr swr{};
    swr.sg_list    = &sge;
    swr.num_sge    = 1;
    swr.opcode     = IBV_WR_SEND;
    swr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr *bad = nullptr;
    if (ibv_post_send(r.qp, &swr, &bad)) die("ibv_post_send");
}

// Poll CQ until one completion arrives
inline void poll_cq(RdmaRes &r)
{
    ibv_wc wc{};
    while (ibv_poll_cq(r.cq, 1, &wc) == 0) {}
    if (wc.status != IBV_WC_SUCCESS) {
        std::cerr << "CQ error: " << ibv_wc_status_str(wc.status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Destroy all RDMA resources
inline void free_resources(RdmaRes &r)
{
    ibv_destroy_qp(r.qp);
    ibv_destroy_cq(r.cq);
    ibv_dereg_mr(r.mr);
    ibv_dealloc_pd(r.pd);
    ibv_close_device(r.ctx);
}
