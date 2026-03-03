# cpp_first_steps

Welcome! This folder is your starting point for working with C++ on HPC hardware at AAU. It takes you from a one-file "Hello World" all the way to real RDMA messaging over a BlueField-3 DPU — in three self-contained steps.

> **Special shout-out to Enrique** — for somehow missing the AI class and thereby earning himself a permanent place in the commit history. Legendary move.

---

## Learning path

```
step0 — check-config          Make sure the hardware and network are ready
  │
  ▼
step1 — hello-cpp             Your first C++ program, runs on any machine
  │
  ▼
step2 — ping-cpp              TCP client/server — two machines talking over a socket
  │
  ▼
step3 — hello-rdma            RDMA Send/Receive — kernel-bypass messaging over BlueField-3
```

Work through them **in order**. Each step builds on the previous.

---

## Prerequisites

You need SSH access to both HPC nodes:

| Node | Hostname | Role |
|------|----------|------|
| `es-dpu-01` | `25.25.25.1` | machine you compile and run from |
| `es-dpu-02` | `25.25.25.3` | remote end for steps 2 and 3 |

If you have not set up SSH keys for GitHub on the remote machine yet, see [cheatsheet.md](cheatsheet.md).

---

## Step 0 — Check the configuration

> **Always run this before any experiment.**

The [check-config/](check-config/) folder contains three scripts that verify the hardware and network are in the expected state.

```bash
# Bring up the dedicated link (needs sudo, run on both machines)
sudo ./check-config/setup_link.sh

# Validate TCP/IP connectivity
./check-config/check_link.sh

# Validate the RDMA device (only needed before step 3)
./check-config/check_rdma.sh
```

Full details: [check-config/README.md](check-config/README.md)

---

## Step 1 — Hello HPC (C++ intro)

> **Concepts:** `main()`, `std::cout`, `argc/argv`, hostname

A single-file C++ program that prints a greeting and the machine's hostname. Good for verifying the compiler works and getting familiar with the toolchain.

```bash
cd step1-hello-cpp
make
./helloHPC
# → Hello, HPC World!
# → Running on host: es-dpu-01
```

Full details: [step1-hello-cpp/README.md](step1-hello-cpp/README.md)

---

## Step 2 — TCP Ping (network intro)

> **Concepts:** sockets, `bind/listen/accept/connect`, `read/write`, `getaddrinfo`

A plain TCP client and server. No RDMA yet — just the POSIX socket API. This is the foundation that every network program is built on.

```bash
# Terminal on es-dpu-02
cd step2-ping-cpp && make
./helloServer

# Terminal on es-dpu-01
cd step2-ping-cpp
./helloClient es-dpu-02
# → [client] Sent: "PING"
# → [client] Reply: "Hello from es-dpu-02!"
```

Full details: [step2-ping-cpp/README.md](step2-ping-cpp/README.md)

---

## Step 3 — RDMA Hello World

> **Concepts:** `libibverbs`, Queue Pairs, RDMA SEND/RECV, RoCEv2, out-of-band handshake via TCP

The real deal. Two machines exchange `"Hello, RDMA World!"` directly over RDMA — the message bypasses the OS kernel and lands straight in the receiver's registered memory.

```bash
# On es-dpu-02
cd step3-hello-rdma && make
./ce-server

# On es-dpu-01
./ce-client es-dpu-02
# → [client] Sent: "Hello, RDMA World!"
```

Full details: [step3-hello-rdma/README.md](step3-hello-rdma/README.md)

---

## Quick reference

| Topic | File |
|-------|------|
| SSH, git, common Linux commands | [cheatsheet.md](cheatsheet.md) |
| Network topology diagram | [check-config/README.md](check-config/README.md) |
| RDMA tunables (`DEVICE_NAME`, `GID_INDEX`, …) | [step3-hello-rdma/rdma_common.h](step3-hello-rdma/rdma_common.h) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Device 'mlx5_2' not found` | Run `ibv_devinfo` and update `DEVICE_NAME` in `rdma_common.h` |
| `connect: Connection refused` | Start the server before the client |
| `Port 9000 in use` | Previous run didn't clean up — wait a moment or reboot the server |
| `ibv_post_recv: Invalid argument` | Posted send before posting recv — always post recv first |
| Link check fails | Re-run `sudo ./check-config/setup_link.sh` on both machines |
