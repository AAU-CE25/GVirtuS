# Step 2 – TCP Ping (Network Intro)

A plain TCP client/server pair written in C++. No RDMA, no special hardware — just the standard POSIX socket API that every network program is built on.

---

## What it does

```
helloClient                          helloServer
     │                                    │
     │──── TCP connect (port 9000) ──────►│
     │──── "PING" ───────────────────────►│
     │◄─── "Hello from es-dpu-02!" ───────│
     │                                    │
   (done)                               (done)
```

1. `helloServer` creates a socket, binds to port 9000, and waits.
2. `helloClient` resolves the server hostname and connects.
3. Client sends `"PING"`.
4. Server replies `"Hello from <hostname>!"`.
5. Both sides print what happened and exit.

---

## C++ concepts introduced

| Concept | Where it appears |
|---------|-----------------|
| `socket() / bind() / listen() / accept()` | `helloServer.cpp` — TCP server boilerplate |
| `getaddrinfo()` | `helloClient.cpp` — resolves hostname **or** IP to a `sockaddr` |
| `connect()` | `helloClient.cpp` — initiates the TCP three-way handshake |
| `read() / write()` | Both files — send and receive raw bytes |
| `std::string` and `+` concatenation | `helloServer.cpp` — building the reply message |
| `ssize_t` vs `int` | Both files — signed return type of `read()` / `write()` |
| `SO_REUSEADDR` | `helloServer.cpp` — avoid "address already in use" on quick restart |
| `constexpr` | Both files — compile-time constants for port and buffer size |
| `reinterpret_cast` | `helloServer.cpp` — required cast between `sockaddr_in*` and `sockaddr*` |
| RAII-style `close()` | Both files — always close file descriptors before exit |

---

## Files

| File | Description |
|------|-------------|
| `helloServer.cpp` | TCP server — waits for a connection and replies |
| `helloClient.cpp` | TCP client — connects, sends PING, reads reply |
| `Makefile` | Builds both binaries with `g++ -std=c++17` |

---

## Prerequisites

No special libraries. Any Linux machine with `g++` installed is enough.

For the two-machine demo you need SSH access to both `es-dpu-01` and `es-dpu-02` (or any two reachable hosts).

---

## Build

```bash
cd step2-ping-cpp
make
```

To clean up the binaries:

```bash
make clean
```

---

## Run

### Single machine (loopback)

```bash
# Terminal 1
./helloServer

# Terminal 2
./helloClient 127.0.0.1
```

### Two machines

```bash
# On es-dpu-02 (server)
cd step2-ping-cpp
./helloServer

# On es-dpu-01 (client)
cd step2-ping-cpp
./helloClient es-dpu-02
```

Expected output:

```
# Server
[server] Running on host: es-dpu-02
[server] Listening on port 9000 …
[server] Client connected!
[server] Received: "PING"
[server] Sent: "Hello from es-dpu-02!"
[server] Done.

# Client
[client] Connecting to es-dpu-02:9000 …
[client] Connected!
[client] Sent: "PING"
[client] Reply: "Hello from es-dpu-02!"
[client] Done.
```

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `connect: Connection refused` | Server not running yet | Start `helloServer` first |
| `bind: Address already in use` | Previous run still holding the port | Wait a few seconds, or the server uses `SO_REUSEADDR` so just restart |
| `Cannot resolve host: es-dpu-02` | Hostname not in `/etc/hosts` | Use IP directly (`25.25.25.3`) or check `/etc/hosts` |

---

## What's next

Step 3 replaces the TCP payload with a **kernel-bypass RDMA SEND** — same idea, but the message goes directly from one machine's memory to the other without touching the OS. See [step3-hello-rdma/README.md](../step3-hello-rdma/README.md).
