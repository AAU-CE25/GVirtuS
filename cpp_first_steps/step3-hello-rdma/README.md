# RDMA Hello World

A minimal **RDMA Send/Receive** example using `libibverbs` with a Reliable-Connected (RC) Queue Pair.

## How it works

```
Client                          Server
  |                               |
  |---[TCP: exchange QP info]---->|
  |<--[TCP: exchange QP info]-----|
  |                               |
  |===[RDMA SEND "Hello..."]=====>|
  |                               |
```

1. Both sides open an RDMA device and create a Queue Pair (QP).
2. They exchange QP numbers, LIDs, and GIDs over a plain TCP socket.
3. Each QP is transitioned **INIT → RTR → RTS**.
4. The client sends `"Hello, RDMA World!"` via an RDMA SEND verb.
5. The server receives and prints it.

## Files

| File | Description |
|------|-------------|
| `rdma_common.h` | Shared helpers: device open, QP setup, post send/recv |
| `ce-server.cpp` | Waits for a connection, receives the RDMA message |
| `ce-client.cpp` | Connects to the server, sends the RDMA message |
| `Makefile` | Build rules |

## Prerequisites

```bash
# Ubuntu / Debian
sudo apt install libibverbs-dev ibverbs-utils rdma-core
```

## Build

```bash
make
```

## Run

On the **server machine**:
```bash
./ce-server
```

On the **client machine** (or the same machine using a loopback RDMA interface):
```bash
./ce-client <server_ip>
```

Expected output:

```
# Server
[server] Listening on port 18515 …
[server] Client connected
[server] QP connected
[server] Waiting for message …
[server] Received: "Hello, RDMA World!"

# Client
[client] Connected to server 192.168.1.10
[client] QP connected
[client] Sent: "Hello, RDMA World!"
```

## Notes

- **GID_INDEX** in `rdma_common.h` controls the GID used for RoCE:
  - `0` → RoCEv1
  - `3` → RoCEv2 (most common for modern NICs)
- **IB_PORT** defaults to `1`; change it if your device uses a different port.
- For loopback testing on the same machine, use a RoCE-capable NIC or a software RDMA device (`rxe` / `siw`):
  ```bash
  sudo rdma link add rxe0 type rxe netdev eth0
  ```
