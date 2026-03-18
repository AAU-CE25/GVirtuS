# communicators/

The **communicator layer** is a pluggable transport abstraction. All transports implement the `Communicator` interface; frontend and backend select a concrete implementation at runtime based on `properties.json`.

## Files & Subdirectories

### Core abstractions
| File | Description |
|------|-------------|
| `Buffer.cpp` | Dynamically growing heap buffer — the serialisation container shared between frontend and backend. Supports `Add<T>()` (append) and `Get<T>()` (consume) for arbitrary POD types and byte ranges. Grows in `blockSize` increments. |
| `CommunicatorFactory.cpp` | Factory that reads the `communicator` field from the JSON endpoint and returns the matching `LD_Lib<Communicator>` instance |
| `EndpointFactory.cpp` | Parses `properties.json` and constructs the correct `Endpoint` subclass (address + port, socket path, etc.) |
| `Result.cpp` | Wraps a return `Buffer` with an exit code; passed from backend plugin handlers back to `Process` |

### Transport implementations

| File / Directory | Transport | Notes |
|-----------------|-----------|-------|
| `tcp/TcpCommunicator.cpp` | Plain TCP socket | Default for LAN/WAN deployments |
| `rdma/RdmaCommunicator.cpp` + `ktmrdma.cpp` | RDMA via `rdma_cm` / `libibverbs` | Supports both InfiniBand and RoCEv2; low-latency option for HPC |
| `AfUnixCommunicator.cpp` | AF_UNIX domain socket | Same-host communication, lower overhead than TCP |
| `ShmCommunicator.cpp` | POSIX shared memory | Fastest same-host option; zero copy |
| `VMShmCommunicator.cpp` | Shared memory for VM guest/host | Variant of SHM for hypervisor scenarios |
| `ZmqCommunicator.cpp` | ZeroMQ | Flexible messaging; useful for multi-endpoint fan-out |
| `VmciCommunicator.cpp` | VMware VMCI | VM↔host on VMware hypervisors |
| `VMSocketCommunicator.cpp` | vsock | Generic hypervisor VM socket |
| `VirtioCommunicator.cpp` | Virtio serial | Paravirtualised channel inside KVM/QEMU guests |
| `hybrid/HybridCommunicator.cpp` | Hybrid (control + data) | Uses a TCP control channel paired with a high-bandwidth RDMA/SHM data channel |
| `Endpoint_Tcp.cpp` / `Endpoint_Rdma.cpp` / `Endpoint_Hybrid.cpp` | Endpoint types | Hold connection parameters for the matching transport |

## Communicator Interface

All transports implement:

```cpp
void   Serve();                          // server: bind and listen
const Communicator* Accept() const;      // server: accept next client
void   Connect();                        // client: connect to server
size_t Read (char* buf, size_t n);       // receive n bytes
size_t Write(const char* buf, size_t n); // send n bytes
void   Sync();                           // flush / barrier
std::string to_string() const;           // human-readable description
```

## Buffer

`Buffer` is the primary data container passed through every layer:

- **Frontend** encodes function ID + arguments into `mpInputBuffer`.
- It is sent via `Communicator::Write()`.
- **Backend** reads it, decodes arguments, executes the CUDA call, encodes the result into an output `Buffer`.
- The output `Buffer` is sent back and decoded by the frontend.

## Selecting a Transport

Set in `properties.json`:
```json
{ "communicator": "tcp" }   // or "rdma", "unix", "shm", "zmq", "hybrid", …
```

`CommunicatorFactory` and `EndpointFactory` handle the rest.

## Key Headers (in `include/gvirtus/communicators/`)

- `Communicator.h` — abstract base class
- `Buffer.h` — serialisation buffer
- `CommunicatorFactory.h` / `EndpointFactory.h`
- `Endpoint.h` — abstract endpoint
