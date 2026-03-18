# GVirtuS — Architecture

## Overview

GVirtuS is a **transparent GPU virtualization** framework. It allows applications that call CUDA APIs to run on a machine with no physical GPU by forwarding every CUDA call over a network transport to a device-equipped backend host.

The core source lives in `src/` and is split into four layers:

```
┌─────────────────────────────────────────────────────────┐
│                    CUDA Application                      │
│            (links against GVirtuS frontend stub)         │
└─────────────────────┬───────────────────────────────────┘
                      │  intercepted CUDA API call
┌─────────────────────▼───────────────────────────────────┐
│                     Frontend                             │
│  • per-thread Frontend instance (thread-safe)           │
│  • serialises call name + args → mpInputBuffer (Buffer) │
│  • sends via Communicator::Write()                       │
│  • blocks until result arrives in mpOutputBuffer        │
└─────────────────────┬───────────────────────────────────┘
                      │  transport (TCP / RDMA / SHM / …)
┌─────────────────────▼───────────────────────────────────┐
│                  Communicator layer                      │
│  • CommunicatorFactory — selects transport at runtime   │
│  • EndpointFactory — parses properties.json endpoint    │
│  • Implementations: TCP, RDMA, AF_UNIX, SHM, ZMQ,       │
│    VMSocket, VMCI, Virtio, Hybrid                        │
│  • Buffer — heap-grown byte buffer with Encoder support  │
└──────────────┬──────────────────────────┬───────────────┘
               │  (client side)           │  (server side)
               │                          ▼
┌──────────────┘          ┌───────────────────────────────┐
│                         │           Backend              │
│                         │  • reads properties.json      │
│                         │  • forks one Process per       │
│                         │    configured endpoint         │
│                         │  • Process: accept loop,       │
│                         │    dispatches to Plugin via    │
│                         │    LD_Lib dynamic loading      │
│                         └──────────────┬────────────────┘
│                                        │  dlopen plugin .so
│                         ┌──────────────▼────────────────┐
│                         │     Plugin (plugins/*)         │
│                         │  e.g. cudart, cublas, cudnn   │
│                         │  Executes real CUDA call on   │
│                         │  physical GPU, serialises      │
│                         │  return value back to Process  │
│                         └───────────────────────────────┘
└──────────────────────────────────────────────────────────
```

---

## Component Details

### Frontend (`frontend/`)

`Frontend` is a **per-thread singleton** managed via a `std::map<pthread_t, Frontend*>`.

On first use it:
1. Reads config from `GVIRTUS_CONFIG`, `GVIRTUS_HOME/etc/properties.json`, or `./properties.json`.
2. Creates an `Endpoint` (via `EndpointFactory`) from the JSON config.
3. Instantiates the matching `Communicator` (via `CommunicatorFactory`) and calls `Connect()`.
4. Allocates three `Buffer` objects: `mpInputBuffer` (args out), `mpOutputBuffer` (result in), `mpLaunchBuffer` (kernel launch params).

Each intercepted CUDA call then:
- Encodes function ID + arguments into `mpInputBuffer`.
- Calls `Execute()`, which writes the buffer to the communicator and reads the result back.
- Decodes the return value from `mpOutputBuffer`.

Optional runtime stats (`GVIRTUS_DUMP_STATS=on`) report bytes sent/received and total routine execution time.

---

### Backend (`backend/`)

`Backend` is the server-side daemon entry point (`main.cpp` → `Backend` → `Process`).

| Class | Role |
|-------|------|
| `Backend` | Reads `properties.json`, creates one `Process` per endpoint, calls `fork()` for each, then `waitpid()` loops. |
| `Process` | Accept loop on the communicator; for each client connection spawns a thread that calls the appropriate plugin handler. Loads plugin shared libraries via `LD_Lib`. |
| `Property` | Typed wrapper around the JSON configuration (endpoints, plugin paths, secure flag). |

Plugin dispatch: `Process` reads the function name string from the incoming `Buffer`, looks it up in the plugin's handler map, calls the handler, and writes the result `Buffer` back.

---

### Communicators (`communicators/`)

The communicator layer is a **strategy pattern** — all transports implement the abstract `Communicator` interface (`Read`, `Write`, `Connect`, `Serve`, `Accept`, `Sync`).

| File / Subdirectory | Transport |
|--------------------|-----------|
| `tcp/TcpCommunicator` | Plain TCP socket |
| `rdma/RdmaCommunicator` | RDMA via `rdma_cm` / `libibverbs` (supports RoCE) |
| `AfUnixCommunicator` | AF_UNIX socket (same host) |
| `ShmCommunicator` / `VMShmCommunicator` | Shared memory |
| `ZmqCommunicator` | ZeroMQ |
| `VmciCommunicator` / `VMSocketCommunicator` | VMware VMCI / VM sockets |
| `VirtioCommunicator` | Virtio serial channel (for hypervisor guests) |
| `hybrid/HybridCommunicator` | Combines a control channel with a high-bandwidth data channel |

`CommunicatorFactory` and `EndpointFactory` read the `type` field from `properties.json` and instantiate the correct subclass at runtime.

`Buffer` is a dynamically growing heap buffer with O(1) amortised `Add` (append) and sequential `Get` (consume) operations. It is shared between frontend and backend as the serialisation container.

---

### Common (`common/`)

Shared, transport-agnostic utilities used by all other layers.

| File | Purpose |
|------|---------|
| `Encoder` / `Decoder` | Base64-style binary-safe codec for transferring binary data as text where required |
| `JSON` | Template header that parses `properties.json` into typed config objects via nlohmann/json |
| `LD_Lib` | RAII wrapper around `dlopen` / `dlsym` for dynamic plugin loading |
| `MessageDispatcher` | Observer-based dispatcher that routes incoming messages to registered handlers |
| `Observable` / `Observer` | Classic observer pattern used between `Process` and plugin handlers |
| `Mutex` | Thin RAII wrapper around `pthread_mutex_t` |
| `SignalException` / `SignalState` | Converts POSIX signals into C++ exceptions so plugins can handle `SIGTERM` gracefully |
| `Util` | Miscellaneous helpers |

---

## Configuration (`properties.json`)

Both frontend and backend are driven by a single JSON file:

```json
{
  "endpoints": [
    {
      "communicator": "tcp",
      "server_address": "192.168.1.100",
      "server_port": 9999
    }
  ],
  "plugins": [
    ["libCudaRt.so"]
  ],
  "secure": false
}
```

The frontend uses the `communicator` + address/port fields to connect; the backend uses the same file to bind/listen and to know which plugin `.so` files to load.

---

## Threading Model

- **Frontend**: one `Frontend` instance per OS thread (`pthread_t` key). Mutex-protected map insertion.
- **Backend `Process`**: single accept loop; a new `std::thread` is spawned for each accepted client connection.
- **Backend `Backend`**: one child process per endpoint (via `fork()`), parent waits with `waitpid()`.
