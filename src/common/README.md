# common/

Shared, transport-agnostic utilities used by **all** other GVirtuS components (frontend, backend, communicators, and plugins).

## Files

| File | Description |
|------|-------------|
| `Encoder.cpp` | Base64-style binary-safe encoder — converts raw bytes to a printable character stream; used when data must cross a text-only channel |
| `Decoder.cpp` | Inverse of `Encoder` — decodes base64 back to raw bytes |
| `JSON.cpp` | Template helper (`JSON<T>`) that reads a `properties.json` file and deserialises it into a typed config object via nlohmann/json |
| `LD_Lib.cpp` | RAII wrapper around `dlopen` / `dlsym` — used by `Process` to load and unload plugin shared libraries at runtime |
| `MessageDispatcher.cpp` | Observer-based message router — maintains a list of `Observer` objects and notifies the correct one when a named message arrives |
| `Observable.cpp` | Base class for objects that emit notifications (e.g. `Process`, plugin handlers) |
| `Observer.cpp` | Base class for objects that receive notifications |
| `Mutex.cpp` | Thin RAII wrapper around `pthread_mutex_t` |
| `SignalException.cpp` | Converts a POSIX signal into a C++ exception, allowing plugin code to handle `SIGTERM` / `SIGINT` gracefully |
| `SignalState.cpp` | Maintains current signal state; used together with `SignalException` |
| `Util.cpp` | Miscellaneous utility functions |

## Key Concepts

### Encoder / Decoder
Used only when the chosen communicator requires text-safe payloads. Most binary transports (TCP, RDMA, SHM) pass raw bytes and do not need encoding.

### LD_Lib
```cpp
// Load a plugin
auto lib = std::make_shared<LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(path);
// Retrieve a symbol
auto fn = lib->sym<handler_t>("handler_map");
```

### Observer Pattern
`Process` (`Observable`) notifies registered plugin `Observer` objects when a function name matches their handler registration. This decouples the dispatch loop from individual CUDA API implementations.

### Signal Handling
`SignalState::Setup()` installs handlers that translate OS signals into `SignalException`. The `Process` accept loop catches these to shut down cleanly.

## Key Headers (in `include/gvirtus/common/`)

- `JSON.h` — template JSON config parser
- `LD_Lib.h` — dynamic library loader
- `Encoder.h` / `Decoder.h`
- `Observable.h` / `Observer.h` / `MessageDispatcher.h`
- `SignalException.h` / `SignalState.h`
