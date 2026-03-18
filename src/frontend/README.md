# frontend/

The **frontend** is a thin shim library that the CUDA application links against instead of the real CUDA runtime (or cuBLAS, cuDNN, etc.). It intercepts every CUDA API call, serialises the arguments, forwards them to the backend over the configured transport, waits for the result, and returns it to the caller — all transparently.

## Files

| File | Description |
|------|-------------|
| `Frontend.cpp` | Core frontend logic — per-thread instance management, config loading, communicator setup, `Execute()` dispatch loop, and optional runtime statistics |

## Per-Thread Design

CUDA applications are frequently multi-threaded. GVirtuS creates one `Frontend` instance per OS thread (`pthread_t`) and stores them in a `std::map` protected by a `std::mutex`. This ensures thread safety without a global lock on the data path.

```
Thread A ──► Frontend instance A ──► Communicator A ──► Backend
Thread B ──► Frontend instance B ──► Communicator B ──► Backend
```

## Initialisation (`Frontend::Init`)

Called automatically the first time a CUDA API is intercepted:

1. Reads the config file path from (in order of priority):
   - `GVIRTUS_CONFIG` environment variable
   - `$GVIRTUS_HOME/etc/properties.json`
   - `./properties.json`
2. Calls `EndpointFactory::get_endpoint()` to parse the JSON config.
3. Calls `CommunicatorFactory::get_communicator()` to create the transport.
4. Calls `communicator->Connect()` to establish the connection to the backend.
5. Allocates three `Buffer` objects:
   - `mpInputBuffer` — serialised function ID + arguments (sent to backend)
   - `mpOutputBuffer` — deserialised result (received from backend)
   - `mpLaunchBuffer` — extra buffer for kernel launch parameters

## Execute Loop

For each intercepted CUDA call (implemented in `plugins/*/src/`):

```
plugin stub:
  frontend.Execute("cudaMemcpy", inputBuffer)
    │
    ├─ Write inputBuffer → Communicator::Write()
    ├─ Communicator::Read() → outputBuffer
    └─ decode return value from outputBuffer → return to caller
```

## Runtime Statistics

Set `GVIRTUS_DUMP_STATS=on` (or `=true` / `=1`) to print per-thread stats on teardown:

```
[GVIRTUS_STATS] Executed 142 routine(s) in 0.031 second(s)
[GVIRTUS_STATS] Sent 3.12 Mb(s) in 0.008 second(s)
[GVIRTUS_STATS] Received 1.05 Mb(s) in 0.006 second(s)
```

## Logging Level

Set `GVIRTUS_LOGLEVEL` to a numeric log4cplus level (e.g. `0` = ALL, `10000` = DEBUG, `20000` = INFO, `40000` = WARN, `50000` = ERROR).

## Key Headers (in `include/gvirtus/frontend/`)

- `Frontend.h` — declares the `Frontend` class and the `Execute()` method used by plugin stubs
