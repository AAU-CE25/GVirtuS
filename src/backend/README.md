# backend/

The **backend** is the GPU-side daemon. It runs on a machine with a physical NVIDIA GPU, listens for incoming connections from frontend clients, and dispatches CUDA API calls to the appropriate plugin.

## Files

| File | Description |
|------|-------------|
| `main.cpp` | Entry point — parses CLI args, constructs a `Backend`, calls `Backend::Start()` |
| `Backend.cpp` | Reads `properties.json`, creates one `Process` per configured endpoint, forks a child process for each, then waits for them with `waitpid()` |
| `Process.cpp` | Runs the accept loop on the communicator; for each accepted client connection it spawns a `std::thread` that reads the incoming function name from the `Buffer`, looks it up in the dynamically loaded plugin map, executes the handler, and writes the result `Buffer` back |
| `Property.cpp` | Typed wrapper around the JSON configuration — exposes `endpoints()`, `plugins()`, `secure()` |

## Lifecycle

```
main()
  └─ Backend(path)          // load config, build Process list
       └─ Backend::Start()  // fork() per endpoint
            └─ Process::Start()   // per child process
                 └─ Communicator::Serve() / Accept() loop
                      └─ thread per client
                           └─ dispatch → Plugin handler → write result
```

## Plugin Loading

`Process` uses `LD_Lib` (`common/`) to `dlopen` each plugin `.so` listed in the config. The plugin registers a map of `function_name → handler_fn`. At dispatch time `Process` looks up the received function name in this map and calls the handler with the input/output `Buffer` pair.

## Key Headers (in `include/gvirtus/backend/`)

- `Backend.h` — declares `Backend` class
- `Process.h` — declares `Process` class (extends `Observable`)
