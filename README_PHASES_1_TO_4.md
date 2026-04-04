# GVirtuS Protocol Evolution (Phases 1-4)

This document explains, in simple terms, what was changed from Phase 1 to Phase 4.
It is meant to be detailed enough for developers, but easy to read for anyone joining the project.

## Why this work was needed

The original request/response path was effectively one-call-at-a-time on a connection. That limits throughput and makes pipelining impossible.

The Phase 1-4 work introduces:

- a safer framed wire protocol,
- more deterministic timeout behavior,
- request ID based response matching,
- asynchronous request posting,
- concurrent backend execution for each client connection.

Goal: allow multiple CUDA calls to be in flight and still return each response to the correct caller.

## Phase 1: Framed protocol foundation

### What was added

- A framed message format with explicit headers.
- Message types for `REQUEST`, `RESPONSE`, `ERROR`, and `RESYNC`.
- CRC validation for frame headers.

### Why it matters

- Protects against desynchronization and corrupted headers.
- Gives the transport enough structure to recover and continue.

### Key outcome

- The transport no longer depends on implicit stream boundaries; each frame is self-described.

## Phase 2: Timeout and recovery behavior

### What was added

- Bounded receive and send wait behavior.
- Explicit timeout handling and cancellation paths.
- Recovery path for stream misalignment (`RESYNC` handling and re-locking to magic).

### Why it matters

- Prevents indefinite hangs.
- Makes failure behavior predictable under partial sends, dropped bytes, or bad framing.

### Key outcome

- Timeout tests can now be run with hard bounds and expected completion behavior.

## Phase 3: Hardening and deterministic stability

### What was improved

- Flaky UCX wait/cancel behavior was stabilized.
- Progress/shutdown behavior was hardened.
- Test assertions were tuned to realistic UCX cancellation timing.
- "Phase" naming in files/targets/scripts was cleaned up to neutral names.

### Why it matters

- Test results became reproducible with filtered runs and bounded execution.
- CI and developer runs now fail for real issues, not random timing noise.

### Key outcome

- Deterministic transport stability checks were achieved (bounded, repeatable green runs).

## Phase 4: Request pipelining core

Phase 4 introduced the core plumbing required for true pipelining.

### Step 1: 32-bit request IDs

- Promoted wire request identity to `uint32_t` request IDs.
- Added monotonic request ID generation in `FramedStream`.

Why: allows many outstanding requests without small-ID rollover constraints.

### Step 2: In-flight request map

- Added a thread-safe in-flight map from `request_id` to pending request state.
- Added request registration and completion paths.

Why: each response can be routed to exactly the caller that initiated it.

### Step 3: Async send API

- Added asynchronous posting API (`SendAsync`) that returns a pending handle.
- Added `PendingRequest::Wait(timeout)` to block only when/where needed.
- Added callback-managed frame lifetime for non-blocking UCX send.

Why: request submission is no longer blocked by immediate response waits.

### Step 4: Dispatch loop response routing

- Added dedicated dispatch loop to receive frames and demultiplex by `request_id`.
- `RESPONSE` frames complete matching pending requests.
- `ERROR` frames complete pending requests with error status.

Why: responses can arrive out of order and still be delivered correctly.

### Step 5: Backend per-connection worker pool

- Added a fixed-size worker pool (size 4) per connection.
- Reader thread deserializes and enqueues; workers execute handlers concurrently.
- Response writing is serialized with a mutex to protect shared communicator writes.

Why: backend can process multiple requests concurrently for one client connection.

## Frontend status after Phase 4

Important detail:

- Transport-level pipelining is implemented in `FramedStream` and backend handling.
- The legacy generic frontend path in `src/frontend/Frontend.cpp` still uses synchronous `Write/Read` style communicator calls.
- CUDA wrapper call sites were adapted to an async-style API shape where possible (`ExecuteAsync(...).Wait()` pattern), but full frontend transport migration to framed async routing is still a follow-up task.

## Testing summary

### Transport-focused tests

- `tests/test_framed_stream.cpp`
- `tests/test_stream_timeout.cpp`

These validate:

- framing correctness,
- request ID handling,
- timeout boundaries,
- resync recovery,
- async handle completion behavior.

### CUDA multithread stress harness

- Added in `tests/test_cudart.cu`:
  - `MultiThreadedMallocFreeStress`
  - 4 threads running `cudaMalloc/cudaFree` loops for 10 seconds.

Purpose: validate no deadlock and no incorrect responses under concurrent call pressure.

## Cleanup and bug fixes during Phase 4

- Fixed a UCX request cleanup edge case in cancel-timeout fallback path to avoid request object leaks/warnings.

## What is complete vs pending

### Complete

- Framed protocol and timeout hardening.
- Request-ID based in-flight matching.
- Async request posting and pending wait API at transport layer.
- Dispatch-based response demux.
- Backend per-connection worker pool execution model.

### Pending follow-up

- Full migration of the generic frontend transport path to directly use framed async request/response routing end-to-end.
- Optional additional out-of-order response stress tests at larger scale.

## Practical impact

After Phases 1-4, the system has the core mechanics required for pipelining:

- multiple requests can be outstanding,
- backend can process requests concurrently,
- responses are matched by request ID,
- timeout and recovery behavior is bounded and testable.

This is a major reliability and scalability upgrade over strictly synchronous one-request-at-a-time behavior.
