---
title: "Contracts: the RMA slot protocol, the placement selector, and the one assumption we do not discharge"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

The evaluation shows the protocol *works*. This document says what it *is*: the state it keeps,
the decision it makes, the transitions it allows, and the invariants that make it safe -- each
with the line of code that discharges it and, where one exists, the counter that proves the
discharge point was reached.

**Nine invariants are discharged. One is not**, and §6 states it as a bounded assumption rather
than a guarantee, because that is the honest form.

Every line reference is to `src/communicators/ucx/UcxCommunicator.cpp` unless another file is
named, and is valid at commit `12b1d7c` on `exp/asplos-correctness-campaign`. Each is given with
the code fragment it points at, so it stays findable once the numbers drift -- they drifted once
already, while this document was being written.

# 1. The state

## 1.1 Wire messages

All four share one 40-byte `EnvelopeHeader` (`UcxAmProtocol.h:99`); the fields are reused per
message type, which is why they need a table rather than a name.

| field | `RmaSetup` | `RmaPosted` | `SlotConsumed` |
|---|---|---|---|
| `status_code` | **epoch** of this layout | `gpu_offset` | -- |
| `payload_size` | number of descriptors | `total` payload bytes | -- |
| `routine_size` | -- | `gpu_size` (0 = host only) | -- |
| `reserved0` | -- | `slot_idx` (server's own index) | `slot_idx` |
| `request_id` | -- | **slot tag** | **slot tag** (echoed verbatim) |

Each `RmaSetup` descriptor carries `remote_addr`, `slot_capacity`, `rkey_size`, and a packed
`reserved0`:

    reserved0 = flags | (server_idx << 16)
      bit 0  kHasGpuShadow     a GPU rkey follows this slot's host rkey
      bit 1  kSlotPersistent   part of the persistent RMA pool, safe to ucp_put into
      31:16  server_idx        the server's OWN index for this slot

**The slot tag is the identity of one transfer**, and it is what makes both guards possible:

    tag = (epoch << 32) | (generation & 0xffffffff)        // UcxAmProtocol.h:88

`epoch` names the layout the transfer was addressed against; `generation` names which use of
that slot it was. The server echoes the tag verbatim in `SlotConsumed`, so an acknowledgement
always carries the identity of the transfer it acknowledges -- never merely the slot it used.

## 1.2 Client state (`UcxCommunicator.h:274`)

    struct RemoteSlot {
        uint64_t   addr, capacity;         // remote (server's view)
        ucp_rkey_h rkey;                   // unpacked on this side
        uint16_t   server_idx;             // identity, NOT our position
        bool       persistent;             // only these may be ucp_put into
        uint64_t   gpu_addr, gpu_capacity; // optional GPU shadow
        ucp_rkey_h gpu_rkey;               //   nullptr => host-only path
        enum class State : uint8_t { Free, InFlight };
        State      state;                  // ownership, not "has data"
        uint64_t   generation;             // bumped on every acquire
    };

    vector<RemoteSlot> remote_slots_;   uint32_t remote_epoch_;    // the LIVE layout
    vector<RemoteSlot> pending_slots_;  uint32_t pending_epoch_;   // a PARKED layout
    bool               rma_swap_pending_;
    vector<ucp_rkey_h> retired_rkeys_;                             // freed outside the callback
    mutex              rma_state_mu_;                              // guards all of the above

**`server_idx` is the load-bearing field.** Identity used to be positional -- the client's slot
*i* had to be the server's slot *i* -- which forced the server to advertise every slot it owned
and broke silently whenever `ucp_rkey_pack` failed for one, since the `continue` shifted every
later index by one.

## 1.3 Server state

`rx_pool_->slots[i]`, guarded by `rx_pool_->mu`. `server_idx` **is** `i`, the position in that
vector -- which is exactly why a rebuild renumbers and a re-advertisement does not
(`N7_EPOCH.md` §1). A slot is advertised only if `rma_persistent && !rma_retired && memh != nullptr`.

# 2. The selector

`include/gvirtus/communicators/RmaPolicy.h`. One entry point, three policies, so the deployable
one can be bounded by an oracle it cannot beat.

    prefer_rma(h2d, pinned, bytes) -> bool
        switch (policy):                       // GVIRTUS_RMA_POLICY
          Quadrant: return bytes >= T[h2d][pinned]
          Oracle:   return oracle_prefers_rma(h2d, pinned, bytes)
          Scalar:   return bytes >= scalar_floor_bytes()    // default 4 MiB

    T[h2d][pinned], bytes:                     // quadrant_threshold()
                    | pageable |  pinned
        D2H  (0)    |   2 MiB  |   1 MiB
        H2D  (1)    |   1 MiB  |   8 KiB

**Why a table and not a scalar**, since this is the contribution and not an implementation
detail: the two directions want thresholds **three orders of magnitude apart**. H2D pinned wins
on RMA from 8 KiB up and the gain grows monotonically to 64 MiB; D2H reverses sign below 1 MiB.
No scalar satisfies both -- a single floor is not merely suboptimal, it is the wrong *shape*.

**One trap is part of the contract**, because ignoring it makes the comparison meaningless.
`GVIRTUS_RMA_MIN_BYTES` does two jobs: it gates the *decision* and it sizes the *pool*. A
"scalar vs quadrant" sweep driven by that one variable measures nothing -- both arms end up with
the same gate (measured: at 8 KiB, 15.21 against 15.22 GB/s). The scalar floor therefore reads
its **own** variable, `GVIRTUS_RMA_SCALAR_FLOOR`, so every policy can run with the pool built
down to the smallest threshold in the table while only the decision differs.

**The oracle is a lookup of the answer** and must never be reported as an achievable
configuration. It exists to bound the quadrant policy.

# 3. The state machine

## 3.1 A slot

    Free  --acquire(++generation, tag = (remote_epoch_, generation))-->  InFlight
      ^                                                                     |
      |                                                                     |
      +---- SlotConsumed, accepted only if  epoch      == remote_epoch_ -----+
                                       AND  generation == slot.generation

**`InFlight` means "the server may still be reading it", not "a put is outstanding."** The
return to `Free` is driven by the server's `SlotConsumed`, **not** by local put completion --
`ucp_put_nbx` completing means only that the source buffer may be reused. Every acknowledgement
that fails either test is dropped, and each drop increments its own counter.

Two exits are not transitions: a transfer that finds no slot large enough **declines** to the
active-message path (`decline_capacity`, 74 observed over 14.78 M operations), and a transfer
that throws returns its slot to `Free` via RAII so a leak can never wedge backpressure.

## 3.2 A layout

    RmaSetup(epoch e) arrives
       |
       +-- no slot InFlight  --> INSTALL NOW
       |                         retire old rkeys; remote_slots_ = new; remote_epoch_ = e
       |
       +-- some slot InFlight --> PARK
                                  pending_slots_ = new; pending_epoch_ = e
                                  rma_swap_pending_ = true
                                  (a second arrival supersedes the parked one
                                   and frees its rkeys rather than leaking them)
                                     |
                                     v
                                  release_remote_slot() sees the last InFlight drain
                                     |
                                     +--> apply_pending_slots_locked()

**The park is the first line of defence and the only one production exercises** -- 81 parks per
run against 0 epoch-guard activations (`N7_EPOCH.md` §3). The epoch guard sits behind it.

`apply_pending_slots_locked` **retires** rkeys into `retired_rkeys_` rather than destroying
them: both callers reach it from inside the AM receive callback, and `ucp_rkey_destroy` there
re-enters the worker lock UCX already holds.

# 4. The invariants, and where each is discharged

| # | invariant | discharge point | evidence it is reached |
|---|---|---|---|
| **I1** | A slot that is `InFlight` is never selected for another transfer | acquire scan skips `state != Free`, under `rma_state_mu_` (`:3563`) | -- |
| **I2** | Slot identity is explicit, never positional | `server_idx` in `reserved0[31:16]`, echoed in `RmaPosted`/`SlotConsumed` | the historical bug it fixes: a failed `rkey_pack` shifted every later index |
| **I3** | An ack minted against a superseded layout never frees a slot of the current one | epoch guard in `release_remote_slot`, `ack_epoch != remote_epoch_` (`:2507`), `rma_ack_dropped_epoch_count_` | **126--280 drops/run** under `readvertise` + `NOPARK` (`N7_EPOCH.md` §3) |
| **I4** | A duplicate or late ack never frees a slot already reassigned | generation check `const bool casa = (s.generation == generation)` (`:2521`), `rma_ack_gen_mismatch_count_` | **38--114/run** with I3 ablated |
| **I5** | The layout is never swapped while a transfer is live | `any_inflight` -> PARK (`:3036`), installed on drain | **81 parks/run**, `rma_swap_parked_count_` |
| **I6** | An rkey is never destroyed from inside the AM callback | retire into `retired_rkeys_` (`:3108`), freed by `drain_retired_rkeys()` | the deadlock it avoids is documented at the call site |
| **I7** | The peer's temporary (eager-AM) slots are never written | `persistent` flag; acquire requires `!any_persistent \|\| persistent` | -- |
| **I8** | A transfer never exceeds slot capacity | acquire requires `capacity >= total`; else decline to AM | `decline_capacity` = **74** in 2 643 921 admissions |
| **I9** | Backpressure terminates | acquire **polls**, and stops early when every slot is free and none fits | the 30 s stall this replaced is measured: 30 470 ms -> 41 ms |
| **I10** | **NIC writes to GPU memory are visible to the CUDA context that reads them** | **not discharged -- see §6** | -- |

**On I3 and I4 the honest reading is layered, not redundant.** Ablating I3 alone changes
nothing observable because I4 catches what it lets through; ablating both produces 114
premature frees per run and still no corrupted bytes, because a sixth timing condition is
needed. `N7_EPOCH.md` §5 states that condition.

**On the cost of all of this**: the lifetime protocol costs **~2.6 us per transfer**
(`SLOT_LIFETIME_RESULTS.md` §B.3). Registration invalidation is demonstrated *necessary* by
ablation, with the exact signature of the historical defect; **generation and epoch are not**
demonstrated necessary by a corruption, only by the counters above.

# 5. What the contracts do not cover

- **Ordering between concurrent transfers on different slots.** There is none, and none is
  needed: slots are independent and the consumer is told which slot each payload is in.
- **Fairness between connections.** Explicitly not a contract; it is FCFS, which is the
  mechanism `N1_SCHEDULER.md` identifies as the cause of the sharing imbalance.
- **Durability of anything.** Nothing here survives a backend restart; reconnection is a
  fresh handshake with epoch restarting at 1.

# 6. I10, bounded: NIC-to-GPU visibility

This is the assumption the system rests on and does not verify. It is stated here in full
because the alternative -- letting a correctness claim rest on it silently -- is worse than
admitting it.

## 6.1 What actually happens

On the GPUDirect path the client `ucp_put_nbx`s the payload into the server's **GPU** shadow
(`slot.gpu_addr`, peer-DMA via `nvidia-peermem`), then sends a small `RmaPosted` active message.
The server's AM handler publishes the region as `msg.gpu_data` and a handler consumes it with
`cudaMemcpyDeviceToDevice` (`:892-912`, `msg.gpu_data = slot.gpu_addr` at `:908`). So a **CUDA
read follows a NIC write to the same device memory, with only an active message between them.**

Two things must hold for that to be correct:

> **A1 (ordering).** The `RmaPosted` AM is observed by the server only *after* the RDMA WRITE
> payload has landed.
>
> **A2 (visibility).** A CUDA read issued after the AM is observed returns the NIC's writes,
> not stale device memory.

## 6.2 Neither is guaranteed by the API, and we say so

**A1 is not a UCX guarantee.** `ucp_put_nbx` completion is **local** -- it means the source
buffer may be reused, not that the bytes have landed -- and `ucp_am_send_nbx` is not ordered
against RDMA writes still in flight. The code says this at the call site (`:4131`). A1 holds in
this deployment because both operations travel the **same RC queue pair** and the InfiniBand
transport delivers messages on one QP in order. **It would break** under multi-rail striping,
under a configuration that carries active messages on a different transport than the PUT (our
`UCX_TLS=rc_mlx5,ud_mlx5,tcp,self` permits `ud_mlx5` and `tcp`), or under any UCX version that
selects different lanes for the two operations.

**A2 is the GPUDirect read-after-write question**, whose documented mitigation is a flush read;
we perform none.

## 6.3 The evidence we have, and exactly what it is worth

- **The flush experiment (2026-07-25).** Adding `ucp_ep_flush_nbx` before the `RmaPosted`
  reproduced the failure under investigation **identically** -- same transfers, same byte counts
  -- while roughly doubling issue time (310 ms against 166 ms for 6 x 64 MB). It was therefore
  not applied. **This is weak evidence for A1**: it shows the flush did not fix *that* bug, not
  that A1 holds. It is recorded as a cost bound, not as a proof.
- **End-to-end bit-exactness.** XSBench reports **identical verification checksums (408237)
  across every arm** at 6e8 lookups, native included. cuDF and PDS-H match native output.
- **Scale without a visibility failure.** 2 643 921 RMA admissions over 14.78 M operations, with
  80 fallbacks and zero corruption attributable to ordering.

## 6.4 The claim, reduced to what the evidence supports

> **We do not claim that Gusto guarantees NIC-to-GPU visibility.** We claim that on the
> evaluated configuration -- ConnectX-7, RoCEv2 200 GbE, **UCX 1.20.0**, drivers 580.95.05 and
> 560.35.05, a single RC lane per connection -- **no visibility failure was observed across
> 2.64 M RMA admissions with end-to-end checksum validation**, and that the two assumptions A1
> and A2 above are **configuration-dependent and neither negotiated nor checked at runtime**.

Three consequences we accept and state:

1. **A deployment that stripes across rails, or that lands active messages on a different
   transport than the PUT, may violate A1.** Our own `UCX_TLS` permits such a selection; we do
   not detect it.
2. **The result is not portable across UCX or driver versions** without re-validation, because
   A1 depends on lane selection and A2 on the peer-memory implementation.
3. **This bounds the GPUDirect claim, not the host-RMA one.** On the host path the payload lands
   in host memory the server reads with the CPU, and A2 does not arise.

## 6.5 What would discharge I10, and what it would cost

Two pieces, in this order:

1. **Capability negotiation at handshake.** The machinery already exists in the adjacent case:
   `GPUDirect` activates only on endpoints that *actually negotiated an RDMA lane*, because
   `ucp_ep_rkey_unpack` returns `UCS_OK` even when the negotiated transport cannot do peer-DMA
   (`ARCHITECTURE_UCX.md` §593-601). The same shape applies here -- each side advertises its
   transport and whether PUT/AM ordering is guaranteed for it -- and the field fits in the
   existing `RmaSetup` header.
2. **A conditional explicit flush**, applied only when negotiation says ordering is not
   guaranteed. Its cost is already measured: **1.9x issue time** (310 ms against 166 for
   6 x 64 MB), which is why it must be conditional rather than unconditional.

Until both exist, I10 stays in §6 rather than in §4, and the paper quotes §6.4 rather than a
guarantee.
