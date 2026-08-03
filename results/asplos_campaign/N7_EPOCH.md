---
title: "The epoch guard: the dangerous state, produced on demand, and what each defence contributes"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

# Status: closed, with one boundary stated precisely

This entry was open because the guard had never been observed to *do* anything. It is now
exercised on demand and measured: **the epoch guard rejects 126 to 280 stale acknowledgements
per run**, and with it removed those acknowledgements free slots that are genuinely in flight.

What is **not** demonstrated is byte-level data corruption. That needs one further condition
which this hardware's timing never grants, and §5 says exactly what it is. The honest headline
is therefore *"the guard is load-bearing and we can show it working"*, not *"we corrupted data
without it"* -- and that is a far stronger position than the previous *"never observed to fire"*.

# 1. Why nothing ever happened: the state could not be built

Three conditions must hold at the same instant:

1. a re-advertisement **reuses** the slot indices (a re-advertisement, not a rebuild);
2. the stale acknowledgement lands **after** the new epoch is installed;
3. the slot it names is **in use** at that moment.

Every previous attempt failed at (1), and the reason is structural: **every pool rebuild
renumbers.** `server_idx` is the slot's position in the backend's `rx_pool_->slots` vector, so a
rebuild that allocates new slots hands out new indices and an acknowledgement from the old epoch
can no longer name anything. The earlier bench (`epochharm.cu`, alternating transfer sizes to
force rebuilds) produced exactly this:

    epoch 1 anuncia server_idx=[0,1,2,3,4,5,6,7]         ->  install now (0/0 in flight)
    epoch 2 anuncia server_idx=[9,10,11,12,13,14,15,16]  ->  7/8 in flight -> PARK

Both dangerous ingredients appear, and never together: identical indices with **nothing live**,
then live traffic with **renumbered** indices.

The only workload ever observed to reuse indices across an epoch change is **cuDF**, which
re-advertises `[0-7]` in epoch 1 and again in epoch 2.

## The fault that builds it

`send_rma_setup()` publishes the **current** `rx_slots` and increments the epoch. Calling it
again *without rebuilding the pool* therefore reproduces cuDF's state exactly: same indices, new
epoch. `GVS_FAULT=readvertise` (with `GVS_FAULT_EVERY=N`) does that every N acknowledgements, on
the backend. It worked on the first run:

    epoch 2 anuncia server_idx=[9,10,11,12,13,14,15,16]  -> 7/8 in flight -> PARK
    epoch 3 anuncia server_idx=[9,10,11,12,13,14,15,16]  -> 5/8 in flight -> PARK
    epoch 4 anuncia server_idx=[9,10,11,12,13,14,15,16]  -> 1/8 in flight -> PARK

Same indices, three different epochs. **Condition 1 is now available on demand**, which it never
was before.

# 2. The second defence nobody had accounted for: the park

`UcxCommunicator.cpp:2966` refuses to swap the slot layout while any transfer is live; it parks
the new layout and installs it once the last one drains. `GVS_FAULT_NOPARK=1` forces the install
regardless -- deliberately unsafe, and the only way to separate the two defences.

# 3. The 2x2, plus the cell where both guards are off

`epochharm 40`, 8 buffers in flight, byte validation, 3 repetitions per cell (identical across
repetitions in every cell). Backend restarted per arm. `slots=8 cap=8 MiB`, `GVS_FAULT_EVERY=4`.

| arm | park | epoch guard | gen guard | **parked** | **epoch_dropped** | **gen_mismatch** | ack_on_free | ack_applied | corrupted bytes |
|---|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|
| A | on | on | on | **81** | **0** | 0 | 0 | 320/320 | 0 |
| D | on | off | on | **81** | 0 | 0 | 0 | 320/320 | 0 |
| B | **off** | on | on | 0 | **126** | 0 | 0 | 194/320 | 0 |
| C | **off** | **off** | on | 0 | 0 | **38** | 81 | 194/320 | 0 |
| E | **off** | **off** | **off** | 0 | 0 | **114** | 81 | 232/320 | 0 |

Read down the `parked` and `epoch_dropped` columns; that is the whole story.

**A and D explain the entire campaign.** With the park on it fires 81 times per run and the
epoch guard fires **zero** times -- in the guarded build *and* in the ablated one. The campaign's
"0 deliveries in 186 opportunities" was never evidence about the guard: **the park removed the
opportunity before the guard could see it.** Ablating a guard that is never reached measures
nothing, which is precisely why arm D is indistinguishable from arm A.

**B is the measurement that was missing.** Defeat the park and the epoch guard **fires 126 times
per run** on acknowledgements it would otherwise have had to trust. At `GVS_FAULT_EVERY=2` this
rises to **280 of 320**.

**The guard's failure mode is a leak, not corruption**, which is worth stating because it is a
design property and not an accident: rejecting an acknowledgement leaves the slot marked
`InFlight`, so arm B applies only 194 of 320 acks and the in-flight accounting inflates to 127
against a pool of 8. The system degrades into backpressure. It does not hand out a live slot.

**C shows the protections are layered.** With the epoch guard off, 38 stale acks reach the slot
loop and the **generation** guard rejects them there; 81 more land on slots that are already free
and are ignored. The epoch guard is not the last line of defence -- there is one behind it.

**E is the cell with nothing left.** `GVS_ABLATE=no_epoch_gen` -- added for this experiment,
because the existing ablation enum made the two mutually exclusive -- turns both off. Now **114
acknowledgements per run name a slot that is genuinely `InFlight` under a newer generation, and
free it.** That is the dangerous event itself, on demand, 114 times a run.

# 4. What five configurations did not produce: corrupted bytes

| slots | cap | `FAULT_EVERY` | slow-read | payloads | dangerous frees | corrupted bytes |
|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 MiB | 4 | -- | 2 / 6 MiB | **114** | 0 |
| 2 | 32 MiB | 2 | -- | 8 / 24 MiB | **89** | 0 |
| 8 | 8 MiB | 2 | -- | 2 / 6 MiB | 0 (all hit free slots) | 0 |
| 8 | 8 MiB | 2 | 3 ms | 2 / 6 MiB | 0 (all hit free slots) | 0 |
| 8 | 8 MiB | 4 | 3 ms | 2 / 6 MiB | 0 (all hit free slots) | 0 |

# 5. The sixth condition, and why the timing never grants it

Freeing a live slot is necessary but not sufficient. For the bytes to be wrong:

> the client must **re-reserve that slot and put new data into it** before the backend has
> finished **reading** the previous occupant.

The backend reads the slot on `RmaPosted` -- a host-to-device copy of a few MiB, well under a
millisecond -- while the client needs a full RPC round trip to come back and reuse it. The
backend wins that race every time, so a premature free lands on a slot whose contents have
already been consumed, and is harmless.

A gate was added to attack exactly this (`GVS_FAULT_SLOWREAD_MS`, delaying the *read* rather than
the acknowledgement) and it **moves the window the wrong way**: delaying the read also delays the
acknowledgement that follows it, so stale acks arrive later still and land on slots that are free
by then. In the two slow-read rows above the dangerous class falls to zero while `ack_on_free`
rises to 195--273. A 25 ms delay stalls the AM progress thread outright and the run never
completes.

**Closing this would need a fault that delays the read without delaying the acknowledgement** --
that is, acking optimistically *before* consuming the slot. That inverts the protocol rather than
perturbing it, and a demonstration built on an inverted protocol proves less than §3 already
does.

# 6. What may be claimed

**May be claimed.**

- The dangerous state -- indices reused across an epoch change -- **occurs in a real workload**
  (cuDF) and can now be produced on demand in a controlled bench.
- **The park is the first defence and the only one production exercises**: 81 parks per run
  against 0 guard activations.
- **The epoch guard is exercised and rejects 126--280 stale acknowledgements per run** once the
  park is removed, and its failure mode is backpressure rather than a slot handed out twice.
- **The protections are layered**: with the epoch guard off the generation guard catches 38 per
  run; with both off, **114 acknowledgements per run free a slot that is in flight**.
- Therefore *"the guard never fires, so perhaps it is unnecessary"* is **refuted**. It never
  fired because the park stands in front of it.

**May not be claimed.**

- That removing the guard corrupts data. Across five configurations byte validation reports
  zero. The premature frees happen; the window does not open.
- The old "0 deliveries in 186 opportunities", in either direction. Those runs never created the
  state.

# 7. What was added to the code, all off by default

| gate | side | what it does |
|---|---|---|
| `GVS_FAULT=readvertise` + `GVS_FAULT_EVERY=N` | backend | re-advertises the current pool every N acks: same indices, new epoch |
| `GVS_FAULT_NOPARK=1` | frontend | installs a new layout with transfers in flight |
| `GVS_FAULT_SLOWREAD_MS=N` | backend | delays the slot read on `RmaPosted` |
| `GVS_ABLATE=no_epoch_gen` | frontend | disables the epoch **and** generation guards together |

**A build hazard found while doing this, recorded because it cost a run.** `ablation_mode()` is
an `inline` function in a header, so every shared object that includes it carries its own copy
and the dynamic linker uses whichever loads first. Rebuilding only
`libgvirtus-communicators-ucx.so` left the stale copy inside `libgvirtus-communicators.so`
interposing, and the run reported `valor no reconocido 'no_epoch_gen'` while every counter still
looked plausible -- an ablation that silently did not happen. **Every `.so` that includes the
header must be rebuilt together**; `strings <lib> | grep <new-token>` is the check that catches
it.

Data: `results/asplos_campaign/epoch_n7/` (`n7_2x2.csv`, 29 client logs with the full
`[GVS IDX]` / `[GVS] rma_setup` sequences, and both harnesses). Bench
`examples/rmatest/epochharm.cu`.
