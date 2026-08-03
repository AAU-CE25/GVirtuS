---
title: "The harmful epoch-collision demonstration -- still open, and now with the reason"
date: "2026-08-03"
geometry: margin=2.3cm
fontsize: 10pt
---

# What was attempted

The epoch guard is demonstrated **reachable**: cuDF re-advertises slot ids `[0-7]` unchanged
across epoch 1 and epoch 2, so an acknowledgement from the old epoch can refer to a slot that
now belongs to the new one. What has never been demonstrated is **harm** -- an old
acknowledgement actually releasing a live slot, with observable corruption. Across three
workloads the previous campaign recorded **0 deliveries in 186 opportunities**.

`examples/rmatest/epochharm.cu` was written to force it: eight buffers, sizes alternating to
force pool rebuilds, no synchronisation between buffers so transfers are in flight while the
epoch changes, byte-level validation. Two arms, guard on and `GVS_ABLATE=no_epoch`, three
repetitions each.

# The result, and what the verification pass found underneath it

| arm | reps | iterations | size changes | corrupted bytes |
|---|---:|---:|---:|---:|
| guard on | 3 | 40 | 20 | **0** |
| guard off (`no_epoch`) | 3 | 40 | 20 | **0** |

An earlier draft of this document concluded that the experiment was **inert** because the epoch
indices renumber. **That conclusion was wrong, and the verification pass caught it.** Reading
the full log rather than a de-duplicated grep shows three announcements, not two:

    epoch 1 anuncia server_idx=[0,1,2,3,4,5,6,7]   ->  install now (0/0 in flight)
    epoch 1 anuncia server_idx=[0,1,2,3,4,5,6,7]   ->  install now (0/0 in flight)
    epoch 2 anuncia server_idx=[9,10,11,12,13,14,15,16]
                                                   ->  7/8 in flight -> PARK -> INSTALLED after park

**Both dangerous conditions did occur -- but never at the same time.**

- A **re-advertisement with identical indices** happened, twice, which is exactly the state
  cuDF produces and the one the guard exists for. Both times it was installed with **0 of 8
  slots in flight**, so there was nothing live for a stale acknowledgement to free.
- **In-flight traffic across an epoch change** also happened, at epoch 2 with **7 of 8 slots
  live** -- but that epoch renumbered, so no index could collide.

## The reason is a mechanism I had not accounted for: the backend parks the install

`UcxCommunicator.cpp:2966` chooses between `PARK` and `install now` on `any_inflight`, and a
parked layout is installed only once the last in-flight transfer drains
(`UcxCommunicator.cpp:2517`). **The backend structurally refuses to swap the slot layout while
transfers are live.**

That is the real explanation for why harm has never materialised across this campaign and the
previous one: **the park prevents the dangerous state from forming**, and the epoch guard sits
behind it as a second line of defence for whatever the park misses.

This is a better result than the one it replaces. It also reframes what the guard is for: not
the common case, which the park already covers, but the residual -- an acknowledgement that was
already in flight when the park released.

# What would actually be needed

Three conditions must hold **at the same instant**, and the park makes the third one hard by
design:

1. the re-advertisement reuses the indices -- a re-advertisement, not a rebuild;
2. the stale acknowledgement lands **after** the new epoch is installed;
3. the slot it names is **in use** at that moment -- which the park exists to prevent.

**So the experiment must defeat the park**, not merely create the re-advertisement. A fault gate
that forces `install now` regardless of `any_inflight` would do it, and would isolate exactly
what the guard contributes beyond the park.

Concretely, one of:

- **Drive it from cuDF**, which is the only workload observed to re-advertise `[0-7]` twice,
  with `GVS_ABLATE=no_epoch` and byte-level validation on its output. This is the cheapest
  route because the workload already produces condition 1 by itself.
- **Add a fault that forces re-advertisement without rebuild** -- a gate in `ensure_rma_pool`
  that re-announces the current pool unchanged -- combined with the existing `epoch_ack_idx`
  fault and `GVS_FAULT_ARM=1` to arm the replay on the first acknowledgement that can collide.

# Status

**Open, and better understood.** The guard is supported by code reading and by the
demonstrated reachability of the state; it is **not** supported by an observed save, and the
reason is now known: the park removes the opportunity in the common case. The honest framing is
that **the slot layout is protected by two mechanisms, and the campaign has only ever exercised
the first.**
The paper should say exactly that, and should not cite either the 186 opportunities or this
run as evidence that the guard is unnecessary -- neither experiment ever created the state the
guard exists for.

Data: `results/asplos_campaign/epoch/` (6 runs with counters and full logs), bench
`examples/rmatest/epochharm.cu`.
