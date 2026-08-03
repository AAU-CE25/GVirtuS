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

# The result, and why it does not count

| arm | reps | iterations | size changes | corrupted bytes |
|---|---:|---:|---:|---:|
| guard on | 3 | 40 | 20 | **0** |
| guard off (`no_epoch`) | 3 | 40 | 20 | **0** |

**This is not evidence that the guard is unnecessary. The experiment is inert, and the logs
prove it.**

The epoch does change -- from 1 to 2 -- but the announced slot indices **renumber**:

    epoch 1 anuncia server_idx=[0,1,2,3,4,5,6,7]
    epoch 2 anuncia server_idx=[9,10,11,12,13,14,15,16]

With renumbering there is no collision to guard against: an old acknowledgement naming slot 3
refers to an index that no longer exists in epoch 2. The counter confirms it --
`ack_epoch_dropped=0` in the guard-on arm, so the guard never fired, and the ablation therefore
had nothing to disable.

**This is the same failure as the previous attempt**, and it is worth stating as a lesson: a
bench that grows the pool triggers a *rebuild*, and a rebuild renumbers. The dangerous state
needs a **re-advertisement without a rebuild**, which is what `ensure_rma_pool` does and what
cuDF triggers naturally. Forcing it by changing transfer sizes cannot produce it, by
construction.

# What would actually be needed

Three conditions must hold **at the same instant**, and the benches so far achieve at most two:

1. the re-advertisement reuses the indices -- a re-advertisement, not a rebuild;
2. the stale acknowledgement lands **after** the new epoch is installed;
3. the slot it names is **in use** at that moment.

Concretely, one of:

- **Drive it from cuDF**, which is the only workload observed to re-advertise `[0-7]` twice,
  with `GVS_ABLATE=no_epoch` and byte-level validation on its output. This is the cheapest
  route because the workload already produces condition 1 by itself.
- **Add a fault that forces re-advertisement without rebuild** -- a gate in `ensure_rma_pool`
  that re-announces the current pool unchanged -- combined with the existing `epoch_ack_idx`
  fault and `GVS_FAULT_ARM=1` to arm the replay on the first acknowledgement that can collide.

# Status

**Open.** The guard remains a necessary invariant supported by code reading and by the
demonstrated reachability of the dangerous state; it is **not** supported by an observed save.
The paper should say exactly that, and should not cite either the 186 opportunities or this
run as evidence that the guard is unnecessary -- neither experiment ever created the state the
guard exists for.

Data: `results/asplos_campaign/epoch/` (6 runs with counters and full logs), bench
`examples/rmatest/epochharm.cu`.
