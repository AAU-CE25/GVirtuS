# RMA round 6 — slot-lifetime audit

## 1. Executive summary

Three results this round, one decisive, one that refutes the leading hypothesis, and one
that undermines every failure-rate number produced so far.

- **The source buffer is exonerated.** With a distinct, write-once source buffer per
  transfer, the corruption still appears with the same provenance: a byte holding the
  *immediately preceding* transfer's value at that offset, from a host buffer the failing
  transfer never touched. The staleness is on the **destination slot**, not the source.
- **Pool size does not behave as premature-reuse predicts.** Failures per 96 transfers:
  2 slots → 22, 8 slots → 6, 16 slots → 27. Non-monotonic. Longer reuse distance does
  not reduce the defect, so "the sender rewrites a slot before the receiver finished with
  it" is not supported in its naive form.
- **The failure rate is highly harness-sensitive and therefore unreliable.** The same
  production build gives 159/160 (0.6%) under `rma3x64` and ~22/96 (23%) under
  `rma_srcprov`. Until that discrepancy is explained, no rate figure in any round of this
  investigation should be quoted as the defect's probability.

**Classification: UNSAFE TO ENABLE.**

## 2. Scope correction carried forward

`integ/rapids-async` is **not** known-good. Measured last round: 159/160 over
10 runs × 16 × 64 MB, clean tree, dpu-01 `d99a04e` / dpu-02 `f0d8c1f`. The lazy pool does
not create the defect. Earlier 16/16 and 48/48 certifications were under-powered.

## 3. Provenance methodology and result

Payload: `byte[offset] = (char)(transfer * 31 + (offset >> 12))`, so a corrupted sample
that reads exactly 31 less than expected identifies the immediately preceding transfer
unambiguously. Values that appear as `delta = +225` are the same relation wrapped in
8-bit signed arithmetic (`225 ≡ −31 mod 256`); **every** mismatch observed across all
runs, without exception, decodes to the immediately preceding transfer.

`examples/rmatest/rma_srcprov.cu` runs the identical workload two ways:

| mode | source buffers | result on the production build |
|---|---|---|
| `reuse` | one buffer, rewritten before each transfer | fails |
| `fresh` | one dedicated buffer per transfer, filled up front, never rewritten | **still fails** |

Decisive instance, `fresh` mode: transfer 13, offset 0, `got=85 want=116 delta=-31`.
`want = (char)(12*31) = 116`; `got = 85 = (11*31) & 0xFF`, i.e. transfer 12's value.
Transfer 12 wrote only into `srcs[11]`, transfer 13 read only `srcs[12]` — different
allocations. The byte that appeared at the destination never existed in the source the
failing transfer used.

This refutes the entire secondary hypothesis class (source lifetime, `ucp_mem_unmap` on
local completion, registration-cache reuse, source visibility) as the carrier of the
stale bytes, and makes experiments M1–M4 unnecessary.

Native sanity check: both modes are clean without GVirtuS (0/4 failures each,
26.4 GB/s), so the test itself is sound.

## 4. Pool-size sweep (R2)

Production build, `reuse` mode, 6 runs × 16 × 64 MB per configuration:

| `GVIRTUS_RMA_SLOTS` | failed transfers / 96 |
|---|---|
| 2 | 22 |
| 8 | 6 |
| 16 | 27 |

No monotonic relationship with reuse distance. Whatever widens the window is not simply
how soon a slot comes round again.

## 5. The rate discrepancy, unexplained

Same production build, same size, same 64 MB transfers, both validating 4 KB-aligned
samples:

| test | failed transfers |
|---|---|
| `rma3x64` | 1 / 160 |
| `rma_srcprov` (`reuse`) | 22 / 96 |

The tests differ only in bookkeeping around an identical transfer sequence. A ~40×
difference in failure rate from that alone is itself evidence about the mechanism, and it
means the "0.6% of transfers" figure quoted in the round-5 report is not a property of
the defect. Reconciling these two harnesses is the highest-value next step: whatever
differs between them is close to the trigger.

## 6. Falsified, cumulative — do not revisit

- Payload interval coverage (single contiguous 64 MiB `ucp_put_nbx`; 66 + 67108864 + 12
  = 67108942 exactly).
- Destination memory type: reproduces with GPUDirect off, host-memory slot.
- Everything GPU-visibility related: `WRITES_ORDERING = NONE`, `SYNC_MEMOPS`,
  `cuFlushGPUDirectRDMAWrites`, CUDA context ownership.
- Sender remote completion: endpoint and worker flush both applied, no effect.
- `wait_request_completion` tri-state handling (correct as written against UCX 1.20.0).
- Slot-index specificity; temporary/persistent slot collision; `std::vector` reference
  invalidation.
- **Source buffer lifetime and reuse (this round).**

## 7. Remaining hypotheses, reordered by evidence

1. **Reconcile the two harnesses.** A 40× rate difference between `rma3x64` and
   `rma_srcprov` on identical transfers localises the trigger better than any further
   protocol theory. Diff them line by line and bisect the difference.
2. **Receiver consumes before the write completes**, rather than the sender rewriting too
   early. The provenance shows stale *destination* content, and pool size does not help,
   which fits "the read happened too soon" better than "the write happened too late".
   Every receiver-side action that costs time still hides it, consistent with this.
3. Control-message ordering (H7): what, in UCX 1.20, orders `ucp_am_send_nbx` against
   prior `ucp_put_nbx` on the same endpoint after a completed flush.

## 8. Workload impact

- **miniBUDE unaffected** — its transfers are below the 4 MB RMA floor and never enter
  this path. Its validated multi-tenant results stand. (Worth confirming from a trace
  rather than configuration alone before publication.)
- **Throughput figures (23–24 GB/s)** were measured on this path without content
  validation. Usable as performance data, not as evidence the transport is correct.
- **XSBench** (192 MB grid) is exposed and has no checksum. Do not run or publish its
  numbers without end-to-end content validation.
- **llama** weight uploads ≥4 MB are exposed and have no checksum. Successful model
  loading does not prove bytewise correctness.

## 9. Safe fallback

Below the 4 MB `GVIRTUS_RMA_MIN_BYTES` floor the eager AM path is used and is not
implicated. Raising that floor above the largest transfer a workload performs disables
the affected path entirely at a known bandwidth cost (~9 GB/s staged instead of ~23).
That is the recommended mitigation for any campaign that must run before this is fixed,
and it should itself be validated with thousands of content-checked transfers first.

## 10. Reproduction

```
cd ~/GVirtuS/examples/setupprobe
# provenance A/B (production build)
./gvrun.sh ucx 256 "" -- "/ex/rmatest/rma_srcprov 67108864 16 reuse"
./gvrun.sh ucx 256 "" -- "/ex/rmatest/rma_srcprov 67108864 16 fresh"
# pool sweep
./gvrun.sh ucx 256 8  -- "/ex/rmatest/rma_srcprov 67108864 16 reuse"
# the lower-rate harness
./gvrun.sh ucx 256 "" -- "/ex/rmatest/rma3x64 67108864 16"
# host-memory destination (still corrupts)
./gvrun.sh ucx_nogds 256 "" -- "/ex/rmatest/rma3x64 67108864 16"
```

## 11. Branches, commits, state

| | |
|---|---|
| production | `integ/rapids-async`, dpu-01 `d99a04e`, dpu-02 `f0d8c1f`, `src/ include/ plugins/` clean |
| this round | `exp/rma-slot-lifetime-audit` (from production); **production sources untouched**, only `examples/rmatest/rma_srcprov.cu` added |
| earlier | `exp/lazy-rma-slots`, `exp/lazy-rma-completion-audit` preserved |
| reports | rounds 2–5 unchanged |

No destructive git operations were used.

## 12. Remaining risks

The defect is in the shipped path, its true probability is unknown because the rate is
harness-dependent, and no receiver-side observation cheap enough to leave the race intact
has yet caught it in the act. Until the two harnesses are reconciled, treat any large-RMA
result as unvalidated.

## Final classification

**UNSAFE TO ENABLE.**
