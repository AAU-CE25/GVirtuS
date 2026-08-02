---
title: "ASPLOS correctness & evaluation campaign — plan"
date: "2026-08-02"
branch: "exp/asplos-correctness-campaign"
base_commit: "3b05300f70a35bd80134bb248529e90d430b63e2"
---

# 0. Scope

Harden Gusto for an ASPLOS submission along seven axes: real code defects, CUDA-semantic
conformance, deterministic fault injection, clean failure recovery, targeted statistical
repetition, GPUDirect completion/visibility validation, and a reproducible artifact.

**Explicitly out of scope** (and not to be re-litigated): advanced scheduling, new generic
workloads, comparisons against systems not on this testbed (rCUDA, Gleam, Cricket), new
hardware, indiscriminate parameter sweeps, and any optimisation before correctness closes.

Results that already exist and are NOT re-measured: GPU-resident RMA at ~92.5–93.0 % of the
network ceiling; GPU RMA ~1.78–1.80× host RMA at 1 GiB; cuDF ~13.6–15.4 % over the same UCX
protocol with a host bounce; the TinyLlama 2×2; allocation-lifetime invalidation vs
pointer-keyed caching; generation checking vs premature release; XSBench / BabelStream /
miniBUDE / CloverLeaf as workload delimiters.

# 1. Testbed (fixed)

| host | role | fabric IP | driver | kernel | persistence |
|---|---|---|---|---|---|
| `es-dpu-01` | backend, owns L40S | `25.25.25.2` | **580.95.05** | 5.15.0-181 | Enabled |
| `es-dpu-02` | frontend | `25.25.25.1` | **560.35.05** | 5.15.0-100 | Disabled |

Both: AMD EPYC 9354P (32c/64t), 251 GiB RAM, NVIDIA L40S 46068 MiB, 2× ConnectX-7 200 Gb
(fw 32.41.1000), RoCEv2 on `mlx5_1`/`ens1f1np1`, MTU 9000, MLNX_OFED 24.04-0.6.6,
Ubuntu 22.04.5, gcc 11.4.0, cmake 3.22.1, git 2.34.1.

UCX: **1.17.0 on both hosts, 1.20.0 inside the backend container.** Backend container CUDA
12.6.85 (cudart 12.6.77).

**Recorded testbed asymmetries** (not defects, but they bound what can be claimed):
- the two hosts run **different NVIDIA drivers** (580.95.05 vs 560.35.05). Phase 7 asks the
  driver for GPUDirect ordering/flush guarantees; the answer is the *backend's* driver, and
  any statement about visibility is a statement about 580.95.05 only.
- `es-dpu-02:/home` is at **95 % (15 GiB free)**; `es-dpu-01:/home` has 198 GiB. Bulk raw
  output goes to dpu-01 or is pruned. Soak tests must not be started without a space check.
- `core_pattern` routes to apport on both hosts. Every container runs `ulimit -c 0`.

# 2. Phase order (mandatory)

1. Baseline & inventory
2. `cudaStreamPerThread`
3. Semantic conformance
4. Failure recovery
5. Lifetime fault injection
6. TinyLlama repetitions
7. Admission-boundary repetitions
8. cuDF missing repetitions
9. Lifetime overhead
10. GPUDirect visibility
11. Sanitizers, then soak **(soak deferred to the very end by user instruction, 2026-08-02)**
12. Artifact reproducibility

# 3. Working rules adopted

- Inspect before modifying; locate every path rather than assuming it.
- No prior result is deleted. Failed runs are kept as data and labelled.
- One experimental dimension at a time.
- Every functional change ships with a test that fails before and passes after.
- Ablations via runtime flags or compile-time debug flags, never by editing a library in place
  (`lib/` mutation for an A/B has already produced two wrong results in this project).
- A conclusion that is not demonstrated is labelled **pending** or **inconclusive**, never
  "resolved because it stopped happening".
- Structured output (JSON/CSV) over parsing human-readable text.
- **Before citing any number, open its raw file.**

# 4. Deliverable layout

```
docs/asplos_campaign_plan.md      this file
docs/asplos_campaign_log.md       dated block-by-block journal
docs/asplos_campaign_results.md   findings, tables, interpretations
docs/asplos_claim_map.md          paper claim -> evidence -> status
results/asplos_campaign/{environment,semantic_conformance,ptds,failure_recovery,
  tinyllama_2x2,admission_boundaries,cudf_completion,lifetime_cost,gpudirect_visibility,
  sanitizers,soak,processed,figures}/
tests/semantic/                   conformance sources
```

# 5. Phase 0 outcome (gate)

**The baseline gate is PASSED**: connect, 1 MiB H2D, `cudaStreamSynchronize`, D2H, clean
teardown, exit 0, over UCX/RoCE to the live backend. Four stream classes (null, explicit,
legacy `0x1`, per-thread `0x2`) all succeed single-threaded.

Four things Phase 0 found that change how later phases must be run — see
`docs/asplos_campaign_results.md` §1 for the evidence:

- **F-01** `GVIRTUS_RMA_MIN_BYTES` does not gate RMA admission on the frontend. Admission is
  `GVIRTUS_RMA_SCALAR_FLOOR` (default **4 MiB**) or `GVIRTUS_RMA_POLICY`. Any experiment
  intended to exercise RMA below 4 MiB must set one of those explicitly.
- **F-02** `[GUSTO CFG] pool efectivo:` reports the *frontend RX* pool (2 slots), not the
  backend slot pool under study (8 × 32 MiB). It is a true statement about the wrong object.
- **F-03** PTDS surface is 5 real `_ptsz` entry points against **49 silent stubs** returning
  `cudaErrorNotSupported`. `cudaGraphLaunch_ptsz`, `cudaEventRecord_ptsz` and
  `cudaStreamWaitEvent_ptsz` are stubs.
- **F-04** `AblationGate.h` differs between the hosts; the backend (which emits the banner)
  carries the *older* copy, so the "no injection" banner fix never reached the side that
  prints it.

# 6. Per-phase acceptance criteria

Each phase closes only when its criterion is met **and** its raw files exist:

| phase | closes when |
|---|---|
| 1 PTDS | root cause named; a test that fails before the fix; 0 aborts / 0 mismatches across 1/2/8/32 threads; ASan+UBSan+TSan clean |
| 2 conformance | every listed property has a row; 0 mismatches on the full system; injected-wrong variants fail and only they |
| 3 fault injection | every fault reproducible from a seed; guard-on vs guard-off table; corruption distinguished from structural violation |
| 4 recovery | 0 segfaults, 0 deadlocks, diagnosable error, resources returned, new client connects; residual limits written down |
| 5 statistics | ≥5 (target 10) independent process-level repetitions, randomised order, CI reported |
| 6 lifetime cost | bookkeeping as % of total, with instrumentation on AND off, proven not to dominate |
| 7 GPUDirect | the exact ordering/flush guarantee named for this driver+UCX; ablation causal or explicitly inconclusive |
| 8 sanitizers/soak | matrix green; soak shows no unbounded growth in any tracked resource |
| 9 observability | every run emits a JSON summary with commit, config, counters, checksum |
| 10 artifact | every figure and table regenerates from raw; no claim marked verified without a path to rebuild it |
