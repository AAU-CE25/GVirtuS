#

> ## 2026-08-01 -- new data
>
> `RESULTS_2026-08-01.md` (paper) and `artifact/data/raw/ablation_2026-08-01/` (raw data,
> harnesses and patches). Includes: the execution-plane 2x2 ablation, correctness ablation
> and cost of the lifetime protocol, the threshold sweep, per-RPC traces for CloverLeaf
> (425,084 rows x 3 seeds) and miniBUDE, and the rebuilt aggregates for XSBench
> (`XSBench_filtered_v2.csv`) and llama (`LLAMA-7B_full.csv`, 1409 rows).
 DATA_INDEX.md -- what to use for the paper (2026-07-31)

## Start here

| file | what it is |
|---|---|
| **`RESULTS.md`** | **The tables.** §1--§10, all re-measured after the 07-26 04:24 cutoff. This is the source for every headline number. |
| **`LLAMA-7B.csv`** | **The curated raw data.** 467 rows, 77 label families. Only current, valid, quotable measurements -- see the filters below. |
| **`MINIBUDE_RESULTS.md`** | miniBUDE contention scaling. Both halves re-measured: GVirtuS arms after the frontend rebuild (change < 2.5%), baremetal with MPS enabled. |
| `OSU_RESULTS.md` | OSU MPI GPU-buffer micro-benchmarks. Unaffected by the cutoff (transfer bandwidth does not use CUDA graphs). |
| **`CLOVERLEAF_RESULTS.md`** | CloverLeaf contention scaling, five arms, N=1/2/4/8 plus the staggered control. Also carries the MPS numerical-corruption result and the RPC-count measurement that explains why GPUDirect does not show up. |
| **`BABELSTREAM_RESULTS.md`** | BabelStream contention scaling, five arms, same matrix. The workload where remoting overtakes default native from N=2 upward. |
| **`CloverLeaf.csv`** | CloverLeaf contention aggregate, **25 rows**, one per (arm, N, launch mode). Different schema from `LLAMA-7B.csv` -- see below. |
| **`BabelStream.csv`** | BabelStream contention aggregate, **37 rows**, same schema. The extra rows over CloverLeaf are the stagger-offset sweep (`stagger0p*`). |
| `OSU.csv` | OSU point-to-point, **92 rows** parsed from the raw log: `osu_latency`/`osu_bw`/`osu_bibw` on GPU buffers plus `osu_bw` on host buffers, 23 sizes each. The collectives are absent because they crashed (not CUDA-aware) -- see §2 of the document. **The GVirtuS data-path numbers (§3) are not in this CSV**: the `transfer_bench` output was never kept as a file, so those four values exist only as transcriptions in `OSU_RESULTS.md`. |
| `XSBench.csv` | XSBench, **one row per measured point** (122), each carrying its `lookups`, `cohort_complete` and `degenerate` flags. Read the caveat below before aggregating from it. |
| `MiniBUDE.csv` | miniBUDE **aggregate**, 20 rows, one per (arm, N): cohort throughput, per-tenant percentiles, and the fairness ratios `jain_mean` / `minmax_ratio` with their per-repetition ranges. Built from the per-tenant `t<i>.log` files, which had never been aggregated. **Do not quote `agg_throughput_*` for this workload** -- see below. |
| **`CUDF_ETL_RESULTS.md`** | **The cuDF data-path campaign -- current.** 78,108 batches, 20 campaigns, seven arms, N=1/2/4/8. The only workload that exercises the data path, so the only one where GPUDirect is measurable. Carries the phase decomposition, the pinned-memory intervention with its TCP placebo, and the controls that refuted four of our own hypotheses. |
| **`ARCHITECTURE_UCX.md`** | **How the transport is built and why every threshold has the value it has.** Wire protocol, slot pool (lazy build, self-reporting trigger, epochs, ABA guard), the five thresholds with their evidence, the registration-lifetime subsystem, and an explicit section on what is *not* proven. |
| `CUDF_RESULTS.md` | **SUPERSEDED** by `CUDF_ETL_RESULTS.md`. Measured 2026-07-29, before the driver-API D2H fix (3.32x -> 1.50x overhead). Kept only as the "before" half of that result. |
| `cuDF.csv` | **SUPERSEDED** -- the pre-fix aggregate, 40 rows. Its schema notes below still describe how the statistics were computed, which is unchanged. |
| **`XSBENCH_RESULTS.md`** | XSBench contention scaling, four systems, plus the MPS comparison and the tenant-density experiment. |

**Do not quote from** `ABSTRACT_NUMBERS.md`, `FINDINGS.md`, `LIMITATIONS.md`, `MANIFEST.md`,
`P2_SATURATION.md`, `README.md`. All six carry a PRE-CUTOFF banner: their GVirtuS numbers were
taken with the `cudaGraphExecUpdate` stub and understate decode throughput.

## How `LLAMA-7B.csv` was filtered

Derived from `summary.csv` (kept intact as the historical record; 841 rows were excluded).

1. **Stale-frontend cutoff.** Every GVirtuS-side family measured before **2026-07-26 04:24**
   is dropped. Until that rebuild the deployed `libcudart.so.12` carried the
   `cudaGraphExecUpdate` stub, so CUDA graphs never collapsed the launch stream. Measured on
   identical configurations: UCX 7B C8 TPOT **18.7 -> 12.5 ms**, TCP **39.3 -> 16.0 ms**.
   Baremetal families are kept regardless of date -- they never load the shim.
2. **Last-5 for re-measured families.** The §7 saturation re-run reused the original labels,
   so each `*_sat5_l*` holds 5 old rows followed by 5 new ones with identical names. Only the
   final 5 per label survive.
3. **Degenerate rows** (13 dropped). A row must satisfy `completed > conc_per`. bench.py's
   closed-mode stability criterion is `fail == 0`, so a run that completed *nothing*, or that
   completed one round while the server was still loading the model (TTFT p50 of 32--56 s), is
   recorded `STABLE` and looks legitimate.

**When aggregating, still filter `stable == STABLE`** -- the file keeps non-STABLE rows that
passed the completion check, because saturation points are *expected* to be unstable and that
is the signal in §7.

## Label families -> table

| family | table | note |
|---|---|---|
| `7b_bm_gon_uni_c*`, `7b_bmHI_c*` | §2 | baremetal isolated serving; HI = n=40 at C2/C4 |
| `7b_ucxgd26_gon_uni_c*`, `7b_ucxHI_c*` | §2 | UCX GPUDirect |
| `7b_truerdma_c*` | §2 | RDMA with GPUDirect off **on the backend** (the frontend flag alone does not disable it) |
| `7b_rdma26b_uni_c*` | control | frontend-flag-only arm -- same system as `ucxgd26`, kept to show they agree |
| `7b_c4noise_*` | control | same-system noise floor at C4: 25 reps span 1.94x |
| `7b_ucxA_*`, `7b_tcpA_*`, `1b_*A_*` | §3 §5 §6 | unified pass; `goff` = graphs off, `fix` = FIXED prompts |
| `7b_*_sat5_l*` | §7 | open-loop lambda, 10 seeds; use the stable-count, not goodput alone |
| `mtbm_*`, `mt_ucx_*` | §8 §8b | multi-tenant; `_l1.0` = flat total load, `_lN` = per-pod-constant |
| `mtmpson_*`, `mtmpsoff_*` | §8b | **the MPS control** -- the experiment that identified the mechanism |
| `7b_cb0_c*`, `7b_cb1_c*` | scope note | cuBLAS async-GEMM ablation. **Null result, deliberately kept**: `cb0` is the control and reproduces the baseline to the decimal, so the null is real and not a broken run |


## `cudf_etl/` -- the current cuDF dataset

Ten CSVs, fourteen figures and the four scripts that produced them. The chain matters:
`consolidate.py` reads the per-batch records and writes **every** CSV; `figures.py` reads
**only those CSVs** and writes every figure. Nothing re-reads the raw logs, so a table and
its figure cannot disagree.

| file | rows | what it is |
|---|---:|---|
| `weak_scaling.csv` | 28 | the headline table: median, time overhead, retention, loss per (arm, N) |
| `cell_summary.csv` | 118 | every cell with bootstrap CI95 **over repetitions** and p50/p95/p99 over batches |
| `per_repetition.csv` | 210 | one row per repetition -- the unit of replication |
| `startup_regression.csv` | 3 | alpha, beta, beta's bootstrap CI95, R2, max residual |
| `phase_breakdown.csv` | 12 | H2D / compute / D2H, both memory types, N=1 and 8 |
| `memory_2x2.csv` | 9 | the pinning intervention and the TCP placebo |
| `jain_fairness.csv` | 168 | per-repetition fairness index |
| `communication_counters.csv` | 1,270 | call counts and size histogram per API and path |
| `system_samples.csv` | 707,636 | 100 ms host telemetry, consolidated from 677 files |
| `exclusions.csv` | 3 | the exclusion ledger -- 817 batches, every one accounted for |
| **`raw_batches.jsonl`** | **80,453** | **the primary record**: every per-batch measurement across 19 campaigns, each tagged with its campaign. Includes the records the exclusion rules drop, so the rules can be re-applied and audited rather than taken on trust |
| `run_summary.csv` | 667 | one row per measured point with its own `wall_s` -- the startup regression is fitted from this, not from the per-batch records |
| `pool_events.jsonl` | 100 | slot-pool lifecycle: growth, epoch arrival, retirement, capacity miss |
| `manifest.csv` | 885 | per-run traceability |
| `metadata.json` | 16 | one object per campaign: detected topology, hosts, container image, software versions |
| `COMMANDS.md` | - | how the campaign was run, and how to re-derive every table and figure from this folder alone |

**Three things to know before quoting from it.**

* **The unit of replication is the repetition, not the batch.** Medians are taken within a
  repetition first, then across repetitions; the bootstrap resamples repetitions. Treating
  60 batches as 60 replicates would inflate n sixtyfold and produce intervals that are
  arithmetically correct and scientifically false.
* **Exclusions are applied by code, by objective rule, never case by case.** A repetition
  with no per-client summary record terminated abnormally and is dropped; so is a point whose
  cohort was incomplete. Nothing is deleted from disk. The ledger is in `exclusions.csv`.
* **Retention and time overhead are different quantities** and are never used
  interchangeably: at N=1 GPUDirect has a time overhead of 11.3% and a retention of 89.8%.

The figures are greyscale-safe -- every series carries a distinct marker and line style, not
only a colour -- so they survive being printed in black and white.

## `cuDF.csv` -- read this before using it

One row per (configuration, tenants, batch size). 32 columns. Three things to know:

* **`lat_median_s` is the median of per-repetition means**, and `lat_ci_lo`/`lat_ci_hi`
  bootstrap the **repetitions** (5 per cell), not the individual batches. Resampling batches
  inflates the sample and narrows the interval artificially: under that error the N=1
  GPUDirect-vs-host-RMA difference looked significant, and it is not. Percentiles are computed
  within each repetition over all its batches, then the median across repetitions is reported.
* **`thr_agg_mibs` is only quotable when `agg_quotable=1`** (measured overlap > 1.5x). At N=1
  the ratio is 0.73 -- cuDF startup and teardown sit inside wall time but outside batch latency.
  Summing non-concurrent per-tenant rates is what produced an impossible 742 GFLOP/s in
  miniBUDE.
* **Two arms are not headline results.** `adaptive` is a 16->64 MiB slot-size variant measured
  before an implementation mistake was caught -- the spec's configuration G is **`adaptreal`**.
  `native_prepatch` and `tcp_norearm` are methodological controls (container-flag symmetry, and
  backend-restart accumulation) kept because they answer questions, not because they are
  results.

`h2d_gpudirect_frac` is the mechanism column: 0.99 for the GPUDirect and adaptive arms, 0.000
for host-RMA, empty for native/MPS/TCP. It is what proves the arms are different experiments
rather than two labels on one configuration.

Its raw artefacts (`raw_cudf/`, `plots/cudf/`) **have been removed**: they belonged to the
pre-fix campaign this file superseded, and keeping two sets of cuDF raw data invited quoting
the wrong one. The current equivalents are all in `cudf_etl/`.

## Two schemas, six files -- do not merge them

`LLAMA-7B.csv` is **LLM-serving only**: its columns are `ttft_*`, `tpot_*`, `goodput`,
`conc_per`, `eff_batch`. None of them apply to a contention workload, which is why the
contention data is a separate file rather than 20 empty columns on both sides.

`CloverLeaf.csv` and `BabelStream.csv` are the contention aggregates, one file per workload,
one row per (arm, N, launch mode), carrying `completion_p50/p95/p99`, `agg_throughput_*`,
`agg_iqr`, `agg_spread_ratio`, `per_client_min/max`, `jain_mean`, `minmax_ratio`,
`reps_complete/requested` and `multimodal_flag`. Rows come only from repetitions in which
**every** client produced a valid result; `reps_complete < reps_requested` marks the points
where that filter bit.

Three things to know before reading it:

* **`completion_p50` is the primary metric when comparing transports.** Throughput is
  secondary and can mislead: at N=8 BabelStream reads TCP 681.2 GB/s against RDMA 687.1 --
  1% apart -- while the TCP wall-clock is 21% worse. BabelStream GB/s is sustained *kernel*
  bandwidth, and the remoting cost sits between the kernels, not inside them.
* **`jain_mean` does not discriminate on its own.** It reads 0.986--1.000 across arms that
  differ by 21% in `min/max`. Read it together with `minmax_ratio` and the percentiles, or
  every arm looks identical.
* **`launch_mode`**: `sync` releases all N tenants within ~1 ms -- the pessimistic bound, and
  a lock-step case no real deployment produces. `stagger` offsets them by 2 s -- the
  optimistic bound. `stagger0p*` are the offset sweep. The honest operating range is between
  the two; the offset sweep shows the gain is monotonic in the offset and that breaking exact
  synchrony is not enough (9--75 ms changes almost nothing).

Raw trees, per workload:
`~/experiments/{cloverleaf,babelstream}/results/<arm>/N<n>/<mode>/seed*/`,
`~/xsbench_campaign/results/` (per-client stdout, `seed_raw.json`, `.attempts/` snapshots of
every retry) and `~/mb_campaign/run_*`.

## `XSBench.csv` -- always filter on `lookups`

The file is one row per point straight from the tree, and the tree holds **two problem
sizes**: 90 rows at 6e8 and 32 at other sizes, because the tenant-density experiment was run
at 1.25e9 on purpose and reused the same `seed*/` directories. Aggregating without filtering
`lookups` averages two different workloads. The authoritative cross-arm table is the one in
`XSBENCH_RESULTS.md`, measured at 6e8 throughout; this CSV is the surviving raw material, not
a replacement for it.

`degenerate=1` marks 14 rows whose makespan is under half what its N should take -- clients
that died at start-up. **`cohort_complete` does not catch them**: two native N=8 runs at 6e8
record `n_done=8/8` with a 1.4 s makespan against a normal ~98 s. Filter on both columns.

## miniBUDE -- fairness is where the arms separate, and two traps

`MiniBUDE.csv` carries the result that cohort wall time cannot show: **native shares
the GPU between tenants and remoting serialises it.** At N=8, Jain is 0.9998 for baremetal
and for baremetal+MPS (min/max 0.97 and 0.98) against 0.65--0.73 for all three GVirtuS arms
(min/max 0.18--0.23) -- no overlap. Under GVirtuS the fastest tenant runs at the full
single-client rate while the slowest sits near a fifth of it.

This separates the *mechanism* from the *policy*. MPS and the GVirtuS backend both funnel N
tenants into one CUDA context -- that is why their cohort times track each other here, in
llama §8b and in BabelStream -- but MPS interleaves and the backend serialises. The
context-consolidation explanation used elsewhere in this campaign stays correct; it just does
not carry fairness with it.

The three remote arms do **not** differ from one another: `rdma` vs `GPUDirect` at n=15 gives
Jain p=0.093 and min/max p=0.419. An n=5 sample had looked convincing (min/max p=0.087, 4 of
5 rdma reps above every gpudirect rep); at n=15 that became 2 of 14. Small-sample artefact.

**Trap 1 -- `agg_throughput_*` is not a throughput here.** It sums per-tenant rates, which is
only meaningful if the tenants overlap in time; under GVirtuS they do not. The sum reaches
742 GFLOP/s at N=8, **3.4x the card's physical capacity** (one tenant alone reaches 216 and
saturates it). Use `wall_s` -- cohort wall time -- as the performance metric for miniBUDE.

**Trap 2 -- `tenants_ok` in `~/mb_campaign/*.csv` is double-counted for the GVirtuS arms.**
`mb_campaign.sh` counts with a `t*.log` glob that also matches the `tenant_*.log` files those
arms write, so N=8 records 16. Harmless for the GFLOP/s statistics (duplicating values changes
neither mean nor min nor max) but useless as a completeness filter, since `16 >= 8` passes
even with dead tenants. `MiniBUDE.csv` indexes `t{i}.log` explicitly and is unaffected.

**When re-running miniBUDE, arm the backend per arm.** `mb_campaign.sh` sends `MTSYS=ucx` for
both `ucx_rdma` and `ucx_gpudirect` -- its own comment says the arms *"differ on the BACKEND,
not here"* -- so with an unarmed backend the two are silently the same experiment.

## XSBench -- raw tree partly overwritten (the tables are fine)

`XSBENCH_RESULTS.md` is quotable as it stands: its main table is measured at **6e8 lookups
for every arm**, n=5 for N=1/2/4 and n=8 for N=8, and the document states that all arms
report identical verification checksums at that size (408237), so remoting is bit-exact.

What *is* damaged is the raw tree. The tenant-density experiment was run at **1.25e9 on
purpose**, and it reused the same `seed*/` directories, so
`~/xsbench_campaign/results/xsbench/<arm>/N<n>/` now holds a mixture of both sizes and only
a few 6e8 runs survive. Anyone re-deriving numbers from the tree -- rather than from the
table -- must filter on `lookups` in `seed_raw.json` or they will average two different
workloads.

Two further notes for anyone re-running it:

* **The harness never arms the backend.** `run_seed_xs.sh` sets client-side flags only
  (`GVIRTUS_GPUDIRECT`, `GVIRTUS_RMA_ZEROCOPY`); as this index notes for the llama campaign,
  the frontend flag alone does not disable GPUDirect. Arm the backend per arm and assert the
  `[GVS] GPUDirect=` line before trusting an arm label.
* **`run_block_xs.sh` does not re-arm between N groups**, and the backend degrades as
  connections accumulate -- measurably, before it dies. Re-measured 2026-07-28 with a re-arm
  before every point: N=1 dispersion fell from 2.9% to 0.7% and the N=8 bimodality (13%
  spread, plus one 2.9 s run that `done=8/8` accepted) disappeared entirely.

## Not included, on purpose

An accept-loop defect in the GVirtuS backend (it stops accepting connections under sustained
8-client load while the container, the process and the exit code all still report healthy) was
diagnosed during this campaign and is documented separately in `ANOMALY_TCP_LISTENER.md`. It is
**an implementation bug in GVirtuS, not a property of the architecture being evaluated**, so it
is not part of this dataset and should not appear in the paper's results. It mattered only as an
operational hazard while measuring: every harness here verifies the listener with `ss` before
trusting a run.

## Three things to carry into the writing

**Scope.** Every fix in this campaign is inside the remoting layer; the applications are
unmodified. The open prefill gap (TTFT ~=2.9x baremetal at C8) would likely close by extending
CUDA-graph capture to prefill -- but that is a llama.cpp change, so it is reported as a measured
cost, not optimised away. See the "Scope" section in `RESULTS.md`.

**Both baselines.** Against *default* native multi-tenancy GVirtuS looks ahead; against
*MPS-configured* native it is at parity or slightly behind, in all three workloads. Quoting only
the first is a straw man. Quoting only the second discards the practical result. Quote both.

**GPUDirect's reach -- corrected 2026-07-31.** This entry used to say that no application in
the campaign exercised the peer-DMA path. That was true of the five workloads it covered and
is no longer true: **cuDF streaming ETL does**, and it is the workload the data path was built
for. It moves 62.5 MiB per column, eight columns per batch, in both directions, continuously.

The other five still do not, and the reason is the same 4 MiB RMA floor: llama's largest
transfer is 3.6 MB, miniBUDE's 256 KB, and XSBench crosses the floor once per run and then
computes for 100 s. That is why `rdma` and `GPUDirect` are statistically indistinguishable in
every cell of both contention workloads -- a control confirming the threshold behaves, not a
disappointment.

On cuDF the ablation is measurable and monotone: against `ucx_host_rma`, which is the same
transport and the same slot pool with the payload bounced through host memory, peer-DMA is
worth **57.6 ms at one tenant growing to 105.8 ms at eight**, and **more** -- not less -- when
the baseline's host memory is pinned (78.0 to 127.5 ms). State the regime where it applies:
above 4 MiB, and repeatedly.

As of 2026-07-28 this has a measured mechanism rather than an observation. CloverLeaf issues
**411,054 RPCs in one 47 s run** (~8,700/s) and its makespan is the RPC count times the
per-RPC cost: 32 µs baremetal, 36 µs over RDMA, 115 µs over TCP. Cross-checked two ways -- the
TCP-minus-RDMA gap is 64 µs per RPC, and the measured round-trip gap is ~40 µs plus tail.
These workloads are **control-bound, not data-bound**: the messages are small *because* the
application is a stream of short calls, so lowering the RMA threshold would not help -- it
would only push small messages down a path built for large ones. `rdma` and `GPUDirect` are
statistically indistinguishable in every cell of both contention workloads, which is the
same fact seen from the other side.

Defending the zero-copy data path needs a workload that moves bulk blocks repeatedly. None of
those five does -- **cuDF does**, and `CUDF_ETL_RESULTS.md` is where that defence lives. The
two facts are complementary rather than contradictory: the contention workloads bound where
the data-path advantage *does not* apply, which is what makes the cuDF result specific instead
of a general claim about virtualisation.

## Fairness -- the audit that rebuilt it from the raw per-tenant records

`FAIRNESS_RESULTS.md` (+ `.pdf`) is the fairness reference. It does not start from the
published Jain indices: it rebuilds them from the raw per-tenant records of every workload,
with controls, and asks whether the inequality is introduced by Gusto or explained by demand,
sample size, Poisson arrivals, start skew or natural variability.

Data it produced, all also under `results/asplos_campaign/fairness/` in the repo:

| file | rows | what it is |
|---|---:|---|
| `tenants_canonico.csv` | 3688 | one row per tenant per run, four fixed-work workloads |
| `fairness_trabajo_fijo_resumen.csv` | -- | Jain per cohort, then median across cohorts |
| `llama_fairness_por_tenant.csv` | 169 | per-tenant serving fractions |
| `tabla_D_minibude_por_tenant.csv` | 352 | per-iteration service-quantum timeline |

### It confirms two claims this index already made, and supplies their proof

- **miniBUDE (§ above): confirmed independently.** A separate pipeline reading the raw tenant
  logs reproduces Jain 0.9998 baremetal / 1.0000 MPS against 0.696 GPUDirect, 0.746 host RMA,
  0.669 TCP, and the "fastest tenant at the full single-client rate while the slowest sits
  near a fifth of it" -- measured as x1.001 beside x4.874 in the same cohort.
- **llama (`RESULTS.md` §8): confirmed and now demonstrated.** That section already said the
  Jain decline 1.000 -> 0.804 is a property of the shared Poisson arrival trace and should not
  be cited as a GVirtuS limitation. It was right, and it was asserted. It is now shown: the
  per-tenant demand imbalance reaches **7.0x** at N=10, and once fairness is computed on
  demand-normalised quantities the index is **1.0000 exactly** across the whole stable regime.
  Under saturation the observed Jain sits inside a permutation null (p = 0.48--0.96).

### What is new

- **The mechanism, measured rather than interpreted.** This index said "MPS interleaves and
  the backend serialises". Dividing each miniBUDE iteration by the solo iteration time turns
  that into a number: native serves **5 of 400** iterations with no wait, Gusto **405 of 1200**,
  with multipliers spanning 1 to 16. Native sits at exactly N on essentially every iteration.
- **A paired equivalence test for serving.** Over 13 cells matched on (N, lambda, rep), the
  paired Jain difference is -0.0027 with a bootstrap CI95 of [-0.0079, +0.0009] -- inside a
  declared margin of +-0.05 -- while completion fraction differs by **+0.0764**, CI95
  [+0.0426, +0.1122], excluding zero. Equal fairness, more service.
- **A metric audit** (Table A) with the numerical impact of each defect.

### Metric defects it found, which affect figures outside this document

1. **`goodput` divides two different windows.** `bench.py:118,132` counts completions in
   `[t_meas, t_end + REQ_TIMEOUT]` (55 s in the multi-tenant configuration) and divides by
   `WINDOW` (30 s). Measured inflation **x1.76**. The ratio between arms survives -- both share
   the denominator -- but the absolute figure is not a steady-state rate. Say the window.
2. **Jain over per-tenant throughput is invalid when demand differs**, which it does here by
   up to 7.0x. Every published fairness index for llama should be recomputed on
   completion fraction, or dropped.
3. **The SLO Jain index under saturation reports starvation as equality.** Where fewer than
   half the tenants have non-zero attainment, it is suppressed rather than printed.
4. **`bench.py` appends to the per-request JSONL and replicates share a label**, so the three
   runs of each multi-tenant cell are concatenated with no separator. They are recoverable by
   cumulative `completed+fail` counts from `summary.csv` -- verified exact on ten labels -- but
   any analysis that read those files per-run without segmenting them was mixing replicates.

### The duplicate-filename trap, both directions

Trap 2 above notes that `mb_campaign.sh` globs `t*.log`, which also matches the `tenant_*.log`
files the remote arms write, so it double-counts them. The audit's first extractor made the
mirror mistake -- it globbed `tenant_*.log`, which **only** the remote arms write, and silently
produced zero rows for baremetal and baremetal+MPS, i.e. it deleted the control arm. Both come
from the same duplicated-filename convention. The shipped extractor globs `t*.log` and
deduplicates by tenant id.

### Still missing, stated rather than filled

- The llama campaign has **no per-request timestamp** (added 2026-08-02, after it ran), so
  completions inside the window cannot be separated per tenant from completions during the
  drain, and no-progress intervals are not reconstructible. A serving timeline needs a re-run.
- **CloverLeaf does not keep its figure of merit**: it writes to `clover.out`, which is not in
  the artifact. Only wall time per tenant survives.
- **XSBench TCP N=8 is unusable**: all 64 tenants lack a `Runtime:` line and some record
  `duration_s = 0.0`. Eight cohorts dropped and counted, not silently skipped.
- Why the sharing is uneven is **not** established. That needs backend queue-entry and
  dispatch timestamps per connection; see `FAIRNESS_RESULTS.md` §6.6.

## XSBench -- the obsolete cohort table is gone, and the MPS row with it

`XSBENCH_RESULTS.md` carried two cohort tables that disagreed. The 2026-08-01 correction block
(baremetal 12.86 / 25.02 / 50.23 / 98.39; TCP does not complete N=8) is the valid one; the
older table below it (baremetal N=8 = 114.92, TCP N=8 = 100.89) has been **deleted, not
reconciled**. Recomputing the makespan from the per-client `status_raw.json` on 2026-08-02
reproduces the correction block in every cell, so the question is settled.

**The MPS row went with it.** It claimed MPS makes native 13-15% faster from N=2, measured
against the obsolete baseline. Against the correct one the margin is -0.4% / +0.2% / 0.0%, and
no raw file anywhere backs the MPS numbers themselves: there is no `baremetal_mps` tree under
`~/xsbench_campaign/`. Its premise fell too -- with correct data baremetal wins at every N
*without* MPS. This is XSBench-specific: the llama and miniBUDE MPS results have raw data and
stand.

**Per-tenant fairness is now packaged.** The per-client stdout was still on the server, so it
was reconstructed without re-running anything: `XSBench_fairness_por_tenant.csv`, 306 rows. At
N=8 the best `ucx` tenant reaches 96% of its solo rate while the worst gets a sixth of it,
where native shares to the third decimal. At N=2 the ratio is exactly 1.99 in all four remote
arms -- serialisation, not contention. Wall-clock hides this entirely (1.15-1.41); the internal
`Runtime:` shows it (4.8-6.0x).

## `GAPS.md` -- what the artifact still does not contain

A living list of what is missing, what claim it blocks, whether it is reconstructible from the
server or needs re-measuring, and the minimum experiment that closes it. The two that need new
runs are the **per-N SLO capacity curve** (the intermediate multi-tenant sweep, harnesses
already written and unrun) and the **Native+MPS memory footprint per tenant**, which is the
column that decides whether the ~463 MiB/tenant saving belongs to remoting or merely to context
consolidation.

## One document per workload (2026-08-03)

llama content was spread across `RESULTS.md` sections 2-8b, this index and
`LLAMA-7B.README.md`. It is now consolidated in **`LLAMA-7B_RESULTS.md`**: the multi-tenant
campaign, the per-pod-constant sweep, the memory footprint with its MPS control, the
capacity-under-SLO curve and the per-tenant serving fairness. `RESULTS.md` keeps a summary of
the four headline numbers and points there; it is the cross-workload summary again, not the
llama document.

The same rule now holds throughout: `MINIBUDE_RESULTS.md`, `XSBENCH_RESULTS.md`,
`BABELSTREAM_RESULTS.md`, `CLOVERLEAF_RESULTS.md`, `CUDF_ETL_RESULTS.md` and
`LLAMA-7B_RESULTS.md` each carry their own workload end to end, including its fairness result.
`FAIRNESS_RESULTS.md` keeps only what is genuinely cross-cutting: the metric audit (table A),
the comparison across workloads (table B) and the synthesis. `GAPS.md` lists what is still
missing and the minimum experiment for each.
