---
title: "Multi-tenant fairness -- audit, method, and cross-workload synthesis"
date: "2026-08-02"
geometry: margin=2.3cm
fontsize: 10pt
---

# 0. What this is, and where the numbers live

A full audit of the project's fairness evaluation. It does **not** start from the published
Jain indices: it rebuilds them from the raw per-tenant records of every workload, with
controls, and asks whether the inequality is introduced by Gusto or explained by demand, sample
size, Poisson arrivals, start skew or natural variability.

Rule applied without exception: **nothing invented, nothing filled in**. Where a metric is not
computable, it says what is missing.

**Per-workload results live in each workload's own document.** This one carries the method, the
metric audit, and the comparison across workloads:

| workload | its document | what it says about fairness |
|---|---|---|
| llama 7B | `LLAMA-7B_RESULTS.md` §4 | serving fairness, demand-normalised, permutation null |
| miniBUDE | `MINIBUDE_RESULTS.md` | the quantum accounting -- the strongest case |
| XSBench | `XSBENCH_RESULTS.md` §3 | per-tenant, reconstructed from server-side stdout |
| BabelStream | `BABELSTREAM_RESULTS.md` | confirmed independently, no separation |
| CloverLeaf | `CLOVERLEAF_RESULTS.md` | no separation; figure of merit missing from artifact |

Artifacts, all under `results/asplos_campaign/fairness/` in the repository:

| file | rows | what it is |
|---|---:|---|
| `tenants_canonico.csv` | **3838** | one row per tenant per run, four fixed-work workloads |
| `fairness_trabajo_fijo_por_corrida.csv` | **523** | Jain per cohort, never pooled |

> **Row counts corrected 2026-08-03.** These two tables read 3688 and 493 until the CSVs were
> counted directly. Both files were regenerated after those numbers were written, to add the
> `baremetal_mps` arms and the CloverLeaf TCP variants, so the extra 150 and 30 rows are new
> cohorts and not duplicates (verified: every row is unique). **No published statistic changes**
> -- the N=8 `sync` table below was recomputed from the 523-row file and reproduces to three
> decimals, miniBUDE GPUDirect included (15 runs, Jain 0.696, slowest/fastest 4.871).
| `fairness_trabajo_fijo_resumen.csv` | -- | median across cohorts |
| `llama_fairness_por_corrida.csv` | 39 | serving runs, with the permutation null |
| `llama_fairness_por_tenant.csv` | 169 | tenant-run rows |
| `tabla_D_minibude_por_tenant.csv` | 352 | per-iteration timeline |
| `XSBench_fairness_por_tenant.csv` | 306 | per-tenant runtime and Mlookups/s |
| `scripts/` | -- | the seven programs that generate all of the above |

# 1. Table A -- metric audit

| # | metric | current formula | problem | corrected formula | measured impact |
|---|---|---|---|---|---|
| A1 | `goodput` (llama) | `tokens(comp)/WINDOW` with `comp` filtered to `tc` in `[t_meas, t_end+TIMEOUT]` (`bench.py:118,132`) | counts completions over a **55 s** window and divides by **30 s** | clip to `tc <= t_end` | **1.76x** on `slo_ucx_n8_l2.0`: 277.3 -> **157.9 t/s**. The **ratio between arms survives** (they share the denominator); the absolute figure is not a steady-state rate |
| A2 | `jain` (llama) | Jain over **per-tenant tokens** (`bench.py:125-126`) | the quantity is absolute service, not normalised by demand; with Poisson arrivals one tenant receives up to **7.0x** more requests than another | Jain over `completed_i/offered_i` | the 0.719--0.804 values at N=6/8/10 become **1.0000** exactly. They were a demand artefact |
| A3 | `jain` over SLO under saturation | Jain over `slo_5s_fraction` | with almost every tenant at zero, the index measures starvation and reports it as equality | suppress if `nonzero < N/2` or mean `< 5%` | 12 cells suppressed; in them 1/4--5/8 tenants have non-zero attainment and the mean is 0.005--0.017 |
| A4 | Jain over runtime (fixed work) | the *temptation* to apply Jain to `duration_s` | a longer runtime is **worse** service: the index rewards the opposite of what it measures | Jain over `progress = t_solo/t_concurrent` | never published; documented so it is not done |
| A5 | grouping of repetitions | key `(system, N, mode, seed)` | two sibling result trees with the same seed merge into **one** cohort of 2N tenants from different campaigns | add `cohort_path` to the key | it invented a **3162 s** start spread; the real cohorts are coordinated to **0.0 s** |
| A6 | duplicate trees | `experiments/babelstream/results_stale/` | an obsolete duplicate, not excluded, merging with `results/` | label, do not delete | root cause of A5 |
| A7 | miniBUDE extraction | glob `tenant_*.log` | the remote arms write `t<i>.log` **and** `tenant_<i>.log`; the baremetal arms write only the first | glob `t*.log` and deduplicate | **it deleted the entire control arm** (0 native rows) |
| A8 | XSBench `exit_code` | used as a success signal | the checksum is pinned to the default lookup count; the code means nothing | use the `Lookups/s` line | already documented in earlier campaigns; confirmed here |
| A9 | `stagger` mode | mixed with `sync` in the summaries | start spread of **14.0 s by design** | always separate by mode | Jain falls to 0.957--0.99 and slowest/fastest rises to 1.24--1.76 **from the skew alone** |
| A10 | `slo_min_tenant_*` (llama) | minimum SLO attainment across tenants | a tenant that receives **no requests** counts as 0% attainment | exclude zero-demand tenants | at lambda=0.25 with N=8 only 5 of 8 tenants have demand; the field read "worst tenant 0%" while the aggregate read 100% |

**Data defects, not formula defects.** XSBench TCP N=8: all 64 tenants lack a `Runtime:` line
and some record `duration_s = 0.0`; 8 cohorts dropped and **counted**, not silenced. CloverLeaf
does not keep its figure of merit (it writes to `clover.out`, absent from the artifact): only
wall time survives. miniBUDE's `epoch_s` has **1-second** resolution against runs of 2.4--12 s.

# 2. Table B -- progress fairness in fixed work

`sync` mode, identical work verified from the input parameters, Jain over normalised progress
computed **per cohort** and summarised with the median across cohorts. Full per-workload
discussion in each workload's document.

| workload | system | N | runs | Jain | worst slowdown | median slowdown | **slowest/fastest** | class |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| miniBUDE | native | 8 | 5 | 0.9998 | 7.95 | 7.83 | **1.032** | -- |
| miniBUDE | native+MPS | 8 | 5 | 1.0000 | 7.96 | 7.95 | **1.016** | -- |
| miniBUDE | **Gusto GPUDirect** | 8 | 15 | **0.696** | 4.87 | 3.75 | **4.871** | **A** |
| miniBUDE | UCX host RMA | 8 | 14 | 0.746 | 4.31 | 3.22 | **4.311** | **A** |
| miniBUDE | TCP | 8 | 5 | 0.669 | 4.87 | 3.93 | **4.619** | **A** |
| XSBench | native | 8 | 4 | 1.0000 | -- | -- | **1.001** | -- |
| XSBench | Gusto AM | 8 | 5 | **0.661** | -- | -- | **5.982** | **A** |
| XSBench | Gusto GPUDirect | 8 | 5 | 0.716 | -- | -- | **4.848** | **A** |
| XSBench | UCX host RMA | 8 | 5 | 0.693 | -- | -- | **4.882** | **A** |
| BabelStream | native | 8 | 5 | 1.0000 | 7.81 | 7.80 | 1.007 | -- |
| BabelStream | Gusto AM | 8 | 5 | 0.9999 | 6.60 | 6.54 | 1.034 | **D** |
| CloverLeaf | native | 8 | 5 | 1.0000 | 8.60 | 8.59 | 1.005 | -- |
| CloverLeaf | Gusto AM | 8 | 5 | 1.0000 | 8.40 | 8.39 | 1.019 | **D** |

Classification: **A** strong evidence of Gusto-introduced unfairness - **B** compatible signal,
modest magnitude - **C** explained by demand or noise - **D** no significant difference -
**E** invalid experiment.

**XSBench TCP N=8 -> E** (unusable data, see §1).

> **XSBench moved from B to A.** An earlier pass classified it B on wall-clock
> slowest/fastest of 1.15--1.41. Wall clock equalises because the cohorts start and end
> together; on the internal `Runtime:` the ratio is 4.8--6.0x. Which metric one reads decides
> the classification, and the internal one is the right one for progress fairness.

The effect appears **identically on TCP**, so it is **not the RMA data path**. And
native+MPS -- which consolidates contexts exactly as Gusto does -- **does not reproduce it**
(1.016), so it is not context consolidation either: it is how the backend **orders** work from
concurrent connections.

# 3. Serving (llama) -- the summary; detail in `LLAMA-7B_RESULTS.md` §4

**Classification: C -- explained by demand and experimental noise.** Demand imbalance from the
Poisson draw reaches **7.0x** at N=10; normalised by demand, Jain is **1.0000 exactly** across
the whole stable regime, and under saturation the observed value sits inside a permutation null
(p = 0.48--0.96 in every cell, both systems). Paired over 13 matched cells, the two systems are
**statistically equivalent** in fairness while Gusto completes significantly more work.

**The fairness problem is in fixed work, not in serving.** In llama the problem is not fairness
but **capacity under an SLO**, which is a different story and lives in `LLAMA-7B_RESULTS.md` §3.

# 4. The five statements

| statement | verdict | evidence |
|---|---|---|
| 1. "Gusto preserves aggregate efficiency but does not guarantee fair per-tenant progress" | **supported** | miniBUDE N=8: median slowdown 3.75 against 7.83, Jain 0.696 against 0.9998 |
| 2. "Under equal fixed work, some tenants run near their single-client rate while others experience multi-fold slowdown" | **supported** | tenant 2 at x1.001 with multiplier 1 on all ten iterations beside tenant 3 at x4.874, same cohort, in 3 repetitions; XSBench best tenant at 96% of solo while worst gets a sixth |
| 3. "The observed llama Jain values are dominated by stochastic arrivals and do not establish scheduler unfairness" | **supported** | demand imbalance up to 7.0x; normalised Jain 1.0000 throughout the stable regime; p = 0.48--0.96 against the permutation null under saturation |
| 4. "Gusto's fairness limitation is workload-dependent rather than universal" | **supported** | 4.87x in miniBUDE and 5.98x in XSBench against 1.03x in BabelStream and CloverLeaf, same N, same system |
| 5. "A fairness-aware scheduler is orthogonal to the semantic-contract contribution" | **unsupported** | a design claim; no data in this audit supports or refutes it |

# 5. Conclusion

## 5.1 The three strongest fairness results

1. **In fixed work Gusto shares the service quantum markedly unevenly and native does not** --
   4.87x against 1.03x slowest/fastest at N=8 in miniBUDE, with 15 runs and both native and
   native+MPS controls, and independently 5.98x against 1.001x in XSBench. The mechanism is
   visible per iteration, not inferred.
2. **In serving, Gusto and native are statistically equivalent in fairness** (paired Jain
   difference CI95 inside +/-0.05) **and Gusto serves significantly more** (+0.0764, CI95
   excludes zero).
3. **The published Jain indices over per-tenant throughput were a demand artefact.**
   Normalised, they are exactly 1.0000 even with an arrival imbalance of 7.0x.

## 5.2 The three biggest methodological problems

1. **The `goodput` denominator** counts 55 s of completions and divides by 30 s (x1.76).
2. **Jain over absolute service** in the presence of unequal demand -- this invalidates every
   published fairness index for llama.
3. **Repetitions are concatenated in the JSONL** with no separator, so any per-run analysis was
   impossible without reconstructing them. And in the fixed-work analysis, grouping by seed
   instead of by cohort path **merges different campaigns**.

## 5.3 What to drop and what to keep

**Drop from the paper:**

- Any Jain index computed over per-tenant throughput or tokens (llama). Replace with
  demand-normalised fractions.
- The SLO Jain index at the saturated points: there it measures starvation, not fairness.
- The absolute goodput figure under saturation quoted as a rate (277.3 / 302.9 t/s), or state
  the real 55 s window.
- The multi-tenant **x1.42**: it is the maximum of three runs. The mean of the three is
  **x1.37**.
- The XSBench MPS row (see `XSBENCH_RESULTS.md` §2) and the obsolete XSBench cohort table.

**Keep:**

- The ratio between arms under saturation: it survives the change of denominator.
- The **x1.37** against default native and the **97--99% parity** against MPS-configured native.
- The fixed-work figures: complete cohorts, verified work, start coordination at 0.0 s in the
  control.
- The memory result, which **survives the MPS control** -- see `LLAMA-7B_RESULTS.md` §2.

## 5.4 The strongest honest claim for ASPLOS

> *API remoting consolidates multi-tenant work into a single CUDA context, which improves both
> aggregate throughput and the fraction of offered work served -- in llama, +0.076 of completion
> fraction against native with a CI95 excluding zero -- **without degrading service fairness**,
> which is statistically equivalent to native's. That same mechanism, however, **does not
> guarantee equitable progress under fixed work**: in miniBUDE at eight tenants, 34% of
> iterations are served with no wait at all while others queue behind up to ten quanta, so one
> tenant finishes at its single-client rate and another five times slower, where native and
> native+MPS share to within 1.03x. The limitation is one of ordering, not of the data path: it
> appears identically over TCP.*

## 5.5 The minimum experiment still missing

**One, and it is cheap.** Instrument the backend to record, per connection, the queue-entry and
dispatch instant of every RPC, and repeat miniBUDE at N=8. That would separate the three
hypotheses the current data cannot distinguish:

- connection arrival order (is the favoured tenant always the one that connects first?);
- monopolisation of a backend thread or stream;
- head-of-line blocking in the dispatcher.

Without that trace I can demonstrate **that** the sharing is uneven and **by how much**, but
not **why**.

**Second gap, declared:** the llama campaign has **no per-request timestamp** (added
2026-08-02, afterwards). So per tenant these are not reconstructible: completions inside the
window against completions during the drain, first completion, and longest no-progress
interval. A serving timeline requires a re-run with the patched `bench.py`.
