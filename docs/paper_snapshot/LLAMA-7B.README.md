# `LLAMA-7B.csv` -- provenance

467 rows, 363 distinct labels. Despite the name it covers both models (`1b_*` and `7b_*` rows).

This file is **documentation**: it is referenced by `data/README.md` and `paper/data_index.md`,
and no analysis code reads it. Experiment e4 supports no claim in `artifact/claim_map.yaml`.

> **2026-08-01**: superseded for analysis by `LLAMA-7B_full.csv`, rebuilt from the master CSV
> (1409 rows, 991 labels) with reconstructed provenance and the three cells that closed §2, §5
> and §6. This file is kept because the documents above cite it.

## AVISO: 194 of 363 labels have no raw data in the artifact

You cannot get from the number back to the measurement that produced it. Arms with **no** raw
file at all:

| arm | labels | rows |
|---|---:|---:|
| `bmHI` | 40 | 80 |
| `c4noise` | 16 | 16 |
| `cb0` | 10 | 10 |
| `cb1` | 10 | 10 |
| `n1` | 2 | 2 |
| `truerdma` | 20 | 20 |
| `ucxHI` | 40 | 40 |

And arms with **partial** backing:

| arm | labels | without raw data |
|---|---:|---:|
| `bm` | 70 | 50 |
| `n2` | 4 | 2 |
| `n4` | 4 | 2 |
| `n8` | 4 | 2 |

The data exists elsewhere (the original campaign) but does not travel here. Until it is
included, or the CSV is regenerated from the 178 JSONL files that do travel, these rows should
be read as a bibliographic reference, not as something verifiable inside the artifact.

## The label is not a unique key

87 labels appear more than once (191 rows in total for those 87 labels). The repeats usually
carry DIFFERENT values, and that is **not corruption**: they are independent runs of the same
configuration that happen to share a name.

The spread is real and documented. `RESULTS.md` measures it and draws the consequence:

> *baremetal C2 spans 125.2--244.6 (**1.95x**) and baremetal C4 spans 170.7--295.8 (**1.73x**)
> with no GVirtuS anywhere in the path ... At these concurrencies report a distribution or
> report nothing.*

That is why **C2 and C4 are withdrawn as retention points**. The remaining defect is one of
labelling: selecting or averaging by `label` silently mixes distinct runs.

> **2026-08-01**: re-measuring C4 confirms this directly -- five repetitions give
> 204.8-204.8-227.6-238.9-238.9, a **bimodal** population that no point estimate describes.
> C8, by contrast, reproduces cleanly (591.6, n=5, against the published 589.3). See §G of
> `RESULTS_2026-08-01.md`.

## Temporal scope

Built from files dated 2026-07-26 and 2026-07-27 only. **No row comes from 2026-07-24**, the
date `RESULTS.md` invalidates -- verified, not assumed. See `../llama/README.md`.

## 2026-08-02 -- the Jain column in `summary.csv` should not be read as fairness

`summary.csv` carries a `jain` column computed over **per-tenant token throughput**
(`bench.py:125-126`). With open-loop Poisson arrivals the tenants do not receive equal demand:
measured imbalance reaches **7.0x** at N=10 and 4.0x at N=8. An index over absolute service
under unequal demand measures the arrival draw, not the scheduler.

Recomputed on demand-normalised quantities, fairness is **1.0000 exactly** across the entire
stable regime, and under saturation the observed value sits inside a permutation null
(p = 0.48-0.96). `RESULTS.md` section 8 already said this decline was an arrival artefact and
should not be cited as a GVirtuS limitation; that reading is now demonstrated rather than
asserted. See `FAIRNESS_RESULTS.md`.

Two further cautions for anyone reading these files per-request:

- `bench.py` appends to `<label>.jsonl` and replicates share a label, so the three runs of a
  multi-tenant cell are **concatenated with no separator**. Segment them by cumulative
  `completed+fail` from `summary.csv` (verified exact on ten labels) before any per-run
  statistic.
- This campaign predates the per-request timestamp fields, so completions inside the window
  cannot be separated per tenant from completions during the drain.
