# BabelStream over GVirtuS -- contention scaling, five arms (2026-07-28)

Memory-bandwidth benchmark, `-s 67108864 -n 1000` (double; 3 x 512 MiB arrays; five kernels
x 1000 iterations). Five arms, N = 1/2/4/8 concurrent tenants against one L40S, 5 seeds per
point, plus a staggered-launch control at N=8.

Harness: `drive.sh block babelstream <arm> 5` -- same code path for every arm. Backend
re-armed and its listener verified before each point.

**Every number comes from repetitions in which all N clients produced a valid result.**
25 points, 25 complete cohorts, no retries.

---

## Wall-clock per tenant (completion p50, s) and overhead vs native

| arm | N=1 | N=2 | N=4 | N=8 sync | N=8 stagger |
|---|---:|---:|---:|---:|---:|
| baremetal (native) | 10.78 | 21.97 | 42.66 | 84.04 | 71.02 |
| baremetal + MPS | 10.58 (0.98x) | 20.14 (0.92x) | 39.01 (0.91x) | 77.12 (0.92x) | 64.58 (0.91x) |
| **GVirtuS rdma** | 12.21 (**1.13x**) | 21.97 (**1.00x**) | 41.65 (**0.98x**) | 80.77 (**0.96x**) | 68.34 (**0.96x**) |
| **GVirtuS GPUDirect** | 12.21 (1.13x) | 21.76 (0.99x) | 40.84 (0.96x) | 79.75 (**0.95x**) | 67.99 (0.96x) |
| GVirtuS tcp | 15.05 (1.40x) | 29.08 (1.32x) | 51.56 (1.21x) | 93.87 (1.12x) | 75.64 (1.06x) |

**From N=2 upward the RDMA arms are faster than default native**, reaching 0.95--0.96x at
N=8. That is not a transport speed-up: the GVirtuS backend consolidates every tenant into a
single CUDA context, which is what MPS does. Native with MPS reaches 0.92x, so remoting
buys most of the MPS effect without configuring MPS -- and stays 4--5% behind a properly
MPS-configured native.

**State both baselines.** Against default native the RDMA arms look ahead; against
MPS-configured native they are slightly behind. Quoting only the first is a straw man;
quoting only the second discards the practical result.

TCP's relative position *improves* with N (1.40x -> 1.12x), the opposite of CloverLeaf,
where it degrades. BabelStream moves bulk blocks with few calls, so per-RPC latency is
amortised; CloverLeaf issues 411k calls per run and is dominated by it.

## Aggregate throughput (GB/s, all five kernels)

| arm | N=1 | N=2 | N=4 | N=8 sync | N=8 stagger |
|---|---:|---:|---:|---:|---:|
| baremetal | 677.6 | 629.4 | 631.1 | 631.8 | 777.5 |
| baremetal + MPS | 679.5 | 687.1 | 687.7 | 686.5 | 854.0 |
| GVirtuS rdma | 673.4 | 687.2 | 686.9 | 687.1 | 823.7 |
| GVirtuS GPUDirect | 673.4 | 690.6 | 691.0 | 687.5 | 829.5 |
| GVirtuS tcp | 597.8 | 596.2 | 649.5 | 681.2 | 785.5 |

**Do not use this table as the primary transport comparison.** At N=8 TCP reads 681.2 GB/s
against RDMA's 687.1 -- a 1% gap -- while its wall-clock is 21% worse. BabelStream's GB/s is
sustained *kernel* bandwidth, i.e. GPU-side work; the remoting cost sits between the
kernels, not inside them, so this metric is nearly blind to the transport. Wall-clock is the
metric that carries the comparison; throughput is secondary.

## Fairness and tail (N=8 sync)

| arm | Jain | min/max | p50 | p95 | p99 | tail spread |
|---|---:|---:|---:|---:|---:|---:|
| baremetal | 1.000 | 0.998 | 84.04 | 84.25 | 84.25 | 0.3% |
| baremetal + MPS | 1.000 | 0.999 | 77.12 | 77.33 | 77.33 | 0.3% |
| GVirtuS rdma | 1.000 | 0.992 | 80.77 | 81.40 | 81.59 | 1.0% |
| GVirtuS GPUDirect | 1.000 | 0.976 | 79.75 | 80.58 | 80.77 | 1.3% |
| GVirtuS tcp | 0.999 | 0.913 | 93.87 | 97.68 | 99.88 | 6.4% |

The RDMA arms hold a native-grade tail -- p50 to p99 within 1.3%. TCP's is five times wider
and its min/max drops to 0.913.

As in CloverLeaf, **Jain does not discriminate** (0.999--1.000 everywhere, including the arm
with an 8.7% min/max gap). Report `min/max` and percentiles alongside it.

---

## Staggered-launch control

| arm | sync | stagger | gain |
|---|---:|---:|---:|
| baremetal | 84.04 | 71.02 | 15.5% |
| baremetal + MPS | 77.12 | 64.58 | 16.3% |
| GVirtuS rdma | 80.77 | 68.34 | 15.4% |
| GVirtuS GPUDirect | 79.75 | 67.99 | 14.7% |
| GVirtuS tcp | 93.87 | 75.64 | 19.4% |

A companion offset sweep (native and GPUDirect, N=8, offsets 0.009 -> 2.0 s) shows the gain
is monotonic in the offset and needs a large offset to appear:

| offset | 0 (sync) | 0.009 s | 0.075 s | 0.5 s | 2.0 s |
|---|---:|---:|---:|---:|---:|
| baremetal, s/tenant | 84.04 | 83.97 | 82.79 | 79.92 | 71.02 |
| min/max | 1.00 | 1.00 | 0.96 | 0.93 | 0.73 |

Breaking exact synchrony is not enough -- offsets of 9--75 ms change almost nothing. So the
contention measured under `sync` is real and not an artefact of releasing the tenants
together. The `min/max` drop from 1.00 to 0.73 is the counterpart: staggered tenants finish
at different times by construction.

---

## Validity

- 5 seeds per point, 25 points, **25 complete cohorts, no retries**.
- Both latency gates bracketing the TCP block passed (36.4 µs before, 41.6 µs after,
  against a 58.5 µs threshold), so the block is internally comparable.
- `baremetal` and `baremetal_rootns` agree within 0.5%, so containerisation is not a
  confound.
- No point raised the multimodality flag (aggregate spread <=1.18x everywhere).
- BabelStream reports min/max/avg per kernel but **no per-iteration distribution**, so there
  are no p95/p99 *of kernel time* for this workload. The percentiles above are of per-tenant
  completion time across clients and repetitions. Nothing is derived from a metric the
  application does not report.

## Raw data

`~/experiments/babelstream/results/<arm>/N<n>/<mode>/seed*/` -- per-client `stdout.log`,
`status_raw.json`, and `seed_raw.json` per point. Parsed with `parse_results.py`,
aggregated with `summarize.py`; the aggregate is `babelstream_summary.csv`.

## Confirmed independently (2026-08-02)

The fairness audit rebuilt this from the raw `status_raw.json` per client through a separate
pipeline, computing Jain over **normalised progress** (solo runtime / concurrent runtime) per
cohort rather than over per-tenant wall time, and it reproduces the conclusion above:
slowest/fastest 1.034 for GVirtuS AM and 1.015 for host RMA at N=8 sync, against 1.007
native -- **no separation**. Classification **D: no significant difference**, in contrast with
miniBUDE (4.87) and XSBench (5.98) under the identical method.

Two cautions this confirms and sharpens:

- **The `stagger` cells cannot carry a fairness claim.** Their start spread is 14.0 s by
  design, and that alone drives Jain to 0.99 and slowest/fastest to 1.34--1.37 in *every* arm
  including native. The `stagger0p009` ... `stagger0p5` sweep in this tree shows the clean
  dose-response: spread 0.085 s -> 3.5 s moves slowest/fastest 1.014 -> 1.079.
- **`experiments/babelstream/results_stale/` is a duplicate of `results/`.** Merging the two
  fabricates a 3162 s start spread that does not exist; the real cohorts are coordinated to
  0.0 s. Exclude it.

Data: `tenants_canonico.csv`, `fairness_trabajo_fijo_resumen.csv`. Method and controls:
`FAIRNESS_RESULTS.md`.
