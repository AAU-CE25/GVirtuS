#!/usr/bin/env python3
"""
Aggregate the rigorous rep campaign (benchmarks/_summary/_rig/*) into
mean +/- 95% CI per (benchmark, config, metric).

- CSV benchmarks (minibude/matrix/transfer/babel): mean +/- 95% CI (t-dist)
  computed from per-rep raw values.
- llama: llama-bench already reports mean +/- sd over r reps; we parse that and
  convert sd -> 95% CI (t, dof=r-1, r=5 => t=2.776; gd transfer r varies).

Outputs:
  mode_comparison.csv     -- one row per (benchmark, metric): mean per mode
  mode_comparison_ci.csv  -- same + ci95 (half-width) per mode
  babel_agg_<mode>.csv    -- per-size Triad mean+ci (for the sweep plot)
  transfer_agg_<mode>.csv -- per-size/dir mean+ci (for the transfer plot)
"""
import os, csv, glob, math, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.join(HERE, "_rig")

# t critical (two-sided 95%) by dof
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 19: 2.093, 20: 2.086}


def mean_ci(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = st.mean(vals)
    if n == 1:
        return m, 0.0, 1
    sd = st.stdev(vals)
    t = T95.get(n - 1, 1.96)
    return m, t * sd / math.sqrt(n), n


def load_col(path, valcol, filt=None):
    """Return list of float values from a rep CSV (skips header)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if filt and not filt(row):
                continue
            try:
                out.append(float(row[valcol]))
            except (ValueError, KeyError, TypeError):
                pass
    return out


def parse_llama(path):
    """Return dict {pp8:(mean,sd), tg16:(mean,sd)} and rep count."""
    import re
    res, reps = {}, 5
    if not os.path.exists(path):
        return res, reps
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "reps=" in line:
                try:
                    reps = int(line.split("reps=")[1].split()[0])
                except Exception:
                    pass
            if "|" not in line:
                continue
            for key in ("pp8", "tg16"):
                if f" {key} " not in line and f"{key} |" not in line:
                    continue
                mobj = re.search(r"([0-9]+\.[0-9]+)\s*[\xb1\ufffd\u00b1±]+\s*([0-9]+\.[0-9]+)", line)
                if mobj:
                    res[key] = (float(mobj.group(1)), float(mobj.group(2)))
    return res, reps


MODES = ["native", "tcp", "rdma", "gd"]
MODE_OUT = {"native": "native", "tcp": "ucx_tcp", "rdma": "ucx_rdma", "gd": "ucx_gpudirect"}

# ---- scalar benchmarks ------------------------------------------------------
rows_mean, rows_ci = {}, {}


def set_val(bench, metric, mode, m, ci):
    rows_mean.setdefault((bench, metric), {})[MODE_OUT[mode]] = m
    rows_ci.setdefault((bench, metric), {})[MODE_OUT[mode]] = ci


# miniBUDE: gflops
for mode in MODES:
    v = load_col(os.path.join(RIG, f"minibude_{mode}.csv"), "gflops")
    m, ci, n = mean_ci(v)
    if m is not None:
        set_val("miniBUDE", "gflops", mode, m, ci)
        print(f"miniBUDE gflops {mode}: {m:.3f} +/- {ci:.3f} (n={n})")

# simple_matrix: host_ms
for mode in MODES:
    v = load_col(os.path.join(RIG, f"matrix_{mode}.csv"), "host_ms")
    m, ci, n = mean_ci(v)
    if m is not None:
        set_val("simple_matrix", "host_ms", mode, m, ci)
        print(f"matrix host_ms {mode}: {m:.2f} +/- {ci:.2f} (n={n})")

# llama: tg16, pp8 (from llama-bench mean+-sd -> CI)
for mode in MODES:
    res, reps = parse_llama(os.path.join(RIG, f"llama_{mode}.log"))
    t = T95.get(reps - 1, 1.96)
    for key in ("tg16", "pp8"):
        if key in res:
            m, sd = res[key]
            ci = t * sd / math.sqrt(reps)
            set_val("llama", key, mode, m, ci)
            print(f"llama {key} {mode}: {m:.2f} +/- {ci:.2f} (reps={reps})")

# ---- write scalar tables ----------------------------------------------------
HB = {("llama", "tg16"): 1, ("llama", "pp8"): 1, ("miniBUDE", "gflops"): 1,
      ("simple_matrix", "host_ms"): 0}
UNIT = {("llama", "tg16"): "tokens/s", ("llama", "pp8"): "tokens/s",
        ("miniBUDE", "gflops"): "GFLOP/s", ("simple_matrix", "host_ms"): "ms"}

with open(os.path.join(HERE, "mode_comparison.csv"), "w", newline="") as f, \
     open(os.path.join(HERE, "mode_comparison_ci.csv"), "w", newline="") as g:
    hdr = ["benchmark", "metric", "unit", "higher_better"] + list(MODE_OUT.values())
    w = csv.writer(f); wc = csv.writer(g)
    w.writerow(hdr)
    wc.writerow(["benchmark", "metric"] + [f"{m}" for m in MODE_OUT.values()]
                + [f"{m}_ci95" for m in MODE_OUT.values()])
    for key in [("llama", "tg16"), ("llama", "pp8"), ("miniBUDE", "gflops"),
                ("simple_matrix", "host_ms")]:
        b, mt = key
        means = rows_mean.get(key, {})
        cis = rows_ci.get(key, {})
        w.writerow([b, mt, UNIT[key], HB[key]]
                   + [f"{means.get(m, ''):.4g}" if means.get(m) is not None else ""
                      for m in MODE_OUT.values()])
        wc.writerow([b, mt]
                    + [f"{means.get(m, ''):.4g}" if means.get(m) is not None else ""
                       for m in MODE_OUT.values()]
                    + [f"{cis.get(m, ''):.4g}" if cis.get(m) is not None else ""
                       for m in MODE_OUT.values()])

# ---- BabelStream Triad sweep: per-size mean+ci ------------------------------
for mode in MODES:
    path = os.path.join(RIG, f"babel_{mode}.csv")
    if not os.path.exists(path):
        continue
    bysize = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["kernel"] != "Triad":
                continue
            n = int(row["n_elements"])
            bysize.setdefault(n, []).append(float(row["mbps"]) / 1000.0)
    out = os.path.join(HERE, f"babel_agg_{MODE_OUT[mode]}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_elements", "gbps_mean", "gbps_ci95", "n"])
        for n in sorted(bysize):
            m, ci, k = mean_ci(bysize[n])
            w.writerow([n, f"{m:.2f}", f"{ci:.2f}", k])
    print(f"babel {mode}: {len(bysize)} sizes -> {os.path.basename(out)}")

# ---- transfer sweep: per-size/dir mean+ci -----------------------------------
for mode in MODES:
    path = os.path.join(RIG, f"transfer_{mode}.csv")
    if not os.path.exists(path):
        continue
    by = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row["dir"], int(row["bytes"]))
            by.setdefault(k, []).append(float(row["gbps"]))
    out = os.path.join(HERE, f"transfer_agg_{MODE_OUT[mode]}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "bytes", "gbps_mean", "gbps_ci95", "n"])
        for (d, b) in sorted(by, key=lambda x: (x[0], x[1])):
            m, ci, k = mean_ci(by[(d, b)])
            w.writerow([d, b, f"{m:.4f}", f"{ci:.4f}", k])
    print(f"transfer {mode}: {len(by)} points -> {os.path.basename(out)}")

print("\nDone. Wrote mode_comparison.csv (+_ci), babel_agg_*, transfer_agg_*")
