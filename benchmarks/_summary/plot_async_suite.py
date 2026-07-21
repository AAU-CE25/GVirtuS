#!/usr/bin/env python3
"""Plots for the async-dispatcher benchmark suite (doc 12).
Generates:
  plots/babelstream_triad_async.png   — Triad GB/s vs size, async off vs on
  plots/async_speedup_by_benchmark.png — async speedup (x) per workload/regime
Run: python plot_async_suite.py   (from data/_async_suite/)
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# ---- BabelStream Triad: GB/s vs size, async off vs on ----
BABEL_COLS = ["config", "function", "num_times", "n_elements", "sizeof",
              "max_mbytes_per_sec", "min_runtime", "max_runtime", "avg_runtime"]
series = {"async_0": {}, "async_1": {}}
with open(os.path.join(HERE, "babel_async_raw.csv")) as f:
    for row in csv.reader(f):
        if len(row) < 6 or row[1] != "Triad":
            continue
        cfg = row[0]
        n = int(row[3])
        gbps = float(row[5]) / 1000.0  # MB/s -> GB/s
        series.setdefault(cfg, {})[n] = gbps

plt.figure(figsize=(7, 4.5))
labels = {"async_0": "async OFF (sync)", "async_1": "async ON"}
for cfg, marker in (("async_0", "o-"), ("async_1", "s-")):
    d = series.get(cfg, {})
    if not d:
        continue
    xs = sorted(d)
    plt.plot(xs, [d[x] for x in xs], marker, label=labels[cfg])
plt.xscale("log", base=2)
plt.xlabel("array size (elements)")
plt.ylabel("Triad bandwidth (GB/s)")
plt.title("BabelStream Triad — async dispatcher (RDMA+GPUDirect)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "babelstream_triad_async.png"), dpi=130)
plt.close()

# ---- Cross-benchmark async speedup bar chart (from summary) ----
bars = []
with open(os.path.join(HERE, "async_suite_summary.csv")) as f:
    for row in csv.DictReader(f):
        if row["benchmark"].startswith("#") or not row.get("async_ratio"):
            continue
        try:
            ratio = float(row["async_ratio"])
        except ValueError:
            continue
        # lower-is-better for simple_matrix ms -> invert to a speedup
        if "ms" in row["metric"]:
            ratio = (float(row["async_off"]) / float(row["async_on"]))
        bars.append((f'{row["benchmark"]}\n{row["metric"]}', ratio, row["regime"]))

colors = {"launch/RPC-bound": "#1f77b4", "launch-bound": "#1f77b4",
          "launch-overhead-bound": "#1f77b4", "transitional": "#ff7f0e",
          "bandwidth-bound": "#7f7f7f", "compute-bound": "#7f7f7f",
          "transfer-bound": "#7f7f7f"}
plt.figure(figsize=(11, 5))
xs = range(len(bars))
plt.bar(xs, [b[1] for b in bars], color=[colors.get(b[2], "#7f7f7f") for b in bars])
plt.axhline(1.0, color="k", lw=0.8, ls="--")
plt.xticks(list(xs), [b[0] for b in bars], rotation=45, ha="right", fontsize=7)
plt.ylabel("async speedup (×)")
plt.title("Async dispatcher speedup by workload (blue=launch-bound, grey=compute/bw/transfer-bound)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "async_speedup_by_benchmark.png"), dpi=130)
plt.close()
print("wrote", os.listdir(OUT))
