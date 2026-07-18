#!/usr/bin/env python3
"""Prototype (local Push/Pop call-config) speedup vs baseline, by workload regime.
Data: prototype_pushpop_summary.csv
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "prototype_pushpop_summary.csv"))))

# one representative bar per (workload,regime): pick the primary metric row
picks = [
    ("miniBUDE", "compute-bound", "gflops", "rdma", "miniBUDE\n(compute-bound)"),
    ("babelstream", "bandwidth-bound", "triad_mbytes_per_s", "rdma", "BabelStream 33M\n(bandwidth-bound)"),
    ("llama-tinyllama-1.1b", "launch-bound", "tg16_tokens_per_s", "rdma", "llama tg16\n(launch-bound)"),
    ("babelstream", "launch-overhead-bound", "triad_mbytes_per_s", "rdma", "BabelStream 262K\n(launch-overhead-bound)"),
]
labels, speedups = [], []
for wl, rg, mt, tr, lab in picks:
    for r in rows:
        if r["workload"] == wl and r["regime"] == rg and r["metric"] == mt and r["transport"] == tr:
            labels.append(lab)
            speedups.append(float(r["speedup"]))
            break

colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(labels)), speedups, color=colors)
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("speedup vs baseline (x)")
ax.set_title("Prototype: local __cudaPush/PopCallConfiguration\nspeedup tracks how launch-bound the workload is")
for b, v in zip(bars, speedups):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0.9, max(speedups) * 1.15)
ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
out = os.path.join(HERE, "plots", "prototype_pushpop_speedup.png")
fig.savefig(out, dpi=130)
print("wrote", out)
