#!/usr/bin/env python3
"""Llama token-gen throughput progression under the cheap frontend optimizations.
baseline -> +rec#1 (local Push/Pop) -> +rec#2 (cache cudaGetDevice/cudaGetLastError).
Reads llama_optimizations_progression.csv, writes plots/llama_optimization_progression.png.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {r["config"]: r for r in csv.DictReader(open(os.path.join(HERE, "llama_optimizations_progression.csv")))}

stages = ["baseline_tg16", "rec1_tg16", "rec12_tg16"]
stage_lbl = ["baseline", "+rec#1\n(local Push/Pop)", "+rec#2\n(cache GetDevice/\nGetLastError)"]
configs = [("gvirtus-rdma", "RDMA", "#1f77b4"), ("gvirtus-rdma-gpudirect", "RDMA+GPUDirect", "#2ca02c"), ("gvirtus-tcp", "TCP", "#d62728")]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(stages))
for cfg, lbl, col in configs:
    vals = [float(rows[cfg][s]) if rows[cfg][s] else np.nan for s in stages]
    ax.plot(x, vals, "o-", color=col, lw=2, ms=8, label=lbl)
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, fontweight="bold", color=col)
# total speedup annotations
rdma = rows["gvirtus-rdma"]
ax.text(2, float(rdma["rec12_tg16"]) * 0.62,
        f"RDMA total: {float(rdma['rec12_tg16'])/float(rdma['baseline_tg16']):.1f}x",
        color="#1f77b4", fontsize=10, fontweight="bold", ha="center")
ax.set_xticks(x); ax.set_xticklabels(stage_lbl)
ax.set_ylabel("token generation tg16 (tokens/s)")
ax.set_title("Llama TinyLlama-1.1B over GVirtuS: cheap frontend optimizations\n"
             "(per-launch RPCs 6.2 -> 4.2 -> ~1.1; correctness preserved)")
ax.grid(axis="y", ls=":", alpha=0.5); ax.legend(fontsize=10)
ax.set_ylim(0, float(rdma["rec12_tg16"]) * 1.15)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
out = os.path.join(HERE, "plots", "llama_optimization_progression.png")
fig.savefig(out, dpi=130)
print("wrote", out)
