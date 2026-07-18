#!/usr/bin/env python3
"""Plot the miniBUDE (compute-bound proxy) results across transports.

Reads minibude.csv and writes into ./plots/:
  * gflops per config (bar) with % of bare metal
  * one-time setup transfer cost (context_ms) per config (bar, log)
Run:  python plot_minibude.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "minibude.csv"))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

ORDER = ["baremetal", "gvirtus-tcp", "gvirtus-rdma", "gvirtus-rdma-gpudirect"]
LABELS = {"baremetal": "bare metal", "gvirtus-tcp": "UCX-TCP",
          "gvirtus-rdma": "UCX-RDMA", "gvirtus-rdma-gpudirect": "RDMA+GPUDirect"}
COLORS = {"baremetal": "#000000", "gvirtus-tcp": "#1f77b4",
          "gvirtus-rdma": "#ff7f0e", "gvirtus-rdma-gpudirect": "#2ca02c"}

d = df[df["deck"] == "data/bm1"].set_index("config")
base = d.loc["baremetal", "gflops"]

# 1) GFLOP/s per config.
plt.figure(figsize=(7, 4.5))
xs = [c for c in ORDER if c in d.index]
vals = [d.loc[c, "gflops"] for c in xs]
bars = plt.bar([LABELS[c] for c in xs], vals, color=[COLORS[c] for c in xs])
for c, v, b in zip(xs, vals, bars):
    plt.text(b.get_x() + b.get_width() / 2, v + 1,
             f"{v:.1f}\n({100*v/base:.1f}%)", ha="center", va="bottom", fontsize=8)
plt.ylabel("GFLOP/s")
plt.ylim(0, base * 1.15)
plt.title("miniBUDE (compute-bound) - throughput across transports\n"
          "GVirtuS overhead < 0.2% (compute dominates, transfer amortized)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "minibude_gflops.png"), dpi=130)
plt.close()

# 2) One-time setup transfer cost (context_ms) - where the transport DOES show.
plt.figure(figsize=(7, 4.5))
vals = [d.loc[c, "context_ms"] for c in xs]
bars = plt.bar([LABELS[c] for c in xs], vals, color=[COLORS[c] for c in xs])
for v, b in zip(vals, bars):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} ms",
             ha="center", va="bottom", fontsize=8)
plt.ylabel("one-time setup transfer (ms, log)")
plt.yscale("log")
plt.title("miniBUDE setup transfer cost (context_ms)\n"
          "transport shows here (TCP ~10ms vs RDMA ~2ms) but is amortized over ~2364ms compute")
plt.grid(True, axis="y", which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "minibude_context_ms.png"), dpi=130)
plt.close()

print("wrote plots to", OUT)
