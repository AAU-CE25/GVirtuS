#!/usr/bin/env python3
"""Plot the BabelStream-over-GVirtuS transport sweep.

Reads babelstream_sweep.csv and writes, into ./plots/:
  * per-kernel BANDWIDTH vs size (all configs)
  * per-kernel LATENCY (per-launch, us) vs size  -> the control-path RPC story
  * Triad GVirtuS-only zoom + bare-metal-normalized overhead
  * control-path RPC latency floor bar chart (smallest size = pure RPC round-trip)
Also writes babelstream_summary_gbps.csv, babelstream_latency_us.csv,
babelstream_rpc_latency_us.csv.

Run:  python plot_sweep.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "babelstream_sweep.csv")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
df["gbps"] = df["max_mbytes_per_sec"] / 1000.0
df["lat_us"] = df["average_runtime"] * 1e6       # avg per-launch time (RPC round-trip + xfer)
df["function"] = df["function"].str.strip()
df["mib"] = df["n_elements"] * 8 / (1024 ** 2)   # per-buffer size (double = 8 B)

CONFIGS = [
    ("baremetal",              "bare metal",            "#000000", "o"),
    ("gvirtus-tcp",            "GVirtuS UCX-TCP",       "#1f77b4", "s"),
    ("gvirtus-rdma",           "GVirtuS UCX-RDMA",      "#ff7f0e", "^"),
    ("gvirtus-rdma-gpudirect", "GVirtuS RDMA+GPUDirect","#2ca02c", "D"),
]
KERNELS = ["Copy", "Mul", "Add", "Triad", "Dot"]


def _lineplot(sub, ycol, configs, logy=False):
    for cfg, label, color, marker in configs:
        s = sub[sub["config"] == cfg].sort_values("n_elements")
        if s.empty:
            continue
        plt.plot(s["mib"], s[ycol], marker=marker, color=color,
                 label=label, linewidth=1.6, markersize=5)
    plt.xscale("log", base=2)
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)


# 1) Per-kernel BANDWIDTH vs size.
for kern in KERNELS:
    plt.figure(figsize=(7, 4.5))
    _lineplot(df[df["function"] == kern], "gbps", CONFIGS)
    plt.xlabel("array size per buffer (MiB, log scale)")
    plt.ylabel("bandwidth (GB/s)")
    plt.title(f"BabelStream {kern} - bandwidth vs size across transports")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"babelstream_{kern.lower()}.png"), dpi=130)
    plt.close()

# 2) Per-kernel LATENCY (per-launch, us) vs size - the control-path RPC story.
for kern in KERNELS:
    plt.figure(figsize=(7, 4.5))
    _lineplot(df[df["function"] == kern], "lat_us", CONFIGS, logy=True)
    plt.xlabel("array size per buffer (MiB, log scale)")
    plt.ylabel("per-launch latency (us, log scale)")
    plt.title(f"BabelStream {kern} - per-launch latency (RPC round-trip + transfer)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"babelstream_{kern.lower()}_latency.png"), dpi=130)
    plt.close()

# 3) Triad GVirtuS-only bandwidth zoom.
plt.figure(figsize=(7, 4.5))
_lineplot(df[df["function"] == "Triad"], "gbps", CONFIGS[1:])
plt.xlabel("array size per buffer (MiB, log scale)")
plt.ylabel("bandwidth (GB/s)")
plt.title("BabelStream Triad - GVirtuS transports (control-path amortization)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "babelstream_triad_gvirtus.png"), dpi=130)
plt.close()

# 4) Overhead relative to bare metal (Triad bandwidth ratio).
plt.figure(figsize=(7, 4.5))
base = (df[(df["function"] == "Triad") & (df["config"] == "baremetal")]
        .set_index("n_elements")["gbps"])
for cfg, label, color, marker in CONFIGS[1:]:
    s = df[(df["function"] == "Triad") & (df["config"] == cfg)].sort_values("n_elements")
    if s.empty:
        continue
    ratio = s["gbps"].values / base.reindex(s["n_elements"]).values
    plt.plot(s["mib"], ratio, marker=marker, color=color, label=label,
             linewidth=1.6, markersize=5)
plt.xscale("log", base=2)
plt.axhline(1.0, color="#000000", linewidth=0.8, linestyle="--", alpha=0.6)
plt.xlabel("array size per buffer (MiB, log scale)")
plt.ylabel("bandwidth fraction of bare metal")
plt.title("BabelStream Triad - GVirtuS bandwidth as fraction of bare metal")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "babelstream_triad_overhead.png"), dpi=130)
plt.close()

# 5) Control-path RPC latency FLOOR (smallest size ~ pure RPC round-trip) - bar chart.
smallest = int(df["n_elements"].min())
floor = df[df["n_elements"] == smallest]
plt.figure(figsize=(7.5, 4.5))
x = np.arange(len(KERNELS))
w = 0.2
for i, (cfg, label, color, _m) in enumerate(CONFIGS):
    vals = [floor[(floor["function"] == k) & (floor["config"] == cfg)]["lat_us"].mean()
            for k in KERNELS]
    plt.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
plt.yscale("log")
plt.xticks(x, KERNELS)
plt.ylabel("per-launch latency (us, log scale)")
plt.title(f"Control-path RPC latency floor (arrays {smallest} elems, transfer negligible)")
plt.legend(fontsize=8)
plt.grid(True, axis="y", which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "babelstream_rpc_latency_floor.png"), dpi=130)
plt.close()

# 6) Tidy summary tables -> CSV.
(df.pivot_table(index=["function", "n_elements"], columns="config", values="gbps")
   .reindex(KERNELS, level=0)
   .to_csv(os.path.join(HERE, "babelstream_summary_gbps.csv")))
(df.pivot_table(index=["function", "n_elements"], columns="config", values="lat_us")
   .reindex(KERNELS, level=0)
   .to_csv(os.path.join(HERE, "babelstream_latency_us.csv")))
(floor.pivot_table(index="function", columns="config", values="lat_us")
      .reindex(KERNELS)
      .to_csv(os.path.join(HERE, "babelstream_rpc_latency_us.csv")))

print("wrote plots to", OUT)
print("wrote summaries: babelstream_summary_gbps.csv, babelstream_latency_us.csv, "
      "babelstream_rpc_latency_us.csv")
