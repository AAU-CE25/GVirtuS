#!/usr/bin/env python3
"""Plot the H2D/D2H transfer-bandwidth sweep (transport data path).

Reads transfer_bw.csv and writes H2D + D2H bandwidth-vs-size plots into ./plots/.
Run:  python plot_transfer.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "transfer_bw2.csv")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)

CONFIGS = [
    ("baremetal",              "bare metal (PCIe)",   "#000000", "o"),
    ("gvirtus-tcp",            "GVirtuS UCX-TCP",     "#1f77b4", "s"),
    ("gvirtus-rdma",           "GVirtuS UCX-RDMA",    "#ff7f0e", "^"),
    ("gvirtus-rdma-gpudirect", "GVirtuS RDMA+GPUDirect", "#2ca02c", "D"),
]

for direction in ["H2D", "D2H"]:
    plt.figure(figsize=(7, 4.5))
    sub = df[df["dir"] == direction]
    for cfg, label, color, marker in CONFIGS:
        s = sub[sub["config"] == cfg].sort_values("bytes")
        if s.empty:
            continue
        plt.plot(s["bytes"] / 1024.0, s["gbps"], marker=marker, color=color,
                 label=label, linewidth=1.6, markersize=5)
    plt.xscale("log", base=2)
    plt.xlabel("transfer size (KiB, log scale)")
    plt.ylabel("bandwidth (GB/s)")
    plt.title(f"cudaMemcpy {direction} bandwidth over GVirtuS transports")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"transfer_{direction.lower()}.png"), dpi=130)
    plt.close()

# RDMA vs RDMA+GPUDirect zoom for D2H (where GPUDirect helps ~2.3x).
plt.figure(figsize=(7, 4.5))
sub = df[df["dir"] == "D2H"]
for cfg, label, color, marker in CONFIGS[2:]:
    s = sub[sub["config"] == cfg].sort_values("bytes")
    if s.empty:
        continue
    plt.plot(s["bytes"] / 1024.0, s["gbps"], marker=marker, color=color,
             label=label, linewidth=1.6, markersize=5)
plt.xscale("log", base=2)
plt.xlabel("transfer size (KiB, log scale)")
plt.ylabel("bandwidth (GB/s)")
plt.title("D2H: GPUDirect ~2.3x faster than plain RDMA (>=4 MiB)")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "transfer_d2h_rdma_vs_gpudirect.png"), dpi=130)
plt.close()

print("wrote transfer plots to", OUT)
