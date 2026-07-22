#!/usr/bin/env python3
"""Plot llama.cpp token-gen / prompt-eval throughput across transports.
Data: llama_bench_transports.csv (TinyLlama-1.1B Q4, llama-bench -p 8 -n 16 -r 3).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "llama_bench_transports.csv"))))

order = ["baremetal", "gvirtus-tcp", "gvirtus-rdma", "gvirtus-rdma-gpudirect"]
labels = {"baremetal": "bare metal", "gvirtus-tcp": "GVirtuS TCP",
          "gvirtus-rdma": "GVirtuS RDMA", "gvirtus-rdma-gpudirect": "GVirtuS RDMA+GPUDirect"}
colors = {"baremetal": "#444", "gvirtus-tcp": "#d62728",
          "gvirtus-rdma": "#1f77b4", "gvirtus-rdma-gpudirect": "#2ca02c"}
by = {r["config"]: r for r in rows}

os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

for metric, sd, title, fname in [
    ("tg16_ts", "tg16_sd", "Token generation (tg16)", "llama_tg16_transports.png"),
    ("pp8_ts", "pp8_sd", "Prompt eval (pp8)", "llama_pp8_transports.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = range(len(order))
    vals = [float(by[c][metric]) for c in order]
    errs = [float(by[c][sd]) for c in order]
    bars = ax.bar(xs, vals, yerr=errs, capsize=4,
                  color=[colors[c] for c in order])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels[c] for c in order], rotation=15, ha="right")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title(f"llama.cpp TinyLlama-1.1B Q4 — {title}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, "plots", fname)
    fig.savefig(out, dpi=130)
    print("wrote", out)

# GVirtuS-only zoom (drop bare metal so the RDMA vs TCP gap is visible)
gv = ["gvirtus-tcp", "gvirtus-rdma", "gvirtus-rdma-gpudirect"]
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = range(len(gv))
vals = [float(by[c]["tg16_ts"]) for c in gv]
errs = [float(by[c]["tg16_sd"]) for c in gv]
bars = ax.bar(xs, vals, yerr=errs, capsize=4, color=[colors[c] for c in gv])
ax.set_xticks(list(xs)); ax.set_xticklabels([labels[c] for c in gv], rotation=15, ha="right")
ax.set_ylabel("token gen (tokens/s)")
ax.set_title("llama.cpp tg16 over GVirtuS — RDMA ~2.2x TCP (RPC-latency bound)")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout()
out = os.path.join(HERE, "plots", "llama_tg16_gvirtus_only.png")
fig.savefig(out, dpi=130); print("wrote", out)
