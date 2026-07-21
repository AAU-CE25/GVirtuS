#!/usr/bin/env python3
"""Per-RPC latency distribution analysis for GVirtuS UCX transports.
Input: lat_{rdma,tcp,gpudirect}.csv (routine,payload_bytes,rt_us,server_us)
captured via GVIRTUS_LATENCY_TRACE during an identical llama-bench run.
Outputs: percentile table CSV + CDF plots + per-routine table + tail-ratio plot.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSPORTS = ["tcp", "rdma", "gpudirect"]
LABEL = {"tcp": "TCP", "rdma": "RDMA", "gpudirect": "RDMA+GPUDirect"}
COLOR = {"tcp": "#d62728", "rdma": "#1f77b4", "gpudirect": "#2ca02c"}
PCTS = [50, 90, 99, 99.9]

def load(t):
    rt, pay, rout = [], [], []
    with open(os.path.join(HERE, f"lat_{t}.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rt.append(float(row["rt_us"])); pay.append(int(row["payload_bytes"]))
                rout.append(row["routine"])
            except (ValueError, KeyError):
                pass
    return np.array(rt), np.array(pay), np.array(rout)

data = {t: load(t) for t in TRANSPORTS}

# --- Percentile table (overall + control-plane subset: payload < 4 KiB) -------
def pct_row(name, arr):
    a = np.sort(arr)
    return [name, len(a), round(a.mean(), 1)] + [round(np.percentile(a, p), 1) for p in PCTS] + [round(a.max(), 1)]

hdr = ["scope", "n", "mean_us"] + [f"p{p}_us" for p in PCTS] + ["max_us"]
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
with open(os.path.join(HERE, "latency_percentiles.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["transport"] + hdr)
    for t in TRANSPORTS:
        rt, pay, rout = data[t]
        w.writerow([LABEL[t]] + pct_row("all", rt))
        ctrl = rt[pay < 4096]
        w.writerow([LABEL[t]] + pct_row("control(<4KiB)", ctrl))
        lk = rt[rout == "cudaLaunchKernel"]
        if len(lk):
            w.writerow([LABEL[t]] + pct_row("cudaLaunchKernel", lk))

# print a readable summary
print(f"{'transport':16} {'scope':18} {'n':>7} {'mean':>7} " + " ".join(f'p{p}'.rjust(8) for p in PCTS) + f"{'max':>9} {'p99/p50':>8}")
for t in TRANSPORTS:
    rt, pay, rout = data[t]
    for name, arr in [("all", rt), ("control(<4KiB)", rt[pay < 4096]),
                      ("cudaLaunchKernel", rt[rout == "cudaLaunchKernel"])]:
        if not len(arr): continue
        p = [np.percentile(arr, x) for x in PCTS]
        tail = p[2] / p[0] if p[0] else 0
        print(f"{LABEL[t]:16} {name:18} {len(arr):7d} {arr.mean():7.1f} " +
              " ".join(f"{v:8.1f}" for v in p) + f"{arr.max():9.1f} {tail:8.2f}")

# --- CDF plot (control-plane RPCs, log-x) -------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for t in TRANSPORTS:
    rt, pay, _ = data[t]
    a = np.sort(rt[pay < 4096])
    y = np.arange(1, len(a) + 1) / len(a)
    ax.plot(a, y, color=COLOR[t], lw=2, label=f"{LABEL[t]} (p50={np.percentile(a,50):.0f}µs, p99={np.percentile(a,99):.0f}µs)")
ax.set_xscale("log"); ax.set_xlabel("per-RPC round-trip latency (µs, log)")
ax.set_ylabel("CDF"); ax.set_ylim(0, 1)
ax.set_title("GVirtuS control-plane RPC latency CDF (payload < 4 KiB)\nidentical llama-bench workload")
ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "plots", "latency_cdf_control.png"), dpi=130)
print("wrote plots/latency_cdf_control.png")

# --- Percentile bar (control-plane), grouped by percentile --------------------
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(PCTS)); w = 0.26
for i, t in enumerate(TRANSPORTS):
    rt, pay, _ = data[t]
    a = rt[pay < 4096]
    vals = [np.percentile(a, p) for p in PCTS]
    b = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[t], label=LABEL[t])
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"p{p}" for p in PCTS])
ax.set_ylabel("latency (µs)"); ax.set_title("Control-plane RPC latency percentiles by transport (payload < 4 KiB)")
ax.legend(fontsize=9); ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "plots", "latency_percentiles_control.png"), dpi=130)
print("wrote plots/latency_percentiles_control.png")
