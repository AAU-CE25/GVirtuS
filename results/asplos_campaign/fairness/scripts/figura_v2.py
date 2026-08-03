#!/usr/bin/env python3
"""Figure 5 (v2): goodput against offered load, aggregated over repetitions.

The v1 version plotted every repetition, so with three points at each lambda the line
zig-zagged vertically and looked like error bars. Here each cell is aggregated: the marker is
the median, the vertical bar is the observed range, and the marker is filled only when the SLO
is met in EVERY repetition of that cell.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics as st
from collections import defaultdict

CSV = os.path.expanduser("~/paper/LLAMA_SLO_capacidad_v2.csv")
OUT = os.path.expanduser("~/paper/figures")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIS = [("bm", "native", "#2a78d6", "o"), ("bmmps", "native+MPS", "#eb6834", "s"),
       ("ucx", "Gusto", "#1baf7a", "^")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False,
})

cel = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    try:
        cel[(r["system"], int(r["N"]), float(r["lam"]))].append(
            dict(gp=float(r["goodput"]), p95=float(r["ttft_p95"]),
                 to=int(float(r["timeouts"] or 0))))
    except (ValueError, TypeError):
        continue

Ns = sorted({k[1] for k in cel})
fig, axs = plt.subplots(1, len(Ns), figsize=(3.15 * len(Ns), 2.8), sharey=True)
for ax, N in zip(axs, Ns):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for key, lab, col, mk in SIS:
        lams = sorted({k[2] for k in cel if k[0] == key and k[1] == N})
        if not lams:
            continue
        med = [st.median([x["gp"] for x in cel[(key, N, l)]]) for l in lams]
        lo = [min(x["gp"] for x in cel[(key, N, l)]) for l in lams]
        hi = [max(x["gp"] for x in cel[(key, N, l)]) for l in lams]
        ok = [all(x["p95"] < 1000 and x["to"] == 0 for x in cel[(key, N, l)]) for l in lams]
        ax.plot(lams, med, color=col, lw=1.8, zorder=2)
        ax.vlines(lams, lo, hi, color=col, lw=1.0, alpha=0.45, zorder=1)
        for l, m, o in zip(lams, med, ok):
            ax.scatter([l], [m], s=44, marker=mk, zorder=4, linewidths=1.2,
                       color=col if o else SURF, edgecolors=SURF if o else col)
        if N == Ns[-1]:
            ax.annotate(lab, (lams[-1], med[-1]), textcoords="offset points",
                        xytext=(7, {"bm": -6, "bmmps": 5, "ucx": 0}[key]),
                        color=INK2, fontsize=7, va="center", annotation_clip=False)
    ax.set_title(f"N = {N} tenants", fontsize=8.5, color=INK, pad=6)
    ax.set_xlabel("total offered load (req/s)", fontsize=7.5)
axs[0].set_ylabel("goodput, strict window (t/s)", fontsize=7.5)
fig.suptitle("Median over 3 repetitions, bar = observed range. Filled = meets the SLO "
             "(TTFT p95 < 1 s, 0 timeouts) in EVERY repetition.",
             fontsize=8.8, color=INK, x=0.02, ha="left", y=1.04)
fig.tight_layout(rect=(0, 0, 0.90, 0.98))
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig5_slo_capacidad_v2.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=130 if ext == "png" else None)
    print("written", p)
