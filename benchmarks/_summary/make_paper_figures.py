#!/usr/bin/env python3
"""
Publication figure set for the GVirtuS benchmark paper (InfoComm 2027).

One distinct figure per benchmark. Each shows GVirtuS across its three UCX
transport modes -- UCX/TCP, UCX/RDMA, UCX/RDMA+GPUDirect -- against a bare-metal
(native CUDA, local L40S) reference. GVirtuS is the shipping build (async
dispatcher on). No GVirtuS-internal before/after comparisons.

Rigorous campaign (2026-07-21): every point is >=1 discarded warm-up + >=4 (mostly
5) measured reps, GVIRTUS_LOGLEVEL=40000 (ERROR), backend restarted fresh per
GPUDirect phase, GPU verified clean on both nodes. Error bars = 95% CI.
Aggregated by aggregate_rig.py:
  bars   : _summary/mode_comparison{,_ci}.csv
  babel  : _summary/babel_agg_{native,ucx_tcp,ucx_rdma}.csv (Triad, mean+95%CI)
  transfer: _summary/transfer_agg_{native,ucx_tcp,ucx_rdma,ucx_gpudirect}.csv
Caveats (honesty, per SKILL rules):
  - BabelStream reports peak of its internal 100-iter loop (its convention).
  - transfer UCX/RDMA (GD=0, zerocopy) hits a UCX local-protection error at
    >=8MB, so its line is capped at 4MB; GPUDirect covers full range.
  - transfer GPUDirect aggregated over 4 clean reps (rep5 reset connection).
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

C_NATIVE = "#4d4d4d"; C_TCP = "#c44e52"; C_RDMA = "#4c72b0"; C_GD = "#55a868"
MODES = ["native", "ucx_tcp", "ucx_rdma", "ucx_gpudirect"]
MODE_LABEL = {"native": "Bare metal\n(native)", "ucx_tcp": "GVirtuS\nUCX/TCP",
              "ucx_rdma": "GVirtuS\nUCX/RDMA", "ucx_gpudirect": "GVirtuS\nUCX/RDMA\n+GPUDirect"}
MODE_COLOR = {"native": C_NATIVE, "ucx_tcp": C_TCP, "ucx_rdma": C_RDMA, "ucx_gpudirect": C_GD}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "legend.frameon": False,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p, BENCH))


def load_modes():
    mean, ci = {}, {}
    with open(os.path.join(HERE, "mode_comparison.csv")) as f:
        for r in csv.DictReader(f):
            mean[(r["benchmark"], r["metric"])] = r
    with open(os.path.join(HERE, "mode_comparison_ci.csv")) as f:
        for r in csv.DictReader(f):
            ci[(r["benchmark"], r["metric"])] = r
    return mean, ci


def read_agg(path, xcol, ycol="gbps_mean", cicol="gbps_ci95", filt=None):
    xs, ys, es = [], [], []
    full = os.path.join(HERE, path)
    if not os.path.exists(full):
        return xs, ys, es
    with open(full) as f:
        for r in csv.DictReader(f):
            if filt and not filt(r):
                continue
            xs.append(int(r[xcol])); ys.append(float(r[ycol])); es.append(float(r[cicol]))
    order = np.argsort(xs)
    return list(np.array(xs)[order]), list(np.array(ys)[order]), list(np.array(es)[order])


def bar_panel(ax, means, cis, higher_better, unit, fmt="{:.1f}"):
    x = np.arange(len(MODES))
    vals = [float(means[m]) for m in MODES]
    errs = [float(cis[m]) for m in MODES]
    bars = ax.bar(x, vals, width=0.66, color=[MODE_COLOR[m] for m in MODES],
                  edgecolor="white", linewidth=1.5,
                  yerr=errs, capsize=5, error_kw=dict(ecolor="#333", lw=1.3))
    for b, v, e in zip(bars, vals, errs):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v + e),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10.5, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[m] for m in MODES], fontsize=9.5)
    ax.set_ylabel(unit); ax.set_ylim(0, (max(vals) + max(errs)) * 1.20)
    nat = float(means["native"])
    best = max(vals[1:]) if higher_better else min(vals[1:])
    return (best / nat if higher_better else nat / best) * 100


def fig_llama(mean, ci):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, metric, title in [(axes[0], "tg16", "Token generation (tg16)"),
                              (axes[1], "pp8", "Prompt processing (pp8)")]:
        r = mean[("llama", metric)]; c = ci[("llama", metric)]
        eff = bar_panel(ax, {m: r[m] for m in MODES}, {m: c[m + "_ci95"] for m in MODES},
                        True, "throughput (tokens/s)", fmt="{:.0f}")
        ax.set_title(title)
        ax.annotate(f"best GVirtuS = {eff:.0f}% of native",
                    (0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=10, color="#555")
    fig.suptitle("llama TinyLlama-1.1B Q4_K_M  \u2014  GVirtuS transports vs bare metal",
                 fontsize=15, weight="bold", y=1.03)
    save(fig, "bench_llama.png")


def fig_minibude(mean, ci):
    r = mean[("miniBUDE", "gflops")]; c = ci[("miniBUDE", "gflops")]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    eff = bar_panel(ax, {m: r[m] for m in MODES}, {m: c[m + "_ci95"] for m in MODES},
                    True, "throughput (GFLOP/s)")
    ax.set_title("miniBUDE (compute-bound)  \u2014  GVirtuS transports vs bare metal")
    ax.annotate(f"all transports within {abs(100 - eff):.1f}% of native (output validated)",
                (0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=10, color="#555")
    save(fig, "bench_minibude.png")


def fig_matrix(mean, ci):
    r = mean[("simple_matrix", "host_ms")]; c = ci[("simple_matrix", "host_ms")]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bar_panel(ax, {m: r[m] for m in MODES}, {m: c[m + "_ci95"] for m in MODES},
              False, "per-iteration host time (ms)  \u2014 lower is better", fmt="{:.0f}")
    ax.set_title("simple_matrix SGEMM N=16000 (transfer-bound)  \u2014  GVirtuS transports vs bare metal")
    ax.annotate("device SGEMM identical (~157 ms); the gap is the remote H2D/D2H  "
                "\u2022  GPUDirect closes it most",
                (0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=9.5, color="#555")
    save(fig, "bench_simple_matrix.png")


def fig_babelstream():
    series = [("babel_agg_native.csv", C_NATIVE, "Bare metal (native)"),
              ("babel_agg_ucx_rdma.csv", C_RDMA, "GVirtuS UCX/RDMA"),
              ("babel_agg_ucx_tcp.csv", C_TCP, "GVirtuS UCX/TCP")]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    rdma_last = None
    for path, color, lab in series:
        xs, ys, es = read_agg(path, "n_elements")
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color=color, lw=2, ms=6,
                    capsize=3, label=lab)
        if "rdma" in path:
            rdma_last = (xs[-1], ys[-1])
    if rdma_last:
        ax.annotate("GVirtuS saturates GPU HBM\nbandwidth (~670 GB/s) at large sizes",
                    rdma_last, xytext=(-20, 45), textcoords="offset points", ha="right",
                    fontsize=9.5, color="#444", arrowprops=dict(arrowstyle="->", color="#999"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#ccc"))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("array size (elements, double)")
    ax.set_ylabel("Triad bandwidth (GB/s)  \u2014 peak of 100 iters")
    ax.set_title("BabelStream Triad (bandwidth-bound)  \u2014  GVirtuS transports vs bare metal")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{int(v/1024)}K" if v < 1_048_576 else f"{int(v/1_048_576)}M")))
    save(fig, "bench_babelstream.png")


def fig_transfer():
    series = [("transfer_agg_native.csv", C_NATIVE, "Bare metal (PCIe)"),
              ("transfer_agg_ucx_gpudirect.csv", C_GD, "GVirtuS UCX/RDMA+GPUDirect"),
              ("transfer_agg_ucx_rdma.csv", C_RDMA, "GVirtuS UCX/RDMA (\u22644MB*)"),
              ("transfer_agg_ucx_tcp.csv", C_TCP, "GVirtuS UCX/TCP")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, d in zip(axes, ["H2D", "D2H"]):
        for path, color, lab in series:
            xs, ys, es = read_agg(path, "bytes", filt=lambda r: r["dir"] == d)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=color, lw=2, ms=4,
                        capsize=2, label=lab)
        ax.set_xscale("log", base=2); ax.set_xlabel("transfer size (bytes)"); ax.set_title(d)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: (f"{int(v/1024)}K" if v < 1_048_576 else f"{int(v/1_048_576)}M")))
    axes[0].set_ylabel("bandwidth (GB/s)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Host\u2194device transfer bandwidth  \u2014  GVirtuS transports vs bare metal (PCIe)",
                 fontsize=14, weight="bold", y=1.02)
    fig.text(0.5, -0.03, "*UCX/RDMA (GD=0, zerocopy) hits a UCX local-protection error at \u22658MB; "
             "GPUDirect covers the full range.", ha="center", fontsize=8.5, color="#777")
    save(fig, "bench_transfer_bw.png")


if __name__ == "__main__":
    mean, ci = load_modes()
    fig_llama(mean, ci)
    fig_minibude(mean, ci)
    fig_matrix(mean, ci)
    fig_babelstream()
    fig_transfer()
    print("\nAll figures written to", os.path.relpath(OUT, BENCH))
