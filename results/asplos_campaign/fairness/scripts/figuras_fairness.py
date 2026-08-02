#!/usr/bin/env python3
"""Fairness figures for the paper. Output in ~/paper/figures/.

Categorical palette in fixed order, validated for colour-vision deficiency (worst adjacent
pair CVD delta-E 9.1, normal vision 19.6). Three of the six colours fall below 3:1 against the
surface, so EVERY series also carries its own marker and a direct label: identity never rests
on colour alone.

NOTE ON KEYS. The first element of each SIS tuple is the value as it appears in the CSV and
must not be translated; only the second element, the label, is English. An earlier pass
translated the key and silently emptied the native series.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

H = os.path.expanduser("~")
F = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness")
OUT = os.path.join(H, "paper/figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
#      csv key            label              colour     marker
SIS = [("baremetal",      "native",          "#2a78d6", "o"),
       ("baremetal_mps",  "native+MPS",      "#eb6834", "s"),
       ("gusto_gpudirect","Gusto GPUDirect", "#1baf7a", "^"),
       ("gusto_am",       "Gusto AM",        "#eda100", "D"),
       ("ucx_host_rma",   "UCX host RMA",    "#e87ba4", "v"),
       ("tcp",            "TCP",             "#4a3aa7", "P")]
WL = [("minibude", "MiniBUDE"), ("xsbench", "XSBench"),
      ("babelstream", "BabelStream"), ("cloverleaf", "CloverLeaf")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False,
})

def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=130 if ext == "png" else None)
        print("written", p)
    plt.close(fig)

res = [r for r in csv.DictReader(open(os.path.join(F, "fairness_trabajo_fijo_resumen.csv")))
       if r["mode"] == "sync" and not r["variante"]]

def series(wl, key, field):
    """`field` is a (preferred, fallback) pair.

    The preferred metric is over the workload's INTERNAL runtime, which is what the documents
    conclude from; wall clock is the fallback for CloverLeaf, whose figure of merit is not in
    the artifact. Plotting wall clock everywhere would contradict the tables: on wall clock
    XSBench reads 1.4 where its internal ratio is 6.0.
    """
    pref, fall = field
    xs, ys = [], []
    for n in (2, 4, 8):
        rows = [r for r in res
                if r["workload"] == wl and r["system"] == key and int(r["N"]) == n]
        val = None
        for f in (pref, fall):
            v = [r[f] for r in rows if r.get(f) not in ("", "None", "missing", None)]
            if v:
                val = float(v[0]); break
        if val is not None:
            xs.append(n); ys.append(val)
    return xs, ys

# ---- Figures 1 and 2: two fairness metrics against tenant count -------------------------
for nfig, (field, title, ylab, lim) in enumerate([
        (("jain_int_mediana", "jain_wall_mediana"),
         "Progress fairness (Jain over normalised progress, internal runtime)",
         "Jain", (0.6, 1.02)),
        (("ratio_lento_rapido_int_mediana", "ratio_lento_rapido_wall_mediana"),
         "Inequality between tenants (slowest / fastest, internal runtime)",
         "slowest / fastest", (0.9, 6.4))], 1):
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.5), sharey=True)
    for ax, (wl, nice) in zip(axs, WL):
        clean(ax)
        ax.axhline(1.0, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        labels = []
        for key, lab, col, mk in SIS:
            xs, ys = series(wl, key, field)
            if not xs:
                continue
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=5,
                    mec=SURF, mew=1.0, zorder=3, clip_on=False)
            if wl == "minibude":
                labels.append((ys[-1], xs[-1], lab, col))
        labels.sort()
        step = (lim[1] - lim[0]) * 0.062
        prev = None
        for y, x, lab, col in labels:
            yy = y if prev is None else max(y, prev + step)
            ax.annotate(lab, (x, yy), textcoords="offset points", xytext=(7, 0),
                        color=INK2, fontsize=6.6, va="center", annotation_clip=False)
            prev = yy
        ax.set_title(nice, fontsize=8.5, color=INK, pad=6)
        ax.set_xticks([2, 4, 8]); ax.set_xlabel("tenants (N)", fontsize=7.5)
        ax.set_ylim(*lim)
    axs[0].set_ylabel(ylab, fontsize=7.5)
    fig.suptitle(title, fontsize=9.5, color=INK, x=0.02, ha="left", y=1.04)
    fig.tight_layout(rect=(0, 0, 0.87, 0.98))
    save(fig, "fig1_jain_vs_N" if nfig == 1 else "fig2_slowest_fastest_vs_N")

# ---- Figure 3: the causal figure, quantum sharing in MiniBUDE --------------------------
d = list(csv.DictReader(open(os.path.join(F, "tabla_D_minibude_por_tenant.csv"))))
def matrix(run):
    fs = sorted([x for x in d if x["run"] == run], key=lambda r: int(r["tenant_id"]))
    return np.array([[float(v) for v in f["multiplicadores"].split()] for f in fs]), fs

fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
for ax, (run, nice) in zip(axs, [("run_baremetal_n8_r1", "native"),
                                 ("run_ucx_gpudirect_n8_r1", "Gusto GPUDirect")]):
    M, fs = matrix(run)
    im = ax.imshow(M, cmap="Blues", vmin=1, vmax=16, aspect="auto")
    ax.set_title(f"{nice}   (N=8, MiniBUDE)", fontsize=8.5, color=INK, pad=6)
    ax.set_xlabel("iteration", fontsize=7.5)
    ax.set_xticks(range(10)); ax.set_xticklabels(range(1, 11))
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"t{f['tenant_id']}  x{float(f['slowdown']):.2f}" for f in fs],
                       fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color=("#ffffff" if M[i, j] > 8.5 else INK))
axs[0].set_ylabel("tenant and its slowdown", fontsize=7.5)
cb = fig.colorbar(im, ax=axs, fraction=0.024, pad=0.02)
cb.set_label("service quanta elapsed per iteration", fontsize=7, color=INK2)
cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6.5)
fig.suptitle("A multiplier of 1 means served with no wait; of N, equal sharing",
             fontsize=9.5, color=INK, x=0.02, ha="left", y=1.02)
save(fig, "fig3_minibude_quantum")

# ---- Figure 4: per-tenant service fraction in llama ------------------------------------
t = list(csv.DictReader(open(os.path.join(F, "llama_fairness_por_tenant.csv"))))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
for ax, (lam, nice) in zip(axs, [(1.0, "useful load (lambda=1, stable)"),
                                 (8.0, "saturation (lambda=8, unstable)")]):
    clean(ax)
    w = 0.38
    # NOTE: "nativo" and "gusto" are the CSV values, not labels. Do not translate them.
    for k, (key, lab, col) in enumerate([("nativo", "native", "#2a78d6"),
                                         ("gusto", "Gusto", "#1baf7a")]):
        v = sorted([float(r["completion_fraction"]) for r in t
                    if r["system"] == key and int(r["N"]) == 8
                    and abs(float(r["lambda_total"]) - lam) < 1e-9 and int(r["rep"]) == 1])
        if not v:
            continue
        x = np.arange(len(v)) + (k - 0.5) * w
        ax.bar(x, v, width=w * 0.92, color=col, zorder=3, label=lab)
    ax.set_title(nice, fontsize=8.5, color=INK, pad=6)
    ax.set_xlabel("tenants, sorted by service received", fontsize=7.5)
    ax.set_xticks(range(8)); ax.set_xticklabels([])
    ax.set_ylim(0, 1.12); ax.yaxis.set_major_locator(MultipleLocator(0.25))
axs[0].set_ylabel("completion fraction", fontsize=7.5)
# The legend goes in the right panel, whose upper half is empty; in the left one every bar
# reaches 1.00 and there is no room.
axs[1].legend(loc="upper left", fontsize=7.5, labelcolor=INK2, handlelength=1.2,
              borderaxespad=0.6)
fig.suptitle("llama 7B, N=8: fairness is preserved; what is lost is capacity",
             fontsize=9.5, color=INK, x=0.02, ha="left", y=1.03)
fig.tight_layout()
save(fig, "fig4_llama_per_tenant")
