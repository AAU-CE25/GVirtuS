#!/usr/bin/env python3
"""fig5 v3 -- SLO attainment, TTFT and backlog, three panels instead of one capacity number.

POR QUE SE REEMPLAZA. La version anterior condensaba el resultado en "capacidad bajo un SLO":
un solo numero por sistema que oculta las tres cosas que un revisor quiere ver por separado --
que fraccion de peticiones cumple el objetivo, cuanto tarda la cola en el p95, y si el sistema
esta acumulando trabajo. Las tres estan en el mismo CSV y ninguna necesita medirse de nuevo.

Unidad experimental = la CORRIDA (rep). Se pinta cada rep y la MEDIANA entre reps, con la
palabra "median" escrita, porque estas distribuciones son sesgadas y una media las describe mal
-- ya paso en esta campana con una celda multimodal.
"""
import csv, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = os.path.expanduser("~")
SRC = os.path.join(H, "paper/LLAMA_SLO_capacidad_v2.csv")
OUT = os.path.join(H, "paper/figures")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False,
})

SYS = [("bm", "native", BLUE, "o"), ("bmmps", "native+MPS", ORANGE, "s"),
       ("ucx", "Gusto (UCX)", GREEN, "^")]
rows = list(csv.DictReader(open(SRC)))
N_FIJO = 8          # el punto de multi-tenencia que discute el paper

def serie(sistema, col):
    """lam -> lista de valores, una por rep, a N fijo."""
    d = collections.defaultdict(list)
    for r in rows:
        if r["system"] == sistema and int(r["N"]) == N_FIJO:
            v = r[col]
            if v not in ("", "None"):
                d[float(r["lam"])].append(float(v))
    return d

PANELES = [
    ("pct_1s",   "(a) SLO attainment\nrequests under 1 s (%)",      "% under 1 s",       (0, 105)),
    ("ttft_p95", "(b) queueing delay\nTTFT p95 (ms, log)",          "TTFT p95 (ms)",     None),
    ("backlog",  "(c) backlog\noffered minus served (tok/s)",       "offered - served",  None),
]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
for ax, (col, titulo, ylab, ylim) in zip(axes, PANELES):
    for key, lab, colr, mk in SYS:
        if col == "backlog":
            a, b = serie(key, "goodput_bruto"), serie(key, "goodput")
            d = {l: [x - y for x, y in zip(a[l], b[l])] for l in a if l in b}
        else:
            d = serie(key, col)
        lams = sorted(d)
        if not lams: continue
        med = [float(np.median(d[l])) for l in lams]
        for l in lams:                                  # cada rep, en claro
            ax.scatter([l] * len(d[l]), d[l], s=13, color=colr, alpha=0.30,
                       linewidths=0, zorder=2)
        ax.plot(lams, med, color=colr, lw=1.7, marker=mk, ms=4.6,
                label=lab, zorder=3)
    # lambda es la carga ofrecida TOTAL, no por inquilino: `bench.py` reparte RATE entre los N
    # clientes. Etiquetarlo "per tenant" multiplica por N la lectura de cada celda y cambia
    # por completo la interpretacion de N=8, lambda=0.5.
    ax.set_xlabel("total offered load lambda (req/s)", fontsize=7.6)
    ax.set_ylabel(ylab, fontsize=7.6)
    ax.set_title(titulo, fontsize=8.2, color=INK, loc="left", pad=8)
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if ylim: ax.set_ylim(*ylim)
    if col == "ttft_p95": ax.set_yscale("log")

axes[0].axhline(95, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
axes[0].annotate("95 % target", (1.5, 95), textcoords="offset points",
                 xytext=(-2, -11), fontsize=6.6, color=MUTED, ha="right")
axes[2].axhline(0, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
axes[0].legend(loc="lower left", fontsize=7.0, labelcolor=INK2, handlelength=1.8)
fig.suptitle(f"llama serving at N={N_FIJO} tenants -- line = median of 3 runs, dots = the runs",
             fontsize=9.4, color=INK, x=0.005, ha="left", y=1.05, fontweight="bold")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig5_slo_attainment_ttft_backlog.{ext}"), dpi=200,
                bbox_inches="tight", facecolor=SURF)
print("  -> fig5_slo_attainment_ttft_backlog")
