#!/usr/bin/env python3
"""Figura 5: capacidad util bajo SLO frente a la carga ofrecida, por numero de tenants.

Un panel por N. Eje x = carga ofrecida total; eje y = goodput de ventana estricta. Los puntos
que CUMPLEN el SLO (TTFT p95 < 1 s y 0 timeouts) van rellenos; los que no, huecos. La frontera
entre unos y otros es la rodilla, y su abscisa es la capacidad util.

Paleta categorica validada: nativo = azul (slot 1), Gusto = aqua (slot 3). Ambas series llevan
marcador propio y etiqueta directa, asi que la identidad no depende solo del color.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = os.path.expanduser("~/paper/LLAMA_SLO_capacidad.csv")
OUT = os.path.expanduser("~/paper/figures")
os.makedirs(OUT, exist_ok=True)

TINTA, TINTA2, MUDO = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIS = [("bm", "native", "#2a78d6", "o"), ("bmmps", "native+MPS", "#eb6834", "s"),
       ("ucx", "Gusto", "#1baf7a", "^")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": EJE, "axes.linewidth": 0.7, "axes.labelcolor": TINTA2,
    "xtick.color": MUDO, "ytick.color": MUDO,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "savefig.facecolor": FONDO, "legend.frameon": False,
})

rs = []
for r in csv.DictReader(open(CSV)):
    if int(r["rep"]) == 0:
        continue                      # el punto de humo, no pertenece a la rejilla
    try:
        rs.append(dict(system=r["system"], N=int(r["N"]), lam=float(r["lam"]),
                       gp=float(r["goodput"]), p95=float(r["ttft_p95"]),
                       to=int(float(r["timeouts"] or 0))))
    except (ValueError, TypeError):
        continue
if not rs:
    raise SystemExit("sin datos")

cumple = lambda r: r["p95"] < 1000 and r["to"] == 0
Ns = sorted({r["N"] for r in rs})

fig, axs = plt.subplots(1, len(Ns), figsize=(3.05 * len(Ns), 2.7), sharey=True)
if len(Ns) == 1:
    axs = [axs]
for ax, N in zip(axs, Ns):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    for key, lab, col, mk in SIS:
        v = sorted([r for r in rs if r["system"] == key and r["N"] == N],
                   key=lambda r: r["lam"])
        if not v:
            continue
        x = [r["lam"] for r in v]; y = [r["gp"] for r in v]
        ax.plot(x, y, color=col, lw=1.8, zorder=2)
        ok = [r for r in v if cumple(r)]
        no = [r for r in v if not cumple(r)]
        if ok:
            ax.scatter([r["lam"] for r in ok], [r["gp"] for r in ok], s=42, marker=mk,
                       color=col, edgecolors=FONDO, linewidths=1.0, zorder=4)
            # la rodilla: ultimo punto que cumple
            k = max(ok, key=lambda r: r["lam"])
            # La capacidad en lambda coincide entre sistemas, asi que lo que discrimina es el
            # goodput A esa carga: se anota eso. El desplazamiento vertical evita que las dos
            # etiquetas se pisen cuando la rodilla cae en el mismo punto.
            dy = 11 if key == "ucx" else -18
            ax.annotate("%.0f t/s" % k["gp"], (k["lam"], k["gp"]),
                        textcoords="offset points", xytext=(-2, dy), ha="center",
                        fontsize=7.2, color=col, weight="bold")
        if no:
            ax.scatter([r["lam"] for r in no], [r["gp"] for r in no], s=42, marker=mk,
                       facecolors=FONDO, edgecolors=col, linewidths=1.3, zorder=3)
        if N == Ns[-1]:
            ax.annotate(lab, (x[-1], y[-1]), textcoords="offset points", xytext=(7, 0),
                        color=TINTA2, fontsize=7, va="center", annotation_clip=False)
    ax.set_title(f"N = {N} tenants", fontsize=8.5, color=TINTA, pad=6)
    ax.set_xlabel("total offered load (req/s)", fontsize=7.5)
axs[0].set_ylabel("goodput, strict window (t/s)", fontsize=7.5)
fig.suptitle("Filled = meets the SLO (TTFT p95 < 1 s, 0 timeouts); hollow = does not. "
             "The figure is goodput at that load.",
             fontsize=9, color=TINTA, x=0.02, ha="left", y=1.04)
fig.tight_layout(rect=(0, 0, 0.93, 0.98))
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig5_slo_capacidad.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=130 if ext == "png" else None)
    print("escrita", p)
