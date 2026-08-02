#!/usr/bin/env python3
"""Figuras de fairness para el paper. Salida en ~/paper/figures/.

Paleta categorica en orden fijo, validada para daltonismo (peor par adyacente CVD
delta-E 9.1, vision normal 19.6). Tres de los cinco colores quedan por debajo de 3:1
frente al fondo, asi que TODAS las series llevan etiqueta directa y marcador propio:
la identidad nunca depende solo del color.
"""
import csv, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

H = os.path.expanduser("~")
F = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness")
OUT = os.path.join(H, "paper/figures")
os.makedirs(OUT, exist_ok=True)

TINTA, TINTA2, MUDO = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIS = [("baremetal", "nativo", "#2a78d6", "o"),
       ("baremetal_mps", "nativo+MPS", "#eb6834", "s"),
       ("gusto_gpudirect", "Gusto GPUDirect", "#1baf7a", "^"),
       ("ucx_host_rma", "UCX host RMA", "#eda100", "D"),
       ("tcp", "TCP", "#e87ba4", "v")]
WL = [("minibude", "MiniBUDE"), ("xsbench", "XSBench"),
      ("babelstream", "BabelStream"), ("cloverleaf", "CloverLeaf")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": EJE, "axes.linewidth": 0.7, "axes.labelcolor": TINTA2,
    "xtick.color": MUDO, "ytick.color": MUDO,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "savefig.facecolor": FONDO, "legend.frameon": False,
})

def limpia(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=REJILLA, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

res = [r for r in csv.DictReader(open(os.path.join(F, "fairness_trabajo_fijo_resumen.csv")))
       if r["mode"] == "sync" and not r["variante"]]

def serie(wl, sys_, campo):
    xs, ys = [], []
    for n in (2, 4, 8):
        v = [r[campo] for r in res
             if r["workload"] == wl and r["system"] == sys_ and int(r["N"]) == n
             and r[campo] not in ("", "None", "missing")]
        if v:
            xs.append(n); ys.append(float(v[0]))
    return xs, ys

# ---- Figura 1 y 2: dos metricas de equidad frente al numero de tenants ------------------
for nfig, (campo, titulo, etiq_y, lim, ref) in enumerate([
        ("jain_wall_mediana", "Equidad de progreso (indice de Jain sobre progreso normalizado)",
         "Jain", (0.6, 1.02), 1.0),
        ("ratio_lento_rapido_wall_mediana", "Desigualdad entre tenants (lento / rapido)",
         "ratio lento/rapido", (0.9, 5.4), 1.0)], 1):
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.5), sharey=True)
    for ax, (wl, nom) in zip(axs, WL):
        limpia(ax)
        ax.axhline(ref, color=EJE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        etiquetas = []
        for key, lab, col, mk in SIS:
            xs, ys = serie(wl, key, campo)
            if not xs:
                continue
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=5,
                    mec=FONDO, mew=1.0, zorder=3, clip_on=False)
            if wl == "minibude":
                etiquetas.append((ys[-1], xs[-1], lab))
        # Separacion minima entre etiquetas para que ninguna tape a otra.
        etiquetas.sort()
        paso = (lim[1] - lim[0]) * 0.062
        prev = None
        for y, x, lab in etiquetas:
            yy = y if prev is None else max(y, prev + paso)
            ax.annotate(lab, (x, yy), textcoords="offset points", xytext=(7, 0),
                        color=TINTA2, fontsize=6.8, va="center", annotation_clip=False)
            prev = yy
        ax.set_title(nom, fontsize=8.5, color=TINTA, pad=6)
        ax.set_xticks([2, 4, 8]); ax.set_xlabel("tenants (N)", fontsize=7.5)
        ax.set_ylim(*lim)
    axs[0].set_ylabel(etiq_y, fontsize=7.5)
    fig.suptitle(titulo, fontsize=9.5, color=TINTA, x=0.02, ha="left", y=1.04)
    fig.tight_layout(rect=(0, 0, 0.88, 0.98))
    p = os.path.join(OUT, f"fig{nfig}_" +
                     ("jain_vs_N" if nfig == 1 else "lento_rapido_vs_N") + ".pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=130); plt.close(fig)
    print("escrita", p)

# ---- Figura 3: la figura causal, el reparto del cuanto en MiniBUDE ---------------------
d = list(csv.DictReader(open(os.path.join(F, "tabla_D_minibude_por_tenant.csv"))))
def matriz(run):
    fs = sorted([x for x in d if x["run"] == run], key=lambda r: int(r["tenant_id"]))
    return np.array([[float(v) for v in f["multiplicadores"].split()] for f in fs]), fs

fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
for ax, (run, nom) in zip(axs, [("run_baremetal_n8_r1", "nativo"),
                                ("run_ucx_gpudirect_n8_r1", "Gusto GPUDirect")]):
    M, fs = matriz(run)
    im = ax.imshow(M, cmap="Blues", vmin=1, vmax=16, aspect="auto")
    ax.set_title(f"{nom}   (N=8, MiniBUDE)", fontsize=8.5, color=TINTA, pad=6)
    ax.set_xlabel("iteracion", fontsize=7.5)
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
                    color=("#ffffff" if M[i, j] > 8.5 else TINTA))
axs[0].set_ylabel("tenant y su slowdown", fontsize=7.5)
cb = fig.colorbar(im, ax=axs, fraction=0.024, pad=0.02)
cb.set_label("cuantos de servicio transcurridos por iteracion", fontsize=7, color=TINTA2)
cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6.5)
fig.suptitle("Un multiplicador de 1 significa servido sin esperar; de N, reparto equitativo",
             fontsize=9.5, color=TINTA, x=0.02, ha="left", y=1.02)
p = os.path.join(OUT, "fig3_minibude_cuanto.pdf")
fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=130); plt.close(fig)
print("escrita", p)

# ---- Figura 4: fraccion de servicio por tenant en llama --------------------------------
t = list(csv.DictReader(open(os.path.join(F, "llama_fairness_por_tenant.csv"))))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
for ax, (lam, nom) in zip(axs, [(1.0, "carga util (lambda=1, estable)"),
                                (8.0, "saturacion (lambda=8, inestable)")]):
    limpia(ax)
    an = 0.38
    for k, (sys_, lab, col) in enumerate([("nativo", "nativo", "#2a78d6"),
                                          ("gusto", "Gusto", "#1baf7a")]):
        v = sorted([float(r["completion_fraction"]) for r in t
                    if r["system"] == sys_ and int(r["N"]) == 8
                    and abs(float(r["lambda_total"]) - lam) < 1e-9 and int(r["rep"]) == 1])
        if not v:
            continue
        x = np.arange(len(v)) + (k - 0.5) * an
        ax.bar(x, v, width=an * 0.92, color=col, zorder=3, label=lab)

    ax.set_title(nom, fontsize=8.5, color=TINTA, pad=6)
    ax.set_xlabel("tenants, ordenados por servicio recibido", fontsize=7.5)
    ax.set_xticks(range(8)); ax.set_xticklabels([])
    ax.set_ylim(0, 1.12); ax.yaxis.set_major_locator(MultipleLocator(0.25))
axs[0].set_ylabel("fraccion completada", fontsize=7.5)
# La leyenda va en el panel derecho, cuya mitad superior esta vacia; en el izquierdo
# todas las barras llegan a 1.00 y no hay hueco.
axs[1].legend(loc="upper left", fontsize=7.5, labelcolor=TINTA2, handlelength=1.2,
              borderaxespad=0.6)
fig.suptitle("llama 7B, N=8: la equidad se conserva; lo que se pierde es capacidad",
             fontsize=9.5, color=TINTA, x=0.02, ha="left", y=1.03)
fig.tight_layout()
p = os.path.join(OUT, "fig4_llama_por_tenant.pdf")
fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=130); plt.close(fig)
print("escrita", p)
