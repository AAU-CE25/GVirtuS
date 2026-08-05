#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig1_h2d_policy_oracle.py -- LA figura de la pagina 1.

Dice una sola cosa: dentro de UNA direccion (H2D), fijada y paginable ponen el cruce a 16 KiB y a
1 MiB -- 64x -- asi que un escalar tiene que elegir cual de los dos regimenes sacrifica, mientras
el selector tipado alcanza al oraculo en los dos.

NO dibuja D2H a proposito: su decision es OTRA (aterrizaje, no colocacion). Mezclarlas es como se
construyo el titular equivocado de la reticula simetrica.

OJO CON LOS NOMBRES DE LOS BRAZOS, que me los cruce una vez y la figura decia lo contrario:
  `am`     = el escalar puesto ALTO (4 MiB): nunca entra en RMA.  Peor caso 31,3 % en FIJADA.
  `scalar` = el escalar puesto BAJO (16 KiB): entra siempre.      Peor caso 22,5 % en PAGINABLE.
Los dos porcentajes del titular salen de brazos DISTINTOS, y ese es justo el argumento: no hay un
solo valor bueno. Comprobado contra verifica_datos.py, que recomputa los dos.
"""
import csv, os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.environ.get("GVS_PAPER") or os.path.expanduser("~/paper")
CSV = os.path.join(RAIZ, "canonica", "micros_politica.csv")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e3e1dc", "#faf9f7"
AZUL, VERDE, AMBAR, ROSA = "#2b6cb0", "#1a9e6f", "#e0a300", "#d1477a"

d = collections.defaultdict(dict)
for r in csv.DictReader(open(CSV)):
    if r["direction"] == "h2d":
        d[(r["mem"], r["arm"])][int(r["bytes"])] = float(r["median_gbps"])
if not d:
    sys.exit("sin filas h2d en %s" % CSV)

def peor(mem, arm):
    o = d[(mem, "oracle")]
    return 100 * min(d[(mem, arm)][b] / o[b] for b in d[(mem, arm)] if o.get(b))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), facecolor=SURF)
CRUCE = {"pinned": (16 << 10, "16 KiB"), "pageable": (1 << 20, "1 MiB")}
# "never"/"always" no son literalmente ciertos en los extremos del barrido: el brazo alto SI
# entra en RMA por encima de 4 MiB, y el bajo no entra por debajo de 16 KiB. Se nombran por su
# VALOR, que es exacto y ademas mas corto de leer en una leyenda.
ARMS = [("am",       "Scalar-4MiB",  AMBAR, "--", "s"),
        ("scalar",   "Scalar-16KiB", ROSA,  ":",  "v"),
        ("oracle",   "oracle (needs the answer in advance)",  VERDE, "-",  ""),
        ("quadrant", "typed selector (deployed)",             AZUL,  "-",  "o")]

for ax, mem in zip(axes, ("pinned", "pageable")):
    ax.set_facecolor(SURF)
    for arm, lab, col, ls, mk in ARMS:
        s = d.get((mem, arm))
        if not s:
            continue
        xs = sorted(s)
        ax.plot(xs, [s[x] for x in xs], ls, color=col, marker=mk, ms=3.6,
                lw=2.2 if arm == "quadrant" else 1.4, label=lab,
                zorder=5 if arm == "quadrant" else 3,
                alpha=1.0 if arm in ("quadrant", "oracle") else 0.9)
    c, etq = CRUCE[mem]
    ax.axvline(c, color=INK2, lw=0.9, ls="-.", zorder=2)
    ax.annotate("crossover %s" % etq, (c, 0), xytext=(5, 12), textcoords="offset points",
                fontsize=7.6, color=INK2, va="bottom", fontweight="bold")
    # El brazo que se hunde en ESTE regimen, dicho sobre la propia figura.
    malo = "am" if mem == "pinned" else "scalar"
    lab_malo = "Scalar-4MiB" if malo == "am" else "Scalar-16KiB"
    ax.set_title("H2D, %s host memory" % mem, fontsize=9.4, color=INK, loc="left",
                 fontweight="bold", pad=16)
    # El brazo que se hunde AQUI, sobre la propia figura: es el numero que hay que leer.
    ax.annotate("%s falls to %.1f%% of the oracle here" % (lab_malo, peor(mem, malo)),
                (0.0, 1.015), xycoords="axes fraction", fontsize=7.8, color=ROSA if malo=="scalar" else AMBAR,
                fontweight="bold", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("transfer size (bytes, log2)", fontsize=8, color=INK2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.4, colors=INK2)
axes[0].set_ylabel("GB/s (median of 3 runs)", fontsize=8, color=INK2)
axes[0].legend(fontsize=7.0, frameon=False, loc="upper left")

fig.suptitle("One direction, two host memory kinds, 64x apart — and no single threshold serves both",
             fontsize=10.8, color=INK, x=0.008, ha="left", y=1.03, fontweight="bold")
fig.text(0.008, -0.13,
         "The two worst cases come from OPPOSITE settings of the same scalar knob: Scalar-4MiB "
         "collapses pinned to %.1f%%;\nScalar-16KiB collapses pageable to %.1f%%. The typed "
         "selector reaches %.1f%% and %.1f%% of the oracle without probing the pointer. "
         "Device-to-host is\ndeliberately absent: its typed decision is landing, not placement "
         "(CONTRACTS.md 2.0c)."
         % (peor("pinned", "am"), peor("pageable", "scalar"),
            peor("pinned", "quadrant"), peor("pageable", "quadrant")),
         fontsize=7.2, color=MUTED, ha="left")
fig.tight_layout()
sal = os.path.join(RAIZ, "figures", "fig1_h2d_policy_oracle.png")
fig.savefig(sal, dpi=200, bbox_inches="tight", facecolor=SURF)
print("  -> %s" % sal)
