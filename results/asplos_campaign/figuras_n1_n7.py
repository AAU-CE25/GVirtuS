#!/usr/bin/env python3
"""Figures for the two results closed on 2026-08-03. Output in ~/paper/figures/.

Same palette, ink and surface tokens as figuras_fairness.py, and the same rule: every series
carries a marker or a direct label, so identity never rests on colour alone.

fig6  N1 -- why the sharing is uneven. Panel (a) is the cross-system context, and it contains
      the control that refuted the last standing hypothesis: MPS puts eight clients in ONE CUDA
      context and stays fair, so a shared context is not sufficient. Panel (b) is the causal
      intervention, both arms from the SAME source so the comparison is paired.

fig7  N7 -- which defence fires in which configuration. Read the rows top to bottom: with the
      park on, the epoch guard never fires at all, which is why the whole campaign saw nothing.

NOTE ON KEYS. CSV keys are never translated; only labels are. An earlier pass translated a key
and silently emptied a series.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = os.path.expanduser("~")
CAMP = os.path.join(H, "GVirtuS/results/asplos_campaign")
OUT = os.path.join(H, "paper/figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GREEN, AMBER, PINK, PURPLE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False,
})


def guardar(fig, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{nombre}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("  ->", nombre)


# ---------------------------------------------------------------- fig6, panel (a)
rows = list(csv.DictReader(open(os.path.join(
    CAMP, "fairness/tabla_D_minibude_por_tenant.csv"))))
coh = {}
for r in rows:
    coh.setdefault((r["system"], r["run"]), []).append(float(r["slowdown"]))
ctx = {}
for (s, _), v in coh.items():
    if len(v) >= 2:
        ctx.setdefault(s, []).append(max(v) / min(v))

# csv key -> label, colour. Order is fixed and meaningful: the two fair systems first.
PANEL_A = [("baremetal", "native\n8 contexts", BLUE),
           ("baremetal_mps", "native+MPS\n1 context", ORANGE),
           ("tcp", "Gusto\nTCP", PURPLE),
           ("ucx_rdma", "Gusto\nhost RMA", PINK),
           ("ucx_gpudirect", "Gusto\nGPUDirect", GREEN)]

# ---------------------------------------------------------------- fig6, panel (b)
ab = list(csv.DictReader(open(os.path.join(
    CAMP, "sched_n1/minibude_n8_fair_ab.csv"))))
arms = {}
for r in ab:
    arms.setdefault(r["arbitraje"], []).append(float(r["desigualdad"]))
PANEL_B = [("fcfs", "FCFS\n(deployed)", GREEN),
           ("deficit_rr_lead1", "deficit RR\n(intervention)", AMBER)]

fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(7.4, 3.5), sharey=True,
    gridspec_kw={"width_ratios": [5, 2], "wspace": 0.08})

rng = np.random.default_rng(7)   # jitter only; never affects a value
for ax, panel, src in ((axa, PANEL_A, ctx), (axb, PANEL_B, arms)):
    for i, (key, lab, col) in enumerate(panel):
        v = src.get(key, [])
        if not v:
            continue
        x = i + (rng.random(len(v)) - 0.5) * 0.26
        ax.scatter(x, v, s=17, color=col, alpha=0.55, linewidths=0, zorder=2)
        med = float(np.median(v))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=col, lw=2.4,
                solid_capstyle="butt", zorder=3)
        # Direct label, and it SAYS it is a median. A bare "4.87x" over a cloud of dots reads
        # as a mean to most people, and the distributions here are skewed enough that the two
        # differ; the word is not decoration.
        ax.annotate(f"median {med:.2f}x", (i, med), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.6, color=INK,
                    fontweight="bold", zorder=4)
        ax.annotate(f"n={len(v)}", (i, 0.28), ha="center", fontsize=6.5,
                    color=MUTED)
    ax.set_xticks(range(len(panel)))
    ax.set_xticklabels([p[1] for p in panel], fontsize=7.2, color=INK2)
    ax.set_xlim(-0.6, len(panel) - 0.4)
    ax.axhline(1.0, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axa.set_ylim(0.2, 7.6)
axa.set_ylabel("inequality: slowest / fastest tenant", color=INK2)
# La leyenda de la marca, porque una linea horizontal sobre puntos no se lee sola.
axa.plot([], [], color=MUTED, lw=2.4, label="bar = median of the runs shown")
axa.scatter([], [], s=17, color=MUTED, alpha=0.55, linewidths=0, label="one run")
axa.legend(loc="upper left", fontsize=6.8, labelcolor=INK2, handlelength=1.6)
axa.set_title("(a) a shared CUDA context is NOT the cause\n"
              "MPS puts 8 clients in one context and stays fair",
              fontsize=8.2, color=INK, loc="left", pad=9)
axb.set_title("(b) adding the missing\narbitration", fontsize=8.2, color=INK,
              loc="left", pad=9)
axa.annotate("perfect sharing", (4.42, 1.0), textcoords="offset points",
             xytext=(0, 5), fontsize=6.8, color=MUTED, ha="right")
fig.suptitle("miniBUDE, 8 tenants, equal fixed work", fontsize=9.4, color=INK,
             x=0.005, ha="left", y=1.045, fontweight="bold")
guardar(fig, "fig6_n1_arbitraje")


# ---------------------------------------------------------------- fig7
#
# REESCRITA 2026-08-03. La version anterior llevaba los cuatro numeros ESCRITOS A MANO en este
# script (81/126/38/114) y dibujaba solo el contador mayor de cada brazo, escondiendo que
# `park_off_guard_off` tiene DOS distintos de cero: 38 rechazos de la guarda de generacion y
# **81 liberaciones prematuras**, que es justamente el numero que el texto cita. Ahora se lee
# del CSV del 2x2 y de los logs del quinto brazo, y se pinta TODO lo que no es cero.
n7 = list(csv.DictReader(open(os.path.join(CAMP, "epoch_n7/n7_2x2.csv"))))

CONT = [("parked",            "layout install deferred (park)",        BLUE),
        ("ack_epoch_dropped", "epoch guard: stale ack rejected",       GREEN),
        ("ack_gen_mismatch",  "generation guard: stale ack rejected",  AMBER),
        ("ack_on_free",       "PREMATURE FREE of a live slot",         PINK)]

ARM_LAB = [("park_on_guard_on",   "park on,  epoch guard on"),
           ("park_on_guard_off",  "park on,  epoch guard off"),
           ("park_off_guard_on",  "park OFF, epoch guard on"),
           ("park_off_guard_off", "park OFF, epoch guard off"),
           ("park_off_both_off",  "park OFF, both guards off")]

# Un valor por (brazo, contador). Se comprueba que las repeticiones COINCIDEN en vez de
# suponerlo: si alguna difiere se pinta el rango y se dice, que es lo que el pie afirma.
vals, spread = {}, []
for arm, _ in ARM_LAB:
    filas = [r for r in n7 if r["arm"] == arm]
    if not filas:      # el quinto brazo no esta en el CSV: se lee de sus logs
        import glob, re
        for f in sorted(glob.glob(os.path.join(CAMP, "epoch_n7", arm + "_r*.log"))):
            t = open(f, errors="replace").read()
            d = {k: int(re.findall(k + r"=(\d+)", t)[-1]) for k, _, _ in
                 [(c[0], 0, 0) for c in CONT] if re.findall(k + r"=(\d+)", t)}
            filas.append(d)
    for key, _, _ in CONT:
        xs = sorted({int(r[key]) for r in filas if key in r and r[key] != ""})
        if not xs: continue
        vals[(arm, key)] = xs[0]
        if len(xs) > 1: spread.append((arm, key, xs))

fig, ax = plt.subplots(figsize=(7.9, 3.4))
y = np.arange(len(ARM_LAB))[::-1]
alto = 0.19
for yy, (arm, _) in zip(y, ARM_LAB):
    activos = [(k, lab, col) for k, lab, col in CONT if vals.get((arm, k))]
    for j, (k, lab, col) in enumerate(activos):
        v = vals[(arm, k)]
        off = (j - (len(activos) - 1) / 2.0) * alto
        ax.barh(yy + off, v, height=alto * 0.86, color=col, linewidth=0.8,
                edgecolor=SURF, zorder=2)
        ax.annotate(f"{v}", (v, yy + off), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=7.8, color=INK, fontweight="bold")
        ax.annotate(lab, (v, yy + off), textcoords="offset points", xytext=(26, 0),
                    va="center", fontsize=6.9, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels([a[1] for a in ARM_LAB], fontsize=7.6, color=INK2,
                   fontfamily="monospace")
ax.set_xlabel("events per run, of 320 slot reservations", color=INK2, fontsize=7.8)
ax.set_xlim(0, 250)
ax.set_xticks([0, 40, 80, 120])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_title("With the park on, the epoch guard is unreachable -- so ablating it measures nothing",
             fontsize=9.4, color=INK, loc="left", pad=10, fontweight="bold")
pie = ("Rows 1-2 are identical: with the park on, no ack ever reaches the epoch guard, so "
       "turning it off changes nothing.\n"
       "Row 4 is the one to read twice -- 81 PREMATURE FREES of live slots, and still "
       "0 corrupted bytes in every arm.\n")
pie += ("Every counter is identical across the 3 repetitions of its arm."
        if not spread else
        "NOT identical across repetitions: " + "; ".join(f"{a}.{k}={x}" for a, k, x in spread))
ax.annotate(pie + "  N7_EPOCH.md section 5 names the condition still missing.",
            (0, -0.36), xycoords="axes fraction", fontsize=7.0, color=MUTED)
guardar(fig, "fig7_n7_epoch")

print("hecho")
