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
        # Direct label: the median, on every series. Never colour alone.
        ax.annotate(f"{med:.2f}x", (i, med), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK,
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

axa.set_ylim(0.2, 7.2)
axa.set_ylabel("inequality: slowest / fastest tenant", color=INK2)
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
ARMS = [("park on,  epoch guard on",   81,   0,   0),
        ("park on,  epoch guard off",  81,   0,   0),
        ("park OFF, epoch guard on",    0, 126,   0),
        ("park OFF, epoch guard off",   0,   0,  38),
        ("park OFF, both guards off",   0,   0, 114)]
SERIES = [("parked (layout install deferred)", BLUE),
          ("epoch guard: stale ack rejected", GREEN),
          ("generation guard: stale ack rejected", AMBER)]

# One bar per arm, not a grouped bar: exactly ONE counter is non-zero in every arm, so
# grouping offsets bars off their own row label and reads as noise. Colour carries which
# defence fired, and every bar is also labelled with that mechanism in words -- identity
# never rests on colour alone.
fig, ax = plt.subplots(figsize=(7.6, 3.2))
y = np.arange(len(ARMS))[::-1]
for yy, a in zip(y, ARMS):
    vals = a[1:]
    k = int(np.argmax(vals))                 # the one that fired
    v, (lab, col) = vals[k], SERIES[k]
    ax.barh(yy, v, height=0.5, color=col, linewidth=0.8, edgecolor=SURF, zorder=2)
    ax.annotate(f"{v}", (v, yy), textcoords="offset points", xytext=(5, 0),
                va="center", fontsize=8.4, color=INK, fontweight="bold")
    ax.annotate(lab, (v, yy), textcoords="offset points", xytext=(5 + 22, 0),
                va="center", fontsize=7.2, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels([a[0] for a in ARMS], fontsize=7.8, color=INK2,
                   fontfamily="monospace")
ax.set_xlabel("events per run, of 320 slot reservations "
              "(3 repetitions, identical in every one)", color=INK2, fontsize=7.8)
ax.set_xlim(0, 235)
ax.set_xticks([0, 40, 80, 120])
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_title("The epoch guard never fired because the park stood in front of it",
             fontsize=9.6, color=INK, loc="left", pad=10, fontweight="bold")
ax.annotate("With the park on, the guard is unreachable -- so ablating it (row 2) measures "
            "nothing.\nByte corruption: 0 in every arm; N7_EPOCH.md section 5 says which "
            "condition is still missing.",
            (0, -0.34), xycoords="axes fraction", fontsize=7.1, color=MUTED)
guardar(fig, "fig7_n7_epoch")

print("hecho")
