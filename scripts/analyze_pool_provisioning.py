#!/usr/bin/env python3
"""Analisis de la matriz de provisioning del pool RMA.

Entrada : results/pool_provisioning/raw/matrix.csv
Salidas : processed/*.csv  y  figures/*.png

Reglas de estadistica, fijadas ANTES de mirar los datos:
  - El estadistico central es la MEDIANA. Con 3 repeticiones la media es muy sensible a un
    punto raro y aqui no hay tamano muestral para justificarla.
  - La dispersion se da como rango observado [min, max], no como intervalo de confianza
    parametrico: con n=3 un IC normal es una ficcion de precision. Cuando n>=5 se anade
    ademas un IC bootstrap del 95 %.
  - NO se descarta ningun punto. Si un punto se sale, se marca y se explica; no se borra.
  - "Pool estable" = cero fallos de checksum, cero rechazos por timeout y backend vivo en
    TODAS las repeticiones de ese punto.
"""
import csv, os, sys, statistics as st
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ethanadams/.claude/jobs/20f7d7dd/tmp/results/pool_provisioning"
RAW = os.path.join(ROOT, "raw", "matrix.csv")
PROC = os.path.join(ROOT, "processed")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(PROC, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

if not os.path.exists(RAW):
    print("no existe %s" % RAW); sys.exit(1)

filas = list(csv.DictReader(open(RAW)))
if not filas:
    print("matrix.csv vacio"); sys.exit(1)
print("filas leidas: %d" % len(filas))

def f(x, d=0.0):
    try: return float(x)
    except Exception: return d
def i(x, d=0):
    try: return int(float(x))
    except Exception: return d

# ------------------------------------------------------------------ agregacion
grupos = defaultdict(list)
for r in filas:
    grupos[(r["politica"], i(r["slots"]), i(r["clientes"]))].append(r)

def boot_ic(vals, n=2000):
    """IC bootstrap del 95 % de la mediana. Solo se usa con n>=5; por debajo devuelve None,
    porque un IC calculado sobre 3 puntos sugiere una precision que no existe."""
    if len(vals) < 5: return None
    import random
    rnd = random.Random(12345)          # semilla fija: el analisis es reproducible
    meds = sorted(st.median([vals[rnd.randrange(len(vals))] for _ in vals]) for _ in range(n))
    return (meds[int(0.025 * n)], meds[int(0.975 * n)])

agg = []
for (pol, slots, cli), rs in sorted(grupos.items()):
    tot = [f(r["completadas_total"]) for r in rs]
    pct = [f(r["rma_pct"]) for r in rs]
    pico = [f(r["peak_inflight"]) for r in rs]
    esperas = sum(i(r["waited"]) for r in rs)
    fallos = sum(i(r["h2d_fail"]) + i(r["d2h_fail"]) for r in rs)
    tmo = sum(i(r["decline_timeout"]) for r in rs)
    vivo = all(r["backend"] == "VIVO" for r in rs)
    jain = [f(r["jain"]) for r in rs]
    ic = boot_ic(tot)
    agg.append(dict(
        politica=pol, slots=slots, clientes=cli, n=len(rs),
        completadas_mediana=round(st.median(tot), 1),
        completadas_min=round(min(tot), 1), completadas_max=round(max(tot), 1),
        ic95_bajo=round(ic[0], 1) if ic else "", ic95_alto=round(ic[1], 1) if ic else "",
        rma_pct_mediana=round(st.median(pct), 2),
        peak_inflight_max=int(max(pico)),
        esperas=esperas, timeouts=tmo, fallos=fallos,
        jain_min=round(min(jain), 4), backend_vivo=vivo,
        estable=(fallos == 0 and tmo == 0 and vivo)))

with open(os.path.join(PROC, "agregado.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
print("processed/agregado.csv: %d grupos" % len(agg))

# ------------------------------------------------------ tabla escalar vs cuadrantes
with open(os.path.join(PROC, "escalar_vs_cuadrantes.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["slots", "clientes", "compl_escalar", "compl_cuadrantes", "cociente",
                "rma_pct_escalar", "rma_pct_cuadrantes", "factor_admision",
                "pico_escalar", "pico_cuadrantes"])
    for slots in sorted({a["slots"] for a in agg}):
        for cli in sorted({a["clientes"] for a in agg}):
            e = next((a for a in agg if a["politica"] == "scalar" and a["slots"] == slots and a["clientes"] == cli), None)
            q = next((a for a in agg if a["politica"] == "quadrant" and a["slots"] == slots and a["clientes"] == cli), None)
            if not e or not q: continue
            coc = q["completadas_mediana"] / e["completadas_mediana"] if e["completadas_mediana"] else 0
            fac = q["rma_pct_mediana"] / e["rma_pct_mediana"] if e["rma_pct_mediana"] else float("inf")
            w.writerow([slots, cli, e["completadas_mediana"], q["completadas_mediana"], round(coc, 3),
                        e["rma_pct_mediana"], q["rma_pct_mediana"],
                        round(fac, 1) if fac != float("inf") else "inf",
                        e["peak_inflight_max"], q["peak_inflight_max"]])
print("processed/escalar_vs_cuadrantes.csv")

# --------------------------------------------------------- menor pool estable
lineas = ["# Menor pool estable por (politica, clientes)", "",
          "Estable = cero fallos de checksum, cero rechazos por timeout y backend vivo en",
          "TODAS las repeticiones del punto.", ""]
lineas.append("| politica | clientes | menor pool estable | pico observado | esperas |")
lineas.append("|---|---:|---:|---:|---:|")
for pol in sorted({a["politica"] for a in agg}):
    for cli in sorted({a["clientes"] for a in agg}):
        cands = sorted([a for a in agg if a["politica"] == pol and a["clientes"] == cli and a["estable"]],
                       key=lambda a: a["slots"])
        if cands:
            c = cands[0]
            lineas.append("| %s | %d | **%d** | %d | %d |" % (pol, cli, c["slots"], c["peak_inflight_max"], c["esperas"]))
        else:
            lineas.append("| %s | %d | ninguno estable | - | - |" % (pol, cli))
open(os.path.join(PROC, "menor_pool_estable.md"), "w").write("\n".join(lineas) + "\n")
print("processed/menor_pool_estable.md")

# ------------------------------------------------------------------ figuras
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"scalar": "#4C6EF5", "quadrant": "#E8590C"}
    NOM = {"scalar": "escalar 4 MiB", "quadrant": "cuadrantes"}

    # Fig 1: throughput vs slots, un panel por nivel de concurrencia, con rango observado
    clis = sorted({a["clientes"] for a in agg})
    fig, axes = plt.subplots(1, len(clis), figsize=(4.2 * len(clis), 3.6), sharey=True)
    if len(clis) == 1: axes = [axes]
    for ax, cli in zip(axes, clis):
        for pol in ("scalar", "quadrant"):
            pts = sorted([a for a in agg if a["politica"] == pol and a["clientes"] == cli],
                         key=lambda a: a["slots"])
            if not pts: continue
            x = [p["slots"] for p in pts]
            y = [p["completadas_mediana"] for p in pts]
            lo = [p["completadas_mediana"] - p["completadas_min"] for p in pts]
            hi = [p["completadas_max"] - p["completadas_mediana"] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=4, capsize=3, lw=1.4,
                        color=COL[pol], label=NOM[pol])
        ax.set_title("%d cliente%s" % (cli, "s" if cli > 1 else ""), fontsize=10)
        ax.set_xlabel("slots del pool"); ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels([4, 8, 16, 32, 64])
        ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel("transferencias completadas (mediana)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Throughput frente al tamano del pool. Barras = rango observado (n=3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_throughput_vs_slots.png"), dpi=160)
    print("figures/fig1_throughput_vs_slots.png")

    # Fig 2: el resultado central -- admisiones suben, ocupacion no
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for pol in ("scalar", "quadrant"):
        pts = sorted([a for a in agg if a["politica"] == pol and a["clientes"] == 1],
                     key=lambda a: a["slots"])
        if not pts: continue
        a1.plot([p["slots"] for p in pts], [p["rma_pct_mediana"] for p in pts],
                marker="o", ms=4, color=COL[pol], label=NOM[pol])
        a2.plot([p["slots"] for p in pts], [p["peak_inflight_max"] for p in pts],
                marker="s", ms=4, color=COL[pol], label=NOM[pol])
    for ax, tit, yl in ((a1, "operaciones admitidas al camino RMA", "% de operaciones"),
                        (a2, "pico de slots simultaneamente vivos", "slots")):
        ax.set_title(tit, fontsize=10); ax.set_xlabel("slots del pool"); ax.set_ylabel(yl)
        ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_xticklabels([4, 8, 16, 32, 64]); ax.grid(alpha=.25, lw=.6)
        ax.legend(fontsize=8, frameon=False)
    a2.set_ylim(bottom=0)
    # Si las dos politicas dan la MISMA fraccion admitida, el tamano de transferencia no las
    # discrimina (cae por encima del suelo escalar, luego ambas la admiten) y la figura no
    # puede sostener la afirmacion del titulo. Se dice en la figura en vez de afirmarla.
    pc = {p: [a["rma_pct_mediana"] for a in agg if a["politica"] == p] for p in ("scalar", "quadrant")}
    degenerado = (pc["scalar"] and pc["quadrant"] and
                  abs(max(pc["scalar"]) - max(pc["quadrant"])) < 0.01)
    if degenerado:
        fig.suptitle("ATENCION: el tamano de transferencia NO discrimina entre politicas\n"
                     "(ambas admiten la misma fraccion; las curvas se solapan)",
                     fontsize=9, color="#C92A2A")
        a1.text(0.5, 0.5, "las dos politicas coinciden", transform=a1.transAxes,
                ha="center", va="center", fontsize=9, color="#C92A2A", alpha=.8)
    else:
        fig.suptitle("El umbral gobierna la admision; la ocupacion la gobierna la concurrencia", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_admision_vs_ocupacion.png"), dpi=160)
    print("figures/fig2_admision_vs_ocupacion.png")

    # Fig 3: coste de memoria frente a demanda observada
    slots_m = [8, 16, 24, 32, 64]
    rss = [2183, 4240, 6297, 8355, 16588]     # medido, footprint2.csv
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(slots_m, rss, marker="o", color="#495057", label="RAM fijada medida")
    pico = max((a["peak_inflight_max"] for a in agg), default=2)
    ax.axvline(pico, color="#E03131", ls="--", lw=1.4,
               label="demanda pico observada = %d" % pico)
    ax.set_xlabel("slots del pool"); ax.set_ylabel("RSS del backend (MiB)")
    ax.set_title("257 MiB por slot; la demanda pico es %d" % pico, fontsize=10)
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_coste_vs_demanda.png"), dpi=160)
    print("figures/fig3_coste_vs_demanda.png")
except ImportError as e:
    print("sin matplotlib, no se generan figuras: %s" % e)

# ------------------------------------------------------------------ resumen
n_inest = [a for a in agg if not a["estable"]]
print("\n== resumen ==")
print("grupos: %d | inestables: %d" % (len(agg), len(n_inest)))
for a in n_inest:
    print("  INESTABLE %s slots=%d clientes=%d fallos=%d timeouts=%d vivo=%s"
          % (a["politica"], a["slots"], a["clientes"], a["fallos"], a["timeouts"], a["backend_vivo"]))
print("pico maximo observado en toda la matriz: %d" % max(a["peak_inflight_max"] for a in agg))
print("esperas totales en toda la matriz: %d" % sum(a["esperas"] for a in agg))
