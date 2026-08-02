#!/usr/bin/env python3
"""Tabla D: evidencia por tenant del caso mas fuerte (MiniBUDE, trabajo fijo).

SEMANTICA DE LOS CAMPOS, verificada aritmeticamente antes de usarlos:
  raw_iterations  10 iteraciones cronometradas, en orden.
  sum_ms          = suma de raw_iterations[2:]  (miniBUDE descarta 2 de calentamiento)
  avg_ms          = sum_ms / 8
  Comprobado en dos tenants: sum(raw[2:]) == sum_ms al milisegundo.

LA MEDIDA CLAVE. El tiempo de iteracion en solitario (N=1) es el cuanto de servicio. Se
define para cada iteracion k del tenant i:

    multiplicador_ik = raw_iterations_ik / t_iter_solo(sistema)

Un multiplicador de 1 significa que esa iteracion se sirvio sin esperar a nadie. Un
multiplicador de m significa que transcurrieron m cuantos hasta completarla, es decir que el
tenant espero (m-1) cuantos. Con N tenants y reparto equitativo el multiplicador deberia
rondar N en TODAS las iteraciones de TODOS los tenants: eso es lo que hace el nativo.

  offered work        identico y verificado por el deck (poses/proteins/ligands)
  completed work      las 10 iteraciones, en todos los tenants (nadie queda a medias)
  first progress      epoch_s + raw_iterations[0]  (resolucion de epoch_s: 1 s)
  longest no-progress = (max multiplicador - 1) * t_iter_solo
"""
import re, io, os, glob, csv, statistics as st
from collections import defaultdict

H = os.path.expanduser("~")
OUT = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness")
MB = os.path.join(H, "mb_campaign")

def _deck(g):
    po = g(r"poses:\s*(\d+)", int)
    pr = g(r"proteins:\s*(\d+)", int)
    li = g(r"ligands:\s*(\d+)", int)
    return "poses={};prot={};lig={}".format(po, pr, li)


def lee(lg):
    s = io.open(lg, errors="replace").read()
    def g(p, c=float):
        m = re.search(p, s, re.M)
        return c(m.group(1)) if m else None
    ri = re.search(r"raw_iterations:\s*\[([^\]]*)\]", s)
    return dict(
        epoch=g(r"epoch_s:(\d+)", int),
        raw=[float(x) for x in ri.group(1).split(",")] if ri else None,
        sum_ms=g(r"^\s*sum_ms:\s*([\d.]+)"),
        avg_ms=g(r"^\s*avg_ms:\s*([\d.]+)"),
        valid=(re.search(r"valid:\s*(\w+)", s).group(1) if re.search(r"valid:\s*(\w+)", s) else None),
        maxdiff=g(r"max_diff_%:\s*([\d.]+)"),
        deck=_deck(g),
    )

def runs(sys_, N):
    for d in sorted(glob.glob(os.path.join(MB, f"run_{sys_}_n{N}_r*"))):
        vistos = {}
        for lg in sorted(glob.glob(os.path.join(d, "t*.log"))):
            m = re.search(r"/(?:tenant_)?t?(\d+)\.log$", lg)
            if m and m.group(1) not in vistos:
                vistos[m.group(1)] = lg
        if vistos:
            yield d, {int(k): lee(v) for k, v in vistos.items()}

# --- cuanto de servicio: iteracion en solitario, por sistema ---------------------------
solo = {}
for s in ("baremetal", "baremetal_mps", "tcp", "ucx_rdma", "ucx_gpudirect"):
    v = [t["avg_ms"] for _, ts in runs(s, 1) for t in ts.values() if t["avg_ms"]]
    if v:
        solo[s] = st.median(v)
print("cuanto de servicio (avg_ms en N=1, mediana):")
for k, v in solo.items():
    print(f"  {k:16s} {v:8.2f} ms")

filas = []
for s in ("baremetal", "baremetal_mps", "ucx_gpudirect", "ucx_rdma", "tcp"):
    q = solo.get(s)
    if not q:
        continue
    for d, ts in runs(s, 8):
        if len(ts) != 8 or any(t["raw"] is None for t in ts.values()):
            continue
        for tid, t in sorted(ts.items()):
            mult = [r / q for r in t["raw"]]
            filas.append(dict(
                system=s, run=os.path.basename(d), tenant_id=tid,
                offered_work=t["deck"], completed_iterations=len(t["raw"]),
                t_start_epoch_s=t["epoch"],
                first_progress_s=round(t["raw"][0] / 1000.0, 3),
                slowdown=round(t["avg_ms"] / q, 3),
                normalized_progress=round(q / t["avg_ms"], 4),
                mult_min=round(min(mult), 2), mult_p50=round(st.median(mult), 2),
                mult_max=round(max(mult), 2),
                iters_sin_espera=sum(1 for m in mult if m < 1.5),
                longest_no_progress_s=round((max(mult) - 1) * q / 1000.0, 3),
                correctness=f"valid={t['valid']};max_diff_%={t['maxdiff']}",
                multiplicadores=" ".join(f"{m:.1f}" for m in mult),
            ))

p = os.path.join(OUT, "tabla_D_minibude_por_tenant.csv")
with open(p, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), extrasaction="ignore")
    w.writeheader()
    for f in filas:
        w.writerow(f)
print(f"\nescrito {p} ({len(filas)} filas tenant-corrida)\n")

# --- resumen por sistema ----------------------------------------------------------------
print("## Reparto del cuanto de servicio, N=8, MiniBUDE\n")
print("| sistema | corridas | mult. min | mult. mediana | mult. max | "
      "iteraciones servidas sin esperar (de 80) | mayor parada (s) |")
print("|---|---:|---:|---:|---:|---:|---:|")
por = defaultdict(list)
for f in filas:
    por[f["system"]].append(f)
for s, fs in por.items():
    nruns = len({f["run"] for f in fs})
    tot_iter = sum(f["completed_iterations"] for f in fs)
    sinesp = sum(f["iters_sin_espera"] for f in fs)
    print(f"| {s} | {nruns} | {min(f['mult_min'] for f in fs):.2f} | "
          f"{st.median([f['mult_p50'] for f in fs]):.2f} | {max(f['mult_max'] for f in fs):.2f} | "
          f"**{sinesp}** de {tot_iter} | {max(f['longest_no_progress_s'] for f in fs):.2f} |")

print("\n## Tabla D — una cohorte, tenant a tenant (run_ucx_gpudirect_n8_r1)\n")
print("| tenant | trabajo ofrecido | iters | slowdown | progreso norm. | "
      "multiplicadores por iteracion | mayor parada (s) | correccion |")
print("|---:|---|---:|---:|---:|---|---:|---|")
for f in sorted([x for x in filas if x["run"] == "run_ucx_gpudirect_n8_r1"],
                key=lambda r: r["tenant_id"]):
    print(f"| {f['tenant_id']} | {f['offered_work']} | {f['completed_iterations']} | "
          f"{f['slowdown']} | {f['normalized_progress']} | `{f['multiplicadores']}` | "
          f"{f['longest_no_progress_s']} | {f['correctness']} |")

print("\n## El mismo corte en el control nativo (run_baremetal_n8_r1)\n")
print("| tenant | slowdown | progreso norm. | multiplicadores por iteracion |")
print("|---:|---:|---:|---|")
for f in sorted([x for x in filas if x["run"] == "run_baremetal_n8_r1"],
                key=lambda r: r["tenant_id"]):
    print(f"| {f['tenant_id']} | {f['slowdown']} | {f['normalized_progress']} | "
          f"`{f['multiplicadores']}` |")
