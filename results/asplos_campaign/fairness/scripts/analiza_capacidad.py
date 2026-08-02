#!/usr/bin/env python3
"""Curva de capacidad util bajo SLO frente al numero de tenants.

Criterio, el que se pidio:

    capacidad(sistema, N) = max goodput  sujeto a  TTFT p95 < 1 s  y  timeouts = 0

Se usan las metricas de VENTANA ESTRICTA del sidecar `.meta.jsonl` (`goodput_strict`,
`ttft_p95_strict`, `slogp_strict_*`), no las de `summary.csv`: estas ultimas cuentan
finalizaciones en [t_meas, t_end+REQ_TIMEOUT] y dividen entre WINDOW, lo que infla la tasa
hasta x1,76 bajo saturacion.

`lambda` es TOTAL. La carga por tenant es lambda/N; la capacidad se reporta en ambas formas
porque la pregunta -- cuanta capacidad util conserva el sistema al crecer los tenants -- se
lee en la de por tenant.
"""
import json, glob, os, csv, statistics as st
from collections import defaultdict


def slo_por_tenant(label, umbral_ms):
    """Atencion de SLO por tenant, EXCLUYENDO a los que no recibieron peticiones.

    `slo_min_tenant_*` del sidecar cuenta a un tenant sin demanda como 0 % de cumplimiento,
    de modo que a carga baja el peor tenant sale 0 % mientras el agregado sale 100 %. Un
    tenant al que no llega nada no esta mal servido: no esta en la muestra. Se recalcula aqui
    desde el JSONL por peticion en vez de tocar bench.py con la campana en marcha.
    """
    f = os.path.join(BASE, label + ".jsonl")
    if not os.path.exists(f):
        return None, None
    por = defaultdict(list)
    for ln in open(f):
        d = json.loads(ln)
        por[d["server"]].append(d)
    frac = []
    for s_, xs in por.items():
        if not xs:
            continue
        ok = sum(1 for x in xs if x["ok"] and x["ttft_ms"] is not None
                 and x["ttft_ms"] <= umbral_ms)
        frac.append(100.0 * ok / len(xs))
    if not frac:
        return None, None
    return min(frac), len(frac)

BASE = os.path.expanduser("~/GVirtuS/results/asplos_campaign/llama_slo_sweep")
OUT = os.path.expanduser("~/paper/LLAMA_SLO_capacidad.csv")

filas = []
for f in sorted(glob.glob(os.path.join(BASE, "*.meta.jsonl"))):
    for ln in open(f):
        d = json.loads(ln)
        p = d["label"].split("_")        # slo_<sys>_n<N>_l<lam>_r<rep>
        if p[0] != "slo":
            continue
        filas.append(dict(
            system=p[1], N=int(p[2][1:]), lam=float(p[3][1:]), rep=int(p[4][1:]),
            goodput=d.get("goodput_strict"), goodput_bruto=d.get("goodput"),
            ttft_p95=d.get("ttft_p95_strict"), ttft_p50=d.get("ttft_p50_strict"),
            slogp_1s=d.get("slogp_strict_1s"), slogp_5s=d.get("slogp_strict_5s"),
            pct_1s=d.get("pct_slo_strict_1s"), pct_5s=d.get("pct_slo_strict_5s"),
            min_tenant_1s=slo_por_tenant(d["label"], 1000)[0],
            tenants_con_demanda=slo_por_tenant(d["label"], 1000)[1],
            min_tenant_5s=slo_por_tenant(d["label"], 5000)[0],
            timeouts=d.get("timeout"), completadas=d.get("completed_strict"),
            ofrecidas=d.get("offered"), estable=d.get("stable")))

if not filas:
    print("sin datos todavia en", BASE)
    raise SystemExit(0)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
    w.writeheader()
    for r in sorted(filas, key=lambda r: (r["system"], r["N"], r["lam"], r["rep"])):
        w.writerow(r)
print(f"escrito {OUT} ({len(filas)} puntos)\n")

NOM = {"bm": "nativo", "bmmps": "nativo+MPS", "ucx": "Gusto"}
CUMPLE = lambda r: (r["ttft_p95"] is not None and r["ttft_p95"] < 1000
                    and (r["timeouts"] or 0) == 0)

print("## Barrido: cada punto y si cumple el SLO (TTFT p95 < 1 s y 0 timeouts)\n")
print("| sistema | N | lambda total | lambda/tenant | goodput | TTFT p95 | timeouts | "
      "% SLO 1 s | peor tenant 1 s | tenants con demanda | **cumple** |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
for r in sorted(filas, key=lambda r: (r["system"], r["N"], r["lam"])):
    print(f"| {NOM.get(r['system'],r['system'])} | {r['N']} | {r['lam']:.2f} | "
          f"{r['lam']/r['N']:.3f} | {r['goodput']:.1f} | {r['ttft_p95']:.0f} ms | "
          f"{r['timeouts']} | {r['pct_1s']:.0f} % | "
          f"{(('%.0f %%' % r['min_tenant_1s']) if r['min_tenant_1s'] is not None else 'n/d')} | "
          f"{r['tenants_con_demanda']}/{r['N']} | "
          f"{'**SI**' if CUMPLE(r) else 'no'} |")

print("\n## Capacidad util bajo SLO frente al numero de tenants\n")
cap = defaultdict(list)
for r in filas:
    if CUMPLE(r):
        cap[(r["system"], r["N"])].append(r)
print("| sistema | N | **lambda maxima con SLO** | por tenant | goodput a esa carga | "
      "puntos que cumplen |")
print("|---|---:|---:|---:|---:|---:|")
for k in sorted(cap, key=lambda k: (k[0], k[1])):
    m = max(cap[k], key=lambda r: r["lam"])
    print(f"| {NOM.get(k[0],k[0])} | {k[1]} | **{m['lam']:.2f} req/s** | "
          f"{m['lam']/k[1]:.3f} | {m['goodput']:.1f} t/s | {len(cap[k])} |")
todos = {(r["system"], r["N"]) for r in filas}
sin = sorted(todos - set(cap))
for k in sin:
    print(f"| {NOM.get(k[0],k[0])} | {k[1]} | **ningun punto cumple** | — | — | 0 |")

print("\n## Retencion: cuanta capacidad conserva cada sistema al crecer los tenants\n")
for s in sorted({k[0] for k in cap}):
    base = cap.get((s, 2))
    if not base:
        continue
    b = max(base, key=lambda r: r["lam"])["lam"]
    linea = [f"**{NOM.get(s,s)}** (base N=2 = {b:.2f} req/s):"]
    for N in (4, 8):
        v = cap.get((s, N))
        linea.append(f"N={N} -> " + (f"{max(v, key=lambda r: r['lam'])['lam']/b:.2f}x" if v
                                     else "0 (ningun punto cumple)"))
    print("- " + "  ".join(linea))
