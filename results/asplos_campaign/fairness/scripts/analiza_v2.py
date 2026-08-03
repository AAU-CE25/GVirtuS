#!/usr/bin/env python3
"""Capacidad util bajo SLO, agregada sobre las repeticiones, con IC bootstrap.

Criterio:  capacidad(sistema, N) = max lambda  con  TTFT p95 < 1 s  y  timeouts = 0

Se agrega POR CELDA (sistema, N, lambda) sobre las repeticiones. Un punto se considera que
cumple el SLO si lo cumple en TODAS sus repeticiones -- criterio conservador y el unico
honesto: un punto que cumple en 2 de 3 no es una capacidad sostenible.

Metricas de ventana ESTRICTA (`goodput_strict`, `ttft_p95_strict`), no las de summary.csv,
que cuentan hasta t_end+REQ_TIMEOUT y dividen entre WINDOW.

El IC de la diferencia emparejada se calcula sobre los pares (N, lambda, rep) que existen en
los dos sistemas, con bootstrap de 10 000 remuestreos.
"""
import json, glob, os, csv, random, math, statistics as st
from collections import defaultdict

random.seed(20260803)
BASE = os.path.expanduser("~/GVirtuS/results/asplos_campaign/llama_slo_sweep_v2")
OUT = os.path.expanduser("~/paper/LLAMA_SLO_capacidad_v2.csv")
B = 10000

def slo_worst_tenant(label, thr):
    """Peor tenant EXCLUYENDO los que no recibieron peticiones (un tenant sin demanda no esta
    mal servido: no esta en la muestra)."""
    f = os.path.join(BASE, label + ".jsonl")
    if not os.path.exists(f):
        return None, None
    por = defaultdict(list)
    for ln in open(f):
        d = json.loads(ln)
        por[d["server"]].append(d)
    fr = []
    for xs in por.values():
        if xs:
            ok = sum(1 for x in xs if x["ok"] and x["ttft_ms"] is not None and x["ttft_ms"] <= thr)
            fr.append(100.0 * ok / len(xs))
    return (min(fr), len(fr)) if fr else (None, None)

filas = []
for f in sorted(glob.glob(os.path.join(BASE, "*.meta.jsonl"))):
    for ln in open(f):
        d = json.loads(ln)
        p = d["label"].split("_")
        if p[0] != "v2":
            continue
        rep = int(p[4][1:])
        if rep == 0:
            continue                       # el punto de humo
        w1, nt = slo_worst_tenant(d["label"], 1000)
        filas.append(dict(system=p[1], N=int(p[2][1:]), lam=float(p[3][1:]), rep=rep,
                          goodput=d.get("goodput_strict"), ttft_p95=d.get("ttft_p95_strict"),
                          pct_1s=d.get("pct_slo_strict_1s"), timeouts=d.get("timeout"),
                          worst_tenant_1s=w1, tenants_con_demanda=nt,
                          goodput_bruto=d.get("goodput"), estable=d.get("stable")))

if not filas:
    raise SystemExit("sin datos")
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
    w.writeheader()
    for r in sorted(filas, key=lambda r: (r["system"], r["N"], r["lam"], r["rep"])):
        w.writerow(r)
print(f"escrito {OUT} ({len(filas)} puntos)\n")

NOM = {"bm": "native", "bmmps": "native+MPS", "ucx": "Gusto"}
CUMPLE = lambda r: (r["ttft_p95"] is not None and r["ttft_p95"] < 1000
                    and (r["timeouts"] or 0) == 0)

cel = defaultdict(list)
for r in filas:
    cel[(r["system"], r["N"], r["lam"])].append(r)

print("## Sweep aggregated over repetitions\n")
print("| system | N | lambda | reps | meets SLO | goodput median | goodput range | "
      "TTFT p95 median | worst tenant 1 s |")
print("|---|---:|---:|---:|:--:|---:|---|---:|---:|")
for k in sorted(cel, key=lambda k: (k[0], k[1], k[2])):
    v = cel[k]
    ok = sum(1 for r in v if CUMPLE(r))
    g = [r["goodput"] for r in v if r["goodput"] is not None]
    p = [r["ttft_p95"] for r in v if r["ttft_p95"] is not None]
    wt = [r["worst_tenant_1s"] for r in v if r["worst_tenant_1s"] is not None]
    print(f"| {NOM.get(k[0],k[0])} | {k[1]} | {k[2]:.2f} | {len(v)} | "
          f"{'**YES**' if ok == len(v) else ('partial %d/%d' % (ok, len(v)) if ok else 'no')} | "
          f"{st.median(g):.1f} | {min(g):.1f}-{max(g):.1f} | {st.median(p):.0f} ms | "
          f"{(st.median(wt) if wt else float('nan')):.0f} % |")

print("\n## Capacity under the SLO (a point counts only if it meets it in EVERY repetition)\n")
cap = defaultdict(list)
for k, v in cel.items():
    if v and all(CUMPLE(r) for r in v):
        cap[(k[0], k[1])].append((k[2], v))
print("| system | N | **max lambda** | per tenant | goodput at that load (median) | reps |")
print("|---|---:|---:|---:|---:|---:|")
best = {}
for k in sorted(cap, key=lambda k: (k[0], k[1])):
    lam, v = max(cap[k], key=lambda x: x[0])
    g = st.median([r["goodput"] for r in v])
    best[k] = (lam, g)
    print(f"| {NOM.get(k[0],k[0])} | {k[1]} | **{lam:.2f}** | {lam/k[1]:.3f} | {g:.1f} t/s | {len(v)} |")

def boot(d):
    if len(d) < 2:
        return (None, None)
    m = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(B))
    return m[int(0.025 * B)], m[int(0.975 * B)]

print("\n## Paired difference Gusto - native, at the knee\n")
print("| N | lambda | pairs | mean difference (t/s) | bootstrap CI95 | excludes zero |")
print("|---:|---:|---:|---:|---|:--:|")
for N in (2, 4, 8):
    kb, ku = ("bm", N), ("ucx", N)
    if kb not in best or ku not in best:
        continue
    lam = min(best[kb][0], best[ku][0])
    d = []
    for rep in (1, 2, 3):
        a = [r["goodput"] for r in cel.get(("ucx", N, lam), []) if r["rep"] == rep]
        b = [r["goodput"] for r in cel.get(("bm", N, lam), []) if r["rep"] == rep]
        if a and b and a[0] is not None and b[0] is not None:
            d.append(a[0] - b[0])
    if not d:
        continue
    lo, hi = boot(d)
    exc = (lo is not None and (lo > 0 or hi < 0))
    print(f"| {N} | {lam:.2f} | {len(d)} | **{sum(d)/len(d):+.1f}** | "
          f"{('[%+.1f, %+.1f]' % (lo, hi)) if lo is not None else 'n/d'} | "
          f"{'**YES**' if exc else 'no'} |")
