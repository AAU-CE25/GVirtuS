#!/usr/bin/env python3
"""Recomputa el makespan de cohorte de XSBench desde los crudos por tenant, para arbitrar
entre las dos tablas que conviven en XSBENCH_RESULTS.md.

makespan de cohorte = max(t_end) - min(t_start) sobre los N clientes de la cohorte.
Es la definicion que corresponde a "tiempo de pared de la cohorte completa"; se da tambien
max(completion_s) por si la otra tabla usaba esa.

Cohorte valida = N clientes presentes Y los N con linea `Runtime:` en stdout (la propia
condicion que el documento exige: `done=N/N` no basta).
"""
import csv, os, statistics as st
from collections import defaultdict

H = os.path.expanduser("~")
IN = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness/tenants_canonico.csv")
rs = [r for r in csv.DictReader(open(IN)) if r["workload"] == "xsbench" and not r["sospechoso"]]

coh = defaultdict(list)
for r in rs:
    coh[(r["system"], int(r["N"]), r["mode"], r["seed"], r["cohort_path"])].append(r)

print("| sistema | N | cohortes validas | makespan mediana | min | max | "
      "max(completion_s) mediana |")
print("|---|---:|---:|---:|---:|---:|---:|")
ag = defaultdict(list)
for k, v in coh.items():
    sys_, N, mode, seed, path = k
    if mode != "sync" or len(v) != N:
        continue
    if any(x["internal_runtime"] in ("missing", "") for x in v):
        continue          # algun cliente no imprimio Runtime: cohorte incompleta
    try:
        t0 = min(float(x["t_start"]) for x in v)
        t1 = max(float(x["t_end"]) for x in v)
        cs = max(float(x["concurrent_runtime"]) for x in v)
    except Exception:
        continue
    ag[(sys_, N)].append((t1 - t0, cs))

for k in sorted(ag, key=lambda k: (k[0], k[1])):
    v = ag[k]
    ms = [a for a, _ in v]
    cs = [b for _, b in v]
    print(f"| {k[0]} | {k[1]} | {len(v)} | **{st.median(ms):.2f}** | {min(ms):.2f} | "
          f"{max(ms):.2f} | {st.median(cs):.2f} |")

print("\n## Fairness por tenant de XSBench (lo que falta en el paquete)\n")
print("| sistema | N | cohortes | Jain (progreso norm.) | lento/rapido | "
      "Mlookups/s peor | mediana | mejor |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
solo = {}
for k, v in coh.items():
    if k[1] == 1 and k[2] == "sync":
        for x in v:
            if x["internal_runtime"] not in ("missing", ""):
                solo.setdefault(k[0], []).append(float(x["internal_runtime"]))
solo = {a: st.median(b) for a, b in solo.items()}

filas = []
por = defaultdict(list)
for k, v in coh.items():
    sys_, N, mode, seed, path = k
    if mode != "sync" or len(v) != N or N == 1 or sys_ not in solo:
        continue
    if any(x["internal_runtime"] in ("missing", "") for x in v):
        continue
    q = solo[sys_]
    prog = [q / float(x["internal_runtime"]) for x in v]
    lk = sorted(float(x["figure_of_merit"]) / 1e6 for x in v
                if x["figure_of_merit"] not in ("missing", ""))
    j = (sum(prog) ** 2) / (len(prog) * sum(p * p for p in prog))
    por[(sys_, N)].append((j, max(prog) / min(prog), lk))
    for x in v:
        filas.append(dict(system=sys_, N=N, seed=seed, cohort=path,
                          tenant_id=x["tenant_id"],
                          runtime_s=x["internal_runtime"],
                          wall_s=x["concurrent_runtime"],
                          lookups_per_s=x["figure_of_merit"],
                          normalized_progress=round(q / float(x["internal_runtime"]), 4),
                          slowdown=round(float(x["internal_runtime"]) / q, 4)))

for k in sorted(por, key=lambda k: (k[0], k[1])):
    v = por[k]
    js = [a for a, _, _ in v]
    rr = [b for _, b, _ in v]
    lk = [c for _, _, c in v if c]
    peor = st.median([min(c) for c in lk]) if lk else float("nan")
    med = st.median([st.median(c) for c in lk]) if lk else float("nan")
    mej = st.median([max(c) for c in lk]) if lk else float("nan")
    print(f"| {k[0]} | {k[1]} | {len(v)} | {st.median(js):.4f} | {st.median(rr):.3f} | "
          f"{peor:.1f} | {med:.1f} | {mej:.1f} |")

OUT = os.path.expanduser("~/paper/XSBench_fairness_por_tenant.csv")
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
    w.writeheader()
    for f in sorted(filas, key=lambda r: (r["system"], r["N"], r["seed"], int(r["tenant_id"]))):
        w.writerow(f)
print(f"\nescrito {OUT} ({len(filas)} filas por tenant)")
