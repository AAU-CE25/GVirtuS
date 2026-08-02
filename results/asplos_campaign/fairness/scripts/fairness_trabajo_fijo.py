#!/usr/bin/env python3
"""Fairness de PROGRESO en los workloads de trabajo fijo.

Reglas aplicadas, todas ellas explicitas:

 1. Jain se calcula sobre PROGRESO NORMALIZADO, nunca sobre runtime crudo. Con runtime, un
    valor mayor es PEOR servicio, asi que el indice premia justo lo contrario de lo que
    se quiere medir.
        normalized_progress_i = solo_runtime / concurrent_runtime_i
        J = (sum p_i)^2 / (N * sum p_i^2)
 2. Jain se calcula POR CORRIDA (system, N, modo, seed) y luego se resume con la MEDIANA
    entre corridas. Nunca se juntan tenants de repeticiones distintas en un solo indice.
 3. solo_runtime es la MEDIANA de las corridas N=1 del MISMO sistema, mismo workload y
    misma metrica. Si no hay N=1 para ese sistema, la celda se marca sin solo y no se
    calcula slowdown.
 4. Se comprueba antes que el trabajo ofrecido sea identico entre tenants (`fixed_work_units`)
    y se mide la coordinacion de arranque (max t_start - min t_start). Una cohorte con
    arranque descoordinado no puede sostener una acusacion de unfairness.
 5. Se calculan DOS variantes: sobre tiempo interno del workload (excluye arranque) y sobre
    tiempo de pared (lo que el tenant vive). La diferencia entre ambas es diagnostica.
"""
import csv, os, statistics as st, json
from collections import defaultdict

H = os.path.expanduser("~")
IN = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness/tenants_canonico.csv")
OUTDIR = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness")
os.makedirs(OUTDIR, exist_ok=True)

# Normalizacion de nombres de sistema. Las variantes viven en arboles hermanos y el parseo
# por ruta las toma como "sistema"; se renombran y se marcan como variantes no canonicas.
ALIAS = {
    "results_tcp_final": ("tcp", "final"),
    "results_tcp_modoff": ("tcp", "modoff"),
    "results_mps_previo": ("baremetal_mps", "previo"),
    "results_diagnostics_night": ("diagnostics", "night"),
    "ucx_rdma": ("ucx_host_rma", ""),
    "ucx_gpudirect": ("gusto_gpudirect", ""),
    "rdma_zc": ("gusto_gpudirect", ""),
    "rdma": ("ucx_host_rma", ""),
    "ucx": ("gusto_am", ""),
}

def jain(v):
    v = [x for x in v if x is not None]
    if not v or sum(v) <= 0:
        return None
    return (sum(v) ** 2) / (len(v) * sum(x * x for x in v))

def f(x):
    try:
        return float(x)
    except Exception:
        return None

filas = list(csv.DictReader(open(IN)))
for r in filas:
    s = r["system"]
    if s in ALIAS:
        r["system"], r["variante"] = ALIAS[s]
    else:
        r["variante"] = ""

# ---- corridas: (workload, system, variante, N, modo, seed) -------------------------------
corridas = defaultdict(list)
for r in filas:
    if r["sospechoso"]:
        continue
    # cohort_path es lo unico unico: sin el, dos arboles hermanos con la misma semilla
    # se funden en una sola corrida y mezclan tenants de campanas distintas.
    k = (r["workload"], r["system"], r["variante"], int(r["N"]) if r["N"].isdigit() else -1,
         r["mode"], r["seed"], r["cohort_path"])
    corridas[k].append(r)

# ---- solo_runtime por (workload, system, variante, metrica) ------------------------------
solo = {}
for met in ("internal_runtime", "concurrent_runtime"):
    tmp = defaultdict(list)
    for k, rs in corridas.items():
        if k[3] != 1:
            continue
        for r in rs:
            v = f(r[met])
            if v is not None and v > 0:
                tmp[(k[0], k[1], k[2], met)].append(v)
    for kk, vv in tmp.items():
        solo[kk] = st.median(vv)

res = []
incompletas = []
rotas = []
for k, rs in sorted(corridas.items()):
    wl, sys_, var, N, mode, seed, coh = k
    if N <= 1:
        continue
    # Cohorte incompleta: faltan tenants respecto de N declarado. No se analiza, se cuenta.
    if len(rs) != N:
        incompletas.append((wl, sys_, N, mode, seed, len(rs)))
        continue
    # Cohorte con tiempos inutilizables (tenant que fallo al instante o sin figura).
    if any((f(r["concurrent_runtime"]) or 0) <= 0.01 for r in rs):
        rotas.append((wl, sys_, N, mode, seed))
        continue
    # ---- comprobaciones previas ----
    obras = {r["fixed_work_units"] for r in rs}
    trabajo_identico = (len(obras) == 1 and "missing" not in obras)
    ts = [f(r["t_start"]) for r in rs]
    ts = [x for x in ts if x is not None]
    spread = (max(ts) - min(ts)) if len(ts) == len(rs) and ts else None
    malos = sum(1 for r in rs if r["exit_status"] not in ("0", "missing"))
    incorrectos = sum(1 for r in rs if "valid=false" in str(r["correctness_status"]).lower())

    fila = {"workload": wl, "system": sys_, "variante": var, "N": N, "mode": mode,
            "seed": seed, "tenants_presentes": len(rs),
            "trabajo_identico": trabajo_identico,
            "start_spread_s": round(spread, 3) if spread is not None else "missing",
            "tenants_exit_no_cero": malos, "tenants_incorrectos": incorrectos}

    for met, etq in (("internal_runtime", "int"), ("concurrent_runtime", "wall")):
        s0 = solo.get((wl, sys_, var, met))
        vals = [f(r[met]) for r in rs]
        if s0 is None or any(v is None or v <= 0 for v in vals) or len(vals) != N:
            fila[f"jain_{etq}"] = "missing"
            fila[f"peor_slowdown_{etq}"] = "missing"
            fila[f"mediana_slowdown_{etq}"] = "missing"
            fila[f"ratio_lento_rapido_{etq}"] = "missing"
            fila[f"cv_{etq}"] = "missing"
            continue
        prog = [s0 / v for v in vals]
        slow = [v / s0 for v in vals]
        fila[f"jain_{etq}"] = round(jain(prog), 4)
        fila[f"peor_slowdown_{etq}"] = round(max(slow), 3)
        fila[f"mediana_slowdown_{etq}"] = round(st.median(slow), 3)
        fila[f"ratio_lento_rapido_{etq}"] = round(max(slow) / min(slow), 3)
        m = st.mean(prog)
        fila[f"cv_{etq}"] = round(st.pstdev(prog) / m, 4) if m else "missing"
        fila[f"cohort_time_{etq}"] = round(max(vals), 3)
    res.append(fila)

cols = list(res[0].keys()) if res else []
p = os.path.join(OUTDIR, "fairness_trabajo_fijo_por_corrida.csv")
with open(p, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in res:
        w.writerow(r)
print(f"escrito {p} ({len(res)} corridas)\n")

# ---- resumen por (workload, system, N, modo): MEDIANA entre corridas ---------------------
ag = defaultdict(list)
for r in res:
    ag[(r["workload"], r["system"], r["variante"], r["N"], r["mode"])].append(r)

print("| workload | sistema | N | modo | corridas | Jain int (mediana) | Jain wall | "
      "peor slowdown wall | mediana slowdown wall | lento/rapido wall | start spread s |")
print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
resumen = []
for k, rs in sorted(ag.items()):
    def med(campo):
        v = [r[campo] for r in rs if isinstance(r.get(campo), (int, float))]
        return round(st.median(v), 4) if v else None
    sp = [r["start_spread_s"] for r in rs if isinstance(r.get("start_spread_s"), (int, float))]
    fila = {"workload": k[0], "system": k[1], "variante": k[2], "N": k[3], "mode": k[4],
            "corridas": len(rs),
            "jain_int_mediana": med("jain_int"), "jain_wall_mediana": med("jain_wall"),
            "peor_slowdown_wall_mediana": med("peor_slowdown_wall"),
            "mediana_slowdown_wall_mediana": med("mediana_slowdown_wall"),
            "ratio_lento_rapido_wall_mediana": med("ratio_lento_rapido_wall"),
            "start_spread_s_mediana": round(st.median(sp), 3) if sp else None,
            "corridas_trabajo_identico": sum(1 for r in rs if r["trabajo_identico"])}
    resumen.append(fila)
    def q(x):
        return "missing" if x is None else x
    print(f"| {k[0]} | {k[1]}{('/'+k[2]) if k[2] else ''} | {k[3]} | {k[4]} | {len(rs)} | "
          f"{q(fila['jain_int_mediana'])} | {q(fila['jain_wall_mediana'])} | "
          f"{q(fila['peor_slowdown_wall_mediana'])} | {q(fila['mediana_slowdown_wall_mediana'])} | "
          f"{q(fila['ratio_lento_rapido_wall_mediana'])} | {q(fila['start_spread_s_mediana'])} |")

p2 = os.path.join(OUTDIR, "fairness_trabajo_fijo_resumen.csv")
with open(p2, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()), extrasaction="ignore")
    w.writeheader()
    for r in resumen:
        w.writerow(r)
print(f"\nescrito {p2}")

print("\n--- cohortes descartadas y por que (no se silencian) ---")
print(f"incompletas (menos tenants que N declarado): {len(incompletas)}")
for x in sorted(set(incompletas))[:12]:
    print("   ", x)
print(f"con tiempos inutilizables (algun tenant <=0.01 s): {len(rotas)}")
for x in sorted(set(rotas))[:12]:
    print("   ", x)
