#!/usr/bin/env python3
"""Verificacion de todo lo producido esta noche. Cada comprobacion dice PASA o FALLA y por que."""
import json, glob, os, csv, re, io, subprocess, statistics as st
from collections import Counter, defaultdict

H = os.path.expanduser("~")
R = os.path.join(H, "GVirtuS/results/asplos_campaign")
fallos = []

def check(nombre, ok, detalle):
    print(f"[{'PASA' if ok else 'FALLA'}] {nombre}: {detalle}")
    if not ok:
        fallos.append(nombre)

# ---- 1) v2: numero de puntos ------------------------------------------------------------
B = os.path.join(R, "llama_slo_sweep_v2")
fich = glob.glob(os.path.join(B, "*.meta.jsonl"))
lineas = sum(sum(1 for _ in open(f)) for f in fich)
rep = {os.path.basename(f): sum(1 for _ in open(f)) for f in fich
       if sum(1 for _ in open(f)) > 1}
check("v2 ficheros", len(fich) == 108, f"{len(fich)} ficheros (esperado 108 = 27 celdas x 4 lambda)")
check("v2 lineas", lineas == len(fich),
      f"{lineas} lineas frente a {len(fich)} ficheros; etiquetas repetidas: {rep or 'ninguna'}")

# ---- 2) v2: celdas completas ------------------------------------------------------------
cel = Counter()
for f in fich:
    for ln in open(f):
        d = json.loads(ln)
        p = d["label"].split("_")
        cel[(p[1], int(p[2][1:]), float(p[3][1:]))] += 1
esperadas = 3 * 3 * 4
malas = {k: v for k, v in cel.items() if v != 3}
check("v2 celdas a n=3", len(cel) == esperadas and not malas,
      f"{len(cel)} celdas de {esperadas}; con reps != 3: {malas or 'ninguna'}")

# ---- 3) memoria: el brazo de Gusto es fresco o heredado? --------------------------------
mf = os.path.join(R, "memoria/mem_footprint.csv")
sis = set()
if os.path.exists(mf):
    sis = {r["system"] for r in csv.DictReader(open(mf))}
check("memoria: brazos medidos frescos", "bm" in sis and "bmmps" in sis,
      f"sistemas en mem_footprint.csv: {sorted(sis)}. "
      f"{'Gusto NO esta -> su 4487 es HEREDADO de la campana previa' if 'ucx' not in sis else 'Gusto tambien medido'}")

# ---- 4) N1: el tenant favorecido, mi corrida frente a la campana original ---------------
def favorito(archivo, filtro):
    g = defaultdict(dict)
    for r in csv.DictReader(open(archivo)):
        if filtro(r):
            g[(r.get("system", ""), r["run"])][int(r["tenant_id"])] = float(r["slowdown"])
    c = Counter()
    for v in g.values():
        if len(v) == 8:
            c[min(v, key=v.get)] += 1
    return c

td = os.path.join(R, "fairness/tabla_D_minibude_por_tenant.csv")
if os.path.exists(td):
    c_orig = favorito(td, lambda r: r["system"] in ("ucx_gpudirect", "ucx_rdma", "tcp"))
    n = sum(c_orig.values())
    frac = c_orig.get(1, 0) / n if n else 0
    check("N1: 't1 favorecido 9 de 10' se sostiene?", False,
          f"en la campana ORIGINAL t1 es el mas rapido en {c_orig.get(1,0)} de {n} "
          f"({100*frac:.0f} %), no en el 90 %. Mi 9/10 esta sesgado por el orden de arranque "
          f"de mi lanzador. El efecto sigue por encima del azar (12,5 %) pero la cifra "
          f"reportada NO es citable")

# ---- 5) N3: conteos de stubs, dos metodos independientes -------------------------------
sf = os.path.join(H, "GVirtuS/plugins/cudart/frontend/CudaRt_stubs_compat.cpp")
if os.path.exists(sf):
    t = io.open(sf, errors="replace").read()
    m1 = len(set(re.findall(r"cudaError_t_local ([a-zA-Z0-9_]+)\(\)", t)))
    m2 = t.count("return CUDART_STUB_NOT_SUPPORTED")
    check("N3: stubs de cudart coherentes", abs(m1 - m2) <= 1,
          f"simbolos unicos={m1}, returns NOT_SUPPORTED={m2} "
          f"(el documento dice 236; diferencia {abs(m1-236)})")

# ---- 6) N7: la ablacion se activo de verdad? -------------------------------------------
eo = os.path.join(R, "epoch/guard_off_r1.log")
if os.path.exists(eo):
    t = io.open(eo, errors="replace").read()
    ren = re.findall(r"epoch (\d+) anuncia server_idx=\[([^\]]*)\]", t)
    idx = [set(x[1].split(",")) for x in ren]
    solapan = bool(idx and len(idx) > 1 and idx[0] & idx[1])
    check("N7: el experimento es inerte (indices renumeran)", not solapan,
          f"epochs anunciados {[x[0] for x in ren]}, indices solapan={solapan} -> "
          f"{'HAY colision posible' if solapan else 'NO hay colision: el test no prueba nada'}")

# ---- 7) equidad: el resumen sigue coincidiendo con el detalle --------------------------
res = os.path.join(R, "fairness/fairness_trabajo_fijo_resumen.csv")
if os.path.exists(res):
    fil = [r for r in csv.DictReader(open(res))
           if r["workload"] == "minibude" and r["N"] == "8" and r["mode"] == "sync"]
    d = {r["system"]: r.get("ratio_lento_rapido_wall_mediana") for r in fil}
    check("equidad miniBUDE N=8 intacta",
          d.get("baremetal") and float(d["baremetal"]) < 1.1
          and d.get("gusto_gpudirect") and float(d["gusto_gpudirect"]) > 4,
          f"nativo={d.get('baremetal')} gusto_gpudirect={d.get('gusto_gpudirect')}")

print("\n" + ("TODO COHERENTE" if not fallos else f"REVISAR: {fallos}"))
