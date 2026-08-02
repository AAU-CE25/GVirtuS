#!/usr/bin/env python3
"""Cierre de la Tabla C: token_service_fraction por tenant, comparacion emparejada
Gusto-nativo, intervalos de confianza y test de equivalencia, y supresion del Jain de SLO
donde casi todo vale cero.

REGLA DE SUPRESION DEL JAIN DE SLO. El indice de Jain sobre una magnitud casi toda nula no
mide equidad: mide que nadie recibe servicio. Se declara SIN POTENCIA cuando
    (a) menos de la mitad de los tenants tiene atencion de SLO no nula, o
    (b) la atencion media es < 5 %.
En esos casos se informa la inanicion, no un numero de equidad.

EMPAREJAMIENTO. Cada celda (N, lambda, repeticion) existe en Gusto y en nativo con la misma
carga nominal, asi que la comparacion es pareada y elimina el efecto de N y de lambda. Sobre
las diferencias pareadas se da un IC del 95 % por bootstrap (10 000 remuestreos) y un test de
equivalencia TOST con margen declarado.

MARGEN DE EQUIVALENCIA: +-0.05 en Jain. Es una eleccion, no un dato; se declara para que el
lector pueda discrepar. Con el numero de pares disponible el test tiene poca potencia y eso
se dice explicitamente en la salida.
"""
import csv, os, random, statistics as st
from collections import defaultdict

random.seed(20260802)
OUT = os.path.expanduser("~/GVirtuS/results/asplos_campaign/fairness")
POR_TEN = os.path.join(OUT, "llama_fairness_por_tenant.csv")
POR_COR = os.path.join(OUT, "llama_fairness_por_corrida.csv")
MARGEN = 0.05
B = 10000

ten = list(csv.DictReader(open(POR_TEN)))
cor = list(csv.DictReader(open(POR_COR)))

def fl(x):
    try:
        return float(x)
    except Exception:
        return None

def jain(v):
    v = [x for x in v if x is not None]
    if not v or sum(v) <= 0:
        return None
    return (sum(v) ** 2) / (len(v) * sum(x * x for x in v))

def boot_ic(d, B=B):
    if len(d) < 2:
        return (None, None)
    m = []
    for _ in range(B):
        s = [random.choice(d) for _ in d]
        m.append(sum(s) / len(s))
    m.sort()
    return (m[int(0.025 * B)], m[int(0.975 * B)])

# ---------------- 1. token_service_fraction por tenant -----------------------------------
print("## C.1 — token_service_fraction por tenant\n")
print("Formula: tokens_completados_i / (peticiones_i * NPRED), con NPRED=128 fijo en todas")
print("las corridas multi-tenant. Fuente: llama_fairness_por_tenant.csv, columnas")
print("`completed_output_tokens` y `offered_output_tokens`.\n")
print("| sistema | N | lambda | rep | est | tsf peor | tsf mediana | tsf mejor | "
      "Jain tsf | tsf peor/mejor |")
print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
g = defaultdict(list)
for r in ten:
    g[(r["system"], int(r["N"]), float(r["lambda_total"]), int(r["rep"]), r["stable"])].append(r)
tsf_por_celda = {}
for k in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3])):
    v = sorted(fl(r["token_service_fraction"]) for r in g[k])
    j = jain(v)
    tsf_por_celda[k] = v
    print(f"| {k[0]} | {k[1]} | {k[2]:.1f} | {k[3]} | {'S' if k[4]=='STABLE' else 'U'} | "
          f"{v[0]:.4f} | {st.median(v):.4f} | {v[-1]:.4f} | "
          f"{(round(j,4) if j else 'n/d')} | {(v[-1]/v[0] if v[0] > 0 else float('inf')):.2f} |")

# ---------------- 2. Jain de SLO: supresion donde no hay potencia -------------------------
print("\n## C.2 — Jain de SLO: donde tiene sentido y donde no\n")
print("| sistema | N | lambda | rep | tenants con SLO>0 | atencion media | Jain SLO 5 s |")
print("|---|---:|---:|---:|---:|---:|---|")
for k in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3])):
    s5 = [fl(r["slo_5s_fraction"]) for r in g[k]]
    nz = sum(1 for x in s5 if x > 0)
    media = sum(s5) / len(s5)
    if nz < len(s5) / 2 or media < 0.05:
        veredicto = f"**SIN POTENCIA** (inanicion: {nz}/{len(s5)} tenants, media {media:.3f})"
    else:
        veredicto = f"{jain(s5):.4f}"
    print(f"| {k[0]} | {k[1]} | {k[2]:.1f} | {k[3]} | {nz}/{len(s5)} | {media:.3f} | {veredicto} |")

# ---------------- 3. comparacion emparejada Gusto vs nativo -------------------------------
print("\n## C.3 — comparacion emparejada Gusto frente a nativo\n")
idx = {}
for r in cor:
    idx[(r["system"], int(r["N"]), float(r["lambda_total"]), int(r["rep"]))] = r
pares = []
for (s, N, lam, rep), r in idx.items():
    if s != "gusto":
        continue
    o = idx.get(("nativo", N, lam, rep))
    if o:
        pares.append((N, lam, rep, r, o))
pares.sort()
print(f"Pares emparejados por (N, lambda, repeticion): **{len(pares)}**.\n")
print("| N | lambda | rep | est | Jain compl. Gusto | nativo | dif | "
      "compl. media Gusto | nativo | dif |")
print("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")

def compl_media(sistema, N, lam, rep):
    v = [fl(r["completion_fraction"]) for r in ten
         if r["system"] == sistema and int(r["N"]) == N
         and abs(float(r["lambda_total"]) - lam) < 1e-9 and int(r["rep"]) == rep]
    return sum(v) / len(v) if v else None

dif_jain, dif_compl = [], []
for N, lam, rep, rg, rn in pares:
    jg, jn = fl(rg["jain_completion_fraction"]), fl(rn["jain_completion_fraction"])
    cg, cn = compl_media("gusto", N, lam, rep), compl_media("nativo", N, lam, rep)
    if None in (jg, jn, cg, cn):
        continue
    dif_jain.append(jg - jn); dif_compl.append(cg - cn)
    print(f"| {N} | {lam:.1f} | {rep} | {'S' if rg['stable']=='STABLE' else 'U'} | "
          f"{jg:.4f} | {jn:.4f} | {jg-jn:+.4f} | {cg:.4f} | {cn:.4f} | {cg-cn:+.4f} |")

# ---------------- 4. IC bootstrap y equivalencia -----------------------------------------
print("\n## C.4 — intervalos de confianza y equivalencia\n")
for nom, d, marg in (("Jain de completion fraction", dif_jain, MARGEN),
                     ("completion fraction media", dif_compl, None)):
    if len(d) < 2:
        print(f"- **{nom}**: menos de 2 pares, no calculable.")
        continue
    m = sum(d) / len(d)
    lo, hi = boot_ic(d)
    print(f"- **{nom}** (Gusto menos nativo), n={len(d)} pares: "
          f"media **{m:+.4f}**, IC95 bootstrap **[{lo:+.4f}, {hi:+.4f}]**.")
    if marg is not None:
        equiv = (lo > -marg) and (hi < marg)
        print(f"  - TOST con margen +-{marg}: {'**EQUIVALENTES**' if equiv else 'no concluyente'}"
              f" (el IC {'cabe' if equiv else 'NO cabe'} dentro de [-{marg}, +{marg}]).")
        print(f"  - Potencia: con n={len(d)} pares el test es debil; un IC que quepa en el "
              f"margen es evidencia de equivalencia, uno que no quepa NO es evidencia de "
              f"diferencia.")
