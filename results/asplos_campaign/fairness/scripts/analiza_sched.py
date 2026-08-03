#!/usr/bin/env python3
"""N1: por que el servicio llega en rafagas, y si el stream por conexion lo arregla.

Dos brazos identicos salvo una variable del BACKEND:
  stream0  GVS_PER_CONN_STREAM=0  -> stream legacy compartido por todas las conexiones
  stream1  GVS_PER_CONN_STREAM=1  -> un stream propio por conexion

CUANTO DE SERVICIO. Se estima como el MINIMO tiempo de iteracion observado en el brazo: la
iteracion mas rapida es, por definicion, la que no espero a nadie. Se autocalibra y evita
depender de una corrida N=1 con este build instrumentado.

  multiplicador(i,k) = iteracion(i,k) / cuanto

Con N tenants y reparto equitativo todas las iteraciones deberian rondar N. Un 1 significa
servido sin esperar; una mezcla de 1 y de valores altos es una rafaga.
"""
import re, io, os, glob, statistics as st
from collections import defaultdict

D = os.path.expanduser("~/GVirtuS/results/asplos_campaign/sched")

def lee(f):
    s = io.open(f, errors="replace").read()
    m = re.search(r"raw_iterations:\s*\[([^\]]*)\]", s)
    if not m:
        return None
    return [float(x) for x in m.group(1).split(",")]

brazos = {}
for arm in ("stream0", "stream1"):
    coh = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(D, f"{arm}_r*_t*.log"))):
        mm = re.search(r"_r(\d+)_t(\d+)\.log$", f)
        if not mm:
            continue
        v = lee(f)
        if v:
            coh[int(mm.group(1))][int(mm.group(2))] = v
    brazos[arm] = {r: t for r, t in coh.items() if len(t) == 8}

print("## N1 — service-quantum sharing, shared legacy stream against per-connection streams\n")
print("| arm | cohorts | quantum (ms) | mult. median | mult. max | "
      "**iterations with no wait** | slowest/fastest |")
print("|---|---:|---:|---:|---:|---:|---:|")
res = {}
for arm, coh in brazos.items():
    if not coh:
        print(f"| {arm} | 0 | — | — | — | — | — |")
        continue
    todas = [x for t in coh.values() for v in t.values() for x in v]
    q = min(todas)
    mult = [x / q for x in todas]
    sinesp = sum(1 for m in mult if m < 1.5)
    lr = []
    for t in coh.values():
        avg = {k: st.mean(v[2:]) for k, v in t.items()}   # miniBUDE descarta 2 de calentamiento
        lr.append(max(avg.values()) / min(avg.values()))
    res[arm] = dict(q=q, med=st.median(mult), mx=max(mult), sin=sinesp,
                    tot=len(mult), lr=st.median(lr))
    print(f"| {arm} | {len(coh)} | {q:.1f} | {st.median(mult):.2f} | {max(mult):.2f} | "
          f"**{sinesp} of {len(mult)}** ({100*sinesp/len(mult):.0f}%) | {st.median(lr):.2f} |")

if "stream0" in res and "stream1" in res:
    a, b = res["stream0"], res["stream1"]
    print(f"\n**Verdict.** slowest/fastest goes {a['lr']:.2f} -> {b['lr']:.2f}; "
          f"iterations served with no wait {100*a['sin']/a['tot']:.0f}% -> "
          f"{100*b['sin']/b['tot']:.0f}%; median multiplier {a['med']:.2f} -> {b['med']:.2f}.")
    if b["lr"] < a["lr"] * 0.6:
        print("The per-connection stream **removes most of the burst**: the hypothesis holds — "
              "the sharing was decided by submission order into one FIFO stream.")
    elif b["lr"] > a["lr"] * 0.9:
        print("The per-connection stream **changes nothing**. The hypothesis is REFUTED: the "
              "burst does not come from sharing the legacy stream, and the cause is elsewhere.")
    else:
        print("Partial effect: the stream explains part of the burst but not all of it.")

print("\n### One cohort, tenant by tenant (multiplier per iteration)\n")
for arm, coh in brazos.items():
    if not coh:
        continue
    r = sorted(coh)[0]
    todas = [x for t in coh.values() for v in t.values() for x in v]
    q = min(todas)
    print(f"**{arm}**, cohort r{r}:\n")
    print("| tenant | slowdown | multipliers |")
    print("|---:|---:|---|")
    for tid in sorted(coh[r]):
        v = coh[r][tid]
        print(f"| {tid} | {st.mean(v[2:])/q:.2f} | `" +
              " ".join(f"{x/q:.0f}" for x in v) + "` |")
    print()
