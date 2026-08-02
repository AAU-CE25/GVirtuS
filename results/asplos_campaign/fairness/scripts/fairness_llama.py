#!/usr/bin/env python3
"""Tabla C: fairness de SERVICIO en llama, normalizada por demanda.

PROCEDENCIA Y LIMITES, primero:

 - Por peticion: ~/results/<label>.jsonl, campos {label, server, ttft_ms, ntok, ok,
   timeout, tpot_ms_mean}. NO hay marca de tiempo por peticion en esta campana
   (tc_rel_s/tarr_rel_s se anadieron el 2026-08-02, despues). Consecuencia: NO se puede
   calcular "completions durante la ventana" frente a "durante el drenaje", ni primera
   completion, ni intervalo maximo sin progreso. Esos campos quedan en missing.

 - bench.py abre el JSONL en modo APPEND y las repeticiones comparten `label`, de modo que
   las 3 corridas de cada celda estan CONCATENADAS. Se segmentan por conteos acumulados
   len(inw) = completed + fail de ~/results/summary.csv. Verificado: la suma coincide
   EXACTAMENTE con las lineas del fichero en las 10 etiquetas comprobadas, con conteos no
   uniformes ([310,306,307]), asi que la segmentacion no es una coincidencia. Supone que el
   orden de append es el de las filas del CSV; ambos los escribe el mismo proceso en orden
   cronologico.

 - `requests_i` = peticiones ASIGNADAS al tenant i cuya finalizacion cayo en la ventana de
   conteo. bench.py elige servidor con random.choice(SERVERS) en la llegada y no registra
   un contador de ofrecidas por tenant, asi que esto es lo mas cercano a "offered_i" que
   existe. No es exactamente la demanda ofrecida: una peticion lanzada dentro de la ventana
   cuya finalizacion cae fuera no aparece.

 - offered_output_tokens_i = requests_i * NPRED, con NPRED=128 fijo en todas las corridas
   multi-tenant (mt_pods*.sh). Formula exacta, no estimacion.

LINEA NULA: permutacion. Dentro de cada corrida se barajan las ETIQUETAS de tenant sobre las
peticiones, conservando el numero de peticiones por tenant, y se recalcula el Jain. Eso da la
distribucion de Jain bajo "el servicio es independiente de la identidad del tenant",
condicionada a la demanda REAL observada. No supone llegadas multinomiales ni tiempos de
servicio iguales, que es justo lo que la formula n/(n+k-1) da por hecho y aqui no se verifica.
"""
import csv, json, os, random, statistics as st
from collections import defaultdict

R = os.path.expanduser("~/results")
OUT = os.path.expanduser("~/GVirtuS/results/asplos_campaign/fairness")
os.makedirs(OUT, exist_ok=True)
NPRED = 128
B = 2000
random.seed(20260802)

ETIQUETAS = ["mt_ucx_n2_l2", "mtbm_n2_l2", "mt_ucx_n4_l4", "mtbm_n4_l4",
             "mt_ucx_n8_l8", "mtbm_n8_l8",
             "mt_ucx_n2_l1.0", "mtbm_n2_l1.0", "mt_ucx_n4_l1.0", "mtbm_n4_l1.0",
             "mt_ucx_n8_l1.0", "mtbm_n8_l1.0", "mt_ucx_n1_l1.0", "mtbm_n1_l1.0",
             "mt_ucx_gpudirect_n8_l1.0", "mt_ucx_gpudirect_n10_l1.0",
             "mt_ucx_gpudirect_n6_l1.0", "mt_ucx_gpudirect_n4_l1.0",
             "mt_ucx_gpudirect_n2_l1.0", "mt_ucx_gpudirect_n1_l1.0"]

def jain(v):
    v = list(v)
    if not v or sum(v) <= 0:
        return None
    return (sum(v) ** 2) / (len(v) * sum(x * x for x in v))

def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p / 100.0 * len(a)))] if a else None

sumario = {r["label"]: [] for r in csv.DictReader(open(os.path.join(R, "summary.csv")))}
for r in csv.DictReader(open(os.path.join(R, "summary.csv"))):
    sumario[r["label"]].append(r)

por_corrida, por_tenant = [], []

for lbl in ETIQUETAS:
    p = os.path.join(R, lbl + ".jsonl")
    if not os.path.exists(p) or lbl not in sumario:
        continue
    lineas = [json.loads(x) for x in open(p)]
    tramos = [int(r["completed"]) + int(r["fail"]) for r in sumario[lbl]]
    if sum(tramos) != len(lineas):
        print(f"[AVISO] {lbl}: no segmentable ({sum(tramos)} vs {len(lineas)}); se omite")
        continue
    sistema = "gusto" if lbl.startswith("mt_ucx") else "nativo"
    if "gpudirect" in lbl:
        sistema = "gusto_gpudirect"
    N = int(sumario[lbl][0]["n_servers"])
    lam = float(sumario[lbl][0]["rate"])

    off = 0
    for rep, k in enumerate(tramos, 1):
        seg = lineas[off:off + k]; off += k
        est = sumario[lbl][rep - 1]["stable"]
        g = defaultdict(list)
        for x in seg:
            g[x["server"]].append(x)
        if len(g) == 0:
            continue

        met = {}
        for s, xs in sorted(g.items()):
            n_i = len(xs)
            comp = [x for x in xs if x["ok"]]
            tmo = [x for x in xs if x.get("timeout")]
            tt = [x["ttft_ms"] for x in comp if x["ttft_ms"] is not None]
            tok_ok = sum(x["ntok"] for x in comp)
            tok_all = sum(x["ntok"] for x in xs)
            met[s] = dict(
                requests=n_i, completed=len(comp), timeouts=len(tmo),
                failed=n_i - len(comp),
                completion_fraction=len(comp) / n_i,
                offered_output_tokens=n_i * NPRED,
                completed_output_tokens=tok_ok,
                delivered_output_tokens=tok_all,
                token_service_fraction=tok_ok / (n_i * NPRED),
                slo_1s_fraction=sum(1 for t in tt if t <= 1000) / n_i,
                slo_5s_fraction=sum(1 for t in tt if t <= 5000) / n_i,
                timeout_fraction=len(tmo) / n_i,
                ttft_p50=pct(tt, 50), ttft_p95=pct(tt, 95), ttft_p99=pct(tt, 99),
                tpot_p50=pct([x["tpot_ms_mean"] for x in comp if x.get("tpot_ms_mean")], 50),
            )
        tot_req = sum(m["requests"] for m in met.values())
        tot_tok = sum(m["completed_output_tokens"] for m in met.values())
        for s, m in met.items():
            m["offered_share"] = m["requests"] / tot_req if tot_req else None
            m["service_share"] = (m["completed_output_tokens"] / tot_tok) if tot_tok else None
            m["share_normalizada"] = ((m["service_share"] / m["offered_share"])
                                      if m["offered_share"] else None)
            por_tenant.append(dict(workload="llama7b", system=sistema, N=N, lambda_total=lam,
                                   label=lbl, rep=rep, stable=est, tenant=s, **m))

        # desbalance de demanda: no es del sistema, es del sorteo de llegadas
        rq = [m["requests"] for m in met.values()]
        fila = dict(workload="llama7b", system=sistema, N=N, lambda_total=lam, label=lbl,
                    rep=rep, stable=est, tenants=len(met),
                    req_min=min(rq), req_max=max(rq),
                    desbalance_demanda=round(max(rq) / min(rq), 3) if min(rq) else None,
                    jain_demanda=round(jain(rq), 4))
        for campo in ("completion_fraction", "token_service_fraction",
                      "slo_1s_fraction", "slo_5s_fraction", "share_normalizada"):
            v = [met[s][campo] for s in met if met[s][campo] is not None]
            j = jain(v) if v else None
            # Jain es 0/0 cuando TODOS los tenants valen cero. Eso no es un fallo del
            # calculo: es inanicion uniforme, y hay que decirlo con esa palabra en vez de
            # dejar un hueco que luego se lea como \"no medido\".
            if j is None:
                fila["jain_" + campo] = ("inanicion_uniforme" if v and max(v) == 0
                                         else "missing")
            else:
                fila["jain_" + campo] = round(j, 4)
        cf = [met[s]["completion_fraction"] for s in met]
        fila["completion_fraction_min"] = round(min(cf), 4)
        fila["completion_fraction_max"] = round(max(cf), 4)
        s5 = [met[s]["slo_5s_fraction"] for s in met]
        fila["slo_5s_min"] = round(min(s5), 4)
        fila["slo_5s_max"] = round(max(s5), 4)
        tf = [met[s]["timeout_fraction"] for s in met]
        fila["timeout_fraction_min"] = round(min(tf), 4)
        fila["timeout_fraction_max"] = round(max(tf), 4)

        # ---- linea nula por permutacion, condicionada a la demanda real ----
        etiquetas = []
        for s, xs in g.items():
            etiquetas += [s] * len(xs)
        for campo, clave in (("completion_fraction", "ok"), ("slo_5s_fraction", "slo5")):
            obs = fila["jain_" + campo]
            if not isinstance(obs, float):
                continue
            exitos = []
            for x in seg:
                if clave == "ok":
                    exitos.append(1 if x["ok"] else 0)
                else:
                    exitos.append(1 if (x["ok"] and x["ttft_ms"] is not None
                                        and x["ttft_ms"] <= 5000) else 0)
            nulos = []
            for _ in range(B):
                random.shuffle(etiquetas)
                acc = defaultdict(lambda: [0, 0])
                for e, ok_ in zip(etiquetas, exitos):
                    acc[e][0] += ok_; acc[e][1] += 1
                j = jain([a / b for a, b in acc.values()])
                if j is not None:
                    nulos.append(j)
            if nulos:
                nulos.sort()
                fila["null_" + campo + "_p50"] = round(pct(nulos, 50), 4)
                fila["null_" + campo + "_p05"] = round(pct(nulos, 5), 4)
                # p de una cola: cuantas permutaciones dan un Jain <= al observado
                fila["p_" + campo] = round(sum(1 for x in nulos if x <= obs) / len(nulos), 4)
        por_corrida.append(fila)

c1 = os.path.join(OUT, "llama_fairness_por_corrida.csv")
with open(c1, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(por_corrida[0].keys()), extrasaction="ignore")
    w.writeheader()
    for r in por_corrida:
        w.writerow(r)
c2 = os.path.join(OUT, "llama_fairness_por_tenant.csv")
with open(c2, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(por_tenant[0].keys()), extrasaction="ignore")
    w.writeheader()
    for r in por_tenant:
        w.writerow(r)
print(f"escrito {c1} ({len(por_corrida)} corridas)")
print(f"escrito {c2} ({len(por_tenant)} filas tenant-corrida)\n")

print("## Tabla C — fairness de servicio en llama, normalizada por demanda\n")
print("| sistema | N | lambda | rep | est | desbal. demanda | Jain compl. | Jain SLO5s | "
      "Jain share norm. | compl. peor-mejor | SLO5s peor-mejor | timeout min-max | "
      "nulo compl. p50 | p |")
print("|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|")
for f in sorted(por_corrida, key=lambda r: (r["system"], r["N"], r["lambda_total"], r["rep"])):
    print(f"| {f['system']} | {f['N']} | {f['lambda_total']:.1f} | {f['rep']} | "
          f"{'S' if f['stable']=='STABLE' else 'U'} | {f['desbalance_demanda']} | "
          f"{f.get('jain_completion_fraction')} | {f.get('jain_slo_5s_fraction')} | "
          f"{f.get('jain_share_normalizada')} | "
          f"{f['completion_fraction_min']:.2f}-{f['completion_fraction_max']:.2f} | "
          f"{f['slo_5s_min']:.2f}-{f['slo_5s_max']:.2f} | "
          f"{f['timeout_fraction_min']:.2f}-{f['timeout_fraction_max']:.2f} | "
          f"{f.get('null_completion_fraction_p50')} | {f.get('p_completion_fraction')} |")
