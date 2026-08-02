#!/usr/bin/env python3
"""Tabla canonica POR TENANT para los workloads de trabajo fijo.

Regla dura: no se inventa ni se estima nada. Si un campo no existe en el crudo, se escribe
"missing". Los unicos campos derivados son los que salen EXACTAMENTE de contadores o marcas
de tiempo presentes, y su formula queda escrita en la columna `derivacion`.

Fuentes, por workload:

  XSBench      xsbench_campaign/results/xsbench/<sys>/N<N>/<modo>/seed<NN>/client<i>/
                 status_raw.json -> idx, exit_code, t_start, t_end, completion_s
                 stdout.log      -> "Runtime: <s> seconds", "Lookups/s: <n>"
  BabelStream  experiments/babelstream/results*/<sys>/N<N>/<modo>/seed<NN>/client<i>/
                 status_raw.json -> idx, exit_code, t_start, t_end, duration_s, cpu_s
                 stdout.log      -> filas function,num_times,n_elements,sizeof,max_MB_per_sec,...
  CloverLeaf   experiments/cloverleaf/results*/<sys>/N<N>/<modo>/seed<NN>/client<i>/
                 status_raw.json -> idx, exit_code, t_start, t_end, duration_s, cpu_s
                 stdout.log      -> NO contiene la figura de merito: CloverLeaf escribe a
                                    clover.out, que NO viaja en el artefacto. Solo hay tiempo
                                    de pared. Limitacion declarada, no rellenada.
  MiniBUDE     mb_campaign/run_<sys>_n<N>_r<rep>/tenant_<i>.log
                 -> time:{epoch_s}, avg_ms, gflop/s, raw_iterations[], outcome.valid,
                    max_diff_%, deck{poses,proteins,ligands}
                 NOTA: t<i>.log es un duplicado byte a byte de tenant_<i>.log.
                 NO hay t_end absoluto; el fin se deriva como epoch_s + sum_ms/1000.
"""
import json, os, re, csv, glob, sys

H = os.path.expanduser("~")
OUT = os.path.join(H, "GVirtuS/results/asplos_campaign/fairness/tenants_canonico.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

COLS = ["workload", "system", "cohort_path", "run_id", "seed", "mode", "N", "tenant_id",
        "t_start", "t_end", "concurrent_runtime", "internal_runtime", "figure_of_merit",
        "fom_name", "fixed_work_units", "cpu_s", "max_rss_kib", "exit_status",
        "correctness_status", "sospechoso", "derivacion"]

filas = []


def sospecha(path):
    """Marca cohortes que el propio arbol senala como no canonicas. No se descartan en
    silencio: se etiquetan y quien analice decide."""
    p = path.lower()
    for m in (".attempts", "results_bad", "premoderacion", "warmup", "results_ab_",
              "results_backlog", "postfix", "rootns", "_keep", "results_stale"):
        if m in p:
            return m
    return ""


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


# ---------------------------------------------------------------- XSBench / Babel / Clover
def cohortes_status(raiz, workload):
    for st in glob.glob(os.path.join(raiz, "**", "status_raw.json"), recursive=True):
        cli = os.path.dirname(st)
        coh = os.path.dirname(cli)
        rel = os.path.relpath(coh, H)
        try:
            d = json.load(open(st))
        except Exception:
            continue

        # sistema / N / modo / seed se leen de la RUTA; se dejan "missing" si no encajan.
        partes = rel.split(os.sep)
        sysname = mode = seed = "missing"
        N = "missing"
        for i, p in enumerate(partes):
            if re.fullmatch(r"N\d+", p):
                N = p[1:]
                if i > 0:
                    sysname = partes[i - 1]
                if i + 1 < len(partes):
                    mode = partes[i + 1]
            if re.fullmatch(r"seed\d+", p):
                seed = p[4:]
        if sysname == "missing" and "sys" in d:
            sysname = d["sys"]

        so = os.path.join(cli, "stdout.log")
        txt = ""
        if os.path.exists(so):
            txt = open(so, errors="replace").read()

        fom = fomname = intern = work = "missing"
        deriv = ""
        if workload == "xsbench":
            m = re.search(r"Lookups/s:\s*([\d,]+)", txt)
            if m:
                fom, fomname = num(m.group(1)), "lookups_per_s"
            m = re.search(r"Runtime:\s*([\d.]+)\s*seconds", txt)
            if m:
                intern = num(m.group(1))
            deriv = "internal_runtime = linea 'Runtime' de stdout.log (excluye setup)"
        elif workload == "babelstream":
            # Triad es el kernel canonico de BabelStream.
            m = re.search(r"^Triad,(\d+),(\d+),(\d+),([\d.]+),", txt, re.M)
            if m:
                work = f"num_times={m.group(1)};n_elements={m.group(2)};sizeof={m.group(3)}"
                fom, fomname = num(m.group(4)), "triad_MB_per_s"
            m2 = re.search(r"^Triad,\d+,\d+,\d+,[\d.]+,[\d.]+,[\d.]+,([\d.]+)", txt, re.M)
            if m2:
                intern = num(m2.group(1))
            deriv = "figure_of_merit = max_MB_per_sec de la fila Triad; internal = avg_runtime"
        elif workload == "cloverleaf":
            fomname = "missing"
            deriv = ("CloverLeaf escribe su figura de merito en clover.out, que NO esta en el "
                     "artefacto. Solo hay tiempo de pared de status_raw.json.")

        filas.append({
            "workload": workload, "system": sysname, "cohort_path": rel,
            "run_id": rel, "seed": seed, "mode": mode, "N": N,
            "tenant_id": d.get("idx", "missing"),
            "t_start": d.get("t_start", "missing"), "t_end": d.get("t_end", "missing"),
            "concurrent_runtime": d.get("completion_s", d.get("duration_s", "missing")),
            "internal_runtime": intern, "figure_of_merit": fom, "fom_name": fomname,
            "fixed_work_units": work, "cpu_s": d.get("cpu_s", "missing"),
            "max_rss_kib": d.get("max_rss_kib", "missing"),
            "exit_status": d.get("exit_code", "missing"),
            "correctness_status": "missing",
            "sospechoso": sospecha(rel), "derivacion": deriv,
        })


# ---------------------------------------------------------------------------- MiniBUDE
def cohortes_minibude():
    for run in sorted(glob.glob(os.path.join(H, "mb_campaign", "run_*"))):
        base = os.path.basename(run)
        m = re.match(r"run_(.+)_n(\d+)_r(\d+)$", base)
        if not m:
            continue
        sysname, N, rep = m.group(1), m.group(2), m.group(3)
        # Los runs remotos escriben t<i>.log Y tenant_<i>.log (duplicados byte a byte);
        # los baremetal escriben SOLO t<i>.log. Globar solo tenant_* borraba el control.
        vistos = {}
        for lg in sorted(glob.glob(os.path.join(run, "t*.log"))):
            mm = re.search(r"/(?:tenant_)?t?(\d+)\.log$", lg)
            if not mm:
                continue
            tid = mm.group(1)
            if tid in vistos:
                continue
            vistos[tid] = lg
        for tid, lg in sorted(vistos.items(), key=lambda kv: int(kv[0])):
            txt = open(lg, errors="replace").read()
            def g(pat, cast=num):
                mm = re.search(pat, txt, re.M)
                return cast(mm.group(1)) if mm else "missing"
            epoch = g(r"epoch_s:(\d+)")
            avg = g(r"^\s*avg_ms:\s*([\d.]+)")
            summ = g(r"^\s*sum_ms:\s*([\d.]+)")
            gfl = g(r"^\s*gflop/s:\s*([\d.]+)")
            valid = re.search(r"valid:\s*(\w+)", txt)
            mdiff = g(r"max_diff_%:\s*([\d.]+)")
            poses = g(r"poses:\s*(\d+)")
            prot = g(r"proteins:\s*(\d+)")
            lig = g(r"ligands:\s*(\d+)")
            tend = "missing"
            if epoch != "missing" and summ != "missing":
                tend = epoch + summ / 1000.0
            filas.append({
                "workload": "minibude", "system": sysname,
                "cohort_path": os.path.relpath(run, H), "run_id": base,
                "seed": rep, "mode": "sync", "N": N, "tenant_id": tid,
                "t_start": epoch, "t_end": tend,
                "concurrent_runtime": (summ / 1000.0) if summ != "missing" else "missing",
                "internal_runtime": (avg / 1000.0) if avg != "missing" else "missing",
                "figure_of_merit": gfl, "fom_name": "gflop_per_s",
                "fixed_work_units": f"poses={poses};proteins={prot};ligands={lig}",
                "cpu_s": "missing", "max_rss_kib": "missing",
                "exit_status": "missing",
                "correctness_status": (f"valid={valid.group(1)};max_diff_%={mdiff}"
                                       if valid else "missing"),
                "sospechoso": sospecha(run),
                "derivacion": ("t_end = epoch_s + sum_ms/1000 (NO hay t_end absoluto en el "
                               "crudo); concurrent_runtime = sum_ms/1000 = las 8 iteraciones "
                               "cronometradas; internal_runtime = avg_ms/1000"),
            })


cohortes_status(os.path.join(H, "xsbench_campaign/results/xsbench"), "xsbench")
cohortes_status(os.path.join(H, "experiments/babelstream"), "babelstream")
cohortes_status(os.path.join(H, "experiments/cloverleaf"), "cloverleaf")
cohortes_minibude()

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for f in filas:
        w.writerow(f)

print(f"escritas {len(filas)} filas por tenant en {OUT}\n")
from collections import Counter
c = Counter((f["workload"], f["system"], f["N"], f["mode"]) for f in filas if not f["sospechoso"])
print("celdas canonicas (workload, sistema, N, modo) -> tenants:")
for k in sorted(c):
    print(f"  {k}: {c[k]}")
s = Counter(f["sospechoso"] for f in filas if f["sospechoso"])
print("\nfilas etiquetadas como no canonicas (NO descartadas):")
for k, v in sorted(s.items()):
    print(f"  {k}: {v}")
