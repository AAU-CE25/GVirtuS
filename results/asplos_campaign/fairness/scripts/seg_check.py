#!/usr/bin/env python3
"""Comprueba si summary.csv permite segmentar el JSONL por corrida.

bench.py abre el JSONL en modo APPEND y las repeticiones comparten `label`, asi que las
3 corridas de cada celda multi-tenant estan concatenadas sin separador. Si por cada corrida
len(inw) = completed + fail y la suma coincide EXACTAMENTE con las lineas del fichero, los
tramos [0,n1), [n1,n1+n2), ... reconstruyen las corridas -- siempre que el orden de append
sea el mismo que el de las filas del CSV.
"""
import csv, os
os.chdir(os.path.expanduser("~/results"))
rows = list(csv.DictReader(open("summary.csv")))
labels = ["mt_ucx_n8_l8", "mtbm_n8_l8", "mt_ucx_n4_l4", "mtbm_n4_l4",
          "mt_ucx_n2_l2", "mtbm_n2_l2", "mt_ucx_n8_l1.0", "mtbm_n8_l1.0",
          "mt_ucx_gpudirect_n8_l1.0", "mt_ucx_gpudirect_n10_l1.0"]
for lbl in labels:
    rs = [r for r in rows if r["label"] == lbl]
    inw = [int(r["completed"]) + int(r["fail"]) for r in rs]
    p = lbl + ".jsonl"
    n = sum(1 for _ in open(p)) if os.path.exists(p) else -1
    ok = "COINCIDE" if (n >= 0 and sum(inw) == n) else "NO COINCIDE"
    print(f"{lbl:26s} corridas={len(rs):2d} inw={inw} suma={sum(inw):5d} jsonl={n:5d}  {ok}")
