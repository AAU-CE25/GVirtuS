#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""estadistica.py <csv> <columna> [etiqueta] -- resumen de una columna de valores CRUDOS.

Se separa del arnes a proposito: el arnes guarda SIEMPRE las repeticiones una a una y esto
resume aparte. Citar un numero sin poder ver sus replicas ya produjo dos errores en esta
campana -- 14 transferencias de UNA corrida contadas como 14 replicas, y una "banda" que en
realidad eran dos configuraciones exactas.

Emite n, media, sd, p50, p90, p95, p99, IQR, min, max. El estadistico que se publique se elige
luego, pero se elige VIENDOLOS TODOS: el documento usa mediana para miniBUDE y media para
llama, y confundirlos ya cambio una cifra publicada.
"""
import csv, sys, statistics as st


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    if len(v) == 1:
        return v[0]
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def main():
    ruta, col = sys.argv[1], sys.argv[2]
    etiqueta = sys.argv[3] if len(sys.argv) > 3 else col
    vals = []
    with open(ruta) as f:
        for fila in csv.DictReader(f):
            try:
                vals.append(float(fila[col]))
            except (ValueError, KeyError, TypeError):
                continue
    if not vals:
        print("  %-28s SIN DATOS (columna %s vacia o ausente en %s)" % (etiqueta, col, ruta))
        return 1
    n = len(vals)
    sd = st.stdev(vals) if n > 1 else 0.0
    print("  %-28s n=%-3d media=%.4g sd=%.3g  p50=%.4g p90=%.4g p95=%.4g p99=%.4g  "
          "IQR=%.3g  min=%.4g max=%.4g"
          % (etiqueta, n, st.fmean(vals), sd, pct(vals, .50), pct(vals, .90),
             pct(vals, .95), pct(vals, .99), pct(vals, .75) - pct(vals, .25),
             min(vals), max(vals)))
    if n < 3:
        print("       (n<3: no se cita percentil de esto, se citan los valores crudos)")
    return 0


sys.exit(main())
