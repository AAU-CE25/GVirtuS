#!/usr/bin/env python3
"""Quita del barrido v2 los puntos duplicados del intento abortado.

El primer intento de cola_v2 (02:39) completo la celda bm_n8_r1 antes de que lo parara y
relanzara a las 02:56. Como bench.py abre los ficheros en modo APPEND y la etiqueta es la
misma, esos cuatro puntos quedaron por duplicado: la celda (bm, N=8) tiene 4 observaciones por
lambda donde el resto tiene 3, lo que desequilibra la comparacion emparejada.

Se conserva la ULTIMA linea de cada fichero -- la del barrido limpio, posterior al relanzado --
y se guarda la descartada aparte en vez de borrarla.
"""
import glob, os, io, json

B = os.path.expanduser("~/GVirtuS/results/asplos_campaign/llama_slo_sweep_v2")
DESC = os.path.join(B, "descartados_intento1")
os.makedirs(DESC, exist_ok=True)
n = 0
for f in sorted(glob.glob(os.path.join(B, "*.meta.jsonl"))):
    lin = io.open(f, encoding="utf-8").read().splitlines()
    if len(lin) <= 1:
        continue
    base = os.path.basename(f)
    io.open(os.path.join(DESC, base), "w", encoding="utf-8").write("\n".join(lin[:-1]) + "\n")
    io.open(f, "w", encoding="utf-8").write(lin[-1] + "\n")
    # el .jsonl por peticion tambien acumula; se recorta al mismo criterio no es posible sin
    # marca de corrida, asi que se deja y se declara: solo el sidecar queda desduplicado.
    n += 1
    print(f"  {base}: {len(lin)} -> 1 linea ({len(lin)-1} movidas a descartados_intento1/)")
print(f"\n{n} ficheros desduplicados; los descartados se conservan en {DESC}")
