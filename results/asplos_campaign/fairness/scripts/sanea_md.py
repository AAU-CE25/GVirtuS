#!/usr/bin/env python3
"""Sanea un markdown para pdflatex sin fuentes unicode.

El contenedor pandoc/latex no trae ninguna fuente del sistema (`fc-list` devuelve 0), asi que
xelatex esta descartado y lualatex con Latin Modern deja caer los glifos que esa fuente no
cubre -- y un glifo ausente no es un aviso cosmetico: el caracter DESAPARECE del PDF, de modo
que "~=220" se imprime como "220".

Aqui se sustituyen por equivalentes ASCII antes de compilar. Se informa de cada sustitucion
para que quede constancia de que se cambio la notacion, no el contenido.
"""
import io, sys, unicodedata
from collections import Counter

REP = {
    "∈": " en ", "−": "-", "≥": ">=", "≤": "<=",
    "⇒": "=>", "→": "->", "×": "x", "·": "-",
    "…": "...", "‑": "-", "–": "--", "—": "--",
    "≈": "~=", "≡": "==", "±": "+/-", "≠": "!=",
    "λ": "lambda", "Δ": "delta", "α": "alpha", "β": "beta",
    "μ": "u", "⁴": "^4", "⁸": "^8", "⁰": "^0",
    "²": "^2", "³": "^3", "¹": "^1", "⁶": "^6",
    "≤": "<=", "≥": ">=", "⚠": "AVISO:",
}

for p in sys.argv[1:]:
    s = io.open(p, encoding="utf-8").read()
    hechas = Counter()
    for k, v in REP.items():
        if k in s:
            hechas[k] = s.count(k)
            s = s.replace(k, v)
    # Barrido final: lo que quede fuera de Latin-1 se nombra en vez de desaparecer.
    resto = sorted({c for c in s if ord(c) > 0xFF})
    for c in resto:
        s = s.replace(c, "[" + unicodedata.name(c, "?").lower() + "]")
    io.open(p, "w", encoding="utf-8").write(s)
    tot = sum(hechas.values())
    print(f"{p}: {tot} sustituciones" +
          (f"; sin mapa y renombrados: {resto}" if resto else ""))
