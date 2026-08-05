#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprueba que cada PDF contiene de verdad el texto de su .md.

Que el PDF sea mas RECIENTE que el .md no basta: un render puede fallar a medias y dejar un
fichero valido. El 2026-08-03 tres PDF del paquete eran renders truncados (20-46 % del texto) y
la frescura por fecha los daba por buenos.

DOS SENALES, y sirven para cosas distintas:

  1. LA COLA (la que decide). Un render que falla a medias pierde el FINAL, asi que se muestrean
     frases del ultimo cuarto del .md y se exige que al menos una aparezca en el PDF. Esto lo
     valida el autocontrol de abajo; la COBERTURA no lo hacia -- con el umbral del 100 % que yo
     tenia puesto, un PDF recortado al 15 % daba 1,83 y pasaba. La cobertura se sigue mostrando,
     pero como dato, no como criterio.

  2. FRASES (informativa, NO decide). Se muestrean frases del .md y se buscan en el PDF. Tiene
     dos limitaciones medidas que producen falsos positivos, y por eso no manda:
       - los tramos de `codigo en linea` los compone LaTeX con una fuente SUBCONJUNTADA, cuyos
         codigos de caracter son arbitrarios: ese texto no se recupera del stream;
       - comparar solo el primer trozo de una frase falla si hay un simbolo cerca del principio.
     Se comparan FRAGMENTOS de prosa por separado, con varias ventanas. Al intentar usar esta
     senal como criterio dio 27, luego 12, luego 23 y luego 15 "fallos" segun como la ajustara,
     sobre documentos que estaban intactos.

AUTOCONTROL. Antes de juzgar nada, la herramienta se planta un truncamiento a si misma y
comprueba que lo detecta. Si no lo detecta, no informa de los documentos: informa de que esta
rota. Un verificador cuyo alcance no se ha comprobado es como no tener verificador.
"""
import zlib, re, os, sys, glob, random, shutil, subprocess, tempfile

# `history/` esta EXCLUIDA a proposito: son documentos RETIRADOS (ver history/LEEME.md).
# El glob no recursa, asi que hoy ya quedarian fuera -- la exclusion se hace EXPLICITA para que
# pasarlo a recursivo mas adelante no vuelva a meter cifras retiradas en la comprobacion.
def _no_historico(rutas):
    return [r for r in rutas if not r.replace("\\", "/").startswith("history/")]



# RAIZ DEL PAQUETE. Antes esto era `os.chdir(os.path.expanduser("~/paper"))` fijo, y eso hacia
# falsa la afirmacion del LEEME: la copia que viaja DENTRO del paquete, al desempaquetarlo en
# cualquier otro sitio, comprobaba el ~/paper de la maquina -- otra carpeta -- o no encontraba
# nada. Un verificador que valida algo distinto de lo que acompana es peor que ninguno.
# Orden: GVS_PAPER, luego el padre del directorio del script (que es como viaja, en
# <paper>/verificacion/), y por ultimo ~/paper.
def _raiz_paper():
    e = os.environ.get("GVS_PAPER")
    if e:
        return os.path.abspath(os.path.expanduser(e))
    aqui = os.path.dirname(os.path.abspath(__file__))
    padre = os.path.dirname(aqui)
    if os.path.basename(aqui) == "verificacion" and os.path.isdir(padre):
        return padre
    return os.path.expanduser("~/paper")

os.chdir(_raiz_paper())
LIT = re.compile(rb"\((?:\\.|[^()\\])*\)", re.S)
UMBRAL = 1.00


def texto_pdf(p):
    d = open(p, "rb").read()
    partes = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", d, re.S):
        try:
            c = zlib.decompress(m.group(1))
        except Exception:
            continue
        for l in LIT.finditer(c):
            partes.append(l.group(0)[1:-1].decode("latin-1", "replace"))
    t = "".join(partes)
    # Las ligaduras (ff/fi/fl/ffi/ffl) ocupan un solo glifo en los slots 0x0B-0x0F: sin
    # expandirlas, toda palabra con "fi" pierde esas letras.
    for cod, exp in ((0x0B, "ff"), (0x0C, "fi"), (0x0D, "fl"), (0x0E, "ffi"), (0x0F, "ffl")):
        t = t.replace(chr(cod), exp)
    return re.sub(r"[^A-Za-z0-9]+", "", t).lower()


def alnum(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def cobertura(md, pdf):
    fuente = len(alnum(open(md, encoding="utf-8", errors="replace").read()))
    if fuente == 0:
        return None
    return len(texto_pdf(pdf)) / fuente


def frases_de_cola(md, T, n=6):
    """Devuelve (encontradas, verificables) sobre el ultimo cuarto de la prosa del .md."""
    cuerpo = open(md, encoding="utf-8", errors="replace").read()
    cand = [l.strip() for l in cuerpo.splitlines()
            if len(l.strip()) > 70 and "|" not in l
            and not l.strip().startswith(("#", "```", "!", "<", ">"))]
    cola = cand[int(len(cand) * 0.75):]
    if not cola:
        return 0, 0
    muestra = cola[-n:]
    hallo = ver = 0
    for fr in muestra:
        encontrada, verificable = False, False
        for frag in re.split(r"`[^`]*`", fr):
            a = alnum(frag)
            if len(a) < 30:
                continue
            verificable = True
            for ini in range(0, max(1, len(a) - 30), 10):
                if a[ini:ini + 30] in T:
                    encontrada = True
                    break
            if encontrada:
                break
        if verificable:
            ver += 1
            hallo += 1 if encontrada else 0
    return hallo, ver


def frases_perdidas(md, pdf, T):
    cuerpo = open(md, encoding="utf-8", errors="replace").read()
    cand = [l.strip() for l in cuerpo.splitlines()
            if len(l.strip()) > 70 and "|" not in l
            and not l.strip().startswith(("#", "```", "!", "<", ">"))]
    if len(cand) < 3:
        return []
    muestra = random.Random(11).sample(cand, min(8, len(cand)))
    perdidas = []
    for fr in muestra:
        hallada, verificables = False, 0
        for frag in re.split(r"`[^`]*`", fr):
            a = alnum(frag)
            if len(a) < 30:
                continue
            verificables += 1
            for ini in range(0, max(1, len(a) - 30), 10):
                if a[ini:ini + 30] in T:
                    hallada = True
                    break
            if hallada:
                break
        if verificables and not hallada:
            perdidas.append(fr)
    return perdidas


def autocontrol():
    """Renderiza un PDF con SOLO el 40 % de un documento y exige que el detector lo cace.

    Recortar el PDF por bytes NO sirve como control y lo comprobe: cortando al 15 % el detector
    seguia viendo la cola (5/6), porque los streams de texto van al principio y lo que ocupa el
    final son las FUENTES incrustadas. El defecto real del 2026-08-03 era otro -- un render que
    se quedaba a medias y producia un PDF valido con menos paginas -- y eso es lo que se simula
    aqui rindiendo una fuente recortada.
    """
    render = os.path.join(_raiz_paper(), "cudf_etl", "md2pdf.py")
    if not os.path.exists(render):
        return False, "no encuentro md2pdf.py, no puedo construir el control"
    sanos = [m for m in _no_historico(sorted(glob.glob("*.md")))
             if os.path.exists(m[:-3] + ".pdf")]
    if not sanos:
        return False, "no hay ningun par .md/.pdf"
    md = sanos[0]
    h0, v0 = frases_de_cola(md, texto_pdf(md[:-3] + ".pdf"))
    if v0 == 0:
        return False, "%s no tiene prosa de cola verificable" % md
    if h0 == 0:
        return False, "%s entero ya falla la cola (0/%d)" % (md, v0)
    d = tempfile.mkdtemp()
    try:
        lineas = open(md, encoding="utf-8", errors="replace").read().splitlines()
        recorte = os.path.join(d, "recorte.md")
        with open(recorte, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas[:int(len(lineas) * 0.40)]))
        salida = os.path.join(d, "recorte.pdf")
        r = subprocess.run([sys.executable, render, recorte, salida],
                           capture_output=True, timeout=300)
        if not os.path.exists(salida):
            return False, "el render de control fallo: %s" % r.stderr.decode()[-160:]
        h1, v1 = frases_de_cola(md, texto_pdf(salida))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if h1 > 0:
        return False, "un PDF con solo el 40 %% de %s sigue mostrando la cola (%d/%d)" % (md, h1, v1)
    return True, "%s: entero cola %d/%d -> render al 40 %% cola %d/%d" % (md, h0, v0, h1, v1)


ok, detalle = autocontrol()
print("AUTOCONTROL: %s -- %s\n" % ("pasa" if ok else "FALLA", detalle))
if not ok:
    print("La herramienta no detecta un truncamiento plantado. No se informa de los documentos.")
    sys.exit(2)

malos, avisos = 0, 0
for md in _no_historico(sorted(glob.glob("*.md"))):
    pdf = md[:-3] + ".pdf"
    if not os.path.exists(pdf):
        print("  SIN PDF  %s" % md)
        malos += 1
        continue
    c = cobertura(md, pdf)
    T = texto_pdf(pdf)
    h, v = frases_de_cola(md, T)
    p = frases_perdidas(md, pdf, T)
    if v > 0 and h == 0:
        print("  TRUNCADO %-32s cola 0/%d ausente  (cobertura %.2f)" % (md, v, c or -1))
        malos += 1
    elif p:
        print("  ok*      %-32s cobertura %.2f  -- %d frase(s) no localizada(s), informativo"
              % (md, c, len(p)))
        avisos += 1
    else:
        print("  ok       %-32s cobertura %.2f" % (md, c))

print("\n  truncados: %d   avisos informativos de frase: %d" % (malos, avisos))
sys.exit(1 if malos else 0)
