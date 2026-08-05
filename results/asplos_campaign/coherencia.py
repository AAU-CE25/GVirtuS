#!/usr/bin/env python3
"""Cross-document consistency check for ~/paper.

Every key figure changed at least once today. This greps each of them across all markdown and
reports where a document still carries a superseded value. It cannot know what is right -- it
only shows disagreement, which is what has to be looked at.
"""
import glob, io, re, os, collections
os.chdir(os.path.expanduser('~/paper'))

# etiqueta -> (patron correcto, [patrones superados])
CHECKS = [
 ("coverage",            r"942 of 2023|942\b.*46\.6|46\.6% of the exported", [r"940 of 2023", r"46\.5% of the exported"]),
 ("capacity goodput",    r"51\.2|51\.7",                                     [r"57\.6 t/s"]),
 ("sweep v2 rows",       r"108 data rows",                                   [r"112 rows"]),
 ("tenants_canonico",    r"3838",                                            [r"3688"]),
 ("fixed-work cohorts",  r"523",                                             [r"\b493\b"]),
 ("memory saving",       r"426|4524|4501|4490",                              [r"~170 MiB", r"4794|4772|4761"]),
 ("lambda=1.0 goodput",  r"\+9\.9%",                                         [r"\+15\.6%"]),
 ("lambda=1.5 goodput",  r"\+27\.5%",                                        [r"\+33%"]),
 ("light-load tail",     r"\+273 ms|2\.1x|0\.96x",                           [r"an order of magnitude of tail"]),
 ("N1 inequality",       r"5\.02|3\.09",                                     []),
 ("visibility bound",    r"do not claim the general|bounded rather than|I10",[r"must be either closed or explicitly bounded"]),
 # Anadidos 2026-08-03 con la captura de grafos. La tabla paso de 10 invariantes (9
 # descargadas) a 12 (11 descargadas), y un recuento suelto en otro documento es justo el
 # fallo que este script existe para pillar -- ya ocurrio con cuatro documentos a la vez.
 # RETIRADA la comprobacion de "eleven invariants": al pasar la tabla a TRECE dejo de
 # aparecer en ningun documento, de modo que pasaba con cero citas -- verde sobre nada.
 # La cuenta viva es "invariantes I10" mas abajo. Lo que se comprueba ahora es que ningun
 # documento conserve un recuento antiguo.
 ("recuentos viejos",    r"thirteen invariants",
                         [r"eleven invariants", r"nine invariants", r"twelve invariants",
                          r"the tenth stated as a bounded"]),
 # El estado de graph_ptds. Se comprueban las FILAS DE ESTADO, no la prosa: CONFORMANCE.md
 # conserva a proposito las secciones 3c y 3d con su diagnostico equivocado, y marcarlas
 # seria un falso positivo en cada ejecucion.
 ("graph capture",       r"I11|I12|3e",  [r"\*\*confirmed, OPEN\*\* \| `graph_ptds`",
                                          r"defect located, HALF fixed"]),
 # La cifra de llama con grafos tras el arreglo, y su referencia.
 ("llama tg16 grafos",   r"531\.3|528\.6",                                   []),
 # I10 dejo de ser una suposicion externa: 13 invariantes, 12 descargadas sin condiciones.
 ("invariantes I10",     r"thirteen invariants",           [r"eleven invariants", r"nine invariants"]),
 # El precio de descargar A2. El 1,9x era de OTRA operacion (ucp_ep_flush, que ataca A1) y
 # solo vale para el regimen fire-and-forget; citarlo como el coste de I10 es el error que
 # esta comprobacion existe para no repetir.
 ("visibilidad I10",     r"0\.09%|0\.729 us|HOST_FLUSH_REQUIRED",
                         [r"priced at the measured \*\*1\.9x\*\*",
                          r"not discharged -- see .6"]),
 # --- anadidas 2026-08-03 (tarde-noche) -------------------------------------------------
 # El defecto de A1 paso de `assume` a `flush`: el sistema evaluado descarga las TRECE.
 # Un documento que siga diciendo "doce de trece" o "assume (default)" esta obsoleto.
 ("A1 por defecto",      r"deployed default|default is `flush`|thirteen invariants",
                         [r"`assume` \(default\)", r"twelve are discharged",
                          r"I13 is discharged only under", r"the default `assume` leaves it"]),
 # El denominador del contador. `admit_rma` cuenta tambien RMA a slot de HOST, que no genera
 # obligacion A2; citar la igualdad con admit_rma es el error que esto atrapa.
 ("denominador I10",     r"gpu_shadow_consumptions",
                         [r"`descargas` = `admit_rma`", r"descargas` equals `admit_rma`",
                          r"discharges == admit_rma"]),
 # El cruce de H2D fijada se re-midio a 16 KiB con 3 corridas independientes. El 8 KiB
 # anterior salia de una sola corrida con una diferencia de x1,01.
 # El cruce de H2D fijada. Los patrones "superados" de la primera version eran DEMASIADO
 # ESPECIFICOS -- exigian el formato exacto de una tabla concreta -- y dejaron pasar el 8 KiB
 # en SEIS documentos, incluida una contradiccion dentro de CONTRACTS.md (§2 decia 8, §6.4c
 # decia 16). Ahora se marca cualquier "8 KiB" que aparezca cerca de una palabra de umbral,
 # que es la forma en que la cifra se afirma de verdad.
 ("cruce H2D fijada",    r"16 KiB",
                         [r"(?i)(crossover|threshold|pinned)[^\n]{0,80}\b8 KiB\b",
                          r"(?i)\b8 KiB\b[^\n]{0,80}(crossover|threshold)",
                          r"H2D  \(1\)    \|   1 MiB  \|   8 KiB",
                          r"\| pinned host \| \*\*8 KiB\*\*",
                          r"\| \*\*pinned\*\* \| 8 KiB"]),
 # El coste del flush de device: un valor con dispersion, no tres decimales de una corrida.
 ("coste flush device",  r"0\.729-0\.738|~0\.73 us|0\.73 us",              []),
]

md = sorted(glob.glob("*.md"))
texts = {f: open(f, encoding="utf-8", errors="replace").read() for f in md}
problemas = 0
vacias = 0
for etiqueta, bueno, malos in CHECKS:
    hits_malos = collections.defaultdict(list)
    for f, t in texts.items():
        for m in malos:
            for mo in re.finditer(m, t):
                ln = t[:mo.start()].count("\n") + 1
                # una cita dentro de una correccion explicita NO es un problema
                # Ventana a AMBOS lados: la marca de retractacion puede ir despues de la
                # cita ("...cut the saving to ~170 MiB ... That correction is withdrawn").
                ctx = t[max(0, mo.start()-380):mo.start()+380].lower()
                if any(k in ctx for k in ("corrected", "retract", "withdraw", "previously read",
                                          # Anadidas 2026-08-03: son las formas en que este
                                          # paquete marca una cifra como SUPERADA al citarla.
                                          # Sin ellas el script marca las propias correcciones,
                                          # que es la otra forma de ser inutil.
                                          "first published", "re-measured", "was 8 kib",
                                          "single run", "when this suite was written",
                                          "earlier version", "no longer", "superseded",
                                          "may not be claimed", "until recomputed", "wrong",
                                          "is the only one where", "artefact", "detour")):
                    continue
                hits_malos[f].append((ln, mo.group(0)))
    if hits_malos:
        problemas += 1
        print("!! %-22s valor superado aun vivo:" % etiqueta)
        for f, v in hits_malos.items():
            for ln, s in v[:4]:
                print("     %-26s :%-4d  %s" % (f, ln, s))
    else:
        n = sum(1 for t in texts.values() if re.search(bueno, t))
        if n == 0:
            # Una comprobacion que no cita NADIE no ha comprobado nada: su patron "correcto"
            # dejo de existir en el paquete, casi siempre porque la cifra evoluciono y el
            # script se quedo atras. Contarlo como "ok" es exactamente el verde-sobre-nada que
            # este script existe para evitar; ya paso con "eleven invariants".
            vacias += 1
            print("?? %-22s VACIA: ningun documento cita el patron correcto -- la"
                  " comprobacion no prueba nada" % etiqueta)
        else:
            print("   %-22s ok (%d documentos lo citan)" % (etiqueta, n))
print()
print("documentos:", len(md), " comprobaciones con conflicto:", problemas,
      " comprobaciones vacias:", vacias)
if vacias:
    print("AVISO: una comprobacion vacia hay que arreglarla o retirarla, no dejarla en verde.")

# ---------------------------------------------------------------------------------------------
# COMPROBACIONES DE ESTADO FINAL
#
# Las de arriba comparan CIFRAS entre documentos. No detectan una contradiccion NARRATIVA: que un
# documento diga "quadrant es el defecto" y otro "el sistema evaluado usa scalar", o que uno cierre
# el aborto y otro lo declare sin explicar. Eso paso: el script daba 0 conflictos mientras cuatro
# documentos contaban la historia anterior.
#
# Cada entrada es una afirmacion que YA NO PUEDE aparecer sin marca de superada. La marca es
# deliberadamente estrecha -- si un documento quiere conservar la frase vieja, tiene que decir en
# la MISMA linea o en la anterior que esta superada, con fecha.
ESTADO = [
    ("escalar sigue desplegado",
     r"(evaluated system runs the \*?scalar|deployed (and evaluated )?configuration is the \*?\*?scalar|4 MiB remains the deployed floor|runs the SCALAR policy)"),
    ("el aborto sigue sin explicar",
     r"(abort remains unexplained|intermittent frontend abort remains|its cause is still open)"),
    ("el coste de quadrant sigue en 1,1 %",
     r"(costs? 1\.1%|1\.1% loss|-1\.1%, outside noise)"),
    ("quadrant no se usa para numeros end-to-end",
     r"(used for any published end-to-end number \| \*\*no|not the configuration any end-to-end number)"),
    ("quadrant colapsa a 8 slots",
     r"(at CONC=8 with 8 slots it collapses)"),
    ("los hilos de un proceso ya solapan",
     r"(single process with N threads is,? (for now,? )?(effectively )?serialis|threads of one process do not overlap|is serialised, and `llama-server --parallel 8`|every concurrency conclusion in this package is about \*\*processes|mechanism unidentified|mechanism not identified|mechanism NOT identified)"),
    # Anadidas 2026-08-05. La primera caza el umbral de ATERRIZAJE D2H descrito como un escalar
    # compilado a 4 MiB e inmutable: dejo de serlo el 08-04 (per-memtype 128 KiB / 512 KiB, y
    # settable por entorno), y cuatro documentos seguian afirmandolo en presente. Ninguna
    # comprobacion de CIFRAS lo veia, porque "4 MiB" aparece legitimamente en muchos sitios --
    # lo que esta superado es la afirmacion sobre el CODIGO, no el numero.
    ("el umbral D2H sigue compilado",
     r"kGpuDirectD2HThreshold"),
    # La segunda caza la atribucion equivocada de la averia del 08-05. Se dio por "fallo de
    # entorno, no atribuido" y por bisecada; era un desajuste de protocolo de este mismo trabajo.
    # Que un documento del paquete diga que no hay codigo implicado es exactamente el tipo de
    # afirmacion que un revisor comprueba.
    ("la averia del 08-05 sigue sin atribuir",
     r"(An environment fault, unattributed|no code from this session is implicated|"
     r"every large transfer fails)"),
    ("el suelo de 4 MiB sigue vigente",
     r"(only ever fires above the 4 MiB|above the 4 MiB RMA floor|"
     r"the same 4 MiB RMA floor|below the 4 MiB RMA floor|"
     r"4 MiB RMA floor \(`(GVIRTUS_RMA_MIN_BYTES|ucx_rma_min_bytes))"),
    # Anadidas 2026-08-05 (tarde). Las de arriba persiguen FRASES concretas; esta persigue la
    # ESTRUCTURA del claim central, que es lo que ninguna comprobacion veia. El sistema NO tiene
    # cuatro cuadrantes de placement: tiene DOS umbrales de placement (H2D fijada/paginable) y
    # DOS de aterrizaje (D2H fijada/paginable). `prefer_rma()` solo se llama desde `WriteIov`,
    # que es H2D; `GetFromRemoteGpu` no consulta ningun umbral (CONTRACTS.md 2.0c). Siete
    # documentos seguian con la reticula simetrica despues de que 2.0c lo refutara.
    ("cuatro cuadrantes de placement",
     r"(four[- ]quadrant|four quadrants|cuatro cuadrantes|all four (placement )?(quadrants|regimes)|"
     r"in all four regimes|four placement (thresholds|quadrants|decisions)|"
     r"four data paths whose measured\s+crossovers)"),
    # Y el rango 16 KiB-2 MiB presentado como si fuera UN eje de placement: sus extremos son de
    # dos mecanismos distintos (16 KiB placement H2D, 2 MiB una celda que no decide nada).
    ("rango 16 KiB-2 MiB como placement",
     r"(16 KiB (against|to|--|-) 2 MiB|16 KiB.{0,40}2 MiB for pageable D2H|span 16 KiB.{0,10}2 MiB)"),
]
MARCAS = re.compile(r"(until 2026-08|superseded|previously read|previously cited|this (row|cell|line|paragraph) "
                    r"(previously|read)|hasta 2026-08|retract|closed 2026-0|cerrado 2026-0|"
                    r"former 4-MiB|under the former|no longer the deployed|"
                    # Citar 2.0c O decir "inert" ES la marca: son las dos formas en que un
                    # documento reconoce que las dos celdas D2H no deciden nada.
                    r"2\.0c|\binert\b|\binertes?\b|is not four|not four placement)", re.I)

print()
print("--- estado final (contradicciones narrativas) ---")
malos_estado = 0
for etiqueta, patron in ESTADO:
    rx = re.compile(patron, re.I)
    culpables = []
    for f in md:
        try:
            lineas = io.open(f, encoding="utf-8", errors="replace").read().split("\n")
        except Exception:
            continue
        for i, ln in enumerate(lineas):
            if rx.search(ln):
                ctx = "\n".join(lineas[max(0, i - 2):i + 2])
                if not MARCAS.search(ctx):
                    culpables.append("%s:%d" % (f, i + 1))
    if culpables:
        malos_estado += 1
        print("XX %-34s SIN MARCA DE SUPERADA en %d sitio(s): %s%s" %
              (etiqueta, len(culpables), ", ".join(culpables[:8]),
               "" if len(culpables) <= 8 else " ... (+%d mas)" % (len(culpables) - 8)))
    else:
        print("   %-34s ok" % etiqueta)
print()
print("contradicciones de estado final:", malos_estado)
if malos_estado:
    print("AVISO: el paquete cuenta a la vez la historia vieja y la nueva. Eso lo ve un revisor.")
