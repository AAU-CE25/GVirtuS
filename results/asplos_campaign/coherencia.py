#!/usr/bin/env python3
"""Cross-document consistency check for ~/paper.

Every key figure changed at least once today. This greps each of them across all markdown and
reports where a document still carries a superseded value. It cannot know what is right -- it
only shows disagreement, which is what has to be looked at.
"""
import glob, re, os, collections
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
 ("invariantes",         r"eleven invariants",                               [r"nine invariants", r"the tenth stated as a bounded"]),
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
]

md = sorted(glob.glob("*.md"))
texts = {f: open(f, encoding="utf-8", errors="replace").read() for f in md}
problemas = 0
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
        print("   %-22s ok (%d documentos lo citan)" % (etiqueta, n))
print()
print("documentos:", len(md), " comprobaciones con conflicto:", problemas)
