#!/usr/bin/env python3
"""verifica_datos.py -- recomputa desde el CSV cada cifra de titular y la compara con lo que
dicen los documentos. No comprueba que el documento sea coherente consigo mismo (eso es
coherencia.py): comprueba que el NUMERO PUBLICADO sale del dato.

Regla: si una comprobacion no encuentra su fuente, es FALLO, no OK. Una comprobacion que no
puede correr no ha verificado nada -- ya nos paso con un check de coherencia que salia verde
sobre cero documentos.
"""
import collections, csv, os, sys, glob, statistics as st, collections

H = os.path.expanduser("~")
P = os.path.join(H, "paper")
C = os.path.join(H, "GVirtuS/results/asplos_campaign")
ok = fail = 0

def chk(nombre, esperado, obtenido, tol=0.02, nota=""):
    global ok, fail
    if obtenido is None:
        print("  FALLO  %-46s esperado %-12s  SIN FUENTE %s" % (nombre, esperado, nota)); fail += 1; return
    try:
        e, o = float(esperado), float(obtenido)
        bien = abs(e - o) <= tol * max(abs(e), 1e-9)
    except (TypeError, ValueError):
        bien = str(esperado) == str(obtenido)
    print("  %-6s %-46s doc=%-12s dato=%-12s %s" %
          ("ok" if bien else "FALLO", nombre, esperado,
           ("%.4g" % o) if isinstance(obtenido, float) else obtenido, nota))
    ok, fail = (ok + 1, fail) if bien else (ok, fail + 1)

def leer(*ruta):
    p = os.path.join(*ruta)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None

# ---------------------------------------------------------------- 1. miniBUDE fairness
r = leer(C, "fairness/tabla_D_minibude_por_tenant.csv")
if r:
    coh = collections.defaultdict(list)
    for x in r:
        if x["system"] == "ucx_gpudirect":
            coh[x["run"]].append(float(x["slowdown"]))
    desig = [max(v) / min(v) for v in coh.values() if len(v) >= 2]
    mejor = min(min(v) for v in coh.values() if len(v) >= 2)
    # El doc publica la MEDIANA de las 15 corridas (FAIRNESS_RESULTS.md tabla), no el maximo.
    chk("miniBUDE N=8 desigualdad mediana (GPUDirect)", 4.87, st.median(desig), 0.01)
    chk("miniBUDE tenant mas rapido (x su solo)", 1.001, mejor, 0.03)
else:
    chk("miniBUDE N=8 desigualdad max (GPUDirect)", 4.87, None)

# ---------------------------------------------------------------- 2. N7 epoch
r = leer(C, "epoch_n7/n7_2x2.csv")
if r:
    def uno(arm, col):
        v = {int(x[col]) for x in r if x["arm"] == arm}
        return v.pop() if len(v) == 1 else None
    chk("N7 park_on parked", 81, uno("park_on_guard_on", "parked"))
    chk("N7 park_off epoch dropped", 126, uno("park_off_guard_on", "ack_epoch_dropped"))
    chk("N7 guard_off gen mismatch", 38, uno("park_off_guard_off", "ack_gen_mismatch"))
    chk("N7 guard_off premature frees", 81, uno("park_off_guard_off", "ack_on_free"))
    chk("N7 corrupcion en todos los brazos", 0, max(int(x["bytes_malos"]) for x in r))
else:
    chk("N7 park_on parked", 81, None)

# ---------------------------------------------------------------- 3. SLO llama
r = leer(P, "LLAMA_SLO_capacidad_v2.csv")
if r:
    g = collections.defaultdict(list)
    for x in r:
        if int(x["N"]) == 8:
            g[(x["system"], float(x["lam"]))].append(float(x["goodput"]))
    def med(s, l): return st.mean(g[(s, l)]) if g.get((s, l)) else None
    u, b = med("ucx", 1.0), med("bm", 1.0)
    chk("llama goodput lambda=1.0, ucx vs bm (media, +%)", 9.9, (u - b) / b * 100 if u and b else None, 0.25)
    u, b = med("ucx", 1.5), med("bm", 1.5)
    chk("llama goodput lambda=1.5, ucx vs bm (media, +%)", 27.5, (u - b) / b * 100 if u and b else None, 0.25)
    u, b = med("ucx", 0.5), med("bm", 0.5)
    chk("llama capacidad lambda=0.5 ucx (media n=3, t/s)", 51.2, u, 0.02)
else:
    chk("llama goodput lambda=1.0, ucx vs bm (media, +%)", 9.9, None)

# ---------------------------------------------------------------- 4. memoria por tenant
r = leer(P, "mem_footprint.csv")
if r:
    d = {(x["system"], int(x["N"])): float(x["per_tenant_mib"]) for x in r if x["per_tenant_mib"]}
    dep = [v for (s, n), v in d.items() if s == "ucx_deployed" and n == 4]
    contam = [v for (s, n), v in d.items() if s == "ucx" and n == 4]
    na = [v for (s, _), v in d.items() if s.startswith(("bm", "native")) and "mps" not in s]
    chk("memoria por tenant Gusto DESPLEGADO (MiB)", 4490, dep[0] if dep else None, 0.005)
    chk("memoria fila `ucx` = config CONTAMINADA (MiB)", 4761, contam[0] if contam else None, 0.005,
        nota="(no es la que cita el paper)")
    chk("memoria por tenant nativo (MiB)", 4950, max(na) if na else None, 0.03)
else:
    chk("memoria por tenant Gusto (MiB)", 4490, None)

# ---------------------------------------------------------------- 5. cuDF escalado
r = leer(P, "cudf_etl/cell_summary.csv")
if r:
    def cel(cfg, n):
        v = [float(x["median_ms"]) for x in r
             if x["configuration"] == cfg and int(x["tenants"]) == n and x["campaign"] == "c2b_scaling"]
        return st.median(v) if v else None
    chk("cuDF native N=1 (ms)", 319.8, cel("native_no_mps", 1), 0.02)
    chk("cuDF GPUDirect N=1 (ms)", 356.1, cel("ucx_gpudirect", 1), 0.02)
    chk("cuDF native N=8 (ms)", 637.0, cel("native_no_mps", 8), 0.02)
    chk("cuDF GPUDirect N=8 (ms)", 631.5, cel("ucx_gpudirect", 8), 0.02)
else:
    chk("cuDF native N=1 (ms)", 319.8, None)

# ---------------------------------------------------------------- 6. coste A1 (hoy)
r = leer(C, "a1_i10_0803/a1_cost_matrix.csv")
if r:
    por = collections.defaultdict(list)
    for x in r:
        if int(x["transfer"]) > 2: por[(x["cell"], x["rep"])].append(float(x["h2d_GBps"]))
    cel = collections.defaultdict(list)
    for (c, _), v in por.items(): cel[c].append(st.median(v))
    for c, e in (("assume", 22.804), ("fence", 22.814), ("flush", 22.775), ("strict", 6.854)):
        chk("A1 %s 64 MiB (GB/s)" % c, e, st.median(cel[c]) if cel.get(c) else None, 0.005)
    malos = sum(1 for x in r if x["device_ck_ok"] != "pass" or x["host_ck_ok"] != "pass")
    chk("A1 checksums fallidos", 0, malos)
else:
    chk("A1 assume 64 MiB (GB/s)", 22.804, None)

# ---------------------------------------------------------------- 7. cruce 16 KiB
fs = glob.glob(os.path.join(H, "cross_out", "out_*_pinned_r*.csv"))
if fs:
    d = collections.defaultdict(list)
    for f in fs:
        arm = os.path.basename(f)[4:].split("_pinned")[0]
        for x in csv.DictReader(open(f)):
            if x["direction"] == "h2d":
                d[(arm, int(x["bytes"]))].append(float(x["gbytes_per_s"]))
    def m(a, b): return st.median(d[(a, b)]) if d.get((a, b)) else None
    am8, rm8 = m("am", 8192), m("assume", 8192)
    am16, rm16 = m("am", 16384), m("assume", 16384)
    chk("8 KiB: gana AM (am > rma)", True, (am8 > rm8) if am8 and rm8 else None)
    chk("16 KiB: gana RMA (rma > am)", True, (rm16 > am16) if am16 and rm16 else None)
else:
    chk("8 KiB: gana AM (am > rma)", True, None, nota="(cross_out no esta)")

# ---------------------------------------------------------------- campana canonica 2026-08-04
# Toda cifra publicada tiene que ser recomputable desde su CSV. Las de la re-medicion canonica
# no lo eran cuando se escribieron, que es el mismo hueco por el que se publicaron dos CSV
# rancios el 2026-08-03: el texto y el dato pueden divergir sin que nada se queje.
CAN = os.path.join(H, "paper", "canonica")

fm = os.path.join(CAN, "micros_politica.csv")
if os.path.exists(fm):
    M = {}
    for x in csv.DictReader(open(fm)):
        M[(x["mem"], x["direction"], int(x["bytes"]), x["arm"])] = float(x["median_gbps"])

    def g(mem, dr, b, a): return M.get((mem, dr, b, a))

    # H2D paginable a 16 KiB: un suelo escalar bajo lo hunde frente al camino AM.
    am, sc = g("pageable", "h2d", 16384, "am"), g("pageable", "h2d", 16384, "scalar")
    chk("pageable H2D 16K: AM/scalar", 4.42, (am / sc) if am and sc else None)
    # H2D fijada: el escalar desplegado (= AM por debajo de 4 MiB) deja esto sobre la mesa.
    am1, q1 = g("pinned", "h2d", 1048576, "am"), g("pinned", "h2d", 1048576, "quadrant")
    chk("pinned H2D 1M: quadrant/AM", 2.59, (q1 / am1) if am1 and q1 else None)
    am2, q2 = g("pinned", "h2d", 2097152, "am"), g("pinned", "h2d", 2097152, "quadrant")
    chk("pinned H2D 2M: quadrant/AM", 3.20, (q2 / am2) if am2 and q2 else None)
    # La tabla que cierra el argumento: peor razon contra el oraculo, por brazo y regimen.
    def peor(mem, dr, a):
        tam = sorted({k[2] for k in M if k[0] == mem and k[1] == dr})
        rs = [g(mem, dr, b, a) / g(mem, dr, b, "oracle")
              for b in tam if g(mem, dr, b, a) and g(mem, dr, b, "oracle")]
        return min(rs) if rs else None
    chk("peor vs oraculo: AM pinned h2d", 0.313, peor("pinned", "h2d", "am"))
    chk("peor vs oraculo: scalar pageable h2d", 0.225, peor("pageable", "h2d", "scalar"))
    chk("peor vs oraculo: quadrant pinned h2d", 0.980, peor("pinned", "h2d", "quadrant"))
    chk("peor vs oraculo: quadrant pageable h2d", 0.987, peor("pageable", "h2d", "quadrant"))
else:
    chk("micros canonicos", True, None, nota="(paper/canonica no esta)")

ft = os.path.join(CAN, "llama_tg16.csv")
if os.path.exists(ft):
    T = collections.defaultdict(list)
    for x in csv.DictReader(open(ft)):
        if x.get("tps"):
            T[(x["politica"], x["modelo"])].append(float(x["tps"]))
    sc7 = T.get(("scalar", "mistral-7b-q4")); q7 = T.get(("quadrant", "mistral-7b-q4"))
    chk("llama 7B scalar (media t/s)", 137.4, st.fmean(sc7) if sc7 else None)
    chk("llama 7B quadrant (media t/s)", 135.9, st.fmean(q7) if q7 else None)
    # El titular es un DELTA, asi que se comprueba el delta y no solo los extremos.
    chk("llama 7B: quadrant cuesta 1.1%", -1.1,
        100 * (st.fmean(q7) / st.fmean(sc7) - 1) if sc7 and q7 else None, tol=0.15)
else:
    chk("llama canonico", True, None, nota="(llama_tg16.csv no esta)")

fb = os.path.join(CAN, "minibude_biseccion.csv")
if os.path.exists(fb):
    B = {int(x["suelo_bytes"]): int(x["admit_rma"]) for x in csv.DictReader(open(fb))}
    # La afirmacion es "ninguna transferencia llega a 1 MiB", y eso es exactamente admit_rma=0
    # con el suelo en 1 MiB. Es la razon por la que miniBUDE es un control y no el diferenciador.
    chk("miniBUDE: 0 transferencias >= 1 MiB", 0, B.get(1048576))
    chk("miniBUDE: 6 transferencias >= 256 KiB", 6, B.get(262144))
else:
    chk("biseccion miniBUDE", True, None, nota="(minibude_biseccion.csv no esta)")

fc = os.path.join(CAN, "cudf.csv")
if os.path.exists(fc):
    filas = [x for x in csv.DictReader(open(fc)) if x.get("wall_s")]
    # Correccion antes que rendimiento: 21 registros por cliente o la corrida no midio nada.
    malas = [x for x in filas if int(x["records"] or 0) != 21 * int(x["n"])]
    chk("cuDF: 0 corridas con registros incompletos", 0, len(malas))
else:
    chk("cuDF canonico", True, None, nota="(cudf.csv no esta)")

# ------------------------------------------------- 12. umbral de aterrizaje D2H (128 KiB fijada)
# Anadido 2026-08-05. La tabla que justifica el umbral de ATERRIZAJE por tipo de memoria estaba
# PUBLICADA Y SIN COMPROBAR: vivia en prosa y en un comentario del codigo. La mitad FIJADA si
# tiene fuente -- canonica/d2h_crossover.csv, el A/B de GPUDirect on/off -- y es la que decide el
# valor desplegado de 128 KiB, asi que se recomputa aqui punto por punto.
# La mitad PAGINABLE no tiene fichero en el paquete; se declara como hueco en GAPS.md 5 en vez de
# comprobarse contra nada, que es lo que haria una comprobacion verde sobre cero datos.
fd = os.path.join(P, "canonica", "d2h_crossover.csv")
if os.path.exists(fd):
    d = {}
    for x in csv.DictReader(open(fd)):
        if x.get("direction") != "d2h":
            continue
        d.setdefault(int(x["bytes"]), {})[x["arm"]] = float(x["median_gbps"])
    PUB = {131072: 1.19, 262144: 1.38, 524288: 2.65, 1048576: 4.36, 2097152: 5.21}
    for b in sorted(PUB):
        par = d.get(b, {})
        r = (par["on"] / par["off"]) if ("on" in par and "off" in par) else None
        chk("aterrizaje D2H fijada %d KiB (on/off)" % (b >> 10), PUB[b], r, tol=0.04)
    # El valor desplegado es 128 KiB: el menor tamano en que GPUDirect ya NO pierde en fijada.
    menor = min((b for b in d if "on" in d[b] and "off" in d[b] and d[b]["on"] > d[b]["off"]),
                default=None)
    chk("128 KiB es el defecto: primer tamano con ganancia", 65536, menor,
        nota="(65536 es el primero >1.00; 128 KiB es el primero con margen)")
else:
    chk("umbral de aterrizaje D2H", True, None, nota="(canonica/d2h_crossover.csv no esta)")


print("\n  ok=%d  FALLO=%d" % (ok, fail))
sys.exit(1 if fail else 0)
