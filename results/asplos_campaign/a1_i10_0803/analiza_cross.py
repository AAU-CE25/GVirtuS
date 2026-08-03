import csv, glob, statistics as st, collections, os

D = "/home/student.aau.dk/ll33pq/cross_out"
datos = collections.defaultdict(list)      # (arm,mem,dir,bytes) -> [GB/s por rep]
for f in sorted(glob.glob(D + "/out_*.csv")):
    base = os.path.basename(f)[4:-4]        # arm_mem_rN
    arm, mem, rep = base.split("_")
    for r in csv.DictReader(open(f)):
        datos[(arm, mem, r["direction"], int(r["bytes"]))].append(float(r["gbytes_per_s"]))

def kib(b):
    return ("%d KiB" % (b >> 10)) if b < (1 << 20) else ("%d MiB" % (b >> 20))

ARMS = ["am", "assume", "fence", "flush"]
tallas = sorted({k[3] for k in datos})

print("Mediana de %d corridas independientes por celda. GB/s.\n" % 3)
resumen = {}
for mem in ("pinned", "pageable"):
    for d in ("h2d", "d2h"):
        print("--- memory=%s direction=%s" % (mem, d))
        print("%10s %9s %9s %9s %9s   %-10s %s" %
              ("size", "AM", "RMA/assume", "RMA/fence", "RMA/flush", "best", "flush vs assume"))
        for b in tallas:
            v = {}
            for a in ARMS:
                xs = datos.get((a, mem, d, b), [])
                v[a] = st.median(xs) if xs else float("nan")
            mejor = max(ARMS, key=lambda a: v[a])
            delta = (v["flush"] - v["assume"]) / v["assume"] * 100.0
            print("%10s %9.3f %9.3f %9.3f %9.3f   %-10s %+7.2f %%" %
                  (kib(b), v["am"], v["assume"], v["fence"], v["flush"], mejor, delta))
            resumen[(mem, d, b)] = v
        # cruce sostenido: primer tamano a partir del cual RMA/flush gana en TODOS los mayores
        for arm in ("assume", "flush"):
            cruce = None
            for i, b in enumerate(tallas):
                if all(resumen[(mem, d, x)][arm] > resumen[(mem, d, x)]["am"] for x in tallas[i:]):
                    cruce = b; break
            print("   cruce sostenido (%s vs AM): %s" % (arm, kib(cruce) if cruce else "ninguno en el rango"))
        print()
