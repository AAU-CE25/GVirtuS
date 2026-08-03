import csv, glob, statistics as st, collections, os
D = "/home/student.aau.dk/ll33pq/cross_out"
datos = collections.defaultdict(dict)
for f in sorted(glob.glob(D + "/out_*.csv")):
    arm, mem, rep = os.path.basename(f)[4:-4].split("_")
    for r in csv.DictReader(open(f)):
        datos[(arm, mem, r["direction"], int(r["bytes"]))][rep] = float(r["gbytes_per_s"])
def kib(b): return ("%dK" % (b >> 10)) if b < (1 << 20) else ("%dM" % (b >> 20))
tallas = sorted({k[3] for k in datos})

print("=== H2D pinned: las tres corridas de cada brazo, para ver si el cruce es robusto ===")
print("%6s | %-22s | %-22s | %-22s | %-22s" % ("size","AM","assume","fence","flush"))
for b in tallas:
    fila = []
    for a in ("am","assume","fence","flush"):
        v = datos[(a,"pinned","h2d",b)]
        fila.append(" ".join("%.3f" % v[r] for r in sorted(v)))
    print("%6s | %-22s | %-22s | %-22s | %-22s" % (kib(b), *fila))

print("\n=== ¿gana AM a TODOS los brazos RMA en cada corrida? (H2D pinned) ===")
for b in tallas:
    am = datos[("am","pinned","h2d",b)]
    for a in ("assume","flush"):
        v = datos[(a,"pinned","h2d",b)]
        gana = sum(1 for r in sorted(v) if v[r] > am[r])
        print("  %6s  %-7s gana a AM en %d/3 corridas" % (kib(b), a, gana), end="")
    print()

print("\n=== coste de flush frente a assume, por corrida (H2D pinned) ===")
print("%6s | %-24s | mediana" % ("size","delta por corrida %"))
for b in tallas:
    A = datos[("assume","pinned","h2d",b)]; F = datos[("flush","pinned","h2d",b)]
    ds = [(F[r]-A[r])/A[r]*100 for r in sorted(A)]
    print("%6s | %-24s | %+6.2f %%" % (kib(b), " ".join("%+6.2f" % d for d in ds), st.median(ds)))
