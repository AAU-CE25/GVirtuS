import csv, statistics as st, random, collections, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/a1_matrix.csv"
WARM = 2          # transferencias 1-2 = construccion del pool, se descartan
rows = list(csv.DictReader(open(PATH)))

# --- correccion: ninguna celda vale si alguna transferencia fallo el checksum ---------
bad = [r for r in rows if r["device_ck_ok"] != "pass" or r["host_ck_ok"] != "pass"]
print("filas totales: %d | checksums fallidos: %d" % (len(rows), len(bad)))
if bad:
    print("  !! ", bad[:3])

por_corrida = collections.defaultdict(list)
for r in rows:
    if int(r["transfer"]) <= WARM:
        continue
    por_corrida[(r["cell"], r["rep"])].append(float(r["h2d_GBps"]))

# un valor por CORRIDA = mediana de sus transferencias en estado estable
celda = collections.defaultdict(list)
for (c, rep), v in sorted(por_corrida.items()):
    celda[c].append(st.median(v))

def boot(xs, it=20000, seed=7):
    rnd = random.Random(seed)
    ms = sorted(st.median([rnd.choice(xs) for _ in xs]) for _ in range(it))
    return ms[int(.025 * it)], ms[int(.975 * it)]

orden = ["assume", "fence", "flush", "strict"]
print("\n%-8s %3s %8s %8s %-18s %8s" % ("celda", "n", "mediana", "min-max", "IC95 bootstrap", "vs assume"))
base = st.median(celda["assume"]) if "assume" in celda else None
for c in orden:
    if c not in celda: continue
    xs = celda[c]; m = st.median(xs); lo, hi = boot(xs)
    d = "" if base is None else ("%+.2f %%" % ((m - base) / base * 100.0))
    print("%-8s %3d %8.3f %8s [%6.3f, %6.3f] %9s" %
          (c, len(xs), m, "%.2f-%.2f" % (min(xs), max(xs)), lo, hi, d))

print("\nper-run medians (GB/s):")
for c in orden:
    if c in celda:
        print("  %-8s %s" % (c, " ".join("%.2f" % x for x in sorted(celda[c]))))
