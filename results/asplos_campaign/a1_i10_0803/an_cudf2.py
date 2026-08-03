import json, glob, os, time, statistics as st, collections
# SOLO los ficheros escritos por esta corrida. /tmp/cudfout/ conserva campanas anteriores con
# los mismos nombres para reps >2, y mezclarlas da una retencion del 24,6 % que es puro
# artefacto: las reps 3-5 de N=8 son de otra campana (2633 ms) y las 1-2 son de esta (639 ms).
CORTE = time.time() - 3600
por, cks, viejos = collections.defaultdict(list), collections.defaultdict(set), []
for f in glob.glob("/tmp/cudfout/*_n*_r*.jsonl"):
    b = os.path.basename(f)[:-6]
    cfg = b.rsplit("_n", 1)[0]
    if cfg not in ("native", "gpudirect"): continue
    if os.path.getmtime(f) < CORTE:
        viejos.append(b); continue
    n = int(b.rsplit("_n", 1)[1].split("_r")[0]); rep = int(b.split("_r")[-1])
    for line in open(f):
        d = json.loads(line)
        if d.get("status") != "ok": continue
        por[(cfg, n, rep)].append(d["latency_s"] * 1000.0)
        c = d.get("checksum")
        if c is not None: cks[(n, d["client_id"])].add(round(c, 6))

print("descartados por antiguedad: %d ficheros  %s" % (len(viejos), sorted(viejos)[:6]))
print("\ncfg,N,rep,batches,mediana_ms")
res = collections.defaultdict(list)
for k in sorted(por):
    m = st.median(por[k]); res[(k[0], k[1])].append(m)
    print("%s,%d,%d,%d,%.1f" % (k[0], k[1], k[2], len(por[k]), m))

pub = {1: (319.8, 356.1, 89.8), 8: (637.0, 631.5, 100.9)}
print("\n%-6s %9s %11s %11s   %s" % ("N", "native", "gpudirect", "retencion", "publicado (60 batches, 5 reps)"))
for n in (1, 8):
    if ("native", n) not in res or ("gpudirect", n) not in res: continue
    nat = st.median(res[("native", n)]); gd = st.median(res[("gpudirect", n)])
    print("N=%-4d %9.1f %11.1f %10.1f %%   nativo %.1f · gd %.1f · ret %.1f %%"
          % (n, nat, gd, nat / gd * 100, pub[n][0], pub[n][1], pub[n][2]))
malos = {k: v for k, v in cks.items() if len(v) != 1}
print("\ncorrectitud: %d de %d (N,cliente) con checksum inconsistente" % (len(malos), len(cks)))
