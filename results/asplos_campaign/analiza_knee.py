#!/usr/bin/env python3
"""Analysis for the knee sweep (lambda 0.55-0.70, windows scaled to >=120 offered requests).

Reports, per system and load: p95 and p99 of TTFT as the mean over repetitions with the observed
range, the SLO attainment, the backlog growth, and capacity under the SAME all-repetitions
criterion the v2 sweep used, so the two are comparable.
"""
import csv, re, os, statistics as st, random, collections
random.seed(13)
D = os.path.expanduser("~/GVirtuS/results/asplos_campaign/llama_slo_knee")
rows = list(csv.DictReader(open(os.path.join(D, "summary.csv"))))

def key(r):
    m = re.match(r"kn_(\w+?)_n(\d+)_l([\d.]+)_r(\d+)$", r["label"])
    return m.groups() if m else None

cel = collections.defaultdict(list)
for r in rows:
    k = key(r)
    if k:
        cel[(k[0], k[1], k[2])].append(r)

LAMS = ["0.55", "0.6", "0.65", "0.7"]
SYS = [("bm", "native"), ("bmmps", "native+MPS"), ("ucx", "Gusto")]
Ns = sorted({k[1] for k in cel})

def f(v, c):
    return [float(x[c]) for x in v]

for N in Ns:
    print("=" * 78)
    print("N =", N)
    print("%-12s %-6s %3s %7s %-17s %-17s %7s %8s" %
          ("system", "lam", "n", "offered", "TTFT p95 [range]", "TTFT p99 [range]",
           "SLO1s%", "backlog"))
    for s, lab in SYS:
        for lam in LAMS:
            v = cel.get((s, N, lam), [])
            if not v:
                continue
            p95, p99 = f(v, "ttft_p95"), f(v, "ttft_p99")
            slo = f(v, "pct_slo_1s")
            bl = f(v, "backlog_end")
            off = f(v, "offered")
            print("%-12s %-6s %3d %7.0f %6.0f [%5.0f-%5.0f] %6.0f [%5.0f-%5.0f] %7.1f %8.1f" %
                  (lab, lam, len(v), st.mean(off), st.mean(p95), min(p95), max(p95),
                   st.mean(p99), min(p99), max(p99), st.mean(slo), st.mean(bl)))
    print()
    print("  capacity, all-repetitions criterion (every rep: TTFT p95 <= 1000 ms and 0 timeouts)")
    for s, lab in SYS:
        best = None
        for lam in LAMS:
            v = cel.get((s, N, lam), [])
            if not v:
                continue
            ok = all(float(x["ttft_p95"]) <= 1000 and int(float(x["timeout"])) == 0 for x in v)
            if ok:
                best = (lam, st.mean(f(v, "goodput")), len(v))
        print("    %-12s -> %s" % (lab, ("lambda=%s, goodput %.1f t/s (n=%d)" % best) if best
                                   else "NO load in this grid meets it in every repetition"))
    print()
    # Paired differences on p95, Gusto minus each baseline, matched on (lambda, rep)
    def boot(d, B=20000):
        m = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(B))
        return m[500], m[19499]
    for other, olab in (("bm", "native"), ("bmmps", "native+MPS")):
        d = []
        for lam in LAMS:
            a = {key(x)[3]: float(x["ttft_p95"]) for x in cel.get(("ucx", N, lam), [])}
            b = {key(x)[3]: float(x["ttft_p95"]) for x in cel.get((other, N, lam), [])}
            for rep in sorted(set(a) & set(b)):
                d.append(a[rep] - b[rep])
        if len(d) >= 3:
            lo, hi = boot(d)
            print("  paired TTFT p95, Gusto - %-11s = %+7.0f ms  CI95 [%+.0f, %+.0f]  %s (n=%d)" %
                  (olab, st.mean(d), lo, hi,
                   "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0", len(d)))
    print()
