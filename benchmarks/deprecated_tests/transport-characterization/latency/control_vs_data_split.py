#!/usr/bin/env python3
"""Control-path (AM) vs data-path (bulk RDMA/GPUDirect) split during llama decode.
Uses lat_rdma.csv (routine,payload_bytes,rt_us,server_us) from GVIRTUS_LATENCY_TRACE.
Answers: is llama slow because AM is slow, or because it is control-path DOMINATED
with the data path barely used?
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

rt, pay, rout = [], [], []
with open(os.path.join(HERE, "lat_rdma.csv")) as f:
    for row in csv.DictReader(f):
        try:
            rt.append(float(row["rt_us"])); pay.append(int(row["payload_bytes"])); rout.append(row["routine"])
        except (ValueError, KeyError):
            pass
rt = np.array(rt); pay = np.array(pay); rout = np.array(rout)

tot_n = len(rt); tot_us = rt.sum()
print(f"TOTAL RPCs: {tot_n}   aggregate client wall time in RPCs: {tot_us/1e6:.2f} s\n")

# payload buckets: control-plane (small AM) vs data-path (bulk)
buckets = [(0,1,"0 B (pure control)"),(1,1024,"1 B–1 KiB"),(1024,4096,"1–4 KiB"),
           (4096,65536,"4–64 KiB"),(65536,1<<20,"64 KiB–1 MiB"),(1<<20,1<<40,">= 1 MiB (bulk)")]
print(f"{'payload bucket':22} {'count':>7} {'%cnt':>6} {'sum_ms':>9} {'%time':>6} {'mean_us':>8}")
for lo,hi,name in buckets:
    m = (pay>=lo)&(pay<hi)
    if not m.any(): 
        print(f"{name:22} {0:7d}"); continue
    c=m.sum(); s=rt[m].sum()
    print(f"{name:22} {c:7d} {100*c/tot_n:5.1f}% {s/1000:8.1f} {100*s/tot_us:5.1f}% {rt[m].mean():8.1f}")

# control (<4KiB) vs data (>=4KiB) summary
ctrl = pay < 4096; data = ~ctrl
print(f"\nCONTROL path (<4 KiB):  {ctrl.sum():6d} RPCs ({100*ctrl.sum()/tot_n:.1f}%),  "
      f"{rt[ctrl].sum()/1e6:.2f} s ({100*rt[ctrl].sum()/tot_us:.1f}% of RPC time)")
print(f"DATA    path (>=4 KiB): {data.sum():6d} RPCs ({100*data.sum()/tot_n:.1f}%),  "
      f"{rt[data].sum()/1e6:.2f} s ({100*rt[data].sum()/tot_us:.1f}% of RPC time)")

# top routines by count and by aggregate time
print("\nTop routines by COUNT:")
names,counts = np.unique(rout, return_counts=True)
for i in np.argsort(counts)[::-1][:8]:
    n=names[i]; m=rout==n
    print(f"  {n:28} {counts[i]:7d}  {100*counts[i]/tot_n:4.1f}%  aggUS={rt[m].sum()/1000:8.1f}ms  meanUS={rt[m].mean():6.1f}")

print("\nTop routines by AGGREGATE TIME:")
agg = {n: rt[rout==n].sum() for n in names}
for n in sorted(agg, key=agg.get, reverse=True)[:8]:
    m=rout==n
    print(f"  {n:28} {rt[m].sum()/1000:8.1f}ms  ({100*rt[m].sum()/tot_us:4.1f}%)  n={m.sum():6d}  meanUS={rt[m].mean():6.1f}  maxpay={pay[m].max()}")

# the ONE-TIME bulk weight-load transfers (largest payloads)
big = pay >= (1<<20)
if big.any():
    print(f"\nBulk (>=1 MiB) transfers: {big.sum()} RPCs, {rt[big].sum()/1000:.1f} ms total, "
          f"largest payload {pay[big].max()/1e6:.0f} MB — these are the ONE-TIME weight load, not per-token.")
