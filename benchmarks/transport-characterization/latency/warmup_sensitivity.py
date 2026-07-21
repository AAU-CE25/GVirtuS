#!/usr/bin/env python3
"""Warmup-sensitivity check for the per-RPC latency distributions.
Recompute control-plane (<4KiB) percentiles three ways per transport:
  (a) ALL samples (as originally reported)
  (b) EXCLUDE the one-time registration/setup phase (cudaRegister*/cuda*Unregister*/
      cudaMalloc/cudaMemcpy weight-load) -> steady-state only
  (c) LAST 50% of samples (pure steady-state generation, warmup fully discarded)
If p50/p99 barely move, the reported tail result is warmup-robust.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSPORTS = ["tcp", "rdma", "gpudirect"]
LABEL = {"tcp": "TCP", "rdma": "RDMA", "gpudirect": "GPUDirect"}
SETUP = ("cudaRegister", "cudaUnregister", "__cudaRegister")

def load(t):
    rt, pay, rout = [], [], []
    with open(os.path.join(HERE, f"lat_{t}.csv")) as f:
        for row in csv.DictReader(f):
            try:
                rt.append(float(row["rt_us"])); pay.append(int(row["payload_bytes"]))
                rout.append(row["routine"])
            except (ValueError, KeyError):
                pass
    return np.array(rt), np.array(pay), np.array(rout)

def pcts(a):
    return [np.percentile(a, p) for p in (50, 90, 99, 99.9)] + [a.max()]

print(f"{'transport':10} {'subset':22} {'n':>7} {'p50':>7} {'p90':>7} {'p99':>8} {'p99.9':>9} {'max':>9} {'p99/p50':>8}")
for t in TRANSPORTS:
    rt, pay, rout = load(t)
    ctrl = pay < 4096
    is_setup = np.array([any(r.startswith(s) for s in SETUP) for r in rout])
    subsets = {
        "(a) all <4KiB": rt[ctrl],
        "(b) steady (no setup)": rt[ctrl & ~is_setup],
        "(c) last 50%": rt[ctrl][len(rt[ctrl])//2:],
    }
    for name, a in subsets.items():
        if not len(a): continue
        p = pcts(a)
        print(f"{LABEL[t]:10} {name:22} {len(a):7d} {p[0]:7.1f} {p[1]:7.1f} {p[2]:8.1f} {p[3]:9.1f} {p[4]:9.1f} {p[2]/p[0]:8.2f}")
    print()
