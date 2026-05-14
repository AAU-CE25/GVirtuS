#!/usr/bin/env python3
"""
Summarize / plot data_copy_bench sweep results.

Input CSV columns:
    transport,size_bytes,run,n_bytes,send_us,recv_us,roundtrip_us

Usage:
    ./plot_sweep.py results.csv               # prints summary table
    ./plot_sweep.py results.csv plot.png      # also writes PNG (requires matplotlib)
"""
import csv
import sys
from collections import defaultdict
from statistics import median

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

csv_path = sys.argv[1]
png_path = sys.argv[2] if len(sys.argv) > 2 else None

# data[transport][size] = list of roundtrip_us, dropping run 0 as warmup
data = defaultdict(lambda: defaultdict(list))
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if int(row["run"]) == 0:
            continue
        data[row["transport"]][int(row["size_bytes"])].append(
            float(row["roundtrip_us"])
        )

transports = sorted(data.keys())
sizes = sorted({s for t in data.values() for s in t.keys()})

# Print table: median roundtrip + effective bandwidth (GB/s)
print(f"{'size':>10}  " + "  ".join(f"{t:>22}" for t in transports))
print(f"{'(bytes)':>10}  " + "  ".join(f"{'med_us  /  GB/s':>22}" for _ in transports))
for s in sizes:
    cells = []
    for t in transports:
        vals = data[t].get(s, [])
        if not vals:
            cells.append(f"{'--':>22}")
            continue
        med = median(vals)
        bw = (s / (med * 1e-6)) / 1e9  # bytes per second -> GB/s
        cells.append(f"{med:>10.1f}  /  {bw:>5.2f}")
    print(f"{s:>10}  " + "  ".join(cells))

if png_path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PNG", file=sys.stderr)
        sys.exit(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for t in transports:
        xs, ys_us, ys_bw = [], [], []
        for s in sizes:
            vals = data[t].get(s, [])
            if not vals:
                continue
            med = median(vals)
            xs.append(s)
            ys_us.append(med)
            ys_bw.append((s / (med * 1e-6)) / 1e9)
        ax1.plot(xs, ys_us, "o-", label=t)
        ax2.plot(xs, ys_bw, "o-", label=t)

    for ax, ylabel in [(ax1, "Roundtrip latency (µs)"),
                       (ax2, "Effective bandwidth (GB/s)")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Payload size (bytes)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax1.set_yscale("log")

    fig.suptitle("data_copy_bench: TCP vs UCX(TCP) vs UCX(RoCE)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}")
