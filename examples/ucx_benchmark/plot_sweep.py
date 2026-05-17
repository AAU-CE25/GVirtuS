#!/usr/bin/env python3
"""
Summarize / plot matrix_sweep benchmark results.

Input: one or more CSV files with lines like:
    transport,run,n,matrix_bytes,h2d_us,compute_us,d2h_us,total_us
(Log lines mixed in are automatically skipped.)

Usage:
    ./plot_sweep.py file1.csv [file2.csv ...]               # prints summary table
    ./plot_sweep.py file1.csv [file2.csv ...] --png out.png # also writes PNG
"""
import sys
from collections import defaultdict
from statistics import median

def parse_csv_files(paths):
    """Parse matrix sweep CSVs, skipping log/header lines.
    Returns data[transport][matrix_bytes] = {h2d: [], compute: [], d2h: [], total: []}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 8:
                    continue
                transport, run_s, n_s, mb_s, h2d_s, comp_s, d2h_s, tot_s = parts
                # Skip header lines and log lines
                try:
                    run = int(run_s)
                    matrix_bytes = int(mb_s)
                    h2d = float(h2d_s)
                    compute = float(comp_s)
                    d2h = float(d2h_s)
                    total = float(tot_s)
                except ValueError:
                    continue
                # Drop run 0 as warmup
                if run == 0:
                    continue
                data[transport][matrix_bytes]["h2d"].append(h2d)
                data[transport][matrix_bytes]["compute"].append(compute)
                data[transport][matrix_bytes]["d2h"].append(d2h)
                data[transport][matrix_bytes]["total"].append(total)
    return data

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

# Separate CSV files from --png argument
csv_paths = []
png_path = None
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--png" and i + 1 < len(sys.argv):
        png_path = sys.argv[i + 1]
        i += 2
    else:
        csv_paths.append(sys.argv[i])
        i += 1

if not csv_paths:
    print("No CSV files specified.")
    sys.exit(1)

data = parse_csv_files(csv_paths)

transports = sorted(data.keys())
sizes = sorted({s for t in data.values() for s in t.keys()})

# Print summary table
print(f"\n{'Matrix':>12}  {'Bytes':>12}  ", end="")
print("  ".join(f"{t:>30}" for t in transports))
print(f"{'N':>12}  {'':>12}  ", end="")
print("  ".join(f"{'h2d / comp / d2h / total (µs)':>30}" for _ in transports))
print("-" * (28 + 32 * len(transports)))

for s in sizes:
    n = int(s ** 0.5 / 4 ** 0.5)  # matrix_bytes = n*n*4
    # Reverse: n = sqrt(matrix_bytes / sizeof(float))
    import math
    n = int(math.sqrt(s / 4))
    cells = []
    for t in transports:
        entry = data[t].get(s)
        if not entry or not entry["total"]:
            cells.append(f"{'--':>30}")
            continue
        mh = median(entry["h2d"])
        mc = median(entry["compute"])
        md = median(entry["d2h"])
        mt = median(entry["total"])
        cells.append(f"{mh:>6.0f} / {mc:>4.0f} / {md:>6.0f} / {mt:>6.0f}")
    print(f"{n:>12}  {s:>12}  " + "  ".join(cells))

# Speedup summary
if len(transports) == 2:
    t1, t2 = transports  # alphabetical: tcp, ucx-rdma
    print(f"\n{'Speedup':>12} ({t1} / {t2}):")
    for s in sizes:
        n = int(math.sqrt(s / 4))
        e1 = data[t1].get(s)
        e2 = data[t2].get(s)
        if e1 and e2 and e1["total"] and e2["total"]:
            sp = median(e1["total"]) / median(e2["total"])
            print(f"  N={n:>5}: {sp:.2f}x")

if png_path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import math
    except ImportError:
        print("matplotlib not available; skipping PNG", file=sys.stderr)
        sys.exit(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("h2d", "Host→Device (µs)"), ("compute", "Compute (µs)"),
               ("d2h", "Device→Host (µs)"), ("total", "Total (µs)")]

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for t in transports:
            xs, ys = [], []
            for s in sizes:
                entry = data[t].get(s)
                if not entry or not entry[metric]:
                    continue
                xs.append(int(math.sqrt(s / 4)))  # N dimension
                ys.append(median(entry[metric]))
            label = t.replace("ucx-rdma", "UCX/RDMA (RoCEv2)").replace("tcp", "TCP")
            ax.plot(xs, ys, "o-", label=label, linewidth=2, markersize=6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Matrix dimension N")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Matrix Multiply Benchmark: TCP vs UCX/RDMA (RoCEv2)\n"
                 "NxN float32 matrices, median of runs 1-9", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"\nWrote {png_path}")
