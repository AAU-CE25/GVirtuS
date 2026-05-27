#!/usr/bin/env python3
import csv
import sys
csv.field_size_limit(sys.maxsize)
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "tcp"
if MODE not in {"tcp", "rdma", "ucx"}:
    print("Usage: ./benchmark.sh {tcp|rdma|ucx}", file=sys.stderr)
    sys.exit(2)

GVIRTUS_HOME = os.environ.get("GVIRTUS_HOME", "/home/student.aau.dk/ul11nh/GVirtuS")
LZ4_HOME = os.environ.get("LZ4_HOME", "/home/student.aau.dk/ul11nh/lz4-install")
OPENCV_HOME = os.environ.get("OPENCV_HOME", "/home/student.aau.dk/ul11nh/opencv-local")
CUDNN_ROOT = os.environ.get("CUDNN_ROOT", "/home/student.aau.dk/ul11nh/cudnn-9.5.1")
NPP_DIR = os.environ.get("NPP_DIR", "/home/student.aau.dk/ul11nh/.local/lib/python3.10/site-packages/nvidia/npp/lib")

RUNS = int(os.environ.get("RUNS", "1"))
WARMUPS = int(os.environ.get("WARMUPS", "0"))
RUN_TIMEOUT = float(os.environ.get("RUN_TIMEOUT", "180"))
FRONTEND_CMD = os.environ.get("FRONTEND_CMD", "./sample")
GVIRTUS_LOGLEVEL = os.environ.get("GVIRTUS_LOGLEVEL", "10000")
IFACES = os.environ.get("IFACES", "ens1f1np1 ens1f0np0 bond0").split()

CONFIGS = {
    "tcp": os.environ.get("TCP_CONFIG", f"{GVIRTUS_HOME}/etc/properties.json"),
    "rdma": os.environ.get("RDMA_CONFIG", f"{GVIRTUS_HOME}/etc/properties_plain_rdma.json"),
    "ucx": os.environ.get("UCX_CONFIG", f"{GVIRTUS_HOME}/etc/properties_ucx.json"),
}
CONFIG = CONFIGS[MODE]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("benchmark_results") / f"opencvyolo_full_metrics_{ts}_{MODE}"
logs_dir = out_dir / "logs"
counters_dir = out_dir / "counters"
system_dir = out_dir / "system"
for d in [logs_dir, counters_dir, system_dir]:
    d.mkdir(parents=True, exist_ok=True)

results_csv = out_dir / "results.csv"
routine_calls_csv = out_dir / "routine_calls.csv"
routine_summary_csv = out_dir / "routine_summary.csv"
nic_csv = out_dir / "nic_counters.csv"

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def read_counter(iface, name):
    p = Path("/sys/class/net") / iface / "statistics" / name
    try:
        return int(p.read_text().strip())
    except Exception:
        return 0

def capture_net():
    return {
        iface: {
            "rx": read_counter(iface, "rx_bytes"),
            "tx": read_counter(iface, "tx_bytes"),
        }
        for iface in IFACES
    }

def delta_net(before, after):
    rx = 0
    tx = 0
    for iface in IFACES:
        rx += max(0, after.get(iface, {}).get("rx", 0) - before.get(iface, {}).get("rx", 0))
        tx += max(0, after.get(iface, {}).get("tx", 0) - before.get(iface, {}).get("tx", 0))
    return rx, tx

def build_env():
    env = os.environ.copy()

    ld_parts = [
        f"{GVIRTUS_HOME}/lib",
        f"{GVIRTUS_HOME}/lib/frontend",
        f"{OPENCV_HOME}/lib",
        f"{OPENCV_HOME}/lib64",
        f"{CUDNN_ROOT}/lib",
        NPP_DIR,
        "/usr/local/cuda-12.6/lib64",
        f"{LZ4_HOME}/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_parts if p])

    env["GVIRTUS_CONFIG"] = CONFIG
    env["GVIRTUS_HOME"] = GVIRTUS_HOME
    env["GVIRTUS_LOGLEVEL"] = GVIRTUS_LOGLEVEL

    env["LD_PRELOAD"] = ":".join([
        f"{GVIRTUS_HOME}/lib/frontend/libcudart.so",
        f"{GVIRTUS_HOME}/lib/frontend/libcublas.so",
        f"{GVIRTUS_HOME}/lib/frontend/libcudnn.so.9",
    ])

    return env

routine_re = re.compile(r"Routine '([^']+)' returned .*?in=(\d+)B .*?out=(\d+)B")
routine_re_loose = re.compile(r"Routine '([^']+)' returned")

def parse_log(text):
    calls = []
    for m in routine_re.finditer(text):
        calls.append((m.group(1), int(m.group(2)), int(m.group(3))))

    # Fallback for logs that contain routine names but not byte fields.
    if not calls:
        for m in routine_re_loose.finditer(text):
            calls.append((m.group(1), 0, 0))

    counts = Counter(name for name, _, _ in calls)
    g_in = sum(x for _, x, _ in calls)
    g_out = sum(x for _, _, x in calls)

    routine_counts = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    ms = ""
    patterns = [
        r"Time taken:\s*([0-9.]+)\s*ms",
        r"Inference(?: time)?[:=]\s*([0-9.]+)\s*ms",
        r"inference_ms[:=]\s*([0-9.]+)",
        r"Detection(?: time)?[:=]\s*([0-9.]+)\s*ms",
    ]
    for pat in patterns:
        mm = re.search(pat, text, re.IGNORECASE)
        if mm:
            ms = mm.group(1)
            break

    return calls, counts, g_in, g_out, routine_counts, ms

def run_one(phase, idx):
    log_path = logs_dir / f"{MODE}_{phase}_{idx}.log"

    # Avoid accepting a stale image.
    try:
        Path("output.jpg").unlink()
    except FileNotFoundError:
        pass

    before = capture_net()
    t0 = time.monotonic()

    proc = subprocess.Popen(
        FRONTEND_CMD,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=build_env(),
        executable="/bin/bash",
    )

    timed_out = False
    try:
        output, _ = proc.communicate(timeout=RUN_TIMEOUT)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        output, _ = proc.communicate()
        exit_code = 124

    wall = time.monotonic() - t0
    after = capture_net()
    nic_rx, nic_tx = delta_net(before, after)

    log_path.write_text(output or "")

    calls, counts, g_in, g_out, routine_counts, inference_ms = parse_log(output or "")

    has_output = Path("output.jpg").exists() and Path("output.jpg").stat().st_size > 0
    valid_output = (exit_code == 0 and "Detection finished" in (output or "") and has_output)
    status = "OK" if valid_output else "FAILED"

    error = ""
    if timed_out:
        error = "timeout"

    with open(nic_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "mode", "phase", "run", "iface", "rx_before", "rx_after", "rx_delta", "tx_before", "tx_after", "tx_delta"])
        for iface in IFACES:
            rb = before.get(iface, {}).get("rx", 0)
            ra = after.get(iface, {}).get("rx", 0)
            tb = before.get(iface, {}).get("tx", 0)
            ta = after.get(iface, {}).get("tx", 0)
            w.writerow([now_iso(), MODE, phase, idx, iface, rb, ra, max(0, ra-rb), tb, ta, max(0, ta-tb)])

    with open(routine_calls_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "mode", "phase", "run", "routine", "in_B", "out_B"])
        for name, ib, ob in calls:
            w.writerow([now_iso(), MODE, phase, idx, name, ib, ob])

    row = {
        "timestamp": now_iso(),
        "mode": MODE,
        "phase": phase,
        "run": idx,
        "status": status,
        "exit_code": exit_code,
        "wall_s": f"{wall:.6f}",
        "detection_ms": inference_ms,
        "valid_output": str(valid_output).lower(),
        "calls": len(calls),
        "gvirtus_in_B": g_in if calls else "",
        "gvirtus_out_B": g_out if calls else "",
        "nic_rx_B": nic_rx,
        "nic_tx_B": nic_tx,
        "config": CONFIG,
        "log_file": str(log_path),
        "error": error,
        "routine_counts": routine_counts,
    }

    with open(results_csv, "a", newline="") as f:
        fieldnames = list(row.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)

    print(
        f"  {status} wall={wall:.6f}s detection_ms={inference_ms or 'NA'} "
        f"calls={len(calls)} in={g_in if calls else 'NA'}B out={g_out if calls else 'NA'}B "
        f"nic_tx={nic_tx}B nic_rx={nic_rx}B"
    )

    return row

print(f"Benchmark output directory:\n{out_dir}")
print()
print("=" * 60)
print(f"Mode: {MODE}")
print(f"Config: {CONFIG}")
print(f"Frontend: {FRONTEND_CMD}")
print(f"Runs: {RUNS} measured + {WARMUPS} warmup")
print(f"IFACES: {' '.join(IFACES)}")
print("=" * 60)

(system_dir / "system_metadata.txt").write_text(
    f"timestamp={now_iso()}\n"
    f"mode={MODE}\n"
    f"config={CONFIG}\n"
    f"frontend_cmd={FRONTEND_CMD}\n"
    f"runs={RUNS}\n"
    f"warmups={WARMUPS}\n"
    f"run_timeout={RUN_TIMEOUT}\n"
    f"gvirtus_home={GVIRTUS_HOME}\n"
    f"gvirtus_loglevel={GVIRTUS_LOGLEVEL}\n"
    f"ifaces={' '.join(IFACES)}\n"
)

all_rows = []
for i in range(1, WARMUPS + 1):
    print(f"[{MODE}] warmup run {i}...")
    all_rows.append(run_one("warmup", i))

for i in range(1, RUNS + 1):
    print(f"[{MODE}] measure run {i}...")
    all_rows.append(run_one("measure", i))

# Aggregate routine summary from routine_calls.csv.
summary = defaultdict(lambda: [0, 0, 0])
if routine_calls_csv.exists():
    with open(routine_calls_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] == "measure":
                summary[r["routine"]][0] += 1
                summary[r["routine"]][1] += int(r["in_B"] or 0)
                summary[r["routine"]][2] += int(r["out_B"] or 0)

with open(routine_summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["routine", "count", "total_in_B", "total_out_B"])
    for routine, (count, total_in, total_out) in sorted(summary.items()):
        w.writerow([routine, count, total_in, total_out])

print()
print("Done.")
print(f"Main CSV:              {results_csv}")
print(f"Individual calls CSV:  {routine_calls_csv}")
print(f"Routine summary CSV:   {routine_summary_csv}")
print(f"NIC CSV:               {nic_csv}")
print(f"Logs:                  {logs_dir}")
print(f"Counters:              {counters_dir}")
print(f"System metadata:       {system_dir}")

# Simple measured summary.
measured_ok = [
    r for r in csv.DictReader(open(results_csv, newline=""))
    if r["phase"] == "measure" and r["status"] == "OK"
]
if measured_ok:
    def avg(field):
        vals = [float(r[field]) for r in measured_ok if str(r.get(field, "")).strip()]
        return sum(vals) / len(vals) if vals else float("nan")
    print()
    print("Measured summary:")
    print(
        f"{MODE}: n={len(measured_ok)} "
        f"mean_wall_s={avg('wall_s'):.6f} "
        f"mean_detection_ms={avg('detection_ms') if str(avg('detection_ms')) != 'nan' else 'NA'} "
        f"mean_calls={avg('calls'):.1f} "
        f"mean_gvirtus_in_B={avg('gvirtus_in_B'):.1f} "
        f"mean_gvirtus_out_B={avg('gvirtus_out_B'):.1f} "
        f"mean_nic_rx_B={avg('nic_rx_B'):.1f} "
        f"mean_nic_tx_B={avg('nic_tx_B'):.1f}"
    )
