#!/usr/bin/env python3
import csv, os, re, sys, time, subprocess, statistics as st
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

MODE = sys.argv[1] if len(sys.argv) > 1 else "tcp"
if MODE not in {"tcp", "rdma", "ucx"}:
    print("Usage: ./benchmark.sh {tcp|rdma|ucx}", file=sys.stderr)
    sys.exit(2)

GVIRTUS_HOME = os.environ.get("GVIRTUS_HOME", "/home/student.aau.dk/ul11nh/GVirtuS")
LZ4_HOME = os.environ.get("LZ4_HOME", "/home/student.aau.dk/ul11nh/lz4-install")
RUNS = int(os.environ.get("RUNS", "1"))
WARMUPS = int(os.environ.get("WARMUPS", "0"))
RUN_TIMEOUT = float(os.environ.get("RUN_TIMEOUT", "300"))
GVIRTUS_LOGLEVEL = os.environ.get("GVIRTUS_LOGLEVEL", "10000")
IFACES = os.environ.get("IFACES", "ens1f1np1 ens1f0np0 bond0").split()
SIZES = [int(x) for x in os.environ.get("SIZES", "1024 2048 4096 8192 16384").split()]
FRONTEND_CMD_TEMPLATE = os.environ.get("FRONTEND_CMD_TEMPLATE", "MATRIX_N={size} ./simple_matrix")

CONFIGS = {
    "tcp": os.environ.get("TCP_CONFIG", f"{GVIRTUS_HOME}/etc/properties.json"),
    "rdma": os.environ.get("RDMA_CONFIG", f"{GVIRTUS_HOME}/etc/properties_plain_rdma.json"),
    "ucx": os.environ.get("UCX_CONFIG", f"{GVIRTUS_HOME}/etc/properties_ucx.json"),
}
CONFIG = CONFIGS[MODE]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("benchmark_results") / f"simplematrix_full_metrics_{ts}_{MODE}"
logs_dir = out_dir / "logs"
counters_dir = out_dir / "counters"
system_dir = out_dir / "system"
for d in (logs_dir, counters_dir, system_dir):
    d.mkdir(parents=True, exist_ok=True)

results_csv = out_dir / "results.csv"
routine_calls_csv = out_dir / "routine_calls.csv"
routine_summary_csv = out_dir / "routine_summary.csv"
nic_csv = out_dir / "nic_counters.csv"

routine_re = re.compile(r"Routine '([^']+)' returned .*?in=(\d+)B .*?out=(\d+)B")
routine_re_loose = re.compile(r"Routine '([^']+)' returned")
error_re = re.compile(r"FAILED|Aborted|dumped core|terminate|Exception|timeout|FATAL|transport retry|Work Request Flushed|Destination is unreachable|Unsupported", re.I)

def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def payload_label(n):
    b = n * n * 4
    if b >= 1024**3 and b % 1024**3 == 0:
        return f"{b // 1024**3}GB"
    if b >= 1024**2 and b % 1024**2 == 0:
        return f"{b // 1024**2}MB"
    if b >= 1024 and b % 1024 == 0:
        return f"{b // 1024}KB"
    return f"{b}B"

def read_counter(iface, name):
    try:
        return int((Path("/sys/class/net") / iface / "statistics" / name).read_text().strip())
    except Exception:
        return 0

def capture_net():
    return {i: {"rx": read_counter(i, "rx_bytes"), "tx": read_counter(i, "tx_bytes")} for i in IFACES}

def delta_net(before, after):
    rx = tx = 0
    for i in IFACES:
        rx += max(0, after.get(i, {}).get("rx", 0) - before.get(i, {}).get("rx", 0))
        tx += max(0, after.get(i, {}).get("tx", 0) - before.get(i, {}).get("tx", 0))
    return rx, tx

def build_env():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([
        f"{GVIRTUS_HOME}/lib",
        f"{GVIRTUS_HOME}/lib/frontend",
        f"{LZ4_HOME}/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["GVIRTUS_CONFIG"] = CONFIG
    env["GVIRTUS_HOME"] = GVIRTUS_HOME
    env["GVIRTUS_LOGLEVEL"] = GVIRTUS_LOGLEVEL
    env["LD_PRELOAD"] = ":".join([
        f"{GVIRTUS_HOME}/lib/frontend/libcudart.so",
        f"{GVIRTUS_HOME}/lib/frontend/libcublas.so",
    ])
    return env

def parse_field(text, name):
    m = re.search(rf"{re.escape(name)}=([^\s]+)", text)
    return m.group(1) if m else ""

def parse_log(text):
    calls = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in routine_re.finditer(text)]
    if not calls:
        calls = [(m.group(1), 0, 0) for m in routine_re_loose.finditer(text)]
    counts = Counter(name for name, _, _ in calls)
    return calls, counts, sum(x for _, x, _ in calls), sum(x for _, _, x in calls), ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))

def run_one(phase, run_idx, n):
    cmd = FRONTEND_CMD_TEMPLATE.format(size=n, matrix_n=n)
    log_path = logs_dir / f"{MODE}_{phase}_n{n}_run{run_idx}.log"

    before = capture_net()
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, executable="/bin/bash", env=build_env())

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

    calls, counts, g_in, g_out, routine_counts = parse_log(output or "")
    actual_n = parse_field(output or "", "BENCHMARK_MATRIX_N") or str(n)
    result_check = parse_field(output or "", "RESULT_CHECK")
    result_ms = parse_field(output or "", "BENCHMARK_RESULT_MS")
    malloc_ms = parse_field(output or "", "STAGE_MALLOC_MS")
    cudamalloc_ms = parse_field(output or "", "STAGE_CUDAMALLOC_MS")
    h2d_ms = parse_field(output or "", "STAGE_H2D_MS")
    cublas_create_ms = parse_field(output or "", "STAGE_CUBLAS_CREATE_MS")
    gemm_ms = parse_field(output or "", "STAGE_GEMM_MS")
    d2h_ms = parse_field(output or "", "STAGE_D2H_MS")
    cleanup_ms = parse_field(output or "", "STAGE_CLEANUP_MS")

    has_error = bool(error_re.search(output or ""))
    valid_output = exit_code == 0 and not has_error and result_check.upper().startswith("PASS")
    status = "OK" if valid_output else "FAILED"

    with open(nic_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp","mode","phase","matrix_n","run","iface","rx_before","rx_after","rx_delta","tx_before","tx_after","tx_delta"])
        for iface in IFACES:
            rb, ra = before[iface]["rx"], after[iface]["rx"]
            tb, ta = before[iface]["tx"], after[iface]["tx"]
            w.writerow([now_iso(), MODE, phase, n, run_idx, iface, rb, ra, max(0, ra-rb), tb, ta, max(0, ta-tb)])

    with open(routine_calls_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp","mode","phase","matrix_n","run","routine","in_B","out_B"])
        for name, ib, ob in calls:
            w.writerow([now_iso(), MODE, phase, n, run_idx, name, ib, ob])

    row = {
        "timestamp": now_iso(),
        "mode": MODE,
        "phase": phase,
        "matrix_n": actual_n,
        "payload_label": payload_label(int(actual_n)),
        "run": run_idx,
        "status": status,
        "exit_code": exit_code,
        "wall_s": f"{wall:.6f}",
        "benchmark_result_ms": result_ms,
        "malloc_ms": malloc_ms,
        "cudamalloc_ms": cudamalloc_ms,
        "h2d_ms": h2d_ms,
        "cublas_create_ms": cublas_create_ms,
        "gemm_ms": gemm_ms,
        "d2h_ms": d2h_ms,
        "cleanup_ms": cleanup_ms,
        "result_check": result_check,
        "valid_output": str(valid_output).lower(),
        "calls": len(calls),
        "gvirtus_in_B": g_in if calls else "",
        "gvirtus_out_B": g_out if calls else "",
        "nic_rx_B": nic_rx,
        "nic_tx_B": nic_tx,
        "config": CONFIG,
        "frontend_cmd": cmd,
        "log_file": str(log_path),
        "error": "timeout" if timed_out else "",
        "routine_counts": routine_counts,
    }

    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)

    print(f"  {status} n={actual_n} payload={payload_label(int(actual_n))} wall={wall:.6f}s result_ms={result_ms or 'NA'} result={result_check or 'NA'} calls={len(calls)} in={g_in if calls else 'NA'}B out={g_out if calls else 'NA'}B nic_tx={nic_tx}B nic_rx={nic_rx}B")

print(f"Benchmark output directory:\n{out_dir}\n")
print("=" * 60)
print(f"Mode: {MODE}")
print(f"Config: {CONFIG}")
print(f"Frontend template: {FRONTEND_CMD_TEMPLATE}")
print(f"Sizes: {' '.join(map(str, SIZES))}")
print(f"Runs: {RUNS} measured + {WARMUPS} warmup per size")
print(f"IFACES: {' '.join(IFACES)}")
print("=" * 60)

(system_dir / "system_metadata.txt").write_text(
    f"timestamp={now_iso()}\nmode={MODE}\nconfig={CONFIG}\nfrontend_cmd_template={FRONTEND_CMD_TEMPLATE}\nsizes={' '.join(map(str, SIZES))}\nruns={RUNS}\nwarmups={WARMUPS}\nrun_timeout={RUN_TIMEOUT}\ngvirtus_home={GVIRTUS_HOME}\ngvirtus_loglevel={GVIRTUS_LOGLEVEL}\nifaces={' '.join(IFACES)}\n"
)

for n in SIZES:
    for i in range(1, WARMUPS + 1):
        print(f"[{MODE}] warmup n={n} run {i}...")
        run_one("warmup", i, n)
    for i in range(1, RUNS + 1):
        print(f"[{MODE}] measure n={n} run {i}...")
        run_one("measure", i, n)

summary = defaultdict(lambda: [0, 0, 0])
if routine_calls_csv.exists():
    with open(routine_calls_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["phase"] == "measure":
                key = (r["matrix_n"], r["routine"])
                summary[key][0] += 1
                summary[key][1] += int(r["in_B"] or 0)
                summary[key][2] += int(r["out_B"] or 0)

with open(routine_summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["matrix_n","routine","count","total_in_B","total_out_B"])
    for (matrix_n, routine), (count, total_in, total_out) in sorted(summary.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        w.writerow([matrix_n, routine, count, total_in, total_out])

print("\nDone.")
print(f"Main CSV:              {results_csv}")
print(f"Individual calls CSV:  {routine_calls_csv}")
print(f"Routine summary CSV:   {routine_summary_csv}")
print(f"NIC CSV:               {nic_csv}")
print(f"Logs:                  {logs_dir}")
print(f"Counters:              {counters_dir}")
print(f"System metadata:       {system_dir}")

rows = list(csv.DictReader(open(results_csv, newline="")))
measured_ok = [r for r in rows if r["phase"] == "measure" and r["status"] == "OK" and r["exit_code"] == "0" and r["valid_output"] == "true"]
if measured_ok:
    print("\nMeasured summary by size:")
    for n in SIZES:
        group = [r for r in measured_ok if int(r["matrix_n"]) == n]
        if not group:
            continue
        def mean(field):
            vals = [float(r[field]) for r in group if str(r.get(field, "")).strip()]
            return st.mean(vals) if vals else float("nan")
        print(f"{MODE},n={n},ok={len(group)},mean_wall_s={mean('wall_s'):.6f},mean_result_ms={mean('benchmark_result_ms'):.3f},mean_calls={mean('calls'):.1f},mean_gvirtus_in_B={mean('gvirtus_in_B'):.1f},mean_gvirtus_out_B={mean('gvirtus_out_B'):.1f},mean_nic_rx_B={mean('nic_rx_B'):.1f},mean_nic_tx_B={mean('nic_tx_B'):.1f}")
