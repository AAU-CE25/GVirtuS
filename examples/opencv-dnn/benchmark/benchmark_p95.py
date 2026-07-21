#!/usr/bin/env python3
import csv
import sys
csv.field_size_limit(sys.maxsize)

import os
import re
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "tcp"
if MODE not in {"tcp", "rdma", "ucx", "baremetal"}:
    print("Usage: ./benchmark_p95.py {tcp|rdma|ucx|baremetal}", file=sys.stderr)
    sys.exit(2)

GVIRTUS_HOME = os.environ.get("GVIRTUS_HOME", "/home/student.aau.dk/ll33pq/GVirtuS")
LZ4_HOME = os.environ.get("LZ4_HOME", "/home/student.aau.dk/ll33pq/lz4-install")
OPENCV_HOME = os.environ.get("OPENCV_HOME", "/home/student.aau.dk/ll33pq/opencv-local")
CUDNN_ROOT = os.environ.get("CUDNN_ROOT", "/home/student.aau.dk/ll33pq/cudnn-9.5.1")
NPP_DIR = os.environ.get("NPP_DIR", "/home/student.aau.dk/ll33pq/.local/lib/python3.10/site-packages/nvidia/npp/lib")

RUNS = int(os.environ.get("RUNS", "50"))
WARMUPS = int(os.environ.get("WARMUPS", "2"))
RUN_TIMEOUT = float(os.environ.get("RUN_TIMEOUT", "3600"))
FRONTEND_CMD = os.environ.get("FRONTEND_CMD", "../run.sh")

INTERNAL_RUN_ENV = "BENCH_INTERNAL_RUNS"

BAREMETAL_IMAGE = os.environ.get(
    "BAREMETAL_IMAGE", "ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6"
)

BAREMETAL_WORKDIR = os.environ.get(
    "BAREMETAL_WORKDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if MODE == "baremetal":
    import glob as _glob

    _build = (
        "[ ! -f dnn_test_native ] && nvcc main.cu -o dnn_test_native -g "
        "$(pkg-config --cflags --libs opencv4) "
        "-L/usr/local/cuda/lib64 -lcudart -lcublas -lcudnn; "
    )

    _host_libs = _glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.*")
    _host_drv_version = (
        _host_libs[0].rsplit(".so.", 1)[1] if _host_libs else "560.35.05"
    )

    _drv_mounts = (
        f" -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.{_host_drv_version}:"
        f"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.{_host_drv_version}:ro "
        f" -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:"
        f"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro "
        f" -v /usr/lib/x86_64-linux-gnu/libcuda.so.{_host_drv_version}:"
        f"/usr/lib/x86_64-linux-gnu/libcuda.so.{_host_drv_version}:ro "
        f" -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:"
        f"/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro "
        f" -v /usr/lib/x86_64-linux-gnu/libcuda.so:"
        f"/usr/lib/x86_64-linux-gnu/libcuda.so:ro "
    )

    FRONTEND_CMD = (
        f"docker run --rm --gpus all "
        f"-e NVIDIA_DRIVER_CAPABILITIES=all "
        f"{_drv_mounts}"
        f"-v {BAREMETAL_WORKDIR}:/app:rw -w /app "
        f"-e LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:"
        f"/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/lib "
        f"-e PKG_CONFIG_PATH=/usr/local/lib/pkgconfig "
        f"--entrypoint bash {BAREMETAL_IMAGE} -c \"{_build}./dnn_test_native\""
    )

GVIRTUS_LOGLEVEL = os.environ.get("GVIRTUS_LOGLEVEL", "10000")
IFACES = os.environ.get("IFACES", "ens1f1np1 ens1f0np0 bond0").split()

CONFIGS = {
    "tcp": os.environ.get("TCP_CONFIG", f"{GVIRTUS_HOME}/etc/properties.json"),
    "rdma": os.environ.get("RDMA_CONFIG", f"{GVIRTUS_HOME}/etc/properties_plain_rdma.json"),
    "ucx": os.environ.get("UCX_CONFIG", f"{GVIRTUS_HOME}/etc/properties_ucx.json"),
    "baremetal": "",
}

CONFIG = CONFIGS[MODE]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("benchmark_results") / f"opencvdnn_p95_metrics_{ts}_{MODE}"
logs_dir = out_dir / "logs"
counters_dir = out_dir / "counters"
system_dir = out_dir / "system"

for d in [logs_dir, counters_dir, system_dir]:
    d.mkdir(parents=True, exist_ok=True)

results_csv = out_dir / "results.csv"
per_run_csv = out_dir / "per_internal_run_latency.csv"
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

    if MODE == "baremetal":
        return env

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

    env["LD_PRELOAD"] = ":".join(
        [
            f"{GVIRTUS_HOME}/lib/frontend/libcudart.so",
            f"{GVIRTUS_HOME}/lib/frontend/libcublas.so",
            f"{GVIRTUS_HOME}/lib/frontend/libcudnn.so.9",
        ]
    )

    return env


def percentile(values, p):
    xs = sorted(float(x) for x in values)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def mean(values):
    xs = [float(x) for x in values]
    return sum(xs) / len(xs) if xs else ""


routine_re = re.compile(r"Routine '([^']+)' returned .*?in=(\d+)B .*?out=(\d+)B")
routine_re_loose = re.compile(r"Routine '([^']+)' returned")


def extract_inference_latencies_ms(text):
    """
    Extracts every per-inference latency printed by dnn_test.

    Supported examples:
      Time taken: 123.45 ms
      Inference time: 123.45 ms
      Inference: 123.45 ms
      inference_ms=123.45
      Detection time: 123.45 ms
      BENCH_INTERNAL_RUN_RESULT run=1 wall_ms=500.0 inference_ms=123.45 status=OK

    Important:
      p95/p99 are only meaningful if dnn_test prints one latency per internal run.
    """
    values = []

    patterns = [
        r"BENCH_INTERNAL_RUN_RESULT\b.*?\binference_ms=([0-9]+(?:\.[0-9]+)?)",
        r"\binference_ms[=:]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bTime taken:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        r"\bInference(?: time)?(?: took)?[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        r"\bDetection(?: time)?[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    ]

    seen_spans = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            # Avoid duplicate captures from the exact same substring.
            span = m.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass

    return values


def extract_internal_wall_latencies_ms(text):
    """
    Optional: extracts end-to-end per internal run wall latency if dnn_test prints:
      BENCH_INTERNAL_RUN_RESULT run=1 wall_ms=1234.5 inference_ms=123.4 status=OK
    """
    values = []
    for m in re.finditer(
        r"BENCH_INTERNAL_RUN_RESULT\b.*?\bwall_ms=([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    ):
        try:
            values.append(float(m.group(1)))
        except ValueError:
            pass
    return values


def parse_log(text):
    calls = []

    for m in routine_re.finditer(text):
        calls.append((m.group(1), int(m.group(2)), int(m.group(3))))

    if not calls:
        for m in routine_re_loose.finditer(text):
            calls.append((m.group(1), 0, 0))

    counts = Counter(name for name, _, _ in calls)
    g_in = sum(x for _, x, _ in calls)
    g_out = sum(x for _, _, x in calls)
    routine_counts = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    inference_latencies_ms = extract_inference_latencies_ms(text)
    wall_latencies_ms = extract_internal_wall_latencies_ms(text)

    # Backward-compatible scalar value: mean if multiple, otherwise single value.
    detection_ms = mean(inference_latencies_ms)

    return (
        calls,
        counts,
        g_in,
        g_out,
        routine_counts,
        detection_ms,
        inference_latencies_ms,
        wall_latencies_ms,
    )


def latency_stats(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "min": min(values) if values else "",
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else "",
    }


def write_per_run_latencies(mode, phase, inference_values, wall_values):
    max_len = max(len(inference_values), len(wall_values), 0)
    if max_len == 0:
        return

    with open(per_run_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(
                [
                    "timestamp",
                    "mode",
                    "phase",
                    "internal_run",
                    "inference_ms",
                    "wall_ms",
                ]
            )

        for i in range(max_len):
            inf = inference_values[i] if i < len(inference_values) else ""
            wall = wall_values[i] if i < len(wall_values) else ""
            w.writerow([now_iso(), mode, phase, i + 1, inf, wall])


def run_batch(phase, n_runs):
    log_path = logs_dir / f"{MODE}_{phase}_batch.log"

    before = capture_net()
    t0 = time.monotonic()

    env = build_env()
    env[INTERNAL_RUN_ENV] = str(n_runs)

    frontend_parts = shlex.split(FRONTEND_CMD)

    # Do not preload GVirtuS shims into Bash itself if run.sh is the frontend.
    if FRONTEND_CMD.endswith(".sh"):
        env.pop("LD_PRELOAD", None)

    proc = subprocess.Popen(
        frontend_parts,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
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

    output = output or ""
    log_path.write_text(output)

    (
        calls,
        counts,
        g_in,
        g_out,
        routine_counts,
        detection_ms,
        inference_values,
        wall_values,
    ) = parse_log(output)

    valid_output = exit_code == 0 and (
        "Final Results" in output
        or "Accuracy" in output
        or "Time taken:" in output
        or "Inference" in output
        or bool(inference_values)
        or len(calls) > 0
    )

    status = "OK" if valid_output else "FAILED"
    error = "timeout" if timed_out else ""

    inf_stats = latency_stats(inference_values)
    wall_stats = latency_stats(wall_values)

    write_per_run_latencies(MODE, phase, inference_values, wall_values)

    with open(nic_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(
                [
                    "timestamp",
                    "mode",
                    "phase",
                    "run",
                    "iface",
                    "rx_before",
                    "rx_after",
                    "rx_delta",
                    "tx_before",
                    "tx_after",
                    "tx_delta",
                ]
            )

        for iface in IFACES:
            rb = before.get(iface, {}).get("rx", 0)
            ra = after.get(iface, {}).get("rx", 0)
            tb = before.get(iface, {}).get("tx", 0)
            ta = after.get(iface, {}).get("tx", 0)

            w.writerow(
                [
                    now_iso(),
                    MODE,
                    phase,
                    "batch",
                    iface,
                    rb,
                    ra,
                    max(0, ra - rb),
                    tb,
                    ta,
                    max(0, ta - tb),
                ]
            )

    with open(routine_calls_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "mode", "phase", "run", "routine", "in_B", "out_B"])

        for name, ib, ob in calls:
            w.writerow([now_iso(), MODE, phase, "batch", name, ib, ob])

    row = {
        "timestamp": now_iso(),
        "mode": MODE,
        "phase": phase,
        "run": "batch",
        "internal_runs": n_runs,
        "status": status,
        "exit_code": exit_code,
        "wall_s": f"{wall:.6f}",
        "wall_per_internal_run_s": f"{wall / n_runs:.6f}" if n_runs else "",
        "detection_ms": detection_ms,
        "inference_samples": inf_stats["n"],
        "inference_mean_ms": inf_stats["mean"],
        "inference_min_ms": inf_stats["min"],
        "inference_p50_ms": inf_stats["p50"],
        "inference_p90_ms": inf_stats["p90"],
        "inference_p95_ms": inf_stats["p95"],
        "inference_p99_ms": inf_stats["p99"],
        "inference_max_ms": inf_stats["max"],
        "wall_samples": wall_stats["n"],
        "internal_wall_mean_ms": wall_stats["mean"],
        "internal_wall_min_ms": wall_stats["min"],
        "internal_wall_p50_ms": wall_stats["p50"],
        "internal_wall_p90_ms": wall_stats["p90"],
        "internal_wall_p95_ms": wall_stats["p95"],
        "internal_wall_p99_ms": wall_stats["p99"],
        "internal_wall_max_ms": wall_stats["max"],
        "valid_output": str(valid_output).lower(),
        "calls": len(calls),
        "calls_per_internal_run": f"{len(calls) / n_runs:.3f}" if n_runs else "",
        "gvirtus_in_B": g_in if calls else "",
        "gvirtus_out_B": g_out if calls else "",
        "gvirtus_in_B_per_internal_run": f"{g_in / n_runs:.3f}" if calls and n_runs else "",
        "gvirtus_out_B_per_internal_run": f"{g_out / n_runs:.3f}" if calls and n_runs else "",
        "nic_rx_B": nic_rx,
        "nic_tx_B": nic_tx,
        "nic_rx_B_per_internal_run": f"{nic_rx / n_runs:.3f}" if n_runs else "",
        "nic_tx_B_per_internal_run": f"{nic_tx / n_runs:.3f}" if n_runs else "",
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
        f"  {status} internal_runs={n_runs} "
        f"wall={wall:.6f}s wall/run={wall / n_runs if n_runs else 0:.6f}s "
        f"inference_samples={inf_stats['n']} "
        f"mean={inf_stats['mean'] if inf_stats['mean'] != '' else 'NA'}ms "
        f"p95={inf_stats['p95'] if inf_stats['p95'] != '' else 'NA'}ms "
        f"p99={inf_stats['p99'] if inf_stats['p99'] != '' else 'NA'}ms "
        f"calls={len(calls)} in={g_in if calls else 'NA'}B out={g_out if calls else 'NA'}B "
        f"nic_tx={nic_tx}B nic_rx={nic_rx}B"
    )

    if inf_stats["n"] < 2:
        print(
            "  WARNING: p95/p99 not meaningful because fewer than 2 inference samples "
            "were parsed from the log. Make dnn_test print one latency per internal run."
        )

    return row


print(f"Benchmark output directory:\n{out_dir}")
print()
print("=" * 60)
print(f"Mode: {MODE}")
print(f"Config: {CONFIG}")
print(f"Frontend: {FRONTEND_CMD}")
print(f"Runs: {RUNS} measured internal runs + {WARMUPS} warmup internal runs")
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
    f"internal_run_env={INTERNAL_RUN_ENV}\n"
)

all_rows = []

if WARMUPS > 0:
    print(f"[{MODE}] warmup batch: {WARMUPS} internal runs...")
    all_rows.append(run_batch("warmup", WARMUPS))

if RUNS > 0:
    print(f"[{MODE}] measure batch: {RUNS} internal runs...")
    all_rows.append(run_batch("measure", RUNS))

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
print(f"Per-internal-run CSV:  {per_run_csv}")
print(f"Individual calls CSV:  {routine_calls_csv}")
print(f"Routine summary CSV:   {routine_summary_csv}")
print(f"NIC CSV:               {nic_csv}")
print(f"Logs:                  {logs_dir}")
print(f"Counters:              {counters_dir}")
print(f"System metadata:       {system_dir}")

measured_ok = []

if results_csv.exists():
    with open(results_csv, newline="") as f:
        measured_ok = [
            r
            for r in csv.DictReader(f)
            if r["phase"] == "measure" and r["status"] == "OK"
        ]

if measured_ok:
    r = measured_ok[-1]
    print()
    print("Measured summary:")
    print(
        f"{MODE}: "
        f"internal_runs={r['internal_runs']} "
        f"wall_s={float(r['wall_s']):.6f} "
        f"wall_per_run_s={float(r['wall_per_internal_run_s']):.6f} "
        f"inference_samples={r['inference_samples']} "
        f"inference_mean_ms={r['inference_mean_ms'] or 'NA'} "
        f"inference_p50_ms={r['inference_p50_ms'] or 'NA'} "
        f"inference_p95_ms={r['inference_p95_ms'] or 'NA'} "
        f"inference_p99_ms={r['inference_p99_ms'] or 'NA'} "
        f"inference_max_ms={r['inference_max_ms'] or 'NA'} "
        f"calls={r['calls']} "
        f"calls_per_run={r['calls_per_internal_run']} "
        f"nic_rx_B_per_run={r['nic_rx_B_per_internal_run']} "
        f"nic_tx_B_per_run={r['nic_tx_B_per_internal_run']}"
    )
