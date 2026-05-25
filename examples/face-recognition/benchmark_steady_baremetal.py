import csv
import io
import os
import re
import runpy
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

RUNS = int(os.environ.get("RUNS", "50"))
WARMUPS = int(os.environ.get("WARMUPS", "5"))
SCRIPT = os.environ.get("FACERECON_SCRIPT", "cnn_baremetal.py")

outdir = Path("benchmark_results/FaceRecon_Steady_Baremetal")
logdir = outdir / "logs"
outdir.mkdir(parents=True, exist_ok=True)
logdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "results.csv"

patterns = {
    "accuracy": re.compile(r"Test Accuracy:\s*([0-9.]+)%", re.IGNORECASE),
    "execution_s": re.compile(r"Execution Time:\s*([0-9.]+)\s*seconds", re.IGNORECASE),
    "average_s": re.compile(r"Average Time:\s*([0-9.]+)\s*seconds", re.IGNORECASE),
}

def last_match(name, text):
    matches = patterns[name].findall(text)
    return matches[-1] if matches else ""

def run_one(phase, run):
    log_file = logdir / f"baremetal_{phase}_{run}.log"

    buf = io.StringIO()
    t0 = time.perf_counter()
    exit_code = 0

    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 1
    except Exception:
        exit_code = 1
        buf.write("\n=== Python exception ===\n")
        buf.write(traceback.format_exc())

    wall_s = time.perf_counter() - t0
    output = buf.getvalue()
    log_file.write_text(output)

    accuracy = last_match("accuracy", output)
    execution_s = last_match("execution_s", output)
    average_s = last_match("average_s", output)

    valid = bool(accuracy and execution_s and average_s)
    status = "OK" if exit_code == 0 and valid else "FAILED"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "configuration": "baremetal_steady_state",
        "phase": phase,
        "run": run,
        "status": status,
        "exit_code": exit_code,
        "wall_s": f"{wall_s:.6f}",
        "execution_s": execution_s,
        "average_s": average_s,
        "accuracy_pct": accuracy,
        "valid_output": str(valid).lower(),
        "log_file": str(log_file),
    }

    print(
        f"[{phase}] run {run}: {status} "
        f"wall={row['wall_s']}s "
        f"execution_s={execution_s or 'NA'} "
        f"average_s={average_s or 'NA'} "
        f"accuracy={accuracy or 'NA'}%"
    )

    return row

rows = []
process_start = time.perf_counter()

for i in range(1, WARMUPS + 1):
    rows.append(run_one("warmup", i))

for i in range(1, RUNS + 1):
    rows.append(run_one("measure", i))

process_wall_s = time.perf_counter() - process_start
(outdir / "process_wall.txt").write_text(f"process_wall_s={process_wall_s:.6f}\n")

fields = [
    "timestamp",
    "configuration",
    "phase",
    "run",
    "status",
    "exit_code",
    "wall_s",
    "execution_s",
    "average_s",
    "accuracy_pct",
    "valid_output",
    "log_file",
]

with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print()
print("Done.")
print(f"Results CSV: {csv_path}")
print(f"Logs:        {logdir}")
print(f"Process wall: {outdir / 'process_wall.txt'}")
