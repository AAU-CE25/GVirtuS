import csv
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUNS = int(os.environ.get("RUNS", "50"))
WARMUPS = int(os.environ.get("WARMUPS", "5"))
TIMEOUT = int(os.environ.get("RUN_TIMEOUT", "300"))
CMD = os.environ.get("FRONTEND_CMD", "./sample_baremetal")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = Path(f"benchmark_results/baremetal_opencvdnn_{ts}")
logdir = outdir / "logs"
outdir.mkdir(parents=True, exist_ok=True)
logdir.mkdir(parents=True, exist_ok=True)

csv_path = outdir / "results.csv"

patterns = {
    "inference_ms": re.compile(r"Time taken:\s*([0-9.]+)\s*ms"),
    "total_images": re.compile(r"Total images:\s*([0-9]+)"),
    "correct": re.compile(r"Correct predictions:\s*([0-9]+)"),
    "accuracy": re.compile(r"Accuracy:\s*([0-9.]+)%"),
    "pred": re.compile(r"Predicted Class ID:\s*([0-9]+)"),
    "conf": re.compile(r"Confidence:\s*([0-9.]+)%"),
}

def last_match(name, text):
    matches = patterns[name].findall(text)
    return matches[-1] if matches else ""

def run_one(phase, run):
    log = logdir / f"baremetal_{phase}_{run}.log"

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("GVIRTUS_CONFIG", None)
    env.pop("GVIRTUS_LOGLEVEL", None)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["bash", "-lc", CMD],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
        output = proc.stdout
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        exit_code = 124

    wall_s = time.perf_counter() - start
    log.write_text(output)

    valid = (
        "Final Results:" in output
        and "Total images:" in output
        and "Saved total timings" in output
    )

    status = "OK" if exit_code == 0 and valid else "FAILED"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "configuration": "baremetal",
        "phase": phase,
        "run": run,
        "status": status,
        "exit_code": exit_code,
        "wall_s": f"{wall_s:.6f}",
        "inference_ms": last_match("inference_ms", output),
        "total_images": last_match("total_images", output),
        "accuracy": last_match("accuracy", output),
        "correct_predictions": last_match("correct", output),
        "predicted_class": last_match("pred", output),
        "confidence_pct": last_match("conf", output),
        "valid_output": str(valid).lower(),
        "log_file": str(log),
    }

    print(
        f"[{phase}] run {run}: {status} "
        f"wall={row['wall_s']}s "
        f"inference_ms={row['inference_ms'] or 'NA'} "
        f"acc={row['accuracy'] or 'NA'}%"
    )

    return row

rows = []

for i in range(1, WARMUPS + 1):
    rows.append(run_one("warmup", i))

for i in range(1, RUNS + 1):
    rows.append(run_one("measure", i))

fields = [
    "timestamp",
    "configuration",
    "phase",
    "run",
    "status",
    "exit_code",
    "wall_s",
    "inference_ms",
    "total_images",
    "accuracy",
    "correct_predictions",
    "predicted_class",
    "confidence_pct",
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
