import csv
import subprocess
import threading
from datetime import datetime

# --- Custom Logger Class ---
class CustomAcceleratedSparkLogger:
    def __init__(self, results_dir, pipeline_name):
        self.pipeline_log = []
        self.gpu_log = []
        self._gpu_logger_thread = None
        self._gpu_logger_stop = threading.Event()
        self._gpu_log_path = os.path.join(results_dir, "rapids_gpu_usage_log.csv")
        self._pipeline_log_path = os.path.join(results_dir, "rapids_pipeline_log.csv")

    def now_str(self):
        return datetime.now().replace(microsecond=0).isoformat()

    def log_event(self, stage, message):
        ts = self.now_str()
        self.pipeline_log.append({"timestamp": ts, "stage": stage, "message": message})

    def start_gpu_logger(self):
        def gpu_logger():
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
                "-l", "1"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            while not self._gpu_logger_stop.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 5:
                    try:
                        ts = parts[0].split(".")[0].replace("/", "-").replace(" ", "T")
                        self.gpu_log.append({
                            "timestamp": ts,
                            "utilization.gpu": parts[1],
                            "memory.used": parts[2],
                            "memory.total": parts[3],
                            "power.draw": parts[4]
                        })
                    except Exception:
                        continue
            proc.terminate()
        self._gpu_logger_stop.clear()
        self._gpu_logger_thread = threading.Thread(target=gpu_logger, daemon=True)
        self._gpu_logger_thread.start()

    def stop_gpu_logger(self):
        self._gpu_logger_stop.set()
        if self._gpu_logger_thread:
            self._gpu_logger_thread.join(timeout=2)

    def save_logs(self):
        os.makedirs(os.path.dirname(self._pipeline_log_path), exist_ok=True)
        with open(self._pipeline_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "stage", "message"])
            writer.writeheader()
            for row in self.pipeline_log:
                writer.writerow(row)
        with open(self._gpu_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "utilization.gpu", "memory.used", "memory.total", "power.draw"])
            writer.writeheader()
            for row in self.gpu_log:
                writer.writerow(row)

    def capture_spark_logs(self):
        log_pattern = re.compile(r"(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) (\w+) (\w+): (.*)")
        while not self._gpu_logger_stop.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            m = log_pattern.match(line)
            if m:
                ts, level, src, msg = m.groups()
                try:
                    dt = datetime.strptime(ts, "%y/%m/%d %H:%M:%S")
                    iso_ts = dt.replace(microsecond=0).isoformat()
                except Exception:
                    iso_ts = self.now_str()
                self.pipeline_log.append({
                    "timestamp": iso_ts,
                    "stage": f"SPARK_{level}",
                    "message": f"{src}: {msg.strip()}"
                })

