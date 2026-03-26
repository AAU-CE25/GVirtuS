# src/pipeline_gvirtus.py

"""
E-Commerce Customer Analytics Pipeline — GVirtuS Remote GPU

Uses the SAME pipeline logic as pipeline_rapids.py and pipeline_hybrid.py,
but runs through GVirtuS: CUDA calls are intercepted by LD_PRELOAD and
forwarded over TCP to a remote GPU server at 24.24.24.1:2222.

Usage:
  # Inside Docker container (GVirtuS frontend installed):
  spark-submit --jars $RAPIDS_JAR pipeline_gvirtus.py

  # Or via docker compose:
  docker compose up spark-gvirtus
"""

import os
import sys
import time
import json
import socket
import ctypes
import subprocess

from pyspark.sql import SparkSession

from config import (
    DATA_DIR, RESULTS_DIR, RAPIDS_JAR,
    SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY,
    GVIRTUS_BACKEND_HOST, GVIRTUS_BACKEND_PORT, GVIRTUS_HOME,
)
from ecomerce_pipeline import EcommercePipeline


class GVirtuSPipeline(EcommercePipeline):
    """
    Pipeline that selectively uses a REMOTE GPU via GVirtuS.

    Inherits all 7 stages from EcommercePipeline.
    Toggles spark.rapids.sql.enabled per stage based on
    whether GPU acceleration is beneficial AND the backend is reachable.
    """

    def __init__(self, spark: SparkSession, gpu_available: bool = True):
        super().__init__(spark)
        self._gpu_available = gpu_available
        self._gpu_enabled = False
        self.mode = "gvirtus-gpu" if gpu_available else "cpu-fallback"

    # ────────────────────────────────────────────
    #  GPU toggle
    # ────────────────────────────────────────────
    def _enable_gpu(self):
        if self._gpu_available and not self._gpu_enabled:
            self.spark.conf.set("spark.rapids.sql.enabled", "true")
            self._gpu_enabled = True

    def _disable_gpu(self):
        if self._gpu_enabled:
            self.spark.conf.set("spark.rapids.sql.enabled", "false")
            self._gpu_enabled = False

    # ────────────────────────────────────────────
    #  Per-stage GPU decisions
    #
    #  GPU ON  = heavy compute, large data
    #  GPU OFF = I/O bound, small data, unsupported ops
    # ────────────────────────────────────────────
    def load_data(self):
        """I/O bound — CPU is fine, no point sending data over network."""
        self._disable_gpu()
        print("    [load_data]          → CPU  (I/O bound)")
        super().load_data()

    def revenue_analytics(self):
        """Large aggregations — GPU accelerates this well."""
        self._enable_gpu()
        print(f"    [revenue_analytics]  → {'GPU 🟢' if self._gpu_enabled else 'CPU'}")
        super().revenue_analytics()

    def customer_360(self):
        """Joins + aggregations on full dataset — GPU."""
        self._enable_gpu()
        print(f"    [customer_360]       → {'GPU 🟢' if self._gpu_enabled else 'CPU'}")
        super().customer_360()

    def rfm_segmentation(self):
        """Window functions + groupBy — GPU."""
        self._enable_gpu()
        print(f"    [rfm_segmentation]   → {'GPU 🟢' if self._gpu_enabled else 'CPU'}")
        super().rfm_segmentation()

    def cohort_analysis(self):
        """Heavy first pass, then small tables — GPU for initial agg."""
        self._enable_gpu()
        print(f"    [cohort_analysis]    → {'GPU 🟢' if self._gpu_enabled else 'CPU'}")
        super().cohort_analysis()

    def funnel_analysis(self):
        """Often smaller filtered subsets — CPU to avoid transfer overhead."""
        self._disable_gpu()
        print("    [funnel_analysis]    → CPU  (small intermediate data)")
        super().funnel_analysis()

    def customer_clustering(self):
        """ML feature prep benefits from GPU. KMeans itself is Spark MLlib (CPU)."""
        self._enable_gpu()
        print(f"    [customer_clustering]→ {'GPU 🟢' if self._gpu_enabled else 'CPU'}")
        super().customer_clustering()

    # ────────────────────────────────────────────
    #  Run
    # ────────────────────────────────────────────
    def run(self):
        print(f"\n{'='*60}")
        print(f"  E-Commerce Analytics — GVirtuS Remote GPU")
        print(f"  Backend: {GVIRTUS_BACKEND_HOST}:{GVIRTUS_BACKEND_PORT}")
        print(f"  Mode: {self.mode}")
        print(f"{'='*60}\n")

        self.load_data()
        self.revenue_analytics()
        self.customer_360()
        self.rfm_segmentation()
        self.cohort_analysis()
        self.funnel_analysis()
        self.customer_clustering()

        self._disable_gpu()

        total_time = sum(self.timings.values())
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete — Total: {total_time:.2f}s")
        print(f"{'='*60}")
        print(f"\n  Stage Timings:")
        for stage, elapsed in self.timings.items():
            pct = (elapsed / total_time) * 100 if total_time > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {stage:<30} {elapsed:>8.2f}s  ({pct:>5.1f}%) {bar}")

        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_file = f"{RESULTS_DIR}/gvirtus_gpu_results.json"
        output = {
            "mode": self.mode,
            "backend": f"{GVIRTUS_BACKEND_HOST}:{GVIRTUS_BACKEND_PORT}",
            "timings": self.timings,
            "results": self.results,
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_file}")

        return self.timings


# ────────────────────────────────────────────────────────
#  GVirtuS health checks
# ────────────────────────────────────────────────────────

def check_backend_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
    """TCP connectivity check to GVirtuS backend."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_gpu_via_gvirtus() -> dict:
    """
    Test if we can actually make CUDA calls through GVirtuS.
    LD_PRELOAD must already be set for this to work.
    """
    result = {"available": False, "device_count": 0, "error": None}
    try:
        cuda = ctypes.CDLL("libcudart.so")
        count = ctypes.c_int(0)
        ret = cuda.cudaGetDeviceCount(ctypes.byref(count))
        if ret == 0 and count.value > 0:
            result["available"] = True
            result["device_count"] = count.value
        else:
            result["error"] = f"cudaGetDeviceCount returned {ret}, count={count.value}"
    except OSError as e:
        result["error"] = f"Cannot load libcudart.so: {e}"
    except Exception as e:
        result["error"] = str(e)
    return result


def verify_ld_preload() -> bool:
    """Check if LD_PRELOAD is set to intercept CUDA calls."""
    ld_preload = os.environ.get("LD_PRELOAD", "")
    return "gvirtus" in ld_preload.lower() or "libcudart" in ld_preload


# ────────────────────────────────────────────────────────
#  Spark session factory
# ────────────────────────────────────────────────────────

def create_gvirtus_spark_session() -> tuple[SparkSession, bool]:
    """
    Create Spark session configured for GVirtuS remote GPU.

    Returns:
        (spark_session, gpu_available)
    """
    print("=" * 60)
    print("  GVirtuS Environment Check")
    print("=" * 60)

    # 1. Check LD_PRELOAD
    ld_ok = verify_ld_preload()
    print(f"  LD_PRELOAD set:        {'✅ Yes' if ld_ok else '❌ No'}")
    if not ld_ok:
        print(f"    Current LD_PRELOAD: {os.environ.get('LD_PRELOAD', '(not set)')}")
        print(f"    Expected: {GVIRTUS_HOME}/lib/frontend/libcudart.so")

    # 2. Check backend connectivity
    print(f"  Backend ({GVIRTUS_BACKEND_HOST}:{GVIRTUS_BACKEND_PORT}):", end=" ")
    backend_ok = check_backend_reachable(GVIRTUS_BACKEND_HOST, GVIRTUS_BACKEND_PORT)
    print(f"{'✅ Reachable' if backend_ok else '❌ Unreachable'}")

    # 3. Check GPU via GVirtuS (only if LD_PRELOAD is set)
    gpu_available = False
    if ld_ok and backend_ok:
        gpu_info = check_gpu_via_gvirtus()
        gpu_available = gpu_info["available"]
        if gpu_available:
            print(f"  Remote GPU devices:    ✅ {gpu_info['device_count']} found")
        else:
            print(f"  Remote GPU devices:    ❌ {gpu_info['error']}")
    elif not backend_ok:
        print(f"  Remote GPU devices:    ⏭️  Skipped (backend unreachable)")

    print(f"  Mode:                  {'🟢 GVirtuS GPU' if gpu_available else '🟡 CPU fallback'}")
    print("=" * 60)

    # Build Spark session
    builder = (
        SparkSession.builder
        .appName(f"GVirtuS Pipeline (sf{os.environ.get('SCALE_FACTOR', '1')})")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
    )

    if gpu_available:
        # RAPIDS with GPU via GVirtuS
        builder = (
            builder
            .config("spark.jars", RAPIDS_JAR)
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "false")  # Toggled per stage
            .config("spark.rapids.sql.explain", "ALL")
            .config("spark.rapids.sql.incompatibleOps.enabled", "true")
            .config("spark.rapids.sql.variableFloatAgg.enabled", "true")
            .config("spark.rapids.sql.hasNans", "false")
            .config("spark.rapids.memory.pinnedPool.size", "1G")
            .config("spark.rapids.sql.concurrentGpuTasks", "2")
        )
    else:
        print("\n  ⚠️  Running without RAPIDS plugin (CPU only)")

    spark = builder.getOrCreate()
    print(f"\n  Spark version: {spark.version}")
    return spark, gpu_available


# ────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark, gpu_available = create_gvirtus_spark_session()
    try:
        pipeline = GVirtuSPipeline(spark, gpu_available=gpu_available)
        pipeline.run()
    finally:
        spark.stop()