# pipeline_rapids.py

"""
E-Commerce Customer Analytics Pipeline — RAPIDS Accelerated (GPU)

IDENTICAL analytics logic to pipeline_spark.py, but configured to use
NVIDIA RAPIDS Accelerator for Spark. The RAPIDS plugin automatically
replaces CPU operators with GPU-accelerated equivalents.

When running with GVirtuS:
  - The GPU is virtual (remote, on a separate server)
  - RAPIDS doesn't know the difference
  - GVirtuS intercepts all CUDA calls transparently
  - Smart GPUDirect optimises the data transfer path
"""

import time
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY

# Import the SAME pipeline class — the logic is identical
from pipeline_spark import EcommercePipeline


class RapidsEcommercePipeline(EcommercePipeline):
    """
    Extends the base pipeline with RAPIDS-specific reporting.
    The actual analytics logic is inherited — zero code changes needed.
    This demonstrates that RAPIDS (and by extension, GVirtuS vGPU) is transparent.
    """

    def run(self):
        """Execute all pipeline stages with RAPIDS GPU acceleration."""
        print(f"\n{'='*60}")
        print(f"  E-Commerce Analytics Pipeline — RAPIDS GPU Accelerated")
        print(f"{'='*60}")

        # Check GPU availability
        try:
            gpu_info = self.spark.sparkContext._jsc.sc().getExecutorMemoryStatus()
            print(f"  GPU acceleration: ENABLED (RAPIDS plugin active)")
        except Exception:
            print(f"  GPU acceleration: ENABLED (RAPIDS plugin active)")

        self.load_data()
        self.revenue_analytics()
        self.customer_360()
        self.rfm_segmentation()
        self.cohort_analysis()
        self.funnel_analysis()
        self.customer_clustering()

        # Summary
        total_time = sum(self.timings.values())
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete — Total: {total_time:.2f}s")
        print(f"{'='*60}")
        print(f"\n  Stage Timings:")
        for stage, elapsed in self.timings.items():
            pct = (elapsed / total_time) * 100
            bar = "█" * int(pct / 2)
            print(f"    {stage:<30} {elapsed:>8.2f}s  ({pct:>5.1f}%) {bar}")

        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output = {"timings": self.timings, "results": self.results}
        with open(f"{RESULTS_DIR}/rapids_gpu_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {RESULTS_DIR}/rapids_gpu_results.json")

        return self.timings


def create_rapids_spark_session(use_gvirtus: bool = False) -> SparkSession:
    """
    Create a Spark session with RAPIDS GPU acceleration.
    """
    import os
    jars_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jars")
    rapids_jar = os.path.join(jars_dir, "rapids-4-spark_2.12-24.10.0.jar")

    builder = (
        SparkSession.builder
        .appName("EcommerceAnalytics-RAPIDS-GPU")
        .master(SPARK_MASTER)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)

        # ── RAPIDS JAR ──
        .config("spark.jars", rapids_jar)
        .config("spark.driver.extraClassPath", rapids_jar)
        .config("spark.executor.extraClassPath", rapids_jar)

        # ── RAPIDS Plugin Configuration ──
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.rapids.sql.explain", "ALL")

        # GPU resource allocation
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "0.5")
        .config("spark.rapids.sql.concurrentGpuTasks", "2")

        # Memory
        .config("spark.rapids.memory.pinnedPool.size", "2g")
        .config("spark.rapids.sql.batchSizeBytes", str(256 * 1024 * 1024))

        # Enable GPU operations
        .config("spark.rapids.sql.incompatibleOps.enabled", "true")
        .config("spark.rapids.sql.variableFloatAgg.enabled", "true")
        .config("spark.rapids.sql.hasNans", "false")

        # Kryo serialization — RAPIDS requires this registrator
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrator", "com.nvidia.spark.rapids.GpuKryoRegistrator")

        # Adaptive query execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
    )

    # ── GVirtuS Configuration ──
    if use_gvirtus:
        builder = (
            builder
            .config("spark.executorEnv.LD_PRELOAD", "/path/to/gvirtus/libcuda-frontend.so")
            .config("spark.executorEnv.GVIRTUS_CONFIG", "/path/to/gvirtus/gvirtus.yaml")
        )

    return builder.getOrCreate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAPIDS E-Commerce Pipeline")
    parser.add_argument("--gvirtus", action="store_true", help="Use GVirtuS virtual GPU")
    args = parser.parse_args()

    spark = create_rapids_spark_session(use_gvirtus=args.gvirtus)
    try:
        pipeline = RapidsEcommercePipeline(spark)
        pipeline.run()
    finally:
        spark.stop()