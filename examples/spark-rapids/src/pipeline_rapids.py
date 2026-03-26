import sys
import re
# pipeline_rapids.py

import time
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
try:
    from .config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY, RAPIDS_JAR_PATH
    from .ecomerce_pipeline import EcommercePipeline
    from .custom_logger import CustomAcceleratedSparkLogger
except ImportError:
    from config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY, RAPIDS_JAR_PATH
    from ecomerce_pipeline import EcommercePipeline
    from custom_logger import CustomAcceleratedSparkLogger


class RapidsEcommercePipeline(EcommercePipeline):
    def __init__(self, spark):
        super().__init__(spark)
        self.logger = CustomAcceleratedSparkLogger(RESULTS_DIR)

    def run(self):
        print(f"\n{'='*60}")
        print(f"  E-Commerce Analytics Pipeline — RAPIDS GPU Accelerated")
        print(f"{'='*60}")
        print(f"  GPU acceleration: ENABLED (RAPIDS plugin active)")

        self.logger.log_event("pipeline", "START")
        self.logger.start_gpu_logger()
        log_thread = threading.Thread(target=self.logger.capture_spark_logs, daemon=True)
        log_thread.start()
        try:
            stages = [
                ("load_data", self.load_data),
                ("revenue_analytics", self.revenue_analytics),
                ("customer_360", self.customer_360),
                ("rfm_segmentation", self.rfm_segmentation),
                ("cohort_analysis", self.cohort_analysis),
                ("funnel_analysis", self.funnel_analysis),
                ("customer_clustering", self.customer_clustering),
            ]
            for stage, func in stages:
                t0 = time.time()
                self.logger.log_event(stage, "START")
                func()
                elapsed = time.time() - t0
                self.logger.log_event(stage, f"END ({elapsed:.2f}s)")

            total_time = sum(self.timings.values())
            print(f"\n{'='*60}")
            print(f"  Pipeline Complete — Total: {total_time:.2f}s")
            print(f"{'='*60}")
            print(f"\n  Stage Timings:")
            for stage, elapsed in self.timings.items():
                pct = (elapsed / total_time) * 100
                bar = "█" * int(pct / 2)
                print(f"    {stage:<30} {elapsed:>8.2f}s  ({pct:>5.1f}%) {bar}")

            os.makedirs(RESULTS_DIR, exist_ok=True)
            output = {"timings": self.timings, "results": self.results}
            with open(f"{RESULTS_DIR}/rapids_gpu_results.json", "w") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"\n  Results saved to {RESULTS_DIR}/rapids_gpu_results.json")
        finally:
            self.logger.log_event("pipeline", "END")
            self.logger.stop_gpu_logger()
            self.logger.save_logs()
            print(f"\n  Pipeline and GPU logs saved to {RESULTS_DIR}")
            try:
                log_thread.join(timeout=2)
            except Exception:
                pass
        return self.timings


def create_rapids_spark_session(use_gvirtus: bool = False) -> SparkSession:
    """
    Create a Spark session with RAPIDS GPU acceleration.
    """
   
    builder = (
        SparkSession.builder
        .appName("EcommerceAnalytics-RAPIDS-GPU")
        .master(SPARK_MASTER)
        .config("spark.local.dir", "/data/wg38up/spark-temp")  # Local temp dir for shuffle spill
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)

        # RAPIDS JAR
        .config("spark.jars", RAPIDS_JAR_PATH)

        # RAPIDS Plugin
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.rapids.sql.explain", "ALL")

        # Memory
        .config("spark.rapids.memory.gpu.pool", "NONE")
        .config("spark.sql.shuffle.partitions", 4)

        .config("spark.sql.session.timeZone", "UTC")
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