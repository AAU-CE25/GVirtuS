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
except ImportError:
    from config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY, RAPIDS_JAR_PATH
    from ecomerce_pipeline import EcommercePipeline


class RapidsEcommercePipeline(EcommercePipeline):

    def run(self):
        """Execute all pipeline stages with RAPIDS GPU acceleration."""
        print(f"\n{'='*60}")
        print(f"  E-Commerce Analytics Pipeline — RAPIDS GPU Accelerated")
        print(f"{'='*60}")
    
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
   
    builder = (
        SparkSession.builder
        .appName("EcommerceAnalytics-RAPIDS-GPU")
        .master(SPARK_MASTER)
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