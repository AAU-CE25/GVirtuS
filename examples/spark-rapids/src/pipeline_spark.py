# pipeline_spark.py
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
    from .config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY
    from .ecomerce_pipeline import EcommercePipeline
except ImportError:
    from config import DATA_DIR, RESULTS_DIR, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY
    from ecomerce_pipeline import EcommercePipeline

class SparkEcommercePipeline(EcommercePipeline):
    def run(self):
        """Execute all pipeline stages and report timings."""
        print(f"\n{'='*60}")
        print(f"  E-Commerce Analytics Pipeline — Native Spark (CPU)")
        print(f"{'='*60}")

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
        with open(f"{RESULTS_DIR}/spark_cpu_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {RESULTS_DIR}/spark_cpu_results.json")

        return self.timings


def create_spark_session() -> SparkSession:
    """Create a Spark session for CPU execution."""
    return (
        SparkSession.builder
        .appName("EcommerceAnalytics-CPU")
        .master(SPARK_MASTER)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = create_spark_session()
    try:
        pipeline = SparkEcommercePipeline(spark)
        pipeline.run()
    finally:
        spark.stop()