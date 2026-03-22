# pipeline_hybrid.py

import argparse
import json
import os
from typing import Dict, List, Set, Tuple

from pyspark.sql import SparkSession

try:
    from .config import RESULTS_DIR, SPARK_DRIVER_MEMORY, SPARK_MASTER, RAPIDS_JAR_PATH
    from .ecomerce_pipeline import EcommercePipeline
except ImportError:
    from config import RESULTS_DIR, SPARK_DRIVER_MEMORY, SPARK_MASTER, RAPIDS_JAR_PATH
    from ecomerce_pipeline import EcommercePipeline


class HybridEcommercePipeline(EcommercePipeline):
    STAGES: List[Tuple[int, str, str]] = [
        (1, "1_data_loading", "load_data"),
        (2, "2_revenue_analytics", "revenue_analytics"),
        (3, "3_customer_360", "customer_360"),
        (4, "4_rfm_segmentation", "rfm_segmentation"),
        (5, "5_cohort_analysis", "cohort_analysis"),
        (6, "6_funnel_analysis", "funnel_analysis"),
        (7, "7_customer_clustering", "customer_clustering"),
    ]

    def _set_stage_mode(self, mode: str):
        rapids_enabled = "true" if mode == "gpu" else "false"
        self.spark.conf.set("spark.rapids.sql.enabled", rapids_enabled)

    def run(self, gpu_stages: Set[int]):
        print(f"\n{'='*60}")
        print("  E-Commerce Analytics Pipeline — HYBRID (CPU + RAPIDS)")
        print(f"{'='*60}")

        execution_plan: Dict[str, str] = {}
        for stage_id, stage_name, _ in self.STAGES:
            execution_plan[stage_name] = "gpu" if stage_id in gpu_stages else "cpu"

        print("\n  Stage execution plan:")
        for _, stage_name, _ in self.STAGES:
            print(f"    {stage_name:<30} {execution_plan[stage_name]}")

        for stage_id, stage_name, method_name in self.STAGES:
            mode = execution_plan[stage_name]
            self._set_stage_mode(mode)
            print(f"\n  → Running {stage_name} in {mode.upper()} mode")
            getattr(self, method_name)()

        total_time = sum(self.timings.values())
        print(f"\n{'='*60}")
        print(f"  Pipeline Complete — Total: {total_time:.2f}s")
        print(f"{'='*60}")
        print("\n  Stage Timings:")
        for stage, elapsed in self.timings.items():
            pct = (elapsed / total_time) * 100
            bar = "█" * int(pct / 2)
            print(f"    {stage:<30} {elapsed:>8.2f}s  ({pct:>5.1f}%) {bar}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        output = {
            "timings": self.timings,
            "results": self.results,
            "stage_execution": execution_plan,
        }
        with open(f"{RESULTS_DIR}/hybrid_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {RESULTS_DIR}/hybrid_results.json")

        return self.timings


def create_hybrid_spark_session() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("EcommerceAnalytics-Hybrid")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.jars", RAPIDS_JAR_PATH)
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.rapids.sql.explain", "ALL")
        .config("spark.rapids.memory.gpu.pool", "NONE")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
    )
    return builder.getOrCreate()


def parse_stage_list(stage_list_arg: str) -> Set[int]:
    if not stage_list_arg.strip():
        return set()

    stages: Set[int] = set()
    for token in stage_list_arg.split(","):
        token = token.strip()
        if not token:
            continue
        stage_id = int(token)
        if stage_id < 1 or stage_id > 7:
            raise ValueError(f"Invalid stage id '{stage_id}'. Valid range is 1..7")
        stages.add(stage_id)
    return stages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid CPU + RAPIDS E-Commerce Pipeline")
    parser.add_argument(
        "--gpu-stages",
        type=str,
        default="2,4,5,6,7",
        help="Comma-separated stage ids to run with RAPIDS (default: 2,4,5,6,7)",
    )
    args = parser.parse_args()

    gpu_stages = parse_stage_list(args.gpu_stages)

    spark = create_hybrid_spark_session()
    try:
        pipeline = HybridEcommercePipeline(spark)
        pipeline.run(gpu_stages=gpu_stages)
    finally:
        spark.stop()
