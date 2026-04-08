#!/usr/bin/env python3
"""
Simple matrix multiplication using Apache Spark.
Generates two random matrices, multiplies them, and saves the result.

Usage:
    python simple_matrix.py local              # Run locally (native)
    python simple_matrix.py docker             # Run through Docker
    python simple_matrix.py gvirtus            # Run using GVirtuS

    python simple_matrix.py local --mode rapids           # RAPIDS GPU acceleration
    python simple_matrix.py local --overwrite no          # Merge results into existing file

The RAPIDS JAR is placed on the JVM classpath automatically via
PYSPARK_SUBMIT_ARGS (set in config.py), so no spark-submit is needed.
"""

import argparse
import logging
import time
import json
import os

# Import config FIRST — it sets PYSPARK_SUBMIT_ARGS so the RAPIDS JAR
# is on the JVM classpath before PySpark boots.
from config import (
    LOG_LEVEL,
    SPARK_LOG_LEVEL,
    SCALE_FACTOR,
    RESULTS_DIR,
    SPARK_MASTER, 
    SPARK_CONFIG,
    SPARK_RAPIDS_CONFIG,
    SPARK_RAPIDS_GVIRTUS_CONFIG,
)

from pyspark.sql import SparkSession

# ── Logger (log4cplus-style format) ──
LOG_FORMAT = "%(asctime)s [%(levelname)-5s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s"
LOG_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
log = logging.getLogger("SimpleMatrix")

# Suppress noisy third-party loggers
logging.getLogger("py4j").setLevel(logging.WARNING)

# Matrix size scales with SCALE_FACTOR
N = 100 * SCALE_FACTOR  # NxN matrices


def create_spark_session(use_rapids=False, env="local"):
    # Choose config based on env and mode
    if env == "gvirtus" and use_rapids:
        config = SPARK_RAPIDS_GVIRTUS_CONFIG
    elif use_rapids:
        config = SPARK_RAPIDS_CONFIG
    else:
        config = SPARK_CONFIG
    
    app_name = f"SimpleMatrixMultiply-{'RAPIDS' if use_rapids else 'CPU'}-{env}"
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
    )

    for k, v in config:
        builder = builder.config(k, v)

    log.debug(f"Spark config for {app_name}: {dict(config)}")
    spark = builder.getOrCreate()

    # Suppress Spark's verbose Java logs (keep only WARN+)
    spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    return spark


def multiply_matrices_df(spark, n):
    """
    Multiply two NxN matrices using Spark DataFrames.
    C[i][j] = sum_k  A[i][k] * B[k][j]
    """
    from pyspark.sql import functions as F

    # Generate matrices directly in Spark (no Python lists)
    coords = spark.range(n).crossJoin(spark.range(n).withColumnRenamed("id", "id2"))
    A = coords.select(F.col("id").alias("row"), F.col("id2").alias("col"), F.rand(42).alias("val"))
    B = coords.select(F.col("id").alias("row"), F.col("id2").alias("col"), F.rand(43).alias("val"))

    t0 = time.time()
    result = (
        A.alias("a")
         .join(B.alias("b"), F.col("a.col") == F.col("b.row"))
         .groupBy(F.col("a.row").alias("i"), F.col("b.col").alias("j"))
         .agg(F.sum(F.col("a.val") * F.col("b.val")).alias("value"))
    )
    elapsed = time.time() - t0

    return result, elapsed


def save_summary(filepath, summary, overwrite=True):
    """
    Save a JSON summary to *filepath*.

    Args:
        filepath:  Destination path.
        summary:   Dict to persist.
        overwrite: If True (default), replace the file entirely.
                   If False, merge *summary* into the existing file
                   (top-level keys are updated/added; the rest is kept).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not overwrite and os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = json.load(f)
        existing.update(summary)
        summary = existing

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Results saved to {filepath} (overwrite={overwrite})")


def main(env, compute_mode, results_overwrite):

    use_rapids = compute_mode == "rapids"
    log.info(f"Env: {env} | Mode: {compute_mode} | Matrix size: {N}x{N}  (scale factor {SCALE_FACTOR})")

    spark = create_spark_session(use_rapids=use_rapids, env=env)
    if env == "gvirtus" and use_rapids:
        config = SPARK_RAPIDS_GVIRTUS_CONFIG
    elif use_rapids:
        config = SPARK_RAPIDS_CONFIG
    else:
        config = SPARK_CONFIG

    t0 = time.time()
    result_df, mul_time_elapsed = multiply_matrices_df(spark, N)
    count = result_df.count()
    elapsed = time.time() - t0

    log.debug(f"Result has {count} elements ({N}x{N} = {N*N} expected)")
    log.debug(f"Elapsed: {elapsed:.4f}s")
    log.debug(f"Matrix multiplication time: {mul_time_elapsed:.4f}s")

    # Save summary
    summary = {
        "env": env,
        "mode": compute_mode,
        "scale_factor": SCALE_FACTOR,
        "matrix_size": N,
        "result_elements": count,
        "elapsed_seconds": round(elapsed, 4),
        "matrix_multiplication_time": round(mul_time_elapsed, 4),
        "spark_config": dict(config),
    }
    out_path = os.path.join(RESULTS_DIR, f"simple_matrix_{env}_{compute_mode}_results.json")
    save_summary(out_path, summary, overwrite=results_overwrite)

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark matrix multiplication")
    parser.add_argument(
        "env",
        choices=["local", "docker", "gvirtus"],
        default="local",
        help="Execution environment: local (native), docker, or gvirtus",
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "rapids"],
        default="cpu",
        help="Execution mode: cpu (default) or rapids (needs GPU)",
    )
    parser.add_argument(
        "--overwrite",
        choices=["yes", "no"],
        default="yes",
        help="Overwrite existing results file (default: yes). If no, merge new results into existing file.",
    )
  
    args = parser.parse_args()
    overwrite = args.overwrite == "yes"
    main(args.env, args.mode, results_overwrite=overwrite)
   
