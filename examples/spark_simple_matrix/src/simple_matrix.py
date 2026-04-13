#!/usr/bin/env python3
"""
Simple matrix multiplication using Apache Spark.
Generates two random matrices, multiplies them, and saves the result.

Usage:
    python simple_matrix.py local              # Run locally (native)
    python simple_matrix.py docker             # Run through Docker
    python simple_matrix.py gvirtus            # Run using GVirtuS

    python simple_matrix.py local --mode rapids           # RAPIDS GPU acceleration
    python simple_matrix.py local --mode rapids --minimal # Run minimal GPU test first
    python simple_matrix.py local --overwrite no          # Merge results into existing file

The RAPIDS JAR is placed on the JVM classpath automatically via
PYSPARK_SUBMIT_ARGS (set in config.py), so no spark-submit is needed.

Minimal Test (--minimal):
    Runs a 3-row DataFrame test before the full matrix multiply.
    Useful for debugging GVirtuS integration - tests:
    - GPU device detection (cudaGetDeviceCount)
    - RMM memory pool allocation (cudaMalloc)
    - Row-to-columnar conversion (GpuRowToColumnar)
    - Basic aggregation (GpuHashAggregate)
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
    SPARK_RAPIDS_CONFIG_WITH_LD_PRELOAD
)

from pyspark.sql import SparkSession

# ── Logger (log4cplus-style format) ──
LOG_FORMAT = "%(asctime)s [%(levelname)-5s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s"
LOG_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    filename="../logs/simple_matrix.log",  # Add this line
    filemode="w"                   # "w" = overwrite, "a" = append
)
log = logging.getLogger("SimpleMatrix")

# Suppress noisy third-party loggers
logging.getLogger("py4j").setLevel(logging.WARNING)

# Matrix size scales with SCALE_FACTOR
N = 100 * SCALE_FACTOR  # NxN matrices


def create_spark_session(custom_config, session_name="SimpleMatrix"):
    config = custom_config
    app_name = session_name
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
    )

    for k, v in config:
        builder = builder.config(k, v)

    log.debug(f"Initializing Spark config for {app_name}...")
    log.info(">>> About to call SparkSession.builder.getOrCreate() - this triggers RAPIDS/CUDA init")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    logging.getLogger().handlers[0].flush()  # Force log flush
    
    spark = builder.getOrCreate()
    
    log.info("<<< SparkSession created successfully")
    log.debug(f"Spark config for {app_name}: {dict(config)}")

    # Suppress Spark's verbose Java logs (keep only WARN+)
    spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    return spark


def minimal_gpu_test(spark):
    """
    Minimal GPU test case for GVirtuS integration.
    
    Creates a tiny DataFrame and triggers a GPU transfer via count().
    This is the simplest possible test to verify:
    - cudaGetDeviceCount / cudaGetDeviceProperties work
    - RMM pool allocation succeeds
    - Row-to-columnar conversion (GpuRowToColumnar) works
    - Basic GPU aggregation (count) works
    - Columnar-to-row conversion (GpuColumnarToRow) works
    
    If this fails, the full matrix multiply will definitely fail.
    """
    from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
    
    log.info("Running minimal GPU test case...")
    
    schema = StructType([
        StructField("row", IntegerType(), False),
        StructField("col", IntegerType(), False),
        StructField("val", DoubleType(), False),
    ])
    
    # Minimal data: 3 rows to verify GPU operations
    data = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)]
    df = spark.createDataFrame(data, schema=schema)
    
    t0 = time.time()
    
    # count() triggers: GpuRowToColumnar → GpuHashAggregate → GpuColumnarToRow
    count = df.count()
    
    # sum() triggers additional GPU compute
    total = df.agg({"val": "sum"}).collect()[0][0]
    
    elapsed = time.time() - t0
    
    log.info(f"Minimal test: count={count}, sum={total}, elapsed={elapsed:.4f}s")
    
    return {
        "count": count,
        "sum": total,
        "elapsed": elapsed,
        "expected_count": 3,
        "expected_sum": 6.0,
        "passed": count == 3 and abs(total - 6.0) < 0.001
    }


def multiply_matrices_df(spark, n):
    """
    Multiply two NxN matrices using Spark DataFrames.

    Uses DataFrame join + groupBy + sum so the RAPIDS SQL plugin can
    accelerate the computation on GPU.  (RDD operations are invisible
    to the plugin — only Catalyst/SQL plans get GPU-offloaded.)

    C[i][j] = sum_k  A[i][k] * B[k][j]
    """
    import random
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

    random.seed(42)

    schema = StructType([
        StructField("row", IntegerType(), False),
        StructField("col", IntegerType(), False),
        StructField("val", DoubleType(), False),
    ])

    # Generate flat list of (row, col, value) entries for A and B
    a_entries = [(i, k, random.random()) for i in range(n) for k in range(n)]
    b_entries = [(k, j, random.random()) for k in range(n) for j in range(n)]

    # Create DataFrames (column names: row, col, val)
    df_a = spark.createDataFrame(a_entries, schema=schema)
    df_b = spark.createDataFrame(b_entries, schema=schema)

    # Rename to avoid ambiguity after join
    a = df_a.alias("a")
    b = df_b.alias("b")

    # Join on A.col == B.row  (the shared "k" dimension)
    # Then compute products and sum per (i, j)
    t0 = time.time()
    result = (
        a.join(b, F.col("a.col") == F.col("b.row"))
         .select(
             F.col("a.row").alias("i"),
             F.col("b.col").alias("j"),
             (F.col("a.val") * F.col("b.val")).alias("product"),
         )
         .groupBy("i", "j")
         .agg(F.sum("product").alias("value"))
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


def main(env, compute_mode, results_overwrite, run_minimal=False):

    use_rapids = compute_mode == "rapids"
    log.info(f"Env: {env} | Mode: {compute_mode} | Matrix size: {N}x{N}  (scale factor {SCALE_FACTOR})")

    config = SPARK_RAPIDS_CONFIG_WITH_LD_PRELOAD if use_rapids else SPARK_CONFIG
    app_name = f"SimpleMatrixMultiply-{compute_mode.upper()}-{env.upper()}"
    spark = create_spark_session(custom_config=config, session_name=app_name)

    # Run minimal test first if requested
    if run_minimal:
        minimal_result = minimal_gpu_test(spark)
        if not minimal_result["passed"]:
            log.error("Minimal GPU test FAILED - aborting full test")
            spark.stop()
            return
        log.info("Minimal GPU test PASSED - proceeding with full test")
    

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
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal GPU test first (3 rows) before full matrix multiply. Useful for GVirtuS debugging.",
    )
  
    args = parser.parse_args()
    overwrite = args.overwrite == "yes"
    main(args.env, args.mode, results_overwrite=overwrite, run_minimal=args.minimal)
   
