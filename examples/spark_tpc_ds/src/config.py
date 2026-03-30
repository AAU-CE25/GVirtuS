"""
config.py — Central configuration for the Spark TPC-DS benchmark.

All tuneable knobs live here so that pipeline scripts stay clean.
"""

from __future__ import annotations


from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = "/data/wg38up/spark_tpc_ds"
RESULTS_DIR  = f"{PROJECT_ROOT}/results/sf{SCALE_FACTOR}"

# ──────────────────────────────────────────────
# TPC-DS scale factor
#   SF1  ≈  1 GB   (~6 M store_sales rows)
#   SF10 ≈ 10 GB   (~60 M store_sales rows)
# ──────────────────────────────────────────────
SCALE_FACTOR: int = 1

# Derived data path  →  data/sf1/  or  data/sf10/
DATA_PATH = f"{DATA_DIR}/sf{SCALE_FACTOR}"

# ──────────────────────────────────────────────
# Row counts per scale factor (approximate TPC-DS spec ratios)
# ──────────────────────────────────────────────
_ROW_SCALE = {
    # table_name: rows_per_sf1
    "store_sales":       5_760_000,
    "store_returns":       288_000,
    "catalog_sales":     1_440_000,
    "catalog_returns":     144_000,
    "web_sales":           720_000,
    "web_returns":          72_000,
    "inventory":        11_745_000,
    "customer":            100_000,
    "customer_address":     50_000,
    "customer_demographics": 1_920_800,
    "date_dim":              73_049,   # fixed, not scaled
    "item":                  18_000,
    "store":                     12,
    "promotion":                300,
    "household_demographics": 7_200,
    "warehouse":                  5,
    "ship_mode":                 20,
    "reason":                    35,
    "income_band":               20,
    "time_dim":              86_400,   # fixed
    "call_center":                6,
    "web_page":                  60,
    "web_site":                  30,
    "catalog_page":          11_718,
}

def row_count(table: str) -> int:
    """Return the target row count for *table* at the current scale factor."""
    base = _ROW_SCALE.get(table, 1000)
    # Dimension tables with fixed counts don't scale
    fixed = {"date_dim", "time_dim", "customer_demographics",
             "household_demographics", "income_band", "ship_mode",
             "reason", "call_center", "web_page", "web_site"}
    if table in fixed:
        return base
    return base * SCALE_FACTOR


# ──────────────────────────────────────────────
# Spark session settings
# ──────────────────────────────────────────────
SPARK_APP_NAME = "TPC-DS Benchmark"

SPARK_CPU_CONF: dict[str, str] = {
    "spark.master":                    os.getenv("SPARK_MASTER", "local[*]"),
    "spark.driver.memory":             os.getenv("SPARK_DRIVER_MEMORY", "4g"),
    "spark.executor.memory":           os.getenv("SPARK_EXECUTOR_MEMORY", "4g"),
    "spark.sql.shuffle.partitions":    "200",
    "spark.sql.adaptive.enabled":      "true",
    "spark.serializer":                "org.apache.spark.serializer.KryoSerializer",
}

# ──────────────────────────────────────────────
# RAPIDS Accelerator (spark-rapids plugin)
# Download from: https://nvidia.github.io/spark-rapids/
# Set via env:  RAPIDS_JAR=/path/to/rapids-4-spark_2.12-24.10.0.jar
# ──────────────────────────────────────────────
RAPIDS_JAR: str = f"{PROJECT_ROOT}/rapids-4-spark_ADD_YOUR_VERSION.jar"

SPARK_RAPIDS_CONF: dict[str, str] = {
    **SPARK_CPU_CONF,
    "spark.plugins":                        "com.nvidia.spark.SQLPlugin",
    "spark.rapids.sql.enabled":             "true",
    "spark.rapids.sql.explain":             "NOT_ON_GPU",
    "spark.rapids.memory.pinnedPool.size":  "2G",
    "spark.sql.files.maxPartitionBytes":    "512m",
    "spark.jars":                           RAPIDS_JAR,
}

# ──────────────────────────────────────────────
# Pipeline stages to run (name → enabled)
# Toggle individual stages for debugging.
# ──────────────────────────────────────────────
STAGES_ENABLED: dict[str, bool] = {
    "scan":              True,
    "filter_project":    True,
    "join":              True,
    "aggregate":         True,
    "window":            True,
    "cross_channel":     True,
    "write":             True,
}
