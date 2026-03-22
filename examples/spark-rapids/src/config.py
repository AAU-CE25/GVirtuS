# config.py

"""
Scalable benchmark configuration.
Adjust SCALE_FACTOR to control data size:
  SCALE_FACTOR = 1    → ~1 GB   (development/testing)
  SCALE_FACTOR = 10   → ~10 GB  (standard benchmark)
  SCALE_FACTOR = 100  → ~100 GB (stress test)
  SCALE_FACTOR = 1000 → ~1 TB   (production scale)
"""

SCALE_FACTOR = 10

# Derived sizes (linear scaling)
NUM_CUSTOMERS = 1_000_00 * SCALE_FACTOR
NUM_PRODUCTS = 100_00 * SCALE_FACTOR
NUM_ORDERS = 10_000_00 * SCALE_FACTOR
NUM_ORDER_ITEMS = 30_000_00 * SCALE_FACTOR
NUM_CLICKSTREAM = 50_000_00 * SCALE_FACTOR

# Paths
DATA_DIR = f"data/sf{SCALE_FACTOR}"
RESULTS_DIR = f"results/sf{SCALE_FACTOR}"

# Spark config
SPARK_MASTER = "local[4]"
SPARK_EXECUTOR_MEMORY = "8g"
SPARK_DRIVER_MEMORY = "4g"

RAPIDS_JAR_PATH = "jars/rapids-4-spark_2.12-25.02.1.jar"
