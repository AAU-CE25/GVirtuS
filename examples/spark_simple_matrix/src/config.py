# config.py

"""
Scalable benchmark configuration.
Adjust SCALE_FACTOR to control data size:

"""

SCALE_FACTOR = 1

#DEBUG 
LOG_LEVEL = "DEBUG"  # "DEBUG" for more verbose output, "INFO" or "WARN" for less
SPARK_LOG_LEVEL = "DEBUG"  # Suppress Spark's verbose Java logs

# Paths
DATA_DIR = f"/data/wg38up/spark_simple_matrix/sf{SCALE_FACTOR}"
RESULTS_DIR = f"../results/sf{SCALE_FACTOR}"

# RAPIDS JAR path (assumes it's already built and placed here)
RAPIDS_JAR_PATH = "/home/student.aau.dk/wg38up/jars/rapids-4-spark_2.12-26.02.1.jar"

# ── Spark config (CPU-only, no RAPIDS plugin enabled) ──
# JAR is on classpath so the JVM doesn't need to restart when switching modes.
SPARK_MASTER = "local[4]"

# Path to log4j2.properties (same directory as this file)
import os
LOG4J_CONF_PATH = os.path.join(os.path.dirname(__file__), "log4j2.properties")

SPARK_CONFIG = [
    ("spark.local.dir", "/data/wg38up/spark-temp"),
    ("spark.executor.memory", "8g"),
    ("spark.driver.memory", "4g"),
    ("spark.jars", RAPIDS_JAR_PATH),
    ("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile=file:{LOG4J_CONF_PATH}"),
    ("spark.executor.extraJavaOptions", f"-Dlog4j.configurationFile=file:{LOG4J_CONF_PATH}"),
]

# ── Spark config (with RAPIDS GPU acceleration) ──
SPARK_RAPIDS_CONFIG = [
    *SPARK_CONFIG,
    ("spark.plugins", "com.nvidia.spark.SQLPlugin"),
    ("spark.rapids.sql.enabled", "true"),
    ("spark.rapids.sql.explain", "ALL"),
    ("spark.rapids.memory.gpu.pool", "NONE"),
    ("spark.sql.shuffle.partitions", "4"),
    ("spark.sql.session.timeZone", "UTC"),
    # In spark config, try disabling JIT compilation:
    ("spark.rapids.sql.castStringToFloat.enabled", "false"),
    ("spark.rapids.sql.expression.HiveSimpleUDF", "false"),
    #Debugging: disable whole-stage codegen to see if it helps with JIT issues:
    ("spark.rapids.memory.gpu.state.debug.enabled", "true"),
]

# ── GVirtuS-specific RAPIDS config ──
# These settings ensure Spark executors use GVirtuS frontend libs
GVIRTUS_HOME_PATH = "/opt/GVirtuS"
GVIRTUS_LIB_PATH = f"{GVIRTUS_HOME_PATH}/lib/frontend:{GVIRTUS_HOME_PATH}/lib"
RAPIDS_NATIVE_PATH = "/tmp/rapids-native"  # Extracted by entrypoint from JAR

SPARK_RAPIDS_CONFIG_WITH_LD_PRELOAD = [
    *SPARK_RAPIDS_CONFIG,
    # Ensure executor/driver use GVirtuS frontend libs + RAPIDS native libs
    ("spark.executor.extraLibraryPath", f"{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}"),
    ("spark.driver.extraLibraryPath", f"{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}"),
    # Pass GVirtuS env vars to executors
    #("spark.executorEnv.GVIRTUS_HOME", GVIRTUS_HOME_PATH),
    #("spark.executorEnv.LD_LIBRARY_PATH", f"{GVIRTUS_LIB_PATH}:{RAPIDS_NATIVE_PATH}"),
    #("spark.executorEnv.LD_PRELOAD", f"{GVIRTUS_HOME_PATH}/lib/frontend/libcudart.so:{GVIRTUS_HOME_PATH}/lib/frontend/libcuda.so"),
]