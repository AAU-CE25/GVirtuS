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

RAPIDS_JAR_PATH_HOST = "/home/student.aau.dk/wg38up/jars/rapids-4-spark_2.12-26.02.1.jar"
RAPIDS_JAR_PATH_DOCKER = "/app/jars/rapids-4-spark_2.12-26.02.1.jar"

# ── Spark config (CPU-only, no RAPIDS plugin enabled) ──
# JAR is on classpath so the JVM doesn't need to restart when switching modes.
SPARK_MASTER = "local[1]"

# ── GVirtuS-specific RAPIDS config ──
# These settings ensure Spark executors use GVirtuS frontend libs
GVIRTUS_HOME_PATH = "/opt/GVirtuS"
GVIRTUS_LIB_PATH = f"{GVIRTUS_HOME_PATH}/lib/frontend:{GVIRTUS_HOME_PATH}/lib"

# Path to log4j2.properties (same directory as this file)
import os
LOG4J_CONF_PATH = os.path.join(os.path.dirname(__file__), "log4j2.properties")

def build_spark_config(jar_path):
    return [
        ("spark.local.dir", "/data/wg38up/spark-temp"),
        ("spark.executor.memory", "8g"),
        ("spark.driver.memory", "4g"),
        ("spark.jars", jar_path),
        ("spark.driver.extraJavaOptions", f"-Dlog4j.configurationFile=file:{LOG4J_CONF_PATH}"),
        ("spark.executor.extraJavaOptions", f"-Dlog4j.configurationFile=file:{LOG4J_CONF_PATH}"),
    ]

def build_spark_rapids_config(jar_path):
    base = build_spark_config(jar_path)
    return base + [
        ("spark.plugins", "com.nvidia.spark.SQLPlugin"),
        ("spark.rapids.sql.enabled", "true"),
        ("spark.rapids.sql.explain", "ALL"),
        ("spark.rapids.memory.gpu.pool", "NONE"),
        ("spark.sql.shuffle.partitions", "4"),
        ("spark.sql.session.timeZone", "UTC"),
        ("spark.rapids.sql.castStringToFloat.enabled", "false"),
        ("spark.rapids.sql.expression.HiveSimpleUDF", "false"),
        ("spark.rapids.memory.gpu.state.debug.enabled", "true"),
        ("spark.executor.extraLibraryPath", f"{GVIRTUS_LIB_PATH}"),
        ("spark.driver.extraLibraryPath", f"{GVIRTUS_LIB_PATH}"),
    ]

def get_spark_config(env, compute_mode):
    using_docker = env == "docker" or env == "gvirtus"
    use_rapids = compute_mode == "rapids" 

    jar_path = RAPIDS_JAR_PATH_DOCKER if using_docker else RAPIDS_JAR_PATH_HOST

    if use_rapids:
        return build_spark_rapids_config(jar_path)
    else:
        return build_spark_config(jar_path)