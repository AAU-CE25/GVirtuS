# ── Spark + RAPIDS container for simple_matrix ──
# Python 3.10 + Java 17 + PySpark 3.5.2 + RAPIDS plugin
#
# Based on the same NVIDIA CUDA base image as the GVirtuS dev container.
# src/, results/, and jars/ are bind-mounted at runtime (see Makefile targets).

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps: Java 17, Python 3.10 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        python3 \
        python3-pip \
        python3-venv \
        procps \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ── Python deps (PySpark, numpy, etc.) ──
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Verify Spark is reachable
RUN python3 -c "import pyspark; print(f'PySpark {pyspark.__version__} OK')"

# ── Working directory ──
# src/, results/, jars/ are mounted under /app at runtime
WORKDIR /app/src

# Default: run simple_matrix.py both
# The RAPIDS JAR path is configured in config.py (spark.jars),
# so no --jars flag is needed on the command line.
CMD ["python3", "simple_matrix.py", "--mode=both"]
