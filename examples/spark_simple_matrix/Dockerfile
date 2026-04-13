# ══════════════════════════════════════════════════════════════════════════════
# Unified Spark + RAPIDS container for simple_matrix benchmark
# ══════════════════════════════════════════════════════════════════════════════
#
# A single image that supports THREE execution modes:
#   1. Local GPU   (--runtime=nvidia) - uses real CUDA libs from the base image
#   2. GVirtuS     (no --runtime=nvidia) - uses GVirtuS frontend stubs
#   3. CPU-only    (no GPU at all)
#
# The switch between local GPU and GVirtuS happens at the Spark config level:
#   - Local GPU: JVM loads real CUDA libs from /usr/local/cuda/lib64
#   - GVirtuS:   JVM loads GVirtuS stubs via spark.executor.extraLibraryPath
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Build GVirtuS frontend stubs ──
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS gvirtus-builder

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxmu-dev \
        libxi-dev \
        libgl-dev \
        libosmesa-dev \
        liblog4cplus-dev \
        librdmacm-dev \
        libibverbs-dev \
        libgtest-dev \
        liblz4-dev \
        cmake \
    && rm -rf /var/lib/apt/lists/*

ENV GVIRTUS_HOME=/opt/GVirtuS

COPY cmake /gvirtus/cmake
COPY etc /gvirtus/etc
COPY include /gvirtus/include
COPY plugins /gvirtus/plugins
COPY src /gvirtus/src
COPY tools /gvirtus/tools
COPY tests /gvirtus/tests
COPY CMakeLists.txt /gvirtus/

WORKDIR /gvirtus/build
RUN cmake .. && make -j$(nproc) && make install

# ── Stage 2: Runtime image (CUDA base + GVirtuS stubs) ──
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps: Java 17, Python 3.10 ──
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        python3 \
        python3-pip \
        python3-venv \
        liblog4cplus-dev \
        librdmacm-dev \
        libibverbs-dev \
        liblz4-dev \
        procps \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ── Copy GVirtuS frontend stubs from builder ──
ENV GVIRTUS_HOME=/opt/GVirtuS
COPY --from=gvirtus-builder ${GVIRTUS_HOME} ${GVIRTUS_HOME}

# ── Python deps (PySpark, numpy, etc.) ──
COPY examples/spark_simple_matrix/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN python3 -c "import pyspark; print(f'PySpark {pyspark.__version__} OK')"

# ── Entrypoint ──
COPY --chmod=755 examples/spark_simple_matrix/entrypoint.sh /entrypoint.sh

WORKDIR /app/src

# Default GVirtuS log level (INFO). Override with -e GVIRTUS_LOGLEVEL=0
ENV GVIRTUS_LOGLEVEL=20000

ENTRYPOINT ["/entrypoint.sh"]
