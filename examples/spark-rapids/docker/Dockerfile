# docker/Dockerfile

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

# ══════════════════════════════════════════════
#  GVirtuS build dependencies
# ══════════════════════════════════════════════
RUN apt update && apt install -y --no-install-recommends \
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

# ══════════════════════════════════════════════
#  Build GVirtuS
# ══════════════════════════════════════════════
WORKDIR /opt/GVirtuS
COPY . .
RUN mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install

# ══════════════════════════════════════════════
#  Spark + Python dependencies
# ══════════════════════════════════════════════
ARG SPARK_VERSION=3.5.2
ARG HADOOP_VERSION=3
ARG JAVA_VERSION=17

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-${JAVA_VERSION}-jdk-headless \
    python3 python3-pip \
    wget curl netcat-openbsd procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" && \
    tar xzf spark-*.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-*.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON=python3

RUN pip3 install --no-cache-dir \
    pyspark==${SPARK_VERSION} \
    numpy \
    pandas

# ══════════════════════════════════════════════
#  RAPIDS jar
# ══════════════════════════════════════════════
COPY examples/spark-rapids/jars/rapids-4-spark_2.12-25.02.1.jar ${SPARK_HOME}/jars/

# ══════════════════════════════════════════════
#  GVirtuS + Spark configuration
# ══════════════════════════════════════════════
ENV GVIRTUS_CONFIG=/etc/gvirtus/gvirtus.properties
ENV LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}"

RUN mkdir -p /etc/gvirtus
COPY examples/spark-rapids/docker/conf/gvirtus.properties  /etc/gvirtus/gvirtus.properties
COPY examples/spark-rapids/docker/conf/spark-defaults.conf  ${SPARK_HOME}/conf/spark-defaults.conf
COPY examples/spark-rapids/docker/conf/spark-env.sh         ${SPARK_HOME}/conf/spark-env.sh
RUN chmod +x ${SPARK_HOME}/conf/spark-env.sh

# ══════════════════════════════════════════════
#  Application code
# ══════════════════════════════════════════════
WORKDIR /app
COPY examples/spark-rapids/src/ /app/
COPY examples/spark-rapids/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
