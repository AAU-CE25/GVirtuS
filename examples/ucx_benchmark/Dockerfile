FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

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
    libucx-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# Copy the LOCAL gvirtus source tree (build context is repo root).
# This avoids the stale `git clone` of the github master branch.
COPY . /opt/GVirtuS-src
RUN cd /opt/GVirtuS-src && mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install && \
    cd /opt && rm -rf /opt/GVirtuS-src

COPY examples/ucx_benchmark ${GVIRTUS_HOME}/examples/ucx_benchmark/

WORKDIR ${GVIRTUS_HOME}/examples/ucx_benchmark
RUN chmod +x frontend.sh run_benchmarks.sh frontend_matrix_only.sh
