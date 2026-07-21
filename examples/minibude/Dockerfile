FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libmlx5-1 \
    ibverbs-providers \
    libgtest-dev \
    liblz4-dev \
    cmake \
    wget \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install UCX with mlx5 + CUDA support
RUN wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && dpkg -i ucx-*.deb \
    && rm -rf ucx-1.20.0* \
    && ldconfig

WORKDIR /opt
COPY . /opt/GVirtuS
RUN cd GVirtuS && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install

# Fetch + configure miniBUDE so the image is self-contained.
RUN bash ${GVIRTUS_HOME}/examples/minibude/setup.sh || true

WORKDIR /opt/GVirtuS
