FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libmlx5-1 \
    ibverbs-providers \
    libgtest-dev \
    liblz4-dev \
    git \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install UCX with mlx5 support (MOFED build — includes libuct_ib_mlx5.so)
RUN wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && dpkg -i ucx-1.20.0.deb \
    && rm -rf ucx-1.20.0* \
    && ldconfig

WORKDIR /opt
RUN git clone https://github.com/ecn-aau/GVirtuS.git && \
    cd GVirtuS && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install

WORKDIR /opt/GVirtuS