## ===== Stage 1: Build GVirtuS from source =====
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS builder

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
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/GVirtuS
WORKDIR /opt/GVirtuS
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install

## ===== Stage 2: Frontend runtime (minimal, no CUDA toolkit needed) =====
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib

RUN apt update && apt install -y --no-install-recommends \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    liblz4-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only frontend-relevant libraries and config
COPY --from=builder ${GVIRTUS_HOME}/lib/libgvirtus-common.so          ${GVIRTUS_HOME}/lib/
COPY --from=builder ${GVIRTUS_HOME}/lib/libgvirtus-communicators*.so   ${GVIRTUS_HOME}/lib/
COPY --from=builder ${GVIRTUS_HOME}/lib/libgvirtus-frontend.so         ${GVIRTUS_HOME}/lib/
COPY --from=builder ${GVIRTUS_HOME}/lib/frontend/                      ${GVIRTUS_HOME}/lib/frontend/
COPY --from=builder ${GVIRTUS_HOME}/etc/                               ${GVIRTUS_HOME}/etc/
COPY --from=builder ${GVIRTUS_HOME}/include/                           ${GVIRTUS_HOME}/include/

WORKDIR /opt/GVirtuS
