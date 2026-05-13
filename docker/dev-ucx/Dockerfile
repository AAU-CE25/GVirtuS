FROM aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libucx-dev \
    ucx-utils \
    rdma-core \
    ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*
