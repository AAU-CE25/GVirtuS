## This Dockerfile is for building a GVirtuS image with CUDA support.
## It does not include the Gvirtus source code, as it is expected to be mounted at runtime (useful for development).
## The image will be pushed at taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 using the `make docker-build-push-dev` command.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Remove broken NVIDIA apt repo (not needed — CUDA already in base image)
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
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
    cmake \
    wget \
    # Dev-only: RDMA/network discovery tools (make ucx-discover-docker)
    rdma-core \
    infiniband-diags \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Install UCX with mlx5 + CUDA support (install all UCX debs from the bundle)
RUN wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && dpkg -i ucx-*.deb \
    && rm -rf ucx-1.20.0* \
    && ldconfig

ENV GVIRTUS_HOME=/usr/local/gvirtus
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}

COPY ./docker/dev/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]