FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    patch \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    libmlx5-1 \
    ibverbs-providers \
    rdma-core \
    ibverbs-utils \
    libgtest-dev \
    liblz4-dev \
    libnuma1 \
    cmake \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && dpkg -i ucx-1.20.0.deb \
    && rm -rf ucx-1.20.0* \
    && ldconfig

RUN python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools \
    && python3 -m pip install --no-cache-dir \
        "numpy<2" \
        "Cython>=0.29.32,<3" \
        fastrlock

ARG CUPY_VERSION=v13.3.0
ENV CUDA_PATH=/usr/local/cuda
ENV NVCC=/usr/local/cuda/bin/nvcc
ENV CUPY_DYNAMIC_CUDART=1

COPY ./docker/dev/RAPIDS/cupy_dynamic_cudart.patch /tmp/cupy_dynamic_cudart.patch
COPY ./docker/dev/RAPIDS/patches/cupy_launch_extra.py /tmp/cupy_launch_extra.py

RUN git clone --branch ${CUPY_VERSION} --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/cupy/cupy.git /tmp/cupy \
    && cd /tmp/cupy \
    && patch -p1 < /tmp/cupy_dynamic_cudart.patch \
    && python3 /tmp/cupy_launch_extra.py /tmp/cupy \
    && echo "[ok] both cupy patches applied"

RUN cd /tmp/cupy \
    && CUPY_NUM_BUILD_JOBS=$(nproc) \
       CUPY_INSTALL_USE_HIP=0 \
       python3 -m pip install -v --no-build-isolation .

RUN cd / && python3 -c "import cupy; print(cupy.__version__, cupy.__file__)"

# Verify cudart is now imported DYNAMICALLY: undefined cudart symbols should appear
RUN echo "=== runtime.cpython UNDEFINED cuda* symbols (should NOT be empty) ===" \
    && nm -D --undefined-only /usr/local/lib/python3.10/dist-packages/cupy_backends/cuda/api/runtime.cpython-*.so \
       | grep -i "cuda\|cudart" | head -10 \
    && echo "=== ldd of runtime.cpython (libcudart.so should appear) ===" \
    && ldd /usr/local/lib/python3.10/dist-packages/cupy_backends/cuda/api/runtime.cpython-*.so | grep -i cudart

RUN rm -rf /tmp/cupy /root/.cache /tmp/pip-* /tmp/cupy_dynamic_cudart.patch

ENV GVIRTUS_HOME=/usr/local/gvirtus
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/build/plugins/cudadr:${GVIRTUS_HOME}/build/plugins/cudart:${GVIRTUS_HOME}/build/plugins/cublas:${GVIRTUS_HOME}/build:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY ./docker/dev/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
