FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS
ENV OPENPOSE_HOME=/opt/openpose
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}


# Prevent tzdata prompt and install dependencies
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    apt-get update && apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libboost-all-dev \
    libhdf5-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libopenblas-dev \
    liblmdb-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# # Install Python packages
RUN pip3 install opencv-python

# Install UCX 1.20 (same release used by the dev backend image) so GVirtuS's
# UCX communicator can compile. The tarball contains three .deb files at root
# level (no top-level dir), so we install them directly via apt to handle
# dependency resolution automatically.
WORKDIR /tmp
RUN apt-get update && \
    wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 && \
    tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 && \
    apt-get install -y ./ucx-1.20.0.deb ./ucx-cuda-1.20.0.deb && \
    rm -f ucx-*.deb ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 && \
    rm -rf /var/lib/apt/lists/*

# Embed local GVirtuS source (our optimized fork with UCX support) — NOT the
# upstream git clone, since that doesn't have our UCX communicator, Fase 4/5
# zero-copy, WriteIovRma argmax, etc.
WORKDIR /opt
COPY gvirtus_src.tar.gz /tmp/gvirtus_src.tar.gz
RUN mkdir -p /opt/GVirtuS_src && \
    cd /opt/GVirtuS_src && \
    tar xzf /tmp/gvirtus_src.tar.gz && \
    rm /tmp/gvirtus_src.tar.gz && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install

# Clone and build OpenPose
WORKDIR /opt
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Copy custom Cuda.cmake
COPY cmake/Cuda.cmake ${OPENPOSE_HOME}/cmake/Cuda.cmake

# Copy patched Caffe (after OpenPose clone)
COPY caffe/ ${OPENPOSE_HOME}/3rdparty/caffe/

# Pre-staged BODY_25 model (face/hand downloads disabled via cmake flags).
# Required because CMU's model server returns empty files (HTTP 22).
COPY models/ ${OPENPOSE_HOME}/models/

WORKDIR ${OPENPOSE_HOME}/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CAFFE=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_TESTS=OFF \
    -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF \
    -DDOWNLOAD_BODY_COCO_MODEL=OFF \
    -DDOWNLOAD_BODY_MPI_MODEL=OFF \
    -DCUDA_INCLUDE_DIRS="${GVIRTUS_HOME}/include;/usr/local/cuda-12.6/include" \
    -DCMAKE_CXX_FLAGS="-L${GVIRTUS_HOME}/lib -L${GVIRTUS_HOME}/lib/frontend" \
    && make -j"$(nproc)"

