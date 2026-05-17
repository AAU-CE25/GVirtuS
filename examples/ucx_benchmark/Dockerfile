# Base on the dev image so we inherit the EXACT same UCX (1.20.0 built from
# source). Apt's libucx-dev (1.12.x) is wire-incompatible with the dev image
# and causes `ERROR failed to unpack address, invalid bandwidth 0.00` during
# endpoint wireup. CUDA, cuDNN, log4cplus, ibverbs, lz4 are all already in
# the dev image too.
FROM aauce25/gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

# Override the dev image's entrypoint so we can run interactive bench commands.
ENTRYPOINT []

RUN apt update && apt install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# Build the local GVirtuS source against the inherited UCX 1.20.0.
COPY . /opt/GVirtuS-src
RUN cd /opt/GVirtuS-src && mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install && \
    cd /opt && rm -rf /opt/GVirtuS-src

COPY examples/ucx_benchmark ${GVIRTUS_HOME}/examples/ucx_benchmark/

WORKDIR ${GVIRTUS_HOME}/examples/ucx_benchmark
RUN chmod +x frontend.sh run_benchmarks.sh frontend_matrix_only.sh
