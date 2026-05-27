# Frontend para opencv-yolo bajo GVirtuS+UCX.
# Base: openpose (CUDA 12.6 + cuDNN). Recompilamos OpenCV 4.10.0 con DNN_BACKEND_CUDA.
# Los modulos opencv_cuda* (cudawarping, cudaarithm, etc) NO los usa yolo --
# solo necesita opencv_dnn (que es separado) -- y rompen el link porque la imagen
# base tiene un cudart antiguo sin cudaDestroyTextureObject.
FROM ul11nh/openpose_gvirtus:cuda12.6

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config \
      libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
      libv4l-dev libxvidcore-dev libx264-dev \
      libjpeg-dev libpng-dev libtiff-dev gfortran \
      libatlas-base-dev wget \
      liblog4cplus-dev \
      librdmacm-dev libibverbs-dev libmlx5-1 ibverbs-providers \
      liblz4-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCV 4.10.0 con DNN-CUDA, todos los opencv_cuda* OFF (yolo no los usa).
RUN cd /tmp && \
    git clone --depth=1 --branch 4.10.0 https://github.com/opencv/opencv.git && \
    git clone --depth=1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN="8.9" \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D WITH_CUBLAS=ON \
        -D CUDA_ARCH_PTX="" \
        -D BUILD_opencv_dnn=ON \
        -D BUILD_opencv_cudev=ON \
        -D BUILD_opencv_cudaarithm=OFF \
        -D BUILD_opencv_cudaimgproc=OFF \
        -D BUILD_opencv_cudaphoto=OFF \
        -D BUILD_opencv_cudawarping=OFF \
        -D BUILD_opencv_cudabgsegm=OFF \
        -D BUILD_opencv_cudafilters=OFF \
        -D BUILD_opencv_cudaobjdetect=OFF \
        -D BUILD_opencv_cudafeatures2d=OFF \
        -D BUILD_opencv_cudastereo=OFF \
        -D BUILD_opencv_cudaoptflow=OFF \
        -D BUILD_opencv_cudacodec=OFF \
        -D BUILD_opencv_cudalegacy=OFF \
        -D BUILD_opencv_photo=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_opencv_python2=OFF \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && rm -rf opencv opencv_contrib

# UCX 1.20
RUN wget -q https://github.com/openucx/ucx/releases/download/v1.20.0/ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && tar xf ucx-1.20.0-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2 \
    && dpkg -i ucx-*.deb \
    && rm -rf ucx-1.20.0* \
    && ldconfig

WORKDIR /opt
COPY . /opt/GVirtuS-src
RUN cd /opt/GVirtuS-src && mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${GVIRTUS_HOME} .. && \
    make -j$(nproc) && make install && \
    rm -rf /opt/GVirtuS-src/build

# Sanity check ligero: archivos clave existen + opencv4.pc cita CUDA libs.
# No usamos python3 cv2 porque carga TODOS los modulos opencv y dispara
# rabbit-holes de simbolos no relacionados con DNN-CUDA.
RUN ls -la ${GVIRTUS_HOME}/lib/frontend/libcudnn.so* ${GVIRTUS_HOME}/lib/frontend/libcublas.so* && \
    ls -la /usr/local/lib/libopencv_dnn.so* /usr/local/lib/libopencv_core.so* && \
    pkg-config --modversion opencv4 && \
    pkg-config --libs opencv4 | grep -q opencv_dnn && \
    echo "OpenCV+DNN+GVirtuS stubs OK"

WORKDIR /app
ENTRYPOINT ["/bin/bash"]
