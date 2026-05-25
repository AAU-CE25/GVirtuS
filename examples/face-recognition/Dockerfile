# Frontend para face-recognition bajo GVirtuS+UCX.
# Reusa la imagen de opencv-yolo (CUDA 12.6 + GVirtuS frontend stubs) y le anade
# PyTorch CPU para entrenamiento y para cargar el .pth en cnn.py.
FROM ll33pq/gvirtus-frontend/opencv-yolo:cuda12.6

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/opt/GVirtuS

# python3 y pip ya estan en la imagen base. Anadimos torch CPU (mas ligero que
# torch CUDA, ~200MB) - solo se usa para cargar/extraer pesos del .pth en
# python; la inferencia real corre por libextension.so + GVirtuS stubs.
RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir numpy pillow

WORKDIR /app
ENTRYPOINT ["/bin/bash"]
