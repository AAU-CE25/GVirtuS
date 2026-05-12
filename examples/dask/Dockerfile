## Dask + CuPy matrix multiplication on GVirtuS frontend
## No physical GPU needed — CUDA calls are forwarded over TCP via GVirtuS.
##
## Build:
##   docker build -t gvirtus-dask .
##
## Run (point at your GVirtuS backend):
##   docker run --rm \
##     -v /path/to/properties.json:/opt/GVirtuS/etc/properties.json \
##     gvirtus-dask --size 4096 --chunks 1024

FROM aauce25/gvirtus-frontend:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /app

# main.py is NOT baked in — mount it at runtime so edits don't require a rebuild:
#   docker run -v $(pwd)/examples/dask:/app ...
# (or use: make run-dask)

ENV GVIRTUS_LOGLEVEL=20000

ENTRYPOINT ["python3", "main.py"]
