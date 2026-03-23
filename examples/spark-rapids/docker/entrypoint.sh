#!/bin/bash
# docker/entrypoint.sh

set -e

BACKEND_HOST="${GVIRTUS_BACKEND_HOST:-24.24.24.1}"
BACKEND_PORT="${GVIRTUS_BACKEND_PORT:-2222}"
SCALE="${SCALE_FACTOR:-1}"

echo ""
echo "============================================"
echo "  Spark + GVirtuS Frontend"
echo "  Backend: ${BACKEND_HOST}:${BACKEND_PORT}"
echo "  Scale:   SF${SCALE}"
echo "============================================"

# ── Verify frontend libs exist ──
if [ ! -f "${GVIRTUS_HOME}/lib/frontend/libcudart.so" ]; then
    echo "  ❌ GVirtuS frontend not found at ${GVIRTUS_HOME}"
    exit 1
fi
echo "  ✅ GVirtuS frontend installed"

# ── Test backend connectivity ──
echo "  Testing backend connection..."
RETRIES=10
for i in $(seq 1 $RETRIES); do
    if nc -z -w3 "${BACKEND_HOST}" "${BACKEND_PORT}" 2>/dev/null; then
        echo "  ✅ Backend reachable"

        # Set LD_PRELOAD to intercept CUDA
        export LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so"
        export LD_LIBRARY_PATH="${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}"
        echo "  ✅ LD_PRELOAD active"

        echo "============================================"
        echo ""

        exec spark-submit \
            --master local[*] \
            --driver-memory "${SPARK_DRIVER_MEMORY:-4g}" \
            --conf spark.jars="${SPARK_HOME}/jars/rapids-4-spark_2.12-25.02.1.jar" \
            /app/pipeline_gvirtus.py
    fi
    echo "    Attempt ${i}/${RETRIES}... waiting 3s"
    sleep 3
done

# ── Fallback: CPU only ──
echo "  ⚠️  Backend unreachable — running CPU-only"
echo "============================================"
echo ""

exec spark-submit \
    --master local[*] \
    --driver-memory "${SPARK_DRIVER_MEMORY:-4g}" \
    /app/pipeline_spark.py