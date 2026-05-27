#! /bin/bash
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}
export UCX_DIAG=${UCX_DIAG:-0}

if [[ "${UCX_DIAG}" == "1" ]]; then
	if command -v ucx_info >/dev/null 2>&1; then
		echo "UCX_DIAG: ucx_info -d (filtered)" >&2
		ucx_info -d 2>/dev/null | grep -E "(rc|mlx5|cuda|gdr|dmabuf)" || true
		echo "UCX_DIAG: ucx_info -f (filtered)" >&2
		ucx_info -f 2>/dev/null | grep -E "UCX_MEMTYPE|UCX_RNDV|UCX_PROTO|UCX_TLS" || true
	else
		echo "UCX_DIAG: ucx_info not found; skipping diagnostics." >&2
	fi
fi

mkdir -p gvirtus/build && cd gvirtus/build && cmake .. && make -j$(nproc) && make install

BACKEND_CONFIG="${BACKEND_CONFIG:-${GVIRTUS_HOME}/etc/properties_ucx.json}"
echo "Backend config: ${BACKEND_CONFIG}" >&2
${GVIRTUS_HOME}/bin/gvirtus-backend "${BACKEND_CONFIG}"
