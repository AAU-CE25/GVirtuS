#! /bin/bash
export GVIRTUS_LOGLEVEL=${GVIRTUS_LOGLEVEL:-20000}  # default to INFO if not set

mkdir gvirtus/build && cd gvirtus/build && cmake .. && make -j$(nproc) && make install
 ${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties_ucx.json
#tail -f /dev/null # for debugging