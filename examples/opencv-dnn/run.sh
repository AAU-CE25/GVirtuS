export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
export GVIRTUS_LOGLEVEL=50000

nvcc main.cu -o sample -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda -lcublas -lcudnn -lcudart `pkg-config --cflags --libs opencv4`

./sample