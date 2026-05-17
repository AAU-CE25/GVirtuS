# Benchmark Makefile — builds binaries into ./build/ (gitignored).
#
# Usage:
#   make                 # build everything (TCP + UCX)
#   make tcp             # plain TCP only (no UCX dependency)
#   make ucx             # UCX-enabled data_copy_bench + transport_probe
#   make clean
#
# Sweep helpers (run on client side; start the matching server on the peer):
#   make server-tcp PORT=7778
#   make server-ucx PORT=7778
#   make sweep-tcp      SERVER=24.24.24.2 PORT=7778 RUNS=20
#   make sweep-ucx-tcp  SERVER=24.24.24.2 PORT=7778 RUNS=20
#   make sweep-ucx-rdma SERVER=24.24.24.2 PORT=7778 RUNS=20
#   make sweep-all      SERVER=24.24.24.2 PORT=7778 RUNS=20   # all three into one CSV

BUILD_DIR := build
RESULTS_DIR := results
CXX       ?= g++
CXXFLAGS  ?= -O2 -Wall

UCX_LIBS  := -lucp -lucs
PTHREAD   := -lpthread

TCP_BIN   := $(BUILD_DIR)/data_copy_bench
UCX_BIN   := $(BUILD_DIR)/data_copy_bench_ucx
PROBE_BIN := $(BUILD_DIR)/transport_probe

# Sweep parameters (override on command line)
PORT    ?= 7778
SERVER  ?= 24.24.24.2
RUNS    ?= 20
RESULTS ?= $(RESULTS_DIR)/results_$(shell date +%Y%m%d_%H%M%S).csv

.PHONY: all tcp ucx probe clean \
        server-tcp server-ucx \
        sweep-tcp sweep-ucx-tcp sweep-ucx-rdma sweep-all \
        plot

all: tcp ucx probe

tcp: $(TCP_BIN)
ucx: $(UCX_BIN) $(PROBE_BIN)
probe: $(PROBE_BIN)

$(BUILD_DIR):
	@mkdir -p $@

$(TCP_BIN): data_copy_bench.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(PTHREAD)

$(UCX_BIN): data_copy_bench.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_UCX -o $@ $< $(UCX_LIBS) $(PTHREAD)

$(PROBE_BIN): transport_probe.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(UCX_LIBS) $(PTHREAD)

clean:
	rm -rf $(BUILD_DIR)

# --- Server targets (build + run) ---------------------------------------------
# Use these on the GPU/server node to expose the echo endpoint.
# Assumes the working tree is already up to date (sync via git before running).

server-tcp: $(TCP_BIN)
	@echo "[server] TCP on port $(PORT)"
	$(TCP_BIN) server tcp $(PORT)

server-ucx: $(UCX_BIN)
	@echo "[server] UCX on port $(PORT) (set UCX_TLS / UCX_IB_GID_INDEX as needed)"
	UCX_IB_GID_INDEX=1 $(UCX_BIN) server ucx $(PORT)

# --- Client sweeps -------------------------------------------------------------

sweep-tcp: $(TCP_BIN)
	./sweep.sh tcp $(SERVER) $(PORT) $(RESULTS) $(RUNS)

sweep-ucx-tcp: $(UCX_BIN)
	./sweep.sh ucx-tcp $(SERVER) $(PORT) $(RESULTS) $(RUNS)

sweep-ucx-rdma: $(UCX_BIN)
	./sweep.sh ucx-rdma $(SERVER) $(PORT) $(RESULTS) $(RUNS)

# Runs all three transports into a single CSV. NOTE: you must restart the
# server between tcp and ucx variants (different binary).
sweep-all: $(TCP_BIN) $(UCX_BIN)
	@echo "[sweep-all] writing combined CSV to $(RESULTS)"
	./sweep.sh tcp       $(SERVER) $(PORT) $(RESULTS) $(RUNS)
	./sweep.sh ucx-tcp   $(SERVER) $(PORT) $(RESULTS) $(RUNS)
	./sweep.sh ucx-rdma  $(SERVER) $(PORT) $(RESULTS) $(RUNS)
	@echo "[sweep-all] done: $(RESULTS)"

plot:
	@if [ -z "$(CSV)" ]; then echo "usage: make plot CSV=path/to/results.csv [PNG=out.png]"; exit 1; fi
	./plot.sh $(CSV) $(PNG)
