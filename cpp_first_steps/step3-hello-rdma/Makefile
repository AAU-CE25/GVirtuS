CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
LDFLAGS  = -libverbs

TARGETS = ce-server ce-client

.PHONY: all clean

all: $(TARGETS)

ce-server: ce-server.cpp rdma_common.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

ce-client: ce-client.cpp rdma_common.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
