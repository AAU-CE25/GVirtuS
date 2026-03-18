CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

TARGETS = helloServer helloClient

.PHONY: all clean

all: $(TARGETS)

helloServer: helloServer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

helloClient: helloClient.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
