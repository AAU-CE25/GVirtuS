CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

TARGET = helloHPC

.PHONY: all clean

all: $(TARGET)

$(TARGET): helloHPC.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
