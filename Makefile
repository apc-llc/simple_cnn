CXXFLAGS = -g -std=c++11 -O3 -ffast-math -I.

default all: example1 example2

example1: examples/example1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

example2: examples/example2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf example1 example2

