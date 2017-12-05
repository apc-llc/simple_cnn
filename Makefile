CXXFLAGS = -g -std=c++11 -O3 -ffast-math -I.

default all: example1 example2

example1: Example_MNIST/example1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

example2: Example_MNIST/example2.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf example1 example2

