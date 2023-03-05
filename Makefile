# A makefile to perform all the unit test.

all: unitApp clean

# A rule that runs the unit tests
unitApp: runner
		./runner --help-tests
		./runner 

# How to build the test runner
runner: runner.cpp src/tensor_inline.cpp
		g++ -o $@ -fopenmp $^

# How to generate the test runner
runner.cpp: unitTest/tensorinline_basic_operation.h unitTest/tensorinline_complex_operation.h
		cxxtestgen -o $@ --error-printer $^

clean:
	rm runner runner.cpp