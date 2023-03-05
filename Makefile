all: lib run_tests clean

# Rules to build your targets
lib: src/tensor_inline.cpp

# A rule that runs the unit tests
run_tests: runner
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