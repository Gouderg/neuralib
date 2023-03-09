# A makefile to perform all the unit test.
test: unitApp clean

# A rule that runs the unit tests
unitApp: runner
		./runner --help-tests
		./runner 

# How to build the test runner
runner: runner.cpp src/tensor_inline.cpp src/loss.cpp src/loss_binary_crossentropy.cpp
		g++ -o $@ -fopenmp $^

# How to generate the test runner
runner.cpp: tests/test_tensorinline_basic_operation.h tests/test_tensorinline_complex_operation.h tests/test_loss_binary_crossentropy.h
		cxxtestgen -o $@ --error-printer $^

clean:
	rm runner runner.cpp