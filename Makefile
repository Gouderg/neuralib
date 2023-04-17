# A makefile to perform all the unit test.
test: unitApp clean


TESTS_H = $(wildcard tests/*.h)
SRC = $(filter-out src/plot.cpp src/statistic.cpp src/model.cpp, $(wildcard src/*.cpp))

# A rule that runs the unit tests
unitApp: runner
		./runner --help-tests
		./runner 

# How to build the test runner
runner: runner.cpp $(SRC)
		g++ -o $@ -fopenmp $^

# How to generate the test runner
runner.cpp: $(TESTS_H)
		cxxtestgen -o $@ --error-printer $^

clean:
	rm runner runner.cpp