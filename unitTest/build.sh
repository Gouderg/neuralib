#/bin/bash

mkdir -p build
cd build
cxxtestgen --error-printer -o unitTest.cpp ../tensorinline_basic_operation.h
g++ -o unitTest -fopenmp unitTest.cpp ../../src/tensor_inline.cpp
./unitTest --help-tests
./unitTest
cd ..
rm -rf build