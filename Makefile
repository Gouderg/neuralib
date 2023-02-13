all: opti clean

opti: main.o tensor.o tensor_inline.o
	g++ $^ -o $@ -fopenmp -march=znver2 -ffast-math

main.o: main.cpp header/tensor.hpp
	g++ -Wall -std=c++17 -O3 -g -c $<

tensor.o: src/tensor.cpp header/tensor.hpp
	g++ -std=c++17 -O3 -c -g  $<

tensor_inline.o: src/tensor_inline.cpp header/tensor_inline.hpp
	g++ -Wall -std=c++17 -O3 -c -g -fopenmp $<

clean:
	rm *.o

r: clean opti