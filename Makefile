all: opti clean

opti: main.o tensor.o tensor_inline.o
	g++ $^ -o $@ -fopenmp -march=znver2 -ffast-math -mavx2

main.o: main.cpp header/tensor.hpp
	g++ -Wall -std=c++17 -O3  -c $<

tensor.o: src/tensor.cpp header/tensor.hpp
	g++ -std=c++17 -O3 -c   $<

tensor_inline.o: src/tensor_inline.cpp header/tensor_inline.hpp
	g++ -Wall -std=c++17 -O3 -c  -fopenmp $<

clean:
	rm *.o

r: clean opti