#include "header/tensor.hpp"
#include "header/dataset.hpp"

int main(int argc, char const *argv[]) {   

    Tensor X, y;

    std::tie(X, y) = Dataset::spiral_data(10, 4);

    Tensor t1(3,4);
    Tensor t2(3,4);

    std::cout << t1 + t2 << std::endl;
    return 0;
}
