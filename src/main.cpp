#include "../header/layer_dense.hpp"


int main(int argc, char const *argv[]) {
    srand(time(NULL));

    Layer_Dense dense1(2, 3);

    std::vector<std::vector<double>> input = {{1.0, 2.0, 3.0, 2.5},{2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}};
    std::vector<std::vector<double>> weight = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    std::vector<double> biases = {2.0, 3.0, 0.5};
    std::vector<std::vector<double>> output;
    weight = Layer_Dense::transposition(weight);

    output = Layer_Dense::dot(input, weight);

    for (auto elt: output) {
        for (auto val: elt) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
