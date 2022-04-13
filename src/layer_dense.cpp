#include "../header/layer_dense.hpp"

Layer_Dense::Layer_Dense(const int n_inputs, const int n_neurons) {

    // Initialize weights.
    for (int i = 0; i < n_inputs; i ++) {
        std::vector<double> line_weight;
        for (int j = 0; j < n_neurons; j++) {
            line_weight.push_back(0.0000001 * (rand() % 200000 - 100000));
        }
        this->weight.push_back(line_weight);
    }
    
    // Initialize biases;
    for (int i = 0; i < n_neurons; i++) {
        this->biases.push_back(0);
    }

}

// void Layer_Dense::forward(std::vector<std::vector<double>> inputs) {

//     std::cout << "Hello";
// }