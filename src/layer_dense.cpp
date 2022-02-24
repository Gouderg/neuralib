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

std::vector<std::vector<double>> Layer_Dense::transposition(std::vector<std::vector<double>> matrix) {

    std::vector<std::vector<double>> output;
    
    // Run through each column.
    for (int j = 0; j < matrix[0].size(); j++) {
        std::vector<double> line_output;
        
        // Run through each line.
        for (int i = 0; i < matrix.size(); i++) {
           line_output.push_back(matrix[i][j]);
        }

        output.push_back(line_output);
    }
    
    return output;
}

std::vector<std::vector<double>> Layer_Dense::dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
    
    std::vector<std::vector<double>> output;
    
    for (int i = 0; i < v1.size(); i ++) {
        for (int j = 0; j < v1[i].size(); j++) {
            output[i][j] = 0;
            for (int k = 0; k < v1[i].size(); k++) {
                output[i][j] += v1[i][k] * v2[k][j];
            }
        }
    }
    return output;    

}