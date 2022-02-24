#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <ctime>


class Layer_Dense {
    
    public:
        // Constructor.
        Layer_Dense(const int n_inputs, const int n_neurons);


        // Setter.
        void setBiases(const std::vector<double> biases) {this->biases = biases;}
        void setWeight(const std::vector<std::vector<double>> weight) {this->weight = weight;}

        // Getter.
        std::vector<double> getBiases() {return this->biases;}
        std::vector<std::vector<double>> getWeight() {return this->weight;}

        // Forward pass.
        void forward(std::vector<std::vector<double>> inputs);

        // Useful function.
        static std::vector<std::vector<double>> transposition(std::vector<std::vector<double>> matrix);
        static std::vector<std::vector<double>> dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

    private:
        std::vector<double> biases;
        std::vector<std::vector<double>> weight, output;
};