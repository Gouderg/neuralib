#pragma once

#include "../header/tensor.hpp"

class Layer_Dense {

    public:

        // Constructor.
        Layer_Dense(){};
        Layer_Dense(const int n_inputs, const int n_neurons);

        // Forward pass.
        void forward(Tensor inputs);


    private:
        Tensor weights, biases, output;
};