#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include "../header/tensor.hpp"

class Activation_ReLU {

    public:

        Activation_ReLU(){}

        // Getter.
        Tensor& getOutput() { return this->output; }

        // Forward pass.
        void forward(Tensor &inputs);

    private:
        Tensor output;
};

#endif