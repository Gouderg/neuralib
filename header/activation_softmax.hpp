#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <cmath>

#include "../header/tensor.hpp"

class Activation_Softmax {
    
    public:

        Activation_Softmax(){}

        // Getter.
        Tensor& getOutput() { return this->output; }

        // Forward pass.
        void forward(Tensor& inputs);

    private:
        Tensor output;
};

#endif