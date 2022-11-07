#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <cmath>

#include "../header/tensor.hpp"

class Activation_Softmax {
    
    public:

        Activation_Softmax(){}

        // Getter.
        Tensor& getOutput() { return this->output; }
        Tensor& getDinputs() { return this->dinputs; }


        // Forward pass.
        void forward(Tensor& inputs);

        // Backward pass.
        void backward(Tensor &dvalues);

    private:
        Tensor output, inputs, dinputs;
};

#endif