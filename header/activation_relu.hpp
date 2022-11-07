#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include "../header/tensor.hpp"

class Activation_ReLU {

    public:

        Activation_ReLU(){}

        // Getter.
        Tensor& getOutput() { return this->output; }
        Tensor& getDinputs() { return this->dinputs; }


        // Forward pass.
        void forward(Tensor &inputs);

        // Backward pass.
        void backward(Tensor &dvalues);

    private:
        Tensor output, inputs;
        Tensor dinputs;
};

#endif