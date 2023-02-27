#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include "../header/tensor_inline.hpp"

class Activation_ReLU {

    public:

        // Getter.
        TensorInline& getOutput() { return this->output; }
        TensorInline& getDinputs() { return this->dinputs; }


        // Forward pass.
        void forward(TensorInline &inputs);

        // Backward pass.
        void backward(TensorInline dvalues);

    private:
        TensorInline output, inputs;
        TensorInline dinputs;
};

#endif