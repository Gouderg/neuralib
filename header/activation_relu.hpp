#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include "../header/tensor_inline.hpp"

class Activation_ReLU {

    public:

        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const { return this->dinputs; }


        // Forward pass.
        void forward(const TensorInline &inputs);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline output, inputs;
        TensorInline dinputs;
};

#endif