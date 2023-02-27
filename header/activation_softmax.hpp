#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <cmath>

#include "../header/tensor_inline.hpp"

class Activation_Softmax {
    
    public:

        // Getter.
        TensorInline& getOutput() { return this->output; }
        TensorInline& getDinputs() { return this->dinputs; }


        // Forward pass.
        void forward(TensorInline& inputs);

        // Backward pass.
        void backward(TensorInline &dvalues);

    private:
        TensorInline output, inputs, dinputs;
};

#endif