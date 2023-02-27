#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <cmath>

#include "../header/tensor_inline.hpp"

class Activation_Softmax {
    
    public:

        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const { return this->dinputs; }


        // Forward pass.
        void forward(const TensorInline& inputs);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline output, inputs, dinputs;
};

#endif