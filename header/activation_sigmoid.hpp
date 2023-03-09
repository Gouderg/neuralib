#ifndef ACTIVATION_SIGMOID_H
#define ACTIVATION_SIGMOID_H

#include "tensor_inline.hpp"
class Activation_Sigmoid {

    public:
        // Getter.
        const TensorInline& getOutput() const { return this->output; }
        const TensorInline& getDinputs() const { return this->dinputs; }


        // Forward pass.
        void forward(const TensorInline &inputs);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline inputs, output;
        TensorInline dinputs;

};
#endif