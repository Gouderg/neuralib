#ifndef ACTIVATION_SIGMOID_H
#define ACTIVATION_SIGMOID_H

#include "../header/tensor_inline.hpp"
#include "../header/layer.hpp"

class Activation_Sigmoid : public Layer {

    public:

        // Forward pass.
        void forward(const TensorInline &inputs, const bool training = false);

        // Backward pass.
        void backward(const TensorInline &dvalues);

    private:
        TensorInline inputs;

};
#endif